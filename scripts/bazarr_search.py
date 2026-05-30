#!/usr/bin/env python3
"""Trigger Bazarr to search for / re-fetch a subtitle for a given episode.

Two-stage auth (because the Bazarr instance is behind Traefik + Authentik):

1. OAuth 2.0 password-grant to Authentik → access token
2. Bazarr API call with ``Authorization: Bearer <token>`` plus, when the
   outpost is in cookie-session mode, a verbatim ``Cookie:`` header from
   your browser session. Plus ``X-API-KEY`` for Bazarr's own auth.

Reads credentials from ``.env.auth`` in the project root:

    OAUTH_ACCESS_TOKEN_URL
    OAUTH_GRANT_TYPE
    OAUTH_CLIENT_ID
    OAUTH_SCOPE
    OAUTH_USERNAME
    OAUTH_PASSWORD
    BAZARR_API_KEY
    BAZARR_BASE_URL              # e.g. https://bazarr.example.com
    AUTHENTIK_COOKIE             # (optional) full ``Cookie:`` header value
                                 # grabbed from browser DevTools after login.

Usage:
    # OAuth-only check.
    py scripts/bazarr_search.py --dry-run

    # List episodes for a series so you can find an episode id.
    py scripts/bazarr_search.py --series-id 46 --list-episodes

    # Trigger a subtitle search for a specific episode.
    py scripts/bazarr_search.py --episode-id 2196

    # Or look up by series + season + episode, then trigger.
    py scripts/bazarr_search.py --series-id 46 --season 4 --episode 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


# ----- token cache -------------------------------------------------------- #
# Reuses the Authentik access token across invocations so we don't burn an
# OAuth round-trip per script run. ``.cache/`` is gitignored.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_CACHE = _PROJECT_ROOT / ".cache" / "bazarr_oauth_token.json"
_REFRESH_LEEWAY_SECONDS = 60  # refresh this many seconds before stated expiry


# ----- redaction ---------------------------------------------------------- #
# Console output may contain redirect URLs and response bodies. Mask anything
# token-shaped before printing so we can paste logs without leaking secrets.

_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OAuth-y query params (case-insensitive key match)
    (re.compile(
        r"((?:access_token|refresh_token|code|state|client_id|"
        r"client_secret|password|api_key|token)=)[^&\s\"'<>]+",
        re.IGNORECASE), r"\1<REDACTED>"),
    # JWTs (three base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
     "<REDACTED-JWT>"),
    # Long opaque tokens (40+ chars of base64/hex; conservative on length)
    (re.compile(r"\b[A-Za-z0-9_/+=-]{40,}\b"), "<REDACTED-TOKEN>"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _safe_print(s: str) -> None:
    print(_redact(s))


# ----- env helpers -------------------------------------------------------- #


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        sys.exit(f"missing required env var in .env.auth: {name}")
    return v


def _load_env() -> None:
    """Load `.env.auth` from the project root (sibling of scripts/)."""
    env_auth = _PROJECT_ROOT / ".env.auth"
    if not env_auth.exists():
        sys.exit(f"missing {env_auth}")
    load_dotenv(env_auth, override=False)


# ----- OAuth -------------------------------------------------------------- #


def _load_cached_token() -> str | None:
    """Return a non-expired cached access token, or None if absent/stale."""
    if not _TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(_TOKEN_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expires_at = float(data.get("expires_at", 0))
    if time.time() >= expires_at:
        return None
    token = data.get("access_token")
    return token if isinstance(token, str) and token else None


def _save_cached_token(token: str, expires_in: int) -> None:
    """Persist the access token + computed expiry. Refresh `_REFRESH_LEEWAY_SECONDS`
    before actual expiry to avoid races where a token expires mid-request.
    """
    _TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    expires_at = time.time() + max(0, expires_in - _REFRESH_LEEWAY_SECONDS)
    payload = {"access_token": token, "expires_at": expires_at}
    _TOKEN_CACHE.write_text(json.dumps(payload), encoding="utf-8")


def authenticate(client: httpx.Client, *, force_refresh: bool = False) -> str:
    """Authentik OAuth 2.0 password grant. Client auth is in the request body
    (per the Authentik client config). Returns the access token. Reuses a
    cached token from ``.cache/bazarr_oauth_token.json`` if non-expired.

    The HTTP call goes through the supplied ``httpx.Client``, so any
    ``Set-Cookie`` returned by the Authentik token endpoint (e.g. an
    ``authentik_proxy_*`` session cookie with ``Domain=eben0.com``) gets
    stored in the client's cookie jar and auto-forwarded to subsequent
    Bazarr calls. That's the cheapest way to satisfy the outpost without
    manually pasting a browser cookie.
    """
    if not force_refresh:
        cached = _load_cached_token()
        if cached:
            return cached
    token_url = _env("OAUTH_ACCESS_TOKEN_URL")
    body = {
        "grant_type": _env("OAUTH_GRANT_TYPE"),
        "client_id": _env("OAUTH_CLIENT_ID"),
        "scope": _env("OAUTH_SCOPE"),
        "username": _env("OAUTH_USERNAME"),
        "password": _env("OAUTH_PASSWORD"),
    }
    r = client.post(token_url, data=body, timeout=30.0)
    if r.status_code != 200:
        sys.exit(f"OAuth failed: {r.status_code} {_redact(r.text[:500])}")
    data = r.json()
    if "access_token" not in data:
        sys.exit("OAuth response missing access_token")
    token: str = data["access_token"]
    # ``expires_in`` is seconds-from-now per RFC 6749 §5.1.
    expires_in = int(data.get("expires_in", 3600))
    _save_cached_token(token, expires_in)
    # Log captured cookie names (not values) so the user can see whether
    # Authentik handed us session cookies we can forward to Bazarr.
    captured = list(client.cookies.keys())
    if captured:
        _safe_print(f"✓ OAuth response set cookies: {captured}")
    else:
        _safe_print(
            "⚠ OAuth response did not set any cookies — the Authentik "
            "outpost will likely reject the Bazarr call unless you also "
            "set AUTHENTIK_COOKIE in .env.auth."
        )
    return token


# ----- Bazarr ------------------------------------------------------------- #


def _bazarr_headers(access_token: str) -> dict[str, str]:
    """Headers for a Bazarr API call.

    Three layers of auth — all three are usually required when Bazarr sits
    behind Traefik + an Authentik outpost in cookie-session mode:

    * ``Authorization: Bearer <oauth_token>`` — Authentik token (some
      configurations accept it; most outposts also want a session cookie).
    * ``Cookie: authentik_proxy_...`` (optional, via ``AUTHENTIK_COOKIE``) —
      the session the Authentik outpost actually checks. Grab from your
      browser DevTools after logging in; expires per the outpost's session
      lifetime so refresh when 302s come back.
    * ``X-API-KEY: <bazarr_api_key>`` — Bazarr's own API gate.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-API-KEY": _env("BAZARR_API_KEY"),
        "Accept": "application/json",
    }
    cookie = os.getenv("AUTHENTIK_COOKIE")
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _get_json(client: httpx.Client, url: str, access_token: str) -> dict:
    r = client.get(url, headers=_bazarr_headers(access_token), timeout=30.0)
    _safe_print(f"GET {url} → {r.status_code}")
    if r.status_code != 200:
        sys.exit(_redact(r.text[:500]))
    ctype = r.headers.get("content-type", "").lower()
    if "json" not in ctype:
        _safe_print(
            f"⚠ response is not JSON (content-type={ctype!r}); first 400 "
            f"chars below — usually means the Authentik outpost intercepted "
            f"with a login page. Set AUTHENTIK_COOKIE in .env.auth (full "
            f"Cookie: header value from browser DevTools)."
        )
        _safe_print(r.text[:400])
        sys.exit(1)
    return r.json()


def list_episodes(
    client: httpx.Client, access_token: str, series_id: int,
) -> list[dict]:
    base = _env("BAZARR_BASE_URL").rstrip("/")
    url = f"{base}/api/episodes?seriesid[]={series_id}"
    return _get_json(client, url, access_token).get("data", [])


def _ep_id(ep: dict) -> int | None:
    """Bazarr response shape varies by version; try the common keys."""
    v = ep.get("sonarrEpisodeId") or ep.get("episodeId") or ep.get("id")
    return int(v) if v is not None else None


def print_episodes(eps: list[dict]) -> None:
    _safe_print(f"{len(eps)} episodes:")
    # Sort by (season, episode) so the listing is stable + scannable.
    for ep in sorted(eps, key=lambda e: (e.get("season", 0), e.get("episode", 0))):
        season = ep.get("season")
        episode = ep.get("episode")
        title = ep.get("title", "")
        _safe_print(f"  S{season:02d}E{episode:02d}  id={_ep_id(ep)}  {title}")


def resolve_episode_id(
    client: httpx.Client, access_token: str,
    series_id: int, season: int, episode: int,
) -> int:
    """Find the Bazarr/Sonarr episode id for ``series:season:episode``."""
    for ep in list_episodes(client, access_token, series_id):
        if ep.get("season") == season and ep.get("episode") == episode:
            ep_id = _ep_id(ep)
            if ep_id is not None:
                return ep_id
    sys.exit(
        f"Could not find S{season:02d}E{episode:02d} in series {series_id}."
    )


def trigger_search(
    client: httpx.Client, access_token: str, episode_id: int,
) -> None:
    """Ask Bazarr to download a subtitle for the given episode id.

    Hits Bazarr's manual-search-and-download path:
      POST /api/providers/episodes?episodeid=<id>

    This triggers an immediate search across enabled providers (whisperai
    among them). The Whisper provider then POSTs to our whisper-gend ASR.
    """
    base = _env("BAZARR_BASE_URL").rstrip("/")
    url = f"{base}/api/providers/episodes?episodeid={episode_id}"
    r = client.post(url, headers=_bazarr_headers(access_token), timeout=30.0)
    _safe_print(f"POST {url} → {r.status_code}")
    if r.status_code >= 400:
        _safe_print(_redact(r.text[:1000]))
        sys.exit(f"Bazarr request failed (HTTP {r.status_code}).")
    ctype = r.headers.get("content-type", "").lower()
    if "json" not in ctype:
        # A 200 with HTML almost always means the Authentik outpost served
        # its landing/login page — the POST never reached Bazarr. Failing
        # loudly is safer than pretending the search triggered.
        _safe_print(
            f"⚠ POST returned 200 but content-type={ctype!r} (not JSON). "
            f"The request likely never reached Bazarr — set AUTHENTIK_COOKIE "
            f"in .env.auth (full Cookie: header value from browser DevTools)."
        )
        _safe_print(_redact(r.text[:400]))
        sys.exit(2)
    _safe_print(_redact(r.text[:1000]))


# ----- entrypoint --------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="OAuth only; do not call Bazarr.")
    parser.add_argument(
        "--list-episodes", action="store_true",
        help="List episodes for --series-id and exit. Use to find episode ids.")
    parser.add_argument(
        "--series-id", type=int,
        help="Sonarr series id. Required for --list-episodes and for "
             "--season/--episode lookup.")
    parser.add_argument(
        "--season", type=int,
        help="Season number (with --series-id and --episode) to resolve the "
             "episode id by lookup.")
    parser.add_argument(
        "--episode", type=int,
        help="Episode number (with --series-id and --season) to resolve the "
             "episode id by lookup.")
    parser.add_argument(
        "--episode-id", type=int,
        help="Bazarr/Sonarr episode id to trigger a subtitle search for "
             "(skips the season+episode lookup).")
    parser.add_argument(
        "--force-refresh-token", action="store_true",
        help="Ignore the cached OAuth token and request a fresh one.")
    args = parser.parse_args()

    _load_env()
    # Single shared client: cookies set by the OAuth response (Authentik)
    # auto-forward to Bazarr requests when their domains share the parent
    # (e.g. ``Domain=eben0.com``). ``follow_redirects=True`` so 302s through
    # the outpost don't trip us up.
    with httpx.Client(follow_redirects=True) as client:
        access_token = authenticate(
            client, force_refresh=args.force_refresh_token,
        )
        _safe_print(f"✓ OAuth ok  (token length={len(access_token)})")

        if args.dry_run:
            return

        if args.list_episodes:
            if args.series_id is None:
                sys.exit("--list-episodes requires --series-id.")
            print_episodes(list_episodes(client, access_token, args.series_id))
            return

        if args.episode_id is not None:
            trigger_search(client, access_token, args.episode_id)
            return

        if (args.series_id is not None
                and args.season is not None
                and args.episode is not None):
            ep_id = resolve_episode_id(
                client, access_token,
                args.series_id, args.season, args.episode,
            )
            _safe_print(
                f"resolved series={args.series_id} "
                f"S{args.season:02d}E{args.episode:02d} → episode id {ep_id}"
            )
            trigger_search(client, access_token, ep_id)
            return

    sys.exit(
        "Need one of: --dry-run | --list-episodes --series-id N | "
        "--episode-id N | --series-id N --season N --episode N"
    )


if __name__ == "__main__":
    main()
