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


def _save_cached_token(
    token: str, expires_in: int, cookie_header: str = "",
) -> None:
    """Persist the access token + (optional) cookie header + computed expiry.
    Refresh ``_REFRESH_LEEWAY_SECONDS`` before actual expiry to avoid races
    where a token expires mid-request.

    ``cookie_header`` is the serialized ``name=value; name=value; …`` form
    of whatever cookies the OAuth response set. Stored verbatim so a later
    invocation can attach it to Bazarr requests without re-doing OAuth.
    """
    _TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    expires_at = time.time() + max(0, expires_in - _REFRESH_LEEWAY_SECONDS)
    payload = {
        "access_token": token,
        "expires_at": expires_at,
        "cookie_header": cookie_header,
    }
    _TOKEN_CACHE.write_text(json.dumps(payload), encoding="utf-8")


def _load_cached_cookie() -> str | None:
    """Return the cookie header persisted alongside the access token, or
    None if the cache is missing/stale or no cookies were captured."""
    if not _TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(_TOKEN_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Tie cookie lifetime to the token's — if the token's expired we should
    # consider the session stale too (cookies often outlive tokens but
    # being conservative avoids stale-session bugs).
    expires_at = float(data.get("expires_at", 0))
    if time.time() >= expires_at:
        return None
    val = data.get("cookie_header")
    return val if isinstance(val, str) and val else None


def authenticate(client: httpx.Client, *, force_refresh: bool = False) -> str:
    """Authentik OAuth 2.0 password grant. Returns the access token, or an
    empty string when OAuth is disabled (no ``OAUTH_ACCESS_TOKEN_URL`` set in
    ``.env.auth`` — typical for direct LAN access to Bazarr where there's no
    reverse-proxy auth in the way).

    When OAuth is configured, reuses a cached token from
    ``.cache/bazarr_oauth_token.json`` if non-expired. The HTTP call goes
    through the supplied ``httpx.Client`` so any ``Set-Cookie`` returned by
    the token endpoint (e.g. an ``authentik_proxy_*`` session cookie) gets
    stored in the client's cookie jar and persisted into the cache for
    automatic forwarding on later runs.
    """
    # Skip OAuth entirely when the token URL isn't configured. Useful for
    # local/LAN setups where Bazarr is exposed directly (X-API-KEY only).
    token_url = os.getenv("OAUTH_ACCESS_TOKEN_URL", "").strip()
    if not token_url:
        _safe_print("(OAuth disabled — OAUTH_ACCESS_TOKEN_URL not set)")
        return ""
    if not force_refresh:
        cached = _load_cached_token()
        if cached:
            return cached
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
    # Capture cookies set by the OAuth response so later invocations can
    # reuse them without re-doing OAuth (and without manually pasting
    # AUTHENTIK_COOKIE). Serialize to a verbatim ``Cookie:`` header value.
    cookie_header = "; ".join(
        f"{name}={value}" for name, value in client.cookies.items()
    )
    _save_cached_token(token, expires_in, cookie_header=cookie_header)
    captured = list(client.cookies.keys())
    if captured:
        _safe_print(f"✓ OAuth response set cookies: {captured} (cached)")
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
        "X-API-KEY": _env("BAZARR_API_KEY"),
        "Accept": "application/json",
    }
    # Authentication for the reverse-proxy layer (skipped on direct LAN
    # access where these headers are unnecessary).
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    # Prefer the OAuth-captured cookies persisted alongside the token;
    # fall back to AUTHENTIK_COOKIE in .env.auth for outposts whose token
    # endpoint doesn't set cookies (the manual-paste path).
    cookie = _load_cached_cookie() or os.getenv("AUTHENTIK_COOKIE")
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
    client: httpx.Client, access_token: str,
    series_id: int, episode_id: int,
    *, provider: str = "whisperai", language: str = "he",
) -> None:
    """Two-step manual-search-and-download against Bazarr's provider API.

    1. ``GET /api/providers/episodes?seriesid=…&episodeid=…&language=…``
       returns the list of provider results (each is a dict with
       ``provider``, ``subtitle`` id, ``language``, etc.). For our
       whisperai provider this returns one row per language it can
       transcribe to.
    2. ``POST /api/providers/episodes`` with ``provider``, ``subtitle``,
       and the required flags downloads that specific result — for
       whisperai this is what actually fires the whisper-gend ASR.

    Bazarr's web UI does the same two steps when you click "Search
    Subtitle" then "Use this".
    """
    base = _env("BAZARR_BASE_URL").rstrip("/")

    # Step 1: enumerate provider results for this episode + language.
    search_url = (f"{base}/api/providers/episodes"
                  f"?seriesid={series_id}&episodeid={episode_id}"
                  f"&language={language}")
    _safe_print(f"GET {search_url} → searching providers...")
    r = client.get(search_url, headers=_bazarr_headers(access_token),
                   timeout=3600.0)  # whisperai may transcribe synchronously (~30 min on a 57-min episode)
    _safe_print(f"  status={r.status_code}")
    if r.status_code >= 400:
        _safe_print(_redact(r.text[:1000]))
        sys.exit(f"Bazarr search failed (HTTP {r.status_code}).")
    results = r.json().get("data", [])
    _safe_print(f"  {len(results)} provider result(s):")
    for entry in results:
        _safe_print(f"    provider={entry.get('provider')!r:25s} "
                    f"language={entry.get('language')!r:5s} "
                    f"score={entry.get('score')}")

    # Step 2: pick the result from the requested provider.
    chosen = next(
        (e for e in results if e.get("provider") == provider),
        None,
    )
    if chosen is None:
        sys.exit(
            f"No '{provider}' result found among {len(results)} results. "
            f"Providers returned: "
            f"{sorted({e.get('provider') for e in results})}"
        )
    subtitle_id = chosen.get("subtitle")
    if not subtitle_id:
        sys.exit(f"Chosen result missing 'subtitle' id: {chosen!r}")
    _safe_print(f"  ✓ picked provider={provider!r} subtitle={subtitle_id!r}")

    # Step 3: download. Bazarr requires all the boolean flags + provider
    # + subtitle id. The whisperai provider then calls our whisper-gend
    # ASR endpoint to actually transcribe.
    dl_url = (f"{base}/api/providers/episodes"
              f"?seriesid={series_id}&episodeid={episode_id}"
              f"&hi=False&forced=False&original_format=False"
              f"&provider={provider}&subtitle={subtitle_id}")
    _safe_print(f"POST {dl_url} → downloading...")
    r = client.post(dl_url, headers=_bazarr_headers(access_token),
                    timeout=3600.0)
    _safe_print(f"  status={r.status_code}")
    if r.status_code >= 400:
        _safe_print(_redact(r.text[:1000]))
        sys.exit(f"Bazarr download failed (HTTP {r.status_code}).")
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
            if args.series_id is None:
                sys.exit("--episode-id requires --series-id (Bazarr's search "
                         "route is series-scoped).")
            trigger_search(
                client, access_token, args.series_id, args.episode_id,
            )
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
            trigger_search(client, access_token, args.series_id, ep_id)
            return

    sys.exit(
        "Need one of: --dry-run | --list-episodes --series-id N | "
        "--series-id N --episode-id N | --series-id N --season N --episode N"
    )


if __name__ == "__main__":
    main()
