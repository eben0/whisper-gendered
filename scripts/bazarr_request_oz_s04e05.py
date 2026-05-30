#!/usr/bin/env python3
"""Trigger Bazarr to re-fetch the Hebrew SRT for OZ S04E05.

Two-stage auth (because the Bazarr instance is behind Traefik + Authentik):

1. OAuth 2.0 password-grant to Authentik → access token
2. Bazarr API call with both ``Authorization: Bearer <token>`` (for the
   reverse proxy) AND ``X-API-KEY: <BAZARR_API_KEY>`` (for Bazarr itself)

Reads credentials from ``.env.auth`` in the project root:

    OAUTH_ACCESS_TOKEN_URL
    OAUTH_GRANT_TYPE
    OAUTH_CLIENT_ID
    OAUTH_SCOPE
    OAUTH_USERNAME
    OAUTH_PASSWORD
    BAZARR_API_KEY
    BAZARR_BASE_URL              # e.g. https://bazarr.example.com
    BAZARR_EPISODE_ID            # Bazarr's internal episode id for OZ S04E05
    AUTHENTIK_COOKIE             # (optional) full ``Cookie:`` header value
                                 # grabbed from browser DevTools after login.
                                 # Needed when the Authentik outpost is in
                                 # cookie-session mode and Bearer alone is
                                 # not enough (302 → /application/o/authorize).

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\bazarr_request_oz_s04e05.py
    .\\.venv\\Scripts\\python.exe scripts\\bazarr_request_oz_s04e05.py --dry-run
    .\\.venv\\Scripts\\python.exe scripts\\bazarr_request_oz_s04e05.py --find-episode

``--dry-run`` exercises only the OAuth step (no Bazarr call) — useful to
verify credentials without burning a translation cycle.

``--find-episode`` lists episodes for the configured Sonarr series id so
you can grab ``BAZARR_EPISODE_ID`` for OZ S04E05. Set ``BAZARR_SERIES_ID``
in ``.env.auth`` first.
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
    project_root = Path(__file__).resolve().parent.parent
    env_auth = project_root / ".env.auth"
    if not env_auth.exists():
        sys.exit(f"missing {env_auth}")
    load_dotenv(env_auth, override=False)


# ----- OAuth + Bazarr ----------------------------------------------------- #


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


def authenticate(*, force_refresh: bool = False) -> str:
    """Authentik OAuth 2.0 password grant. Client auth is in the request body
    (per the Authentik client config). Returns the access token. Reuses a
    cached token from ``.cache/bazarr_oauth_token.json`` if non-expired.
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
    r = httpx.post(token_url, data=body, timeout=30.0)
    if r.status_code != 200:
        sys.exit(f"OAuth failed: {r.status_code} {_redact(r.text[:500])}")
    data = r.json()
    if "access_token" not in data:
        sys.exit("OAuth response missing access_token")
    token: str = data["access_token"]
    # ``expires_in`` is seconds-from-now per RFC 6749 §5.1.
    expires_in = int(data.get("expires_in", 3600))
    _save_cached_token(token, expires_in)
    return token


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


def find_episode(access_token: str) -> None:
    """List episodes for ``BAZARR_SERIES_ID`` so you can pick the OZ S04E05 id.

    Uses Bazarr's ``GET /api/episodes`` endpoint with ``seriesid``.
    """
    base = _env("BAZARR_BASE_URL").rstrip("/")
    series_id = _env("BAZARR_SERIES_ID")
    url = f"{base}/api/episodes?seriesid[]={series_id}"
    r = httpx.get(url, headers=_bazarr_headers(access_token), timeout=30.0,
                  follow_redirects=True)
    _safe_print(f"GET {url} → {r.status_code}")
    if r.status_code != 200:
        sys.exit(_redact(r.text[:500]))
    ctype = r.headers.get("content-type", "").lower()
    if "json" not in ctype:
        _safe_print(
            f"⚠ response is not JSON (content-type={ctype!r}); "
            f"first 400 chars below — this usually means the Authentik "
            f"outpost intercepted with a login page despite the Bearer "
            f"token. The proxy may require a session cookie instead of "
            f"OAuth headers for non-browser API calls."
        )
        _safe_print(r.text[:400])
        sys.exit(1)
    eps = r.json().get("data", [])
    _safe_print(f"{len(eps)} episodes:")
    for ep in eps:
        # Bazarr response shape varies by version; try the common keys.
        season = ep.get("season")
        episode = ep.get("episode")
        sonarr_id = ep.get("sonarrEpisodeId") or ep.get("episodeId") or ep.get("id")
        title = ep.get("title", "")
        _safe_print(f"  S{season:02d}E{episode:02d}  id={sonarr_id}  {title}")


def trigger_search(access_token: str) -> None:
    """Ask Bazarr to download a subtitle for the configured episode.

    Bazarr's manual-search-and-download path:
      POST /api/providers/episodes?episodeid=<id>

    This triggers an immediate search across enabled providers (whisperai
    among them) for the configured episode. The Whisper provider then
    POSTs to our whisper-gend ASR endpoint.

    If your Bazarr is on a version where this endpoint differs, the
    likely alternatives are:
      POST /api/episodes/wanted?episodeid=<id>
      POST /api/episodes/subtitles    (body with episodeid + provider)
    """
    base = _env("BAZARR_BASE_URL").rstrip("/")
    episode_id = _env("BAZARR_EPISODE_ID")
    url = f"{base}/api/providers/episodes?episodeid={episode_id}"
    r = httpx.post(url, headers=_bazarr_headers(access_token), timeout=30.0,
                   follow_redirects=True)
    _safe_print(f"POST {url} → {r.status_code}")
    _safe_print(_redact(r.text[:1000]))
    if r.status_code >= 400:
        sys.exit(f"Bazarr request failed (HTTP {r.status_code}).")


# ----- entrypoint --------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="OAuth only; do not call Bazarr.")
    parser.add_argument("--find-episode", action="store_true",
                        help="List episodes for BAZARR_SERIES_ID so you can "
                             "grab the right BAZARR_EPISODE_ID.")
    parser.add_argument("--force-refresh-token", action="store_true",
                        help="Ignore the cached OAuth token and request a "
                             "fresh one. Use after credential changes.")
    args = parser.parse_args()

    _load_env()
    access_token = authenticate(force_refresh=args.force_refresh_token)
    _safe_print(f"✓ OAuth ok  (token length={len(access_token)})")
    if args.dry_run:
        return
    if args.find_episode:
        find_episode(access_token)
        return
    trigger_search(access_token)


if __name__ == "__main__":
    main()
