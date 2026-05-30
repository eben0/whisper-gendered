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
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


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


def authenticate() -> str:
    """Authentik OAuth 2.0 password grant. Client auth is in the request body
    (per the Authentik client config). Returns the access token."""
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
        sys.exit(f"OAuth failed: {r.status_code} {r.text[:500]}")
    data = r.json()
    if "access_token" not in data:
        sys.exit(f"OAuth response missing access_token: {data}")
    return data["access_token"]


def _bazarr_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "X-API-KEY": _env("BAZARR_API_KEY"),
        "Accept": "application/json",
    }


def find_episode(access_token: str) -> None:
    """List episodes for ``BAZARR_SERIES_ID`` so you can pick the OZ S04E05 id.

    Uses Bazarr's ``GET /api/episodes`` endpoint with ``seriesid``.
    """
    base = _env("BAZARR_BASE_URL").rstrip("/")
    series_id = _env("BAZARR_SERIES_ID")
    url = f"{base}/api/episodes?seriesid[]={series_id}"
    r = httpx.get(url, headers=_bazarr_headers(access_token), timeout=30.0)
    print(f"GET {url} → {r.status_code}")
    if r.status_code != 200:
        sys.exit(r.text[:500])
    eps = r.json().get("data", [])
    print(f"{len(eps)} episodes:")
    for ep in eps:
        # Bazarr response shape varies by version; try the common keys.
        season = ep.get("season")
        episode = ep.get("episode")
        sonarr_id = ep.get("sonarrEpisodeId") or ep.get("episodeId") or ep.get("id")
        title = ep.get("title", "")
        print(f"  S{season:02d}E{episode:02d}  id={sonarr_id}  {title}")


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
    r = httpx.post(url, headers=_bazarr_headers(access_token), timeout=30.0)
    print(f"POST {url} → {r.status_code}")
    body = r.text[:1000]
    print(body)
    if r.status_code >= 400:
        sys.exit(f"Bazarr request failed (HTTP {r.status_code}).")


# ----- entrypoint --------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="OAuth only; do not call Bazarr.")
    parser.add_argument("--find-episode", action="store_true",
                        help="List episodes for BAZARR_SERIES_ID so you can "
                             "grab the right BAZARR_EPISODE_ID.")
    args = parser.parse_args()

    _load_env()
    access_token = authenticate()
    print(f"✓ OAuth ok  token={access_token[:8]}...{access_token[-8:]}")
    if args.dry_run:
        return
    if args.find_episode:
        find_episode(access_token)
        return
    trigger_search(access_token)


if __name__ == "__main__":
    main()
