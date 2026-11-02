"""Client for the official Idle Champions game API (``post.php``).

These are the same endpoints ic.byteglow.com calls from the browser:

* ``getPlayServerForDefinitions`` (master server) -> which play server to use
* ``getDefinitions``      -> static game data (champions, upgrades, effects,
                              adventures/variants, formation layouts)
* ``getuserdetails``      -> account state (owned champions, loot, feats, ...)
* ``getallformationsaves``-> saved formations

User calls need your ``user_id`` and ``hash`` (in-game: Settings -> Support,
or from the game's local data). Keep them on your machine; they grant full
account access. Be polite to the servers: cache aggressively, don't loop.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .cache import Cache

MASTER_URL = "https://master.idlechampions.com/~idledragons/"
CLIENT_VERSION = 999
NETWORK_ID = 11  # value used by the web tools; identifies the client platform
USER_AGENT = "icfo/0.1 (formation optimizer; personal use)"


class ApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    user_id: str
    hash: str

    @classmethod
    def from_env(cls) -> "Credentials":
        uid, h = os.environ.get("IC_USER_ID"), os.environ.get("IC_HASH")
        if not uid or not h:
            raise ApiError("Set IC_USER_ID and IC_HASH environment variables.")
        return cls(uid, h)


class IdleChampionsClient:
    def __init__(self, cache: Cache | None = None, timeout_s: float = 120.0):
        self.cache = cache or Cache()
        self.timeout_s = timeout_s
        self._play_server: str | None = None

    # -- transport ---------------------------------------------------------
    def _get(self, base: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{base}post.php?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ApiError(f"Unexpected response from {params.get('call')}")
        return data

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Call the play server, following one ``switch_play_server`` redirect."""
        for _ in range(2):
            data = self._get(self.play_server(), params)
            switch = data.get("switch_play_server")
            if switch:
                self._play_server = switch if switch.endswith("/") else switch + "/"
                continue
            if data.get("success") is False:
                raise ApiError(f"{params.get('call')} failed: {data.get('failure_reason', data)}")
            return data
        raise ApiError("Play server redirected more than once")

    def play_server(self) -> str:
        if self._play_server is None:
            data = self._get(
                MASTER_URL,
                {
                    "call": "getPlayServerForDefinitions",
                    "mobile_client_version": CLIENT_VERSION,
                    "network_id": NETWORK_ID,
                },
            )
            self._play_server = data["play_server"]
        return self._play_server

    # -- endpoints ---------------------------------------------------------
    def definitions(self, refresh: bool = False) -> dict[str, Any]:
        if not refresh:
            cached = self.cache.load("definitions")
            if cached is not None:
                return cached
        data = self._call(
            {
                "call": "getDefinitions",
                "new_achievements": 1,
                "mobile_client_version": 99999,
            }
        )
        # Drop large fields the optimizer never uses to keep the cache small.
        for key in ("graphic_defines", "sound_defines", "text_defines", "music_track_defines"):
            data.pop(key, None)
        self.cache.save("definitions", data)
        return data

    def user_details(self, creds: Credentials, max_age_s: float = 600) -> dict[str, Any]:
        cached = self.cache.load("user_details", max_age_s=max_age_s)
        if cached is not None:
            return cached
        data = self._call(
            {
                "call": "getuserdetails",
                "instance_key": 1,
                "user_id": creds.user_id,
                "hash": creds.hash,
                "mobile_client_version": CLIENT_VERSION,
            }
        )
        self.cache.save("user_details", data)
        return data

    def campaign_details(self, creds: Credentials, max_age_s: float = 3600) -> dict[str, Any]:
        """Campaign data incl. blessing definitions (reset_upgrade_defines)."""
        cached = self.cache.load("campaign_details", max_age_s=max_age_s)
        if cached is not None:
            return cached
        data = self._call(
            {
                "call": "getcampaigndetails",
                "game_instance_id": 1,
                "instance_id": 1,
                "user_id": creds.user_id,
                "hash": creds.hash,
                "mobile_client_version": CLIENT_VERSION,
            }
        )
        self.cache.save("campaign_details", data)
        return data

    def formation_saves(self, creds: Credentials) -> dict[str, Any]:
        data = self._call(
            {
                "call": "getallformationsaves",
                "user_id": creds.user_id,
                "hash": creds.hash,
                "mobile_client_version": CLIENT_VERSION,
            }
        )
        self.cache.save("formation_saves", data)
        return data
