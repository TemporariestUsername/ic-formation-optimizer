"""Typed view over the ``getuserdetails`` payload.

Only the fields the optimizer needs are parsed; everything else stays in
``raw`` for later use (gear, legendaries, familiars, blessings, ...).

Confirmed shape (same fields ic.byteglow.com reads)::

    {"details": {"heroes": [{"hero_id": "1", "owned": "1", ...}, ...],
                 "loot": [{"hero_id": 1, ...}, ...], ...}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserState:
    owned: set[int]
    hero_raw: dict[int, dict] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_details(cls, payload: dict[str, Any]) -> "UserState":
        details = payload.get("details", payload)
        owned: set[int] = set()
        hero_raw: dict[int, dict] = {}
        for h in details.get("heroes") or []:
            try:
                hid = int(h.get("hero_id"))
            except (TypeError, ValueError):
                continue
            hero_raw[hid] = h
            if str(h.get("owned")) == "1":
                owned.add(hid)
        return cls(owned=owned, hero_raw=hero_raw, raw=details)

    @classmethod
    def everything(cls, hero_ids: set[int]) -> "UserState":
        """A hypothetical account that owns every champion (theory-crafting)."""
        return cls(owned=set(hero_ids))
