"""Formation layouts as a graph of slots.

Layouts come from ``game_changes`` entries of type ``formation`` on campaigns
(default layout) and adventures/variants (override). Each slot has screen
coordinates, a column index and an explicit adjacency list::

    {"x": 80, "y": 30, "col": 0, "row": 2, "adj": [1, 2]}

Column 0 is the FRONT (closest to the enemies, right-hand side on screen);
higher columns are further back.

Positional effect targets (``adj``, ``col``, ``next_col``, ...) are resolved
here into sets of slot indices relative to the source champion's slot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Slot:
    index: int
    x: float
    y: float
    col: int
    adj: frozenset[int]


@dataclass(frozen=True)
class Layout:
    name: str
    slots: tuple[Slot, ...]
    _cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    @classmethod
    def from_game_change(cls, gc: dict[str, Any]) -> "Layout":
        slots = []
        for i, raw in enumerate(gc.get("formation") or []):
            slots.append(
                Slot(
                    index=i,
                    x=float(raw.get("x", 0)),
                    y=float(raw.get("y", 0)),
                    col=int(raw.get("col", 0)),
                    adj=frozenset(int(a) for a in raw.get("adj") or []),
                )
            )
        return cls(name=str(gc.get("name") or "formation"), slots=tuple(slots))

    def __len__(self) -> int:
        return len(self.slots)

    @property
    def all(self) -> frozenset[int]:
        return frozenset(range(len(self.slots)))

    def column(self, col: int) -> frozenset[int]:
        return frozenset(s.index for s in self.slots if s.col == col)

    def distances(self, src: int) -> dict[int, int]:
        key = ("dist", src)
        if key not in self._cache:
            dist = {src: 0}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nxt in self.slots[cur].adj:
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            self._cache[key] = dist
        return self._cache[key]

    def targets(self, relation: Any, src: int) -> frozenset[int] | None:
        """Slots targeted by ``relation`` from slot ``src``.

        Returns ``None`` for relations this layout model doesn't understand,
        so callers can count them as unmodelled instead of silently wrong.
        Occupant-dependent targets (``by_tags``, ``attack_type``) resolve to all
        slots here; the engine applies the occupant filter.
        """
        key = (repr(relation), src)
        if key in self._cache:
            return self._cache[key]
        result = self._targets_uncached(relation, src)
        self._cache[key] = result
        return result

    def _targets_uncached(self, relation: Any, src: int) -> frozenset[int] | None:
        s = self.slots[src]
        adj = s.adj
        if isinstance(relation, dict):
            kind = relation.get("type")
            if kind == "distance":
                d = int(relation.get("distance", 1))
                return frozenset(i for i, k in self.distances(src).items() if 0 < k <= d)
            if kind in ("by_tags", "attack_type"):
                return self.all
            return None
        rel = str(relation)
        col = s.col
        table = {
            "self": frozenset({src}),
            "self_slot": frozenset({src}),
            "adj": adj,
            "self_and_adj": adj | {src},
            "non_adj": self.all - adj - {src},
            "other": self.all - {src},
            "all": self.all,
            "all_slots": self.all,
            "col": self.column(col),
            "next_col": self.column(col - 1),
            "prev_col": self.column(col + 1),
            "col_and_prev_col": self.column(col) | self.column(col + 1),
            "prev_two_col": self.column(col + 1) | self.column(col + 2),
            "back_2_columns": self.column(col + 1) | self.column(col + 2),
            "ahead": adj & self.column(col - 1),
            "behind": adj & self.column(col + 1),
        }
        return table.get(rel)

    def render(self, labels: dict[int, str], width: int = 10) -> str:
        """ASCII picture: front (col 0) on the right, as in game."""
        cols = sorted({s.col for s in self.slots}, reverse=True)
        ys = sorted({s.y for s in self.slots})
        grid = {(s.col, s.y): s.index for s in self.slots}
        lines = []
        for y in ys:
            cells = []
            for c in cols:
                idx = grid.get((c, y))
                if idx is None:
                    cells.append(" " * width)
                else:
                    cells.append(f"[{labels.get(idx, '·')[: width - 2]:^{width - 2}}]")
            lines.append(" ".join(cells).rstrip())
        lines.append("back" + " " * max(0, (width + 1) * len(cols) - 9) + "front")
        return "\n".join(lines)
