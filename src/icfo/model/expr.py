"""Evaluator for the game's hero expressions used in variant restrictions.

Examples from live data::

    HasTag(`support`)
    hero_id == 33 || hero_id == 80
    GetStat(`wis`) > GetStat(`dex`)
    (!HasTag(`dps`)&&hero_seat_id!=10)||(HasTag(`dps`)&&hero_seat_id==10)
    HasTag(`speed`) || STR>=16
    max(INT,CHA)>=14

The expression is translated to Python syntax and evaluated by walking a
whitelisted AST: no ``eval``, no attribute access, only known names and
functions. Anything else raises :class:`UnsupportedExpr`.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any, Callable

from .definitions import Champion

STATS = ("str", "dex", "con", "int", "wis", "cha")


class UnsupportedExpr(ValueError):
    pass


def _translate(expr: str) -> str:
    s = re.sub(r"`([^`]*)`", lambda m: repr(m.group(1)), expr)
    s = s.replace("&&", " and ").replace("||", " or ")
    s = re.sub(r"!(?!=)", " not ", s)
    return s


@lru_cache(maxsize=4096)
def compile_expr(expr: str) -> ast.Expression:
    try:
        return ast.parse(_translate(expr), mode="eval")
    except SyntaxError as exc:
        raise UnsupportedExpr(f"cannot parse {expr!r}") from exc


def _names(champ: Champion) -> dict[str, Any]:
    names: dict[str, Any] = {"hero_id": champ.id, "hero_seat_id": champ.seat}
    for stat in STATS:
        names[stat.upper()] = champ.stats.get(stat, 0)
    names["total_ability_score"] = sum(champ.stats.get(s, 0) for s in STATS)
    return names


def _functions(champ: Champion) -> dict[str, Callable[..., Any]]:
    return {
        "HasTag": lambda tag: tag in champ.tags,
        "GetStat": lambda stat: champ.stats.get(str(stat).lower(), 0),
        "max": max,
        "min": min,
    }


_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


def evaluate(expr: str, champ: Champion) -> bool:
    tree = compile_expr(expr)
    names, funcs = _names(champ), _functions(champ)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = (ev(v) for v in node.values)
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not ev(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = ev(right_node)
                fn = _CMP.get(type(op))
                if fn is None or not fn(left, right):
                    return False if fn else _unsupported(op)
                left = right
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            if node.keywords:
                _unsupported(node)
            return funcs[node.func.id](*(ev(a) for a in node.args))
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise UnsupportedExpr(f"unknown name {node.id!r}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            return node.value
        return _unsupported(node)

    return bool(ev(tree))


def _unsupported(node: Any) -> Any:
    raise UnsupportedExpr(f"unsupported syntax: {type(node).__name__}")
