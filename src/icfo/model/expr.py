"""Evaluator for the game's small expression language.

Used for variant restrictions, effect filters, per-champion stack counts and
amount formulas. Examples from live data::

    HasTag(`support`)
    hero_id == 33 || hero_id == 80
    (!HasTag(`dps`)&&hero_seat_id!=10)||(HasTag(`dps`)&&hero_seat_id==10)
    max(INT,CHA)>=14
    clamp(min_stat_amount+1-str,0,1)          # per-hero stack count
    has_tag_speed                             # per-hero stack count
    upgrade_amount(12982,0)                   # amount formula
    upgrade_amount(1,0)*upgrade_amount(2,0)*0.5

The expression is translated to Python syntax and evaluated by walking a
whitelisted AST: no ``eval``, no attribute access, only names and functions the
caller provides. Anything else raises :class:`UnsupportedExpr`.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from .definitions import Champion

STATS = ("str", "dex", "con", "int", "wis", "cha")

Names = Callable[[str], Any]  # raises KeyError for unknown names


class UnsupportedExpr(ValueError):
    pass


def _translate(expr: str) -> str:
    s = re.sub(r"`([^`]*)`", lambda m: repr(m.group(1)), expr)
    s = s.replace("&&", " and ").replace("||", " or ")
    s = re.sub(r"!(?!=)", " not ", s)
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s)
    return s


@lru_cache(maxsize=8192)
def compile_expr(expr: str) -> ast.Expression:
    try:
        return ast.parse(_translate(expr.strip()).strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsupportedExpr(f"cannot parse {expr!r}") from exc


_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: lambda a, b: a / b if b else 0.0,
    ast.Mod: lambda a, b: a % b if b else 0.0,
}

BASE_FUNCS: dict[str, Callable[..., Any]] = {
    "max": max,
    "min": min,
    "clamp": lambda v, lo, hi: max(lo, min(hi, v)),
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
}


def evaluate_with(expr: str, names: Names, funcs: Mapping[str, Callable[..., Any]]) -> Any:
    tree = compile_expr(expr)

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(ev(v) for v in node.values)
            return any(ev(v) for v in node.values)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                fn = _CMP.get(type(op))
                if fn is None:
                    raise UnsupportedExpr(f"unsupported comparison {type(op).__name__}")
                right = ev(right_node)
                if not fn(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = funcs.get(node.func.id)
            if fn is None or node.keywords:
                raise UnsupportedExpr(f"unknown function {node.func.id!r}")
            return fn(*(ev(a) for a in node.args))
        if isinstance(node, ast.Name):
            try:
                return names(node.id)
            except KeyError:
                raise UnsupportedExpr(f"unknown name {node.id!r}") from None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            return node.value
        raise UnsupportedExpr(f"unsupported syntax: {type(node).__name__}")

    return ev(tree)


# -- champion expressions ---------------------------------------------------

def champion_names(champ: "Champion", params: Mapping[str, Any] | None = None) -> Names:
    """Names visible in a per-champion expression.

    ``STR``/``str``, ``hero_id``, ``hero_seat_id``, ``total_ability_score``,
    ``has_tag_<tag>``, plus numeric parameters from the effect key itself
    (e.g. ``min_stat_amount``).
    """

    def lookup(name: str) -> Any:
        low = name.lower()
        if low in STATS:
            return champ.stats.get(low, 0)
        if name == "hero_id":
            return champ.id
        if name == "hero_seat_id":
            return champ.seat
        if name == "total_ability_score":
            return sum(champ.stats.get(s, 0) for s in STATS)
        if name.startswith("has_tag_"):
            return name[len("has_tag_"):] in champ.tags
        if name.startswith("has_base_attack_dmg_type_"):
            return name[len("has_base_attack_dmg_type_"):] in champ.attack_types
        if params is not None and name in params and isinstance(params[name], (int, float)):
            return params[name]
        raise KeyError(name)

    return lookup


def champion_funcs(champ: "Champion") -> dict[str, Callable[..., Any]]:
    return {
        **BASE_FUNCS,
        "HasTag": lambda tag: tag in champ.tags,
        "GetStat": lambda stat: champ.stats.get(str(stat).lower(), 0),
        "GetBaseStat": lambda stat: champ.stats.get(str(stat).lower(), 0),
        "HasAttackDamageType": lambda t: str(t) in champ.attack_types,
    }


def evaluate(expr: str, champ: "Champion", params: Mapping[str, Any] | None = None) -> Any:
    """Evaluate a champion expression. Returns whatever the expression yields
    (bool for conditions, a number for stack counts)."""
    return evaluate_with(expr, champion_names(champ, params), champion_funcs(champ))
