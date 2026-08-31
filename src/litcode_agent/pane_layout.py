"""A small binary split tree independent from Textual rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["left", "right", "up", "down"]
Axis = Literal["horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class PaneLeaf:
    pane_id: str


@dataclass(frozen=True, slots=True)
class PaneBranch:
    axis: Axis
    first: PaneNode
    second: PaneNode


PaneNode = PaneLeaf | PaneBranch


class PaneLayout:
    def __init__(self, first_pane_id: str) -> None:
        self.root: PaneNode = PaneLeaf(first_pane_id)

    def pane_ids(self) -> tuple[str, ...]:
        result: list[str] = []

        def visit(node: PaneNode) -> None:
            if isinstance(node, PaneLeaf):
                result.append(node.pane_id)
                return
            visit(node.first)
            visit(node.second)

        visit(self.root)
        return tuple(result)

    def split(self, target: str, direction: Direction, new_pane_id: str) -> None:
        if new_pane_id in self.pane_ids():
            raise ValueError(f"pane already exists: {new_pane_id}")
        replacement: PaneNode | None = None

        def replace(node: PaneNode) -> PaneNode:
            nonlocal replacement
            if isinstance(node, PaneLeaf):
                if node.pane_id != target:
                    return node
                new = PaneLeaf(new_pane_id)
                axis: Axis = (
                    "horizontal" if direction in {"left", "right"} else "vertical"
                )
                replacement = (
                    PaneBranch(axis, new, node)
                    if direction in {"left", "up"}
                    else PaneBranch(axis, node, new)
                )
                return replacement
            return PaneBranch(node.axis, replace(node.first), replace(node.second))

        self.root = replace(self.root)
        if replacement is None:
            raise KeyError(target)

    def close(self, pane_id: str) -> None:
        if len(self.pane_ids()) == 1:
            raise ValueError("cannot close the last pane")

        def remove(node: PaneNode) -> PaneNode | None:
            if isinstance(node, PaneLeaf):
                return None if node.pane_id == pane_id else node
            first = remove(node.first)
            second = remove(node.second)
            if first is None:
                return second
            if second is None:
                return first
            return PaneBranch(node.axis, first, second)

        updated = remove(self.root)
        if updated == self.root:
            raise KeyError(pane_id)
        assert updated is not None
        self.root = updated

    def focus_from(self, pane_id: str, direction: Direction) -> str | None:
        rectangles = _rectangles(self.root)
        if pane_id not in rectangles:
            raise KeyError(pane_id)
        current = rectangles[pane_id]
        candidates: list[tuple[float, float, str]] = []
        for candidate_id, candidate in rectangles.items():
            if candidate_id == pane_id:
                continue
            primary = _primary_distance(current, candidate, direction)
            if primary is None:
                continue
            secondary = _secondary_distance(current, candidate, direction)
            candidates.append((primary, secondary, candidate_id))
        if not candidates:
            return None
        return min(candidates)[2]


Rect = tuple[float, float, float, float]


def _rectangles(root: PaneNode) -> dict[str, Rect]:
    result: dict[str, Rect] = {}

    def visit(node: PaneNode, rect: Rect) -> None:
        if isinstance(node, PaneLeaf):
            result[node.pane_id] = rect
            return
        left, top, right, bottom = rect
        if node.axis == "horizontal":
            middle = (left + right) / 2
            visit(node.first, (left, top, middle, bottom))
            visit(node.second, (middle, top, right, bottom))
        else:
            middle = (top + bottom) / 2
            visit(node.first, (left, top, right, middle))
            visit(node.second, (left, middle, right, bottom))

    visit(root, (0.0, 0.0, 1.0, 1.0))
    return result


def _primary_distance(current: Rect, candidate: Rect, direction: Direction) -> float | None:
    left, top, right, bottom = current
    other_left, other_top, other_right, other_bottom = candidate
    if direction == "right" and other_left >= right:
        return other_left - right
    if direction == "left" and other_right <= left:
        return left - other_right
    if direction == "down" and other_top >= bottom:
        return other_top - bottom
    if direction == "up" and other_bottom <= top:
        return top - other_bottom
    return None


def _secondary_distance(current: Rect, candidate: Rect, direction: Direction) -> float:
    if direction in {"left", "right"}:
        current_center = (current[1] + current[3]) / 2
        candidate_center = (candidate[1] + candidate[3]) / 2
    else:
        current_center = (current[0] + current[2]) / 2
        candidate_center = (candidate[0] + candidate[2]) / 2
    return abs(current_center - candidate_center)
