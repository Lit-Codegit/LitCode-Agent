import pytest

from litcode_agent.pane_layout import PaneLayout


def test_splits_in_requested_direction_and_focuses_by_geometry() -> None:
    layout = PaneLayout("a")

    layout.split("a", "right", "b")
    layout.split("b", "down", "c")

    assert layout.pane_ids() == ("a", "b", "c")
    assert layout.focus_from("a", "right") == "b"
    assert layout.focus_from("b", "down") == "c"
    assert layout.focus_from("c", "left") == "a"
    assert layout.focus_from("a", "left") is None


def test_closes_leaf_without_losing_remaining_layout() -> None:
    layout = PaneLayout("a")
    layout.split("a", "right", "b")
    layout.split("b", "down", "c")

    layout.close("b")

    assert layout.pane_ids() == ("a", "c")
    assert layout.focus_from("a", "right") == "c"


def test_resize_moves_the_nearest_nested_divider() -> None:
    layout = PaneLayout("a")
    layout.split("a", "right", "b")
    layout.split("b", "down", "c")

    layout.set_ratio("c", 0.7)
    root = layout.root
    assert root.ratio == 0.5  # type: ignore[union-attr]
    nested = root.second  # type: ignore[union-attr]
    assert nested.ratio == 0.7  # type: ignore[union-attr]

    layout.resize("c", -0.2)
    nested = layout.root.second  # type: ignore[union-attr]
    assert nested.ratio == pytest.approx(0.9)  # type: ignore[union-attr]
