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
