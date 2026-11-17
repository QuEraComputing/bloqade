from unittest.mock import MagicMock, patch

import pytest
import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt

from bloqade.gemini.visualize import debug as debug_mod
from bloqade.gemini.visualize.debug import (
    AnimatorController,
    StaticDebuggerController,
)


@pytest.fixture
def dummy_ax():
    ax = MagicMock(spec=mpl_axes.Axes)
    ax.cla = MagicMock()
    return ax


@pytest.fixture
def dummy_draw():
    return MagicMock()


@pytest.fixture
def dummy_get_renderer():
    def get_renderer(step_index):
        return 3, MagicMock()

    return get_renderer


def test_static_debugger_on_next_prev_exit(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_next(MagicMock())
    assert ctrl.step_index == 1
    # Handlers must NOT clear the axes themselves — doing so caused a visual
    # flicker where the axes resized to matplotlib defaults for one paint
    # frame before run() repainted. run() owns ax.cla() so clear+draw is atomic.
    assert not dummy_ax.cla.called
    # Reset updated to allow prev
    ctrl.updated = False
    ctrl.on_prev(MagicMock())
    assert ctrl.step_index == 0
    ctrl.on_exit(MagicMock())
    assert not ctrl.running
    assert not ctrl.waiting


def test_static_debugger_on_key_dispatch(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_key(MagicMock(key="right"))
    assert ctrl.step_index == 1
    # Reset updated to allow left
    ctrl.updated = False
    ctrl.on_key(MagicMock(key="left"))
    assert ctrl.step_index == 0
    ctrl.updated = False
    ctrl.on_key(MagicMock(key="escape"))
    assert not ctrl.running


def test_static_debugger_reset(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.step_index = 3
    ctrl.running = False
    ctrl.waiting = False
    ctrl.updated = True
    ctrl.reset()
    assert ctrl.step_index == 0
    assert ctrl.running
    assert ctrl.waiting
    assert not ctrl.updated


def test_animator_controller_next_prev_exit(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.on_next(MagicMock())
    assert ctrl.animation_step == 1
    ctrl.on_prev(MagicMock())
    assert ctrl.animation_step == -1
    ctrl.on_exit(MagicMock())
    assert not ctrl.running
    assert not ctrl.waiting


def test_animator_controller_reset(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.step_index = 3
    ctrl.animation_step = -1
    ctrl.running = False
    ctrl.waiting = False
    ctrl.updated = True
    ctrl.reset()
    assert ctrl.step_index == 0
    assert ctrl.animation_step == 1
    assert ctrl.running
    assert ctrl.waiting
    assert not ctrl.updated


def test_static_debugger_on_slider_change_jumps_to_step(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_slider_change(3)
    assert ctrl.step_index == 3
    # Handler must not cla() — see test_static_debugger_on_next_prev_exit.
    assert not dummy_ax.cla.called
    assert ctrl.updated
    assert not ctrl.waiting


def test_static_debugger_on_slider_change_clamps_out_of_range(dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_slider_change(99)
    assert ctrl.step_index == 4  # clamped to num_steps - 1
    ctrl.updated = False  # allow the next change
    ctrl.on_slider_change(-7)
    assert ctrl.step_index == 0  # clamped to 0


def test_static_debugger_on_slider_change_noop_on_same_value(dummy_ax, dummy_draw):
    """Idempotent: clicking the current step should not break out of the
    wait loop (which would trigger a redundant run-loop redraw)."""
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.step_index = 2
    dummy_ax.cla.reset_mock()
    ctrl.on_slider_change(2)
    assert ctrl.step_index == 2
    assert not ctrl.updated
    assert ctrl.waiting


def test_static_debugger_on_next_calls_sync_slider(dummy_ax, dummy_draw):
    """Buttons / keys must keep the slider visual in sync."""
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    sync_calls: list[int] = []
    ctrl.sync_slider = lambda step_index: sync_calls.append(step_index)
    ctrl.on_next(MagicMock())
    assert sync_calls == [1]
    ctrl.updated = False
    ctrl.on_prev(MagicMock())
    assert sync_calls == [1, 0]


def test_animator_controller_on_slider_change(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.animation_step = -1  # opposite direction
    ctrl.on_slider_change(2)
    assert ctrl.step_index == 2
    # Slider jumps reset the play direction to forward.
    assert ctrl.animation_step == 1
    # Handler must not cla() — run() owns clear+redraw so it's atomic.
    assert not dummy_ax.cla.called
    assert ctrl.updated
    assert not ctrl.waiting


def test_animator_controller_on_next_prev_calls_sync_slider(
    dummy_ax, dummy_get_renderer
):
    """Animator's on_next/on_prev calls sync_slider whenever they actually
    mutate step_index. The default direction is forward (animation_step=1),
    so the very first on_next() advances and syncs immediately."""
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    sync_calls: list[int] = []
    ctrl.sync_slider = lambda step_index: sync_calls.append(step_index)
    ctrl.on_next(MagicMock())
    assert sync_calls == [1]


def test_animator_controller_direction_flip_syncs_slider_on_advance(
    dummy_ax, dummy_get_renderer
):
    """When the previous direction was opposite, the first on_next/on_prev
    only flips animation_step (no advance, no sync). The next call in the
    same direction does the actual advance and the sync."""
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.step_index = 2
    ctrl.animation_step = -1  # last action was a backward step
    sync_calls: list[int] = []
    ctrl.sync_slider = lambda step_index: sync_calls.append(step_index)

    # First on_next: only flips direction; no step change, no sync.
    ctrl.on_next(MagicMock())
    assert ctrl.animation_step == 1
    assert ctrl.step_index == 2
    assert sync_calls == []

    # Second on_next (same iteration window): direction matches, but the
    # 'updated' flag from the first call still gates this one. Reset to
    # simulate the run loop processing in between.
    ctrl.updated = False
    ctrl.on_next(MagicMock())
    assert ctrl.step_index == 3
    assert sync_calls == [3]


def test_static_debugger_slider_respects_updated_guard(dummy_ax, dummy_draw):
    """Race scenario: a button event already updated the state within the
    same plt.pause window, then the slider fires. The slider must NOT
    clobber the button intent — first event in the window wins, matching
    the existing on_next / on_prev guard pattern."""
    ctrl = StaticDebuggerController(dummy_ax, 5, dummy_draw)
    ctrl.on_next(MagicMock())
    assert ctrl.step_index == 1
    assert ctrl.updated is True
    # Slider event arrives in the same window — must be ignored.
    ctrl.on_slider_change(3)
    assert ctrl.step_index == 1
    assert ctrl.updated is True


def test_animator_slider_respects_updated_guard(dummy_ax, dummy_get_renderer):
    """Same race-guard semantics for the animator: a previous handler's
    update wins over a slider event in the same iteration window."""
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)
    ctrl.on_next(MagicMock())  # advances to step 1, sets updated=True
    assert ctrl.step_index == 1
    ctrl.on_slider_change(4)
    assert ctrl.step_index == 1, "slider must not clobber a pending button update"


def test_animator_controller_on_key_dispatch(dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 5, dummy_get_renderer)

    # Simulate key events
    class Event:
        def __init__(self, key: str):
            self.key = key

    # Initially, animation_step should be 1
    ctrl.animation_step = 0
    ctrl.on_key(Event("right"))
    assert (
        ctrl.animation_step == 1
    ), "Expected animation_step to be set to 1 on 'right' key"
    ctrl.on_key(Event("left"))
    assert (
        ctrl.animation_step == -1
    ), "Expected animation_step to be set to -1 on 'left' key"
    ctrl.running = True
    ctrl.waiting = True
    ctrl.on_key(Event("escape"))
    assert not ctrl.running, "Expected running to be False on 'escape' key"
    assert not ctrl.waiting, "Expected waiting to be False on 'escape' key"


def test_static_debugger_run(monkeypatch, dummy_ax, dummy_draw):
    ctrl = StaticDebuggerController(dummy_ax, 2, dummy_draw)
    call_count = {"draw": 0, "pause": 0}
    order: list[str] = []

    def fake_draw(idx):
        order.append("draw")
        call_count["draw"] += 1
        # Simulate waiting loop for pause
        ctrl.waiting = True
        if call_count["draw"] == 2:
            ctrl.running = False
            ctrl.waiting = False

    monkeypatch.setattr(ctrl, "draw", fake_draw)
    dummy_ax.cla.side_effect = lambda: order.append("cla")

    def fake_pause(t):
        call_count["pause"] += 1
        # End waiting after first pause
        ctrl.waiting = False

    monkeypatch.setattr(plt, "pause", fake_pause)
    ctrl.run()
    assert call_count["draw"] >= 1
    assert call_count["pause"] >= 1
    # Every draw must be immediately preceded by a cla() — no intervening
    # pause/paint frame. This is the invariant that prevents the flicker.
    for i, token in enumerate(order):
        if token == "draw":
            assert i > 0 and order[i - 1] == "cla"


def test_animator_controller_run(monkeypatch, dummy_ax, dummy_get_renderer):
    ctrl = AnimatorController(dummy_ax, 2, dummy_get_renderer)
    call_count = {"renderer": 0, "pause": 0}

    def fake_get_renderer(idx):
        def fake_renderer(frame):
            call_count["renderer"] += 1
            if call_count["renderer"] == 2:
                ctrl.running = False
                ctrl.waiting = False
            else:
                ctrl.waiting = False

        return 2, fake_renderer

    ctrl.get_renderer = fake_get_renderer
    monkeypatch.setattr(
        plt, "pause", lambda t: call_count.update({"pause": call_count["pause"] + 1})
    )
    ctrl.run()
    assert call_count["renderer"] >= 1
    assert call_count["pause"] >= 1


def test_debugger_noninteractive(monkeypatch, dummy_ax, dummy_draw):
    called = []

    def draw(idx):
        called.append(idx)

    with patch("bloqade.gemini.visualize.debug.get_drawer", return_value=(draw, 2)):
        monkeypatch.setattr(plt, "pause", lambda t: None)
        monkeypatch.setattr(dummy_ax, "cla", lambda: None)
        debug_mod.debugger(
            mt=MagicMock(),
            arch_spec=MagicMock(),
            interactive=False,
            pause_time=0.01,
            atom_marker="o",
            ax=dummy_ax,
        )
    assert called == [0, 1], f"Expected draw to be called with [0, 1], got {called}"


def test_animated_debugger_noninteractive(monkeypatch, dummy_ax):
    # Patch render_generator to return dummy get_renderer and 2 steps
    def fake_get_renderer(idx):
        def fake_renderer(frame):
            pass

        return 2, fake_renderer

    monkeypatch.setattr(
        debug_mod, "render_generator", lambda *a, **kw: (fake_get_renderer, 2)
    )
    monkeypatch.setattr(plt, "pause", lambda t: None)
    monkeypatch.setattr(dummy_ax, "cla", lambda: None)
    debug_mod.animated_debugger(
        mt=MagicMock(),
        arch_spec=MagicMock(),
        interactive=False,
        atom_marker="o",
        ax=dummy_ax,
        fps=30,
    )
