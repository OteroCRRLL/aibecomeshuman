"""Hands for the agent: a virtual Xbox controller, and an eye on the operator's real one.

- `VirtualPad` presses buttons / tilts the right stick through ViGEmBus (vgamepad). The
  game sees it as one more Xbox controller.
- `OperatorWatch` reads the other (physical) controllers with XInput: any input from them
  means the operator is acting, and the agent must not press. Holding the View (Back)
  button for a second toggles the agent's control (pause / resume), from the sofa.
- `RecordingPad` does nothing but remember presses (tests, recordings).

QTEs are never pressed by the agent: it only answers menus with several named options.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from typing import Protocol

from aibecomeshuman.config import ConfigError

BUTTONS = ("Y", "X", "B", "A")
DIRECTIONS = ("up", "down", "left", "right")
PRESS_S = 0.12
TILT_S = 0.45
VIEW_HOLD_S = 1.0


class Pad(Protocol):
    def press(self, button: str) -> None: ...
    def tilt(self, direction: str) -> None: ...
    def release_all(self) -> None: ...
    def close(self) -> None: ...


class RecordingPad:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def press(self, button: str) -> None:
        self.actions.append(button)

    def tilt(self, direction: str) -> None:
        self.actions.append(f"stick:{direction}")

    def release_all(self) -> None:
        pass

    def close(self) -> None:
        pass


class _XInputState(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_uint32),
        ("wButtons", ctypes.c_uint16),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


def _xinput():
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            return getattr(ctypes.windll, name)
        except (AttributeError, OSError):
            continue
    return None


def connected_controllers() -> set[int]:
    lib = _xinput()
    if lib is None:
        return set()
    state = _XInputState()
    return {i for i in range(4) if lib.XInputGetState(i, ctypes.byref(state)) == 0}


class VirtualPad:
    def __init__(self) -> None:
        try:
            import vgamepad as vg
        except Exception as e:  # missing package or ViGEmBus driver
            raise ConfigError(
                f"virtual controller unavailable ({e}). It needs the ViGEmBus driver and "
                "the vgamepad package (`uv sync`)."
            ) from e
        before = connected_controllers()
        self._vg = vg
        self._pad = vg.VX360Gamepad()
        time.sleep(0.5)  # let Windows register it
        new = connected_controllers() - before
        # The virtual controller's XInput slot, so the operator's real one can be told apart.
        self.index: int | None = min(new) if new else None
        self._lock = threading.Lock()
        buttons = vg.XUSB_BUTTON
        self._map = {
            "Y": buttons.XUSB_GAMEPAD_Y,
            "X": buttons.XUSB_GAMEPAD_X,
            "B": buttons.XUSB_GAMEPAD_B,
            "A": buttons.XUSB_GAMEPAD_A,
        }

    def press(self, button: str) -> None:
        with self._lock:
            self._pad.press_button(button=self._map[button])
            self._pad.update()
            time.sleep(PRESS_S)
            self._pad.release_button(button=self._map[button])
            self._pad.update()

    def tilt(self, direction: str) -> None:
        x, y = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0),
                "right": (1.0, 0.0)}[direction]
        with self._lock:
            self._pad.right_joystick_float(x_value_float=x, y_value_float=y)
            self._pad.update()
            time.sleep(TILT_S)
            self._pad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._pad.update()

    def release_all(self) -> None:
        with self._lock:
            self._pad.reset()
            self._pad.update()

    def close(self) -> None:
        self.release_all()
        del self._pad


class OperatorWatch:
    """Polls the physical controllers (every XInput slot but the virtual one)."""

    STICK_DEADZONE = 12000
    TRIGGER_DEADZONE = 40
    VIEW = 0x0020  # XINPUT_GAMEPAD_BACK

    def __init__(self, ignore: int | None, on_view_hold: Callable[[], None]) -> None:
        self._lib = _xinput()
        self._ignore = ignore
        self._on_view_hold = on_view_hold
        self.last_activity = -1e9
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> OperatorWatch:
        if self._lib is not None:
            self._thread.start()
        return self

    def active_within(self, seconds: float) -> bool:
        return time.monotonic() - self.last_activity <= seconds

    def _run(self) -> None:
        state = _XInputState()
        view_since: float | None = None
        fired = False
        while not self._stop.is_set():
            pressed_view = False
            for i in range(4):
                if i == self._ignore or self._lib.XInputGetState(i, ctypes.byref(state)) != 0:
                    continue
                busy = (
                    state.wButtons
                    or max(abs(state.sThumbLX), abs(state.sThumbLY), abs(state.sThumbRX),
                           abs(state.sThumbRY)) > self.STICK_DEADZONE
                    or max(state.bLeftTrigger, state.bRightTrigger) > self.TRIGGER_DEADZONE
                )
                if busy:
                    self.last_activity = time.monotonic()
                pressed_view = pressed_view or bool(state.wButtons & self.VIEW)
            now = time.monotonic()
            if pressed_view:
                view_since = view_since or now
                if not fired and now - view_since >= VIEW_HOLD_S:
                    fired = True
                    self._on_view_hold()
            else:
                view_since, fired = None, False
            time.sleep(1 / 60)

    def stop(self) -> None:
        self._stop.set()
