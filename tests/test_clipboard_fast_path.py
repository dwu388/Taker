from __future__ import annotations

from types import SimpleNamespace

import pytest

from profit_taker import axiom_migrated_runner_base as runner


class _FakeClipboard:
    def __init__(self, values):
        self.values = list(values)
        self.value = ""

    def copy(self, value):
        self.value = value

    def paste(self):
        if self.values:
            item = self.values.pop(0)
            if item is not None:
                self.value = item
        return self.value


class _FakeGui:
    def __init__(self, clipboard, payload):
        self.clipboard = clipboard
        self.payload = payload
        self.hotkeys = []
        self.moves = []

    def moveTo(self, x, y):
        self.moves.append((x, y))

    def mouseDown(self, button="left"):
        return None

    def mouseUp(self, button="left"):
        return None

    def hotkey(self, *keys):
        self.hotkeys.append(keys)
        if keys == ("ctrl", "c"):
            self.clipboard.value = self.payload


def _payload():
    # Structural validation is monkeypatched in these unit tests; the payload only
    # needs to be distinguishable from the sentinel.
    return "fresh axiom payload"


def test_fast_capture_accepts_fresh_payload_without_fixed_multi_second_waits(monkeypatch):
    cb = _FakeClipboard([])
    gui = _FakeGui(cb, _payload())
    monotonic = [0.0]
    sleeps = []

    monkeypatch.setitem(__import__("sys").modules, "pyperclip", cb)
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", gui)
    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: text == _payload())
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: (sleeps.append(seconds), monotonic.__setitem__(0, monotonic[0] + seconds)))
    monkeypatch.setattr(runner.time, "monotonic", lambda: monotonic[0])

    text, captured_at = runner._capture_clipboard({"capture": {}}, 1)
    assert text == _payload()
    assert captured_at
    assert ("ctrl", "a") in gui.hotkeys
    assert ("ctrl", "c") in gui.hotkeys
    # Normal capture should no longer contain the old 1.857/2.0/2.5 second waits.
    assert max(sleeps) < 1.0
    assert sum(sleeps) < 1.5


def test_capture_retries_copy_when_browser_misses_first_chord(monkeypatch):
    cb = _FakeClipboard([])

    class RetryGui(_FakeGui):
        def __init__(self, clipboard, payload):
            super().__init__(clipboard, payload)
            self.copy_count = 0

        def hotkey(self, *keys):
            self.hotkeys.append(keys)
            if keys == ("ctrl", "c"):
                self.copy_count += 1
                if self.copy_count >= 2:
                    self.clipboard.value = self.payload

    gui = RetryGui(cb, _payload())
    monotonic = [0.0]
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", cb)
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", gui)
    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: text == _payload())
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: monotonic.__setitem__(0, monotonic[0] + seconds))
    monkeypatch.setattr(runner.time, "monotonic", lambda: monotonic[0])

    text, _ = runner._capture_clipboard({"capture": {"copy_retry_after_seconds": 0.25, "clipboard_poll_seconds": 0.05}}, 1)
    assert text == _payload()
    assert gui.copy_count == 2


def test_capture_never_accepts_unchanged_sentinel(monkeypatch):
    cb = _FakeClipboard([])

    class NoCopyGui(_FakeGui):
        def hotkey(self, *keys):
            self.hotkeys.append(keys)

    gui = NoCopyGui(cb, _payload())
    monotonic = [0.0]
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", cb)
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", gui)
    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: True)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: monotonic.__setitem__(0, monotonic[0] + seconds))
    monkeypatch.setattr(runner.time, "monotonic", lambda: monotonic[0])

    with pytest.raises(RuntimeError, match="fresh clipboard text"):
        runner._capture_clipboard({"capture": {"clipboard_read_timeout_seconds": 1.0, "copy_retries": 0}}, 1)


def test_rare_refresh_preserves_refresh_before_selection(monkeypatch):
    cb = _FakeClipboard([])
    gui = _FakeGui(cb, _payload())
    monotonic = [0.0]
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", cb)
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", gui)
    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: text == _payload())
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: monotonic.__setitem__(0, monotonic[0] + seconds))
    monkeypatch.setattr(runner.time, "monotonic", lambda: monotonic[0])

    runner._capture_clipboard({"capture": {}}, runner.REFRESH_EVERY_CYCLES)
    assert gui.hotkeys.index(("ctrl", "shift", "r")) < gui.hotkeys.index(("ctrl", "a"))
