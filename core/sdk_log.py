"""
core/sdk_log.py

Quiets and accounts for the Infineon SDK's own console chatter.

The lines look like:

    [2026-09-08 08:06:17] INFO: Data read thread - Packet loss

They are not errors and they do not come from this app. Infineon's transport
layer (libstrata_shared) writes them straight to std::cout from C++, so neither
logging.getLogger() nor contextlib.redirect_stdout can reach them — the only
handle Python has on C-level output is the underlying file descriptor.

install() puts a pipe in front of stdout and reads it on a background thread.
SDK INFO lines are counted instead of printed; everything else, including SDK
warnings and errors, is written straight through, so print() still behaves.

"Packet loss" means the radar dropped USB packets on the way to the host.
RadarStream already skips a dropped frame and keeps streaming, so it is not
fatal — but a steady stream of them means frames are arriving slower than the
configured 10 fps, and a 3-second capture will take longer than 3 seconds. That
is worth knowing, so the messages are summarized on a timer rather than thrown
away, and the running totals stay available through stats().
"""

import os
import re
import sys
import threading
import time
from collections import Counter

# "[2026-09-08 08:06:17] INFO: Data read thread - Packet loss"
_SDK_LINE = re.compile(
    rb"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s+([A-Z]+):\s*(.*)$"
)

# Only routine chatter is held back. A warning or an error from the SDK still
# reaches the console, because those are the ones worth interrupting for.
_QUIET_LEVELS = {b"INFO", b"DEBUG", b"TRACE"}

# Longest gap between "the radar is still dropping packets" summaries.
SUMMARY_INTERVAL_S = 30.0

_state = None
_lock = threading.Lock()


class _Tap:
    """Reads a pipe standing in for one file descriptor, filtering as it goes."""

    def __init__(self, fd: int):
        self.fd = fd
        self.counts = Counter()
        self.total = 0
        self.first_seen = None
        self._reported = 0          # total already mentioned on the console
        self._last_summary = 0.0
        self._closed = False

        # Keep the real destination, then point the fd at a pipe we own.
        self._saved = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)
        self._read_fd = read_fd

        self._thread = threading.Thread(
            target=self._run, name=f"sdk-log-{fd}", daemon=True)
        self._thread.start()

    # ── passthrough ──────────────────────────────────────────────────────────

    def _emit(self, data: bytes):
        saved = self._saved
        if saved is None:
            return
        try:
            os.write(saved, data)
        except OSError:
            pass

    def _note(self, level: bytes, message: bytes):
        now = time.monotonic()
        with _lock:
            self.counts[message.decode("utf-8", "replace").strip()] += 1
            self.total += 1
            if self.first_seen is None:
                self.first_seen = now
                self._last_summary = now
                self._emit(
                    b"[SensDS] The Infineon SDK is reporting dropped radar "
                    b"packets. This is the SDK's own status message, not an "
                    b"app error; streaming continues. Further notices are "
                    b"summarized below.\n"
                )
                self._reported = self.total
                return
            due = now - self._last_summary >= SUMMARY_INTERVAL_S
            new = self.total - self._reported
            if not (due and new):
                return
            gap = now - self._last_summary
            self._last_summary = now
            self._reported = self.total
        self._emit(
            f"[SensDS] Infineon SDK: {new} dropped-packet notices in the last "
            f"{gap:.0f}s ({self.total} total). Frames are arriving slower than "
            f"10 fps, so captures will run long.\n".encode()
        )

    def _handle(self, line: bytes):
        m = _SDK_LINE.match(line)
        if m and m.group(1) in _QUIET_LEVELS:
            self._note(m.group(1), m.group(2))
        else:
            self._emit(line + b"\n")

    def _run(self):
        buf = b""
        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            # Hold an unterminated tail until its newline arrives, so a line
            # split across two reads is still matched as one line.
            *lines, buf = buf.split(b"\n")
            for line in lines:
                self._handle(line)
        if buf:
            self._handle(buf)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Order matters. dup2 puts the real destination back on the fd and in
        # doing so closes the pipe's only write end, which is what lets the
        # reader see EOF. Draining has to finish before the saved fd is closed,
        # or anything still in the pipe is lost and the reader ends up writing
        # to a descriptor number that has since been handed to something else.
        os.dup2(self._saved, self.fd)
        self._thread.join(timeout=1.0)
        saved, self._saved = self._saved, None
        try:
            os.close(saved)
        except OSError:
            pass
        try:
            os.close(self._read_fd)
        except OSError:
            pass


def install() -> bool:
    """
    Start filtering SDK chatter out of stdout. Safe to call more than once.

    Returns False when there is no usable stdout to tap — a windowed PyInstaller
    build on Windows has none — in which case nothing is changed.
    """
    global _state
    if _state is not None:
        return True
    fd = 1
    try:
        os.fstat(fd)
    except OSError:
        return False
    try:
        _state = _Tap(fd)
    except OSError:
        _state = None
        return False
    return True


def uninstall():
    """Restore stdout. Mainly for tests; the app leaves the tap in place."""
    global _state
    if _state is None:
        return
    tap, _state = _state, None
    # Flush Python's own buffer through the pipe before taking it away.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    tap.close()


def stats() -> dict:
    """
    What the SDK has said so far.

    Returns {"installed", "total", "by_message", "since_s"} — since_s counts
    from the first suppressed message, so total/since_s is the rate.
    """
    if _state is None:
        return {"installed": False, "total": 0, "by_message": {}, "since_s": 0.0}
    with _lock:
        since = (time.monotonic() - _state.first_seen
                 if _state.first_seen is not None else 0.0)
        return {
            "installed": True,
            "total": _state.total,
            "by_message": dict(_state.counts),
            "since_s": since,
        }
