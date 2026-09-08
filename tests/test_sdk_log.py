"""
Tests for the Infineon SDK console filter.

The SDK writes dropped-packet notices to stdout from C++, several a second,
which buries everything else in the console. core/sdk_log.py taps the file
descriptor to hold those back and summarize them.

Every case runs in a subprocess: the module rewires file descriptor 1, which
would swallow pytest's own output if it ran in-process.

Run:  python -m pytest tests/test_sdk_log.py -v
      python tests/test_sdk_log.py            # standalone
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRELUDE = f"""
import sys, os, time
sys.path.insert(0, {str(ROOT)!r})
from core import sdk_log
def w(b): os.write(1, b)
def p(s): os.write(1, (s + "\\n").encode())
LOSS = b"[2026-09-08 08:06:17] INFO: Data read thread - Packet loss\\n"
"""


def run(body: str) -> str:
    """Run a snippet with sdk_log available; return its combined output."""
    proc = subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_repeated_notices_do_not_reach_the_console():
    out = run("""
        sdk_log.install()
        for _ in range(50):
            w(LOSS)
        time.sleep(0.3)
        sdk_log.uninstall()
        p("END")
    """)
    assert "Data read thread - Packet loss" not in out
    assert "END" in out
    # One explanation, and no per-message spam.
    assert out.count("[SensDS]") == 1


def test_the_first_notice_explains_itself():
    out = run("""
        sdk_log.install()
        w(LOSS)
        time.sleep(0.3)
        sdk_log.uninstall()
    """)
    assert "not an app error" in out


def test_summary_is_rate_limited():
    """Notices arriving continuously produce occasional summaries, not a flood."""
    out = run("""
        sdk_log.SUMMARY_INTERVAL_S = 0.4
        sdk_log.install()
        for _ in range(6):
            for _ in range(25):
                w(LOSS)
            time.sleep(0.2)
        time.sleep(0.3)
        sdk_log.uninstall()
    """)
    lines = [l for l in out.splitlines() if l.startswith("[SensDS]")]
    # 150 notices over ~1.2 s at a 0.4 s interval: a handful of lines, not 150.
    assert 1 <= len(lines) <= 6, lines


def test_sdk_errors_and_warnings_still_show():
    """Only routine chatter is held back; a real problem must stay visible."""
    out = run("""
        sdk_log.install()
        w(b"[2026-09-08 08:06:18] ERROR: Data read thread - Packet type error: 0x9\\n")
        w(b"[2026-09-08 08:06:19] WARNING: something worth seeing\\n")
        time.sleep(0.3)
        sdk_log.uninstall()
    """)
    assert "ERROR: Data read thread - Packet type error: 0x9" in out
    assert "WARNING: something worth seeing" in out


def test_ordinary_output_passes_through_in_order():
    out = run("""
        sdk_log.install()
        p("FIRST")
        w(LOSS)
        time.sleep(0.1)
        p("SECOND")
        print("THIRD", flush=True)
        print("FOURTH")           # still in Python's buffer at uninstall
        sdk_log.uninstall()
        p("FIFTH")
    """)
    seen = [l for l in out.splitlines()
            if l in ("FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH")]
    assert seen == ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH"], out


def test_uninstall_does_not_lose_buffered_output():
    """
    Regression: close() used to shut the saved descriptor before draining the
    pipe, so anything still in flight vanished.
    """
    out = run("""
        sdk_log.install()
        print("BUFFERED-BEFORE-UNINSTALL")
        sdk_log.uninstall()
        p("AFTER")
    """)
    assert "BUFFERED-BEFORE-UNINSTALL" in out
    assert "AFTER" in out


def test_a_notice_split_across_two_writes_is_still_matched():
    """The C++ side can flush a line in pieces; a partial read must not leak."""
    out = run("""
        sdk_log.install()
        w(b"[2026-09-08 08:06:19] INFO: Data read thre")
        time.sleep(0.1)
        w(b"ad - Packet loss\\n")
        time.sleep(0.3)
        s = sdk_log.stats()
        sdk_log.uninstall()
        p("COUNT=%d" % s["total"])
    """)
    assert "Packet loss" not in out.replace("dropped-packet", "")
    assert "COUNT=1" in out


def test_stats_counts_by_message():
    out = run("""
        sdk_log.install()
        for _ in range(3):
            w(LOSS)
        w(b"[2026-09-08 08:06:20] INFO: Data read thread - dumped packet\\n")
        time.sleep(0.3)
        s = sdk_log.stats()
        sdk_log.uninstall()
        p("TOTAL=%d" % s["total"])
        p("KINDS=%d" % len(s["by_message"]))
        p("INSTALLED=%s" % s["installed"])
    """)
    assert "TOTAL=4" in out
    assert "KINDS=2" in out


def test_stats_before_install_is_empty_not_an_error():
    out = run("""
        s = sdk_log.stats()
        p("INSTALLED=%s TOTAL=%d" % (s["installed"], s["total"]))
    """)
    assert "INSTALLED=False TOTAL=0" in out


def test_install_is_idempotent():
    out = run("""
        a = sdk_log.install()
        b = sdk_log.install()
        sdk_log.uninstall()
        p("A=%s B=%s" % (a, b))
    """)
    assert "A=True B=True" in out


def test_uninstall_restores_the_real_stdout():
    out = run("""
        sdk_log.install()
        sdk_log.uninstall()
        w(LOSS)                  # no tap left, so this must appear verbatim
        time.sleep(0.1)
    """)
    assert "Data read thread - Packet loss" in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
