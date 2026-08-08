"""
Regression tests for subject discovery in the Train tab.

Guards the failure that broke training on a real dataset: a saved HuggingFace
checkpoint left in DATA_ROOT (a `model/` folder holding config.json +
model.safetensors, with no sub-folders) was treated as a student.  When the
subject split happened to pick it as the validation subject, the val set came
out empty and training died with "Train or val dataset is empty."

Run:  python -m pytest tests/test_subject_discovery.py -v
      python tests/test_subject_discovery.py            # standalone
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui.train_tab as TT  # noqa: E402


def _png(path: Path):
    """Write a minimal valid PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000a49444154789c63000100000500010d0a2db40000"
            "000049454e44ae426082"
        )
    )


def _make_root() -> Path:
    """Mirror the real layout that triggered the bug."""
    root = Path(tempfile.mkdtemp(prefix="sensds_subj_"))
    for subj, gestures in (
        ("Michael", ("idle", "push", "swipe_right")),
        ("Vincent", ("idle", "push", "swipe_right")),
        ("Crawford", ("push",)),          # partial: only one gesture
    ):
        for g in gestures:
            for i in range(3):
                _png(root / subj / g / f"s{i}.png")

    # A saved HF checkpoint dumped in DATA_ROOT — files, no sub-folders.
    (root / "model").mkdir()
    (root / "model" / "config.json").write_text("{}")
    (root / "model" / "model.safetensors").write_bytes(b"\x00")

    # The real models/ output tree, which contains nested dirs.
    (root / "models" / "Michael_v1" / "model").mkdir(parents=True)
    (root / "models" / "Michael_v1" / "model" / "config.json").write_text("{}")

    # Assorted junk that must never count as a student.
    (root / ".DS_Store").write_bytes(b"\x00")
    (root / ".hidden_subject" / "push").mkdir(parents=True)
    _png(root / ".hidden_subject" / "push" / "a.png")
    (root / "temp" / "push").mkdir(parents=True)
    _png(root / "temp" / "push" / "a.png")
    (root / "empty_student").mkdir()                       # no gestures
    (root / "no_pngs" / "push").mkdir(parents=True)        # gesture dir, no PNGs
    return root


def test_stray_checkpoint_is_not_a_subject():
    root = _make_root()
    try:
        subjects = TT.discover_subjects(root)
        assert subjects == ["Crawford", "Michael", "Vincent"], subjects
        assert "model" not in subjects
        assert "models" not in subjects
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_junk_directories_excluded():
    root = _make_root()
    try:
        subjects = TT.discover_subjects(root)
        for bad in ("temp", "empty_student", "no_pngs", ".hidden_subject"):
            assert bad not in subjects, f"{bad} should not be a subject"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_split_never_yields_empty_val():
    """
    The original failure mode: with the stray `model/` dir included, seed 42
    picked it as the sole validation subject.  Every subject the split returns
    must now own at least one sample.
    """
    root = _make_root()
    try:
        for seed in range(25):
            for n_val in (1, 2):
                train, val = TT._split_subjects(root, val_subjects=n_val, seed=seed)
                assert train and val, f"empty split at seed={seed} n_val={n_val}"
                for s in list(train) + list(val):
                    n = sum(1 for _ in (root / s).glob("*/*.png"))
                    assert n > 0, f"subject {s} has no samples (seed={seed})"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_scan_dataset_agrees_with_discovery():
    """The status panel and the trainer must see the same students."""
    root = _make_root()
    orig = TT.DATA_ROOT
    TT.DATA_ROOT = str(root)
    try:
        counts, students = TT.scan_dataset()
        assert sorted(students) == TT.discover_subjects(root)
        assert "model" not in students
        # Crawford contributes only push: 3+3+3 push, 3+3 idle, 3+3 swipe_right
        assert counts == {"push": 9, "idle": 6, "swipe_right": 6}, counts
    finally:
        TT.DATA_ROOT = orig
        shutil.rmtree(root, ignore_errors=True)


def test_empty_root_is_handled():
    root = Path(tempfile.mkdtemp(prefix="sensds_empty_"))
    try:
        assert TT.discover_subjects(root) == []
        try:
            TT._split_subjects(root)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on an empty root")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
