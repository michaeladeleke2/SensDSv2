"""
tools/verify_bundle.py

Check a PyInstaller output tree before it is shipped.

A bundle that lost a dependency looks exactly like a good one: the exe is
there, it is the right size, and it starts. The failure only shows up when a
student opens the tab that needed the missing piece, which is the worst
possible time to find out. This walks the tree and fails the build instead.

    python tools/verify_bundle.py dist/SensDSv2

Checks the packaged files only. It cannot run the Windows exe from CI, so it
looks for the things that have actually gone missing in past builds.
"""

import sys
from pathlib import Path

# (label, [path fragments that satisfy it], why it matters)
REQUIRED = [
    ("PyQt6 platform plugins", ["PyQt6/Qt6/plugins/platforms", "PyQt6/Qt6/plugins"],
     "the window cannot open without them"),
    ("torch", ["torch/_C", "torch/version.py", "torch/__init__.py"],
     "training and inference"),
    ("torch.cuda", ["torch/cuda/__init__.py", "torch/cuda"],
     "`import torch` loads it even in the CPU-only build; excluding it "
     "makes the app die at startup"),
    ("transformers", ["transformers/__init__.py", "transformers"],
     "loading and fine-tuning the model"),
    ("matplotlib", ["matplotlib/__init__.py", "matplotlib"],
     "the Visualize tab's reference view"),
    ("matplotlib data", ["matplotlib/mpl-data", "mpl-data"],
     "matplotlib will not import without its rc files"),
    ("pyqtgraph", ["pyqtgraph/__init__.py", "pyqtgraph"],
     "every live plot"),
    ("scipy", ["scipy/signal", "scipy"],
     "all spectrogram processing"),
    ("PIL", ["PIL/__init__.py", "PIL"],
     "building the image handed to the model"),
    ("vex package", ["vex/aim.py", "vex/aim.pyc", "vex"],
     "the VEX AIM tab, imported too late for PyInstaller to notice"),
    ("vex/settings.json", ["vex/settings.json"],
     "vex.settings reads it from beside itself at import time"),
    ("websocket-client", ["websocket/__init__.py", "websocket"],
     "how vex.aim talks to the robot"),
    ("app assets", ["assets/SensDSLogo.png", "assets"],
     "window icon and branding"),
]

# Missing these is a limitation, not a broken build.
OPTIONAL = [
    ("Infineon radar SDK", ["ifxradarsdk/__init__.py", "ifxradarsdk"],
     "live radar streaming will be unavailable"),
    ("bundled base model", ["models/vit-small-patch16-224/config.json",
                            "models/vit-base-patch16-224/config.json"],
     "the Train tab will need internet the first time"),
]


def find(root: Path, fragments):
    """True if any fragment exists anywhere in the tree."""
    for frag in fragments:
        parts = frag.split("/")
        # PyInstaller 6 puts everything under _internal/; check both.
        for base in (root, root / "_internal"):
            if (base.joinpath(*parts)).exists():
                return True
        # Fall back to a search, since layouts move between versions.
        matches = list(root.rglob(parts[-1]))
        if matches:
            return True
    return False


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_bundle.py <dist/SensDSv2>")
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    files = sum(1 for f in root.rglob("*") if f.is_file())
    print(f"bundle: {root}  ({files} files, {total / 1e9:.2f} GB)\n")

    exes = [p for p in root.iterdir()
            if p.suffix.lower() in (".exe", "") and p.is_file()]
    print("executable:", ", ".join(p.name for p in exes) or "NONE FOUND")

    missing = []
    print("\nrequired:")
    for label, frags, why in REQUIRED:
        ok = find(root, frags)
        print(f"  {'ok  ' if ok else 'MISS'}  {label}")
        if not ok:
            missing.append((label, why))

    print("\noptional:")
    for label, frags, why in OPTIONAL:
        ok = find(root, frags)
        print(f"  {'ok  ' if ok else '--  '}  {label}"
              + ("" if ok else f"   ({why})"))

    if missing:
        print("\nBUILD IS INCOMPLETE:")
        for label, why in missing:
            print(f"  {label}: needed for {why}")
        raise SystemExit(1)
    print("\nAll required components present.")


if __name__ == "__main__":
    main()
