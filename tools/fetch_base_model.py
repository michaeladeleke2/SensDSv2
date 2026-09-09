"""
tools/fetch_base_model.py

Download the fine-tuning base model into models/ so it can be packaged.

Without this, the Train tab reaches HuggingFace the first time a student
trains. That is exactly the thing that fails on a locked-down classroom
network, and it is invisible until the moment someone tries to train. Running
this before a build puts the weights inside the executable, and the app then
trains with no network at all.

    python tools/fetch_base_model.py            # the Small model (~88 MB)
    python tools/fetch_base_model.py --all      # Small and Base (~430 MB)

Adds roughly its own size to the packaged app.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Only the files the trainer actually loads. Most repos publish the same weights
# several ways; taking safetensors alone rather than everything saves 131 MB per
# model, and transformers prefers safetensors anyway. Repos that predate the
# format get a second pass below.
CONFIG = ["*.json", "*.txt"]
SAFETENSORS = CONFIG + ["*.safetensors"]
PICKLE = CONFIG + ["*.bin"]
IGNORE = ["*.msgpack", "*.h5", "*.onnx", "*flax*", "*tf_model*"]


def fetch(model_id: str, dest_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = dest_root / model_id.split("/")[-1]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"downloading {model_id} -> {dest}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        allow_patterns=SAFETENSORS,
        ignore_patterns=IGNORE,
    )
    if not any(dest.glob("*.safetensors")):
        print("  no safetensors in this repo; taking the pickled weights")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(dest),
            allow_patterns=PICKLE,
            ignore_patterns=IGNORE,
        )
    if not (dest / "config.json").exists():
        raise SystemExit(
            f"{model_id}: no config.json arrived, so the app will not see this "
            f"as a usable local model. Check the download and retry."
        )
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  ok: {size / 1e6:.0f} MB in {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="fetch the Base model too, not just Small")
    args = ap.parse_args()

    from ui.train_tab import _MODEL_OPTIONS

    wanted = ["Small"] + (["Base"] if args.all else [])
    dest_root = ROOT / "models"
    dest_root.mkdir(exist_ok=True)

    for key in wanted:
        fetch(_MODEL_OPTIONS[key], dest_root)

    print(f"\nmodels/ is ready. `pyinstaller main.spec` will now bundle it, "
          f"and the Train tab will report the model as available offline.")


if __name__ == "__main__":
    main()
