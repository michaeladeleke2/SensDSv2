"""
core/starter_model.py

The starter model: a six-gesture model (idle, push, swipe_down, swipe_left,
swipe_right, swipe_up) that students can use in the Test tab and the Maze
before training their own.

At 343 MB it is too big for the git repository, so it is published once, with
the web version of SensDS, as float16 weights split into parts (predictions
match the float32 original). The first time it is used it is downloaded from
there, each part checked against the manifest, widened back to float32, and
saved as an ordinary model folder:

    ~/SensDSv2_data/models/starter_six_class/

which "Load Model" can also open. A packaged app can carry the same folder as
models/starter_six_class (main.spec bundles models/), and then nothing is
downloaded.
"""

import hashlib
import json
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request

STARTER_URL = "https://michaeladeleke2.github.io/sensds-web/assets/models/six_class/"
STARTER_NAME = "Starter model: six gestures"
FOLDER = "starter_six_class"
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")
STARTER_DIR = os.path.join(MODELS_ROOT, FOLDER)
_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _bundled_dir() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "models", FOLDER)


def _complete(folder: str) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in _FILES)


def find_starter() -> str | None:
    """The starter model folder if it is already here (bundled or downloaded)."""
    for folder in (_bundled_dir(), STARTER_DIR):
        if _complete(folder):
            return folder
    return None


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url: str, progress=None, done=0, total=0, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SensDSv2"})
    out = bytearray()
    with urllib.request.urlopen(req, timeout=timeout, context=_context()) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out += chunk
            if progress and total:
                progress(done + len(out), total)
    return bytes(out)


def download_starter(progress=None, base_url: str = STARTER_URL, dest: str = STARTER_DIR) -> str:
    """
    Download, check and save the starter model; returns its folder.
    progress(done_bytes, total_bytes) is called as it downloads.
    """
    import numpy as np
    from safetensors.numpy import load as load_bytes, save_file

    manifest = json.loads(_get(base_url + "manifest.json"))
    weights = manifest["weights"]
    total = int(weights["bytes"])
    parts, done = [], 0
    for part in weights["parts"]:
        data = _get(base_url + part["file"], progress, done, total)
        if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise RuntimeError(f"The starter model download was damaged ({part['file']}). Try again.")
        parts.append(data)
        done += len(data)
    blob = b"".join(parts)
    if hashlib.sha256(blob).hexdigest() != weights["sha256"]:
        raise RuntimeError("The starter model download was damaged. Try again.")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".starter-", dir=os.path.dirname(dest))
    try:
        tensors = load_bytes(blob)
        save_file({k: v.astype(np.float32) for k, v in tensors.items()},
                  os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
        for name in ("config.json", "preprocessor_config.json"):
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(_get(base_url + name))
        with open(os.path.join(tmp, "starter_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest
