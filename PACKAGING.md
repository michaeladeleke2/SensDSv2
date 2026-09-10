# Packaging SensDSv2 for Windows

Goal: a folder a student can copy from a flash drive, double-click, and use.
No Python, no `pip install`, no internet.

---

## The Infineon SDK

**PyInstaller cannot cross-compile.** A Windows executable has to be built on
Windows, and the Infineon SDK ships native libraries that must match the build
platform: `.dll` on Windows, `.dylib` on macOS. A macOS wheel inside a Windows
build gives an app that starts, trains and tests fine but can never open the
radar.

The Windows wheel is committed in `vendor/`:

```
vendor/ifxradarsdk-3.6.4+4b4a6245-py3-none-win_amd64.whl
```

That is deliberate. The wheel is MIT-licensed (`vendor/LICENSE-ifxradarsdk.txt`),
which permits redistribution, in this repository and inside the built app, on
the condition that the copyright notice goes with it. PyInstaller copies the
package's `dist-info` into the bundle, so the notice ships inside every build
automatically.

---

## Route 1 — GitHub Actions (recommended)

Builds on a real Windows VM, so you never need a Windows machine, and the
result is downloadable straight away. `.github/workflows/build_windows.yml`
already does this.

### One-time setup: none

CI installs whatever Windows wheel is in `vendor/`. To upgrade the SDK, replace
that file with a newer `ifxradarsdk-*-win_amd64.whl` and push. If `vendor/`
ever holds no Windows wheel, the build still succeeds and is named
`SensDSv2-Windows-NoRadar` so the two can't be confused.

### Every build

Push to `main`. Then **Actions → Build Windows Executable → the latest run →
Artifacts**.

### A release students can download

```bash
git tag v1.0
git push --tags
```

The zip is attached to a GitHub Release, giving a permanent public URL. That is
your hosting: no server to run, no storage to pay for.

---

## Route 2 — build on a Windows machine

Use this if you want a build in the next ten minutes and have a Windows laptop.

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pyinstaller
pip install vendor\ifxradarsdk-3.6.4+4b4a6245-py3-none-win_amd64.whl
python tools\fetch_base_model.py
pyinstaller main.spec
python tools\verify_bundle.py dist\SensDSv2
```

Install CPU-only torch **first**. Plain `pip install torch` pulls the CUDA
build, which needs NVIDIA driver DLLs that no Surface has, and the app dies at
startup with a `c10.dll` error. `build_local.bat` wraps these steps.

---

## What ends up in the folder

```
SensDSv2\
  SensDSv2.exe          ← students double-click this
  _internal\            ← ~22,000 files: Python, Qt, PyTorch, the base model
```

Measured on the equivalent macOS build: **1.91 GB unpacked, 680 MB zipped**.
Windows should land slightly under that, since the CPU-only torch wheel is
smaller than the macOS one. Almost all of it is PyTorch and Qt. Stay under 2 GB
zipped or GitHub Releases will reject the upload.

Ship the **whole folder**. `SensDSv2.exe` on its own does nothing — it is a
launcher for `_internal\`.

### Why a folder and not a single .exe

One-file mode unpacks every DLL into a temp directory on each launch: a 30–60
second black screen, every single time, on the slowest machines in the room.
The folder build opens immediately.

---

## Checking a build before class

```bat
SensDSv2.exe --self-test
```

Imports every subsystem and reports what this copy can do. The Windows build
has no console, so the report opens in a dialog; click **Show Details** for the
full list:

```
  ok    PyTorch
  ok    reference view (matplotlib)
  ok    VEX AIM robot
  --    live radar (Infineon SDK) unavailable (ModuleNotFoundError)

  ok    base model Small: bundled, trains offline
```

Worth running once on a new machine and once on a fresh copy from the flash
drive. Most of the app's heavy imports happen lazily inside worker threads, so
a build that lost one starts perfectly and only fails later, on whichever tab a
student happens to open. CI runs `tools/verify_bundle.py` for the same reason —
it fails the build rather than shipping a bundle with a hole in it.

---

## Flash drive

1. Unzip on the flash drive, or copy the unzipped `SensDSv2\` folder onto it.
2. Plug in, open the folder, double-click `SensDSv2.exe`.

**Copy it to the desktop first if the drive is slow.** Launching reads several
hundred MB of DLLs; from USB 2.0 that is a minute or more of apparent hang,
from an internal SSD it is a few seconds. Use a USB 3.0 drive.

Student recordings go to `C:\Users\<name>\SensDSv2_data`, on the computer, not
the drive. Nothing is written back to the flash drive, so one drive can go
round a whole classroom.

---

## Two things that will happen on the day

**SmartScreen.** An unsigned executable downloaded from the internet shows
"Windows protected your PC". Students must click **More info → Run anyway**.
Copying from a flash drive rather than downloading usually avoids it. The only
real fix is an EV code-signing certificate (a few hundred dollars a year) —
worth it if this goes past your own classroom, not otherwise.

**Antivirus.** PyInstaller's launcher is used by malware often enough that
heuristic scanners sometimes quarantine it. If your school pushes a managed
antivirus, get the folder allow-listed by IT before the first class rather than
during it.

---

## Offline training

`tools/fetch_base_model.py` downloads the fine-tuning weights into `models/`
before the build, and `main.spec` packages them. The Train tab then reports
*local bundle* as its model source and never touches the network.

```bash
python tools/fetch_base_model.py          # Small, ~88 MB
python tools/fetch_base_model.py --all    # adds Base, ~330 MB more
```

Skip this and the app still works, but the first student to click Train needs
internet — which is exactly what fails on a locked-down school network, and it
fails at the least convenient moment.

---

## If a build misbehaves

Run `SensDSv2.exe --self-test` first; it names the missing piece.

Past failures worth knowing about, all now fixed in `main.spec`:

| Symptom | Cause |
|---|---|
| App exits instantly, no window | `excludes=['torch.cuda']`. `import torch` loads `torch.cuda` even in the CPU-only build, so excluding it raises `ModuleNotFoundError` before the window opens. |
| "Compare with reference view" does nothing | matplotlib was neither installed by CI nor collected by the spec. |
| VEX AIM tab cannot connect | `vex` is imported inside a method, so PyInstaller never saw it, and `vex/settings.json` was left behind. |
| `c10.dll` / `torch_cuda.dll` error | The CUDA build of torch got installed. Install the CPU-only one first. |
| Radar button does nothing | No `ifxradarsdk` in the bundle, or the wrong platform's wheel. |

The dependency list lives in `requirements.txt` and nowhere else. The old
workflow repeated it by hand and drifted, which is how matplotlib went missing.
