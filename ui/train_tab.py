import os
import glob
import time
import datetime
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from ui import app_colors, HintCard, _scrollable_left


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration: '42s', '3m 07s', '1h 12m 30s'."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _train_style(c: dict) -> str:
    return f"""
    QWidget#train_root {{ background: {c['bg']}; }}
    QWidget#left_panel {{
        background: {c['panel']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#heading {{
        font-size: 16px;
        font-weight: bold;
        color: {c['accent']};
    }}
    QLabel#field_label {{
        font-size: 12px;
        font-weight: bold;
        color: {c['subtext']};
    }}
    QLabel#status_ok {{
        font-size: 12px;
        color: #27ae60;
        font-weight: bold;
    }}
    QLabel#status_warn {{
        font-size: 12px;
        color: #e67e22;
        font-weight: bold;
    }}
    QLabel#status_err {{
        font-size: 12px;
        color: #c0392b;
        font-weight: bold;
    }}
    QSpinBox, QDoubleSpinBox {{
        border: 1px solid {c['input_border']};
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: {c['input_bg']};
        color: {c['text']};
        max-height: 30px;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {c['accent']};
    }}
    QPushButton#train_btn {{
        background-color: {c['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton#train_btn:hover {{ background-color: #245080; }}
    QPushButton#train_btn:disabled {{ background-color: #555; color: #999; }}
    QPushButton#stop_btn {{
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton#stop_btn:hover {{ background-color: #e74c3c; }}
    QRadioButton {{
        font-size: 13px;
        color: {c['text']};
    }}
    QCheckBox {{
        font-size: 13px;
        color: {c['text']};
    }}
    QPlainTextEdit {{
        background: {c['panel']};
        color: {c['text']};
        font-family: monospace;
        font-size: 12px;
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 4px;
    }}
    QLabel#section_heading {{
        font-size: 13px;
        font-weight: bold;
        color: {c['accent']};
    }}
"""


MIN_SAMPLES = 20
MIN_CLASSES = 3
DATA_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data")
MODELS_DIR = os.path.join(DATA_ROOT, "models")

# Available base models for fine-tuning
# Keys are shown in the UI dropdown; values are the HuggingFace model IDs.
_MODEL_OPTIONS = {
    "Small": "WinKawaks/vit-small-patch16-224",   # ~22 M params — fast on Surface / CPU
    "Base":  "google/vit-base-patch16-224",        # ~86 M params — highest accuracy
}
_DEFAULT_MODEL_KEY = "Small"

# Legacy constant kept for backwards compatibility (used by DownloadWorker default)
_HERE = Path(__file__).resolve().parent
_HF_MODEL_ID = _MODEL_OPTIONS[_DEFAULT_MODEL_KEY]


def _model_local_dir(model_id: str) -> Path:
    """Return the local bundle path for a given HuggingFace model ID."""
    model_name = model_id.split("/")[-1]   # e.g. "vit-small-patch16-224"
    return _HERE.parent / "models" / model_name


def _hf_cache_snapshot(model_id: str) -> Path | None:
    """Return the path to the locally cached HF snapshot for model_id, if it exists."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    # HF converts "/" to "--" in the cache folder name
    cache_key = model_id.replace("/", "--")
    snapshots = cache_root / "hub" / f"models--{cache_key}" / "snapshots"
    if snapshots.exists():
        for snap in reversed(sorted(snapshots.iterdir())):
            if (snap / "config.json").exists():
                return snap
    return None


def _resolve_model(model_id: str) -> tuple[str, str]:
    """Return (model_path_or_id, human_label) for the best available source."""
    local = _model_local_dir(model_id)
    if local.exists() and (local / "config.json").exists():
        return str(local), f"local bundle ({local.name})"
    snap = _hf_cache_snapshot(model_id)
    if snap:
        return str(snap), f"HF cache ({snap.parent.parent.name[:20]}…)"
    return model_id, "HuggingFace (requires internet)"


def model_is_available_offline(model_id: str | None = None) -> bool:
    mid = model_id or _HF_MODEL_ID
    local = _model_local_dir(mid)
    if local.exists() and (local / "config.json").exists():
        return True
    return _hf_cache_snapshot(mid) is not None


# Directories under DATA_ROOT that are never student data.
EXCLUDE_DIRS = {'models', 'model', 'predict_temp', 'temp', 'test', 'checkpoints'}


def is_subject_dir(path: Path) -> bool:
    """
    True if `path` looks like a student's sample folder.

    Structural rather than name-based: a subject must contain at least one
    gesture sub-folder holding PNGs.  A name blocklist alone is whack-a-mole —
    a saved HF checkpoint dropped in DATA_ROOT (config.json + model.safetensors,
    no sub-folders) was silently treated as a student, and if the split picked
    it as the validation subject the val set came out empty and training died
    with "Train or val dataset is empty".
    """
    if not path.is_dir() or path.name.startswith('.'):
        return False
    if path.name in EXCLUDE_DIRS:
        return False
    try:
        for child in path.iterdir():
            if child.is_dir() and next(child.glob('*.png'), None) is not None:
                return True
    except OSError:
        return False
    return False


def discover_subjects(root_dir=None) -> list[str]:
    """
    Sorted names of every real subject folder under root_dir.

    Single source of truth shared by scan_dataset() and TrainWorker so the
    dataset-status panel and the trainer can never disagree about who counts
    as a student.
    """
    root = Path(root_dir or DATA_ROOT)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if is_subject_dir(d))


def scan_dataset(student_filter=None):
    if not os.path.exists(DATA_ROOT):
        return {}, []

    students = discover_subjects(DATA_ROOT)

    class_counts = {}
    for student in students:
        if student_filter and student not in student_filter:
            continue
        student_dir = os.path.join(DATA_ROOT, student)
        for gesture in os.listdir(student_dir):
            gesture_dir = os.path.join(student_dir, gesture)
            if not os.path.isdir(gesture_dir):
                continue
            pngs = glob.glob(os.path.join(gesture_dir, "*.png"))
            if pngs:
                class_counts[gesture] = class_counts.get(gesture, 0) + len(pngs)

    return class_counts, students


def _split_subjects(root_dir, val_subjects=1, seed=42):
    """Subject-wise train/val split — mirrors phygo split_subjects()."""
    subjects = discover_subjects(root_dir)
    if not subjects:
        raise ValueError(f"No subjects found in {root_dir}")
    if len(subjects) < 2:
        return subjects, subjects  # single-subject → random split inside dataset
    rng = np.random.default_rng(seed)
    arr = np.array(subjects)
    rng.shuffle(arr)
    n_val = max(1, min(val_subjects, len(arr) - 1))
    return list(arr[n_val:]), list(arr[:n_val])


class DownloadWorker(QtCore.QObject):
    """Downloads a HuggingFace ViT model into the local models/ directory."""
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, model_id: str):
        super().__init__()
        self._model_id = model_id

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from huggingface_hub import snapshot_download
            dest = str(_model_local_dir(self._model_id))
            self.progress.emit(f"Downloading {self._model_id} from HuggingFace…")
            snapshot_download(
                repo_id=self._model_id,
                local_dir=dest,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            self.progress.emit(f"✓ Model saved to {dest}")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QtCore.QObject):
    """
    Runs the same training pipeline as phygo/scripts/train_vit.py:
    - HuggingFace Trainer with per-epoch eval
    - Subject-wise train/val split (no leakage)
    - Spectrogram-safe augmentation
    - Accuracy + macro F1 metrics
    - load_best_model_at_end with f1_macro
    - Saves in HF format (model + processor + labels.json)
    """
    log = QtCore.pyqtSignal(str)
    epoch_done = QtCore.pyqtSignal(int, float, float, float)  # epoch, loss, acc, f1
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, student_filter, epochs, batch_size, lr, val_subjects, seed,
                 output_dir, model_id: str = _HF_MODEL_ID):
        super().__init__()
        self._student_filter = student_filter
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._val_subjects = val_subjects
        self._seed = seed
        self._output_dir = output_dir
        self._model_id = model_id
        self._running = False
        # Set just before trainer.train(); the epoch callback reads it as the
        # baseline for the first epoch's duration.
        self._t_train_start = time.monotonic()
        # True only while trainer.train() is running.  The final
        # trainer.evaluate() also fires on_evaluate, and without this guard it
        # would emit an extra epoch_done and append a phantom point to the
        # charts (re-evaluating the BEST model, so loss appears to jump back up).
        self._in_training = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        t_begin = time.monotonic()
        try:
            # Guard against the CUDA PyTorch DLL issue on Windows devices
            # that don't have an NVIDIA GPU (Surface, school laptops, etc.).
            try:
                import torch
                import torchvision.transforms as T
                from torch.utils.data import Dataset
            except ImportError as ie:
                self.error.emit(
                    "PyTorch could not load — a required DLL is missing.\n\n"
                    "Fix: run  setup_windows.bat  to install the CPU-only build of "
                    "PyTorch, which works on all Windows devices without a GPU.\n\n"
                    f"(Technical detail: {ie})"
                )
                return

            from transformers import (
                AutoImageProcessor, AutoModelForImageClassification,
                TrainingArguments, Trainer,
            )
            import json
            from PIL import Image

            # --- Device (centralised via platform_utils) ---
            from core.platform_utils import get_device, device_label
            _dev = get_device()
            device = str(_dev) if _dev is not None else "cpu"
            self.log.emit(f"Device: {device_label()}")

            # --- Model source ---
            model_src, model_label = _resolve_model(self._model_id)
            if model_src == self._model_id:
                self.log.emit(
                    f"⚠  No local copy of {self._model_id} found — "
                    "attempting HuggingFace download.\n"
                    "   Connect to the internet or click 'Download Model' first."
                )
            else:
                self.log.emit(f"Model: {model_label}  ({self._model_id})")

            processor = AutoImageProcessor.from_pretrained(model_src)

            # --- Subject split ---
            # discover_subjects() only accepts folders that actually contain
            # gesture sub-folders with PNGs, so stray directories (e.g. a saved
            # HF checkpoint) can't be mistaken for a student and produce an
            # empty validation split.
            root = Path(DATA_ROOT)
            all_subjects = discover_subjects(root)
            if not all_subjects:
                raise ValueError(
                    f"No student folders with samples found in {DATA_ROOT}.\n"
                    "Expected layout:  <student>/<gesture>/*.png"
                )
            if self._student_filter:
                all_subjects = [s for s in all_subjects if s in self._student_filter]
                if not all_subjects:
                    raise ValueError(
                        "None of the selected students have any samples."
                    )

            if len(all_subjects) < 2:
                train_subj, val_subj = all_subjects, all_subjects
                self.log.emit("Single subject — using random 80/20 split.")
            else:
                rng = np.random.default_rng(self._seed)
                arr = np.array(all_subjects)
                rng.shuffle(arr)
                n_val = max(1, min(self._val_subjects, len(arr) - 1))
                val_subj = list(arr[:n_val])
                train_subj = list(arr[n_val:])

            self.log.emit(f"Train subjects: {train_subj}")
            self.log.emit(f"Val subjects:   {val_subj}")

            # --- Discover classes ---
            # Only from the subjects actually taking part, and only gesture
            # folders that hold at least one PNG — an empty folder left behind
            # by a cancelled collection run would otherwise become a class that
            # no sample ever belongs to.
            gestures = set()
            for subj in all_subjects:
                subj_dir = root / subj
                if not subj_dir.is_dir():
                    continue
                for g in subj_dir.iterdir():
                    if g.is_dir() and next(g.glob('*.png'), None) is not None:
                        gestures.add(g.name)
            label_names = sorted(gestures)
            if not label_names:
                raise ValueError(
                    "No gesture folders with PNG samples found for the "
                    "selected students."
                )
            label2id = {n: i for i, n in enumerate(label_names)}
            id2label = {i: n for n, i in label2id.items()}
            self.log.emit(f"Classes ({len(label_names)}): {label_names}")

            # --- Transforms (spectrogram-safe augmentation) ---
            train_transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224), scale=(0.85, 1.0)),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.ToTensor(),
                T.Normalize(mean=processor.image_mean, std=processor.image_std),
            ])
            val_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=processor.image_mean, std=processor.image_std),
            ])

            # --- Dataset ---
            class GestureDataset(Dataset):
                def __init__(self_, subjects, transform, is_train):
                    self_.items = []
                    for subj in subjects:
                        subj_dir = root / subj
                        if not subj_dir.exists():
                            continue
                        for g_dir in subj_dir.iterdir():
                            if g_dir.is_dir() and g_dir.name in label2id:
                                for p in g_dir.glob("*.png"):
                                    self_.items.append((p, label2id[g_dir.name]))
                    # random split when same subjects used for train and val
                    if train_subj == val_subj:
                        rng2 = np.random.default_rng(self._seed)
                        idx = np.arange(len(self_.items))
                        rng2.shuffle(idx)
                        cut = int(len(idx) * 0.8)
                        keep = idx[:cut] if is_train else idx[cut:]
                        self_.items = [self_.items[i] for i in keep]
                    self_._transform = transform

                def __len__(self_):
                    return len(self_.items)

                def __getitem__(self_, idx):
                    p, lbl = self_.items[idx]
                    img = Image.open(p).convert("RGB")
                    return {"pixel_values": self_._transform(img), "labels": lbl}

            train_ds = GestureDataset(train_subj, train_transform, is_train=True)
            val_ds   = GestureDataset(val_subj,   val_transform,   is_train=False)
            self.log.emit(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

            if len(train_ds) == 0 or len(val_ds) == 0:
                which = "Training" if len(train_ds) == 0 else "Validation"
                subj = train_subj if len(train_ds) == 0 else val_subj
                counts = ", ".join(
                    f"{s}: {sum(1 for _ in (root / s).glob('*/*.png'))} samples"
                    for s in all_subjects
                )
                raise ValueError(
                    f"{which} set is empty.\n\n"
                    f"{which} subjects: {subj}\n"
                    f"Samples per student — {counts}\n\n"
                    f"Each student folder needs gesture sub-folders with PNGs:\n"
                    f"    {DATA_ROOT}/<student>/<gesture>/*.png\n"
                    f"If a student has no samples, untick them under "
                    f"'Select students' or collect data for them first."
                )

            # A validation subject missing some classes still trains, but the
            # reported accuracy/F1 only cover the classes they actually have.
            val_classes = {
                g.name for s in val_subj for g in (root / s).iterdir()
                if g.is_dir() and next(g.glob('*.png'), None) is not None
            } if val_subj else set()
            missing = sorted(set(label_names) - val_classes)
            if missing:
                self.log.emit(
                    f"⚠  Validation subject(s) {val_subj} have no samples for: "
                    f"{missing}. Accuracy/F1 only reflect the classes they do have."
                )

            # --- Model ---
            self.log.emit("Loading ViT model...")
            model = AutoModelForImageClassification.from_pretrained(
                model_src,
                num_labels=len(label_names),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            # --- Metrics (pure numpy — no network, no sklearn) ---
            # evaluate.load("accuracy"/"f1") downloads the metric script from
            # the HuggingFace Hub at runtime and the f1 script imports sklearn;
            # both fail on offline school machines, so we compute them directly.
            n_classes = len(label_names)

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds  = np.argmax(logits, axis=1)
                labels = np.asarray(labels)
                acc = float((preds == labels).mean()) if len(labels) else 0.0

                f1s = []
                for c in range(n_classes):
                    tp = int(((preds == c) & (labels == c)).sum())
                    fp = int(((preds == c) & (labels != c)).sum())
                    fn = int(((preds != c) & (labels == c)).sum())
                    denom = 2 * tp + fp + fn
                    f1s.append(2 * tp / denom if denom > 0 else 0.0)
                return {"accuracy": acc, "f1_macro": float(np.mean(f1s))}

            def collate_fn(batch):
                import torch
                return {
                    "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                    "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
                }

            # Callback to emit epoch signals back to the UI thread
            worker_self = self

            from transformers import TrainerCallback

            # Mutable holder so the callback can time each epoch without
            # needing its own __init__ (TrainerCallback is instantiated by us
            # but re-entered by the Trainer).
            epoch_clock = {"last": None}

            class _UICallback(TrainerCallback):
                def on_evaluate(self_, args, state, control, metrics=None, **kwargs):
                    if not worker_self._running:
                        control.should_training_stop = True
                        return
                    # Ignore the final standalone evaluate() — it is not an epoch.
                    if not worker_self._in_training:
                        return
                    if metrics and state.epoch is not None:
                        now = time.monotonic()
                        prev = epoch_clock["last"] or worker_self._t_train_start
                        epoch_s = now - prev
                        epoch_clock["last"] = now

                        epoch = int(round(state.epoch))
                        loss = metrics.get("eval_loss", 0.0)
                        acc  = metrics.get("eval_accuracy", 0.0)
                        f1   = metrics.get("eval_f1_macro", 0.0)
                        msg = (
                            f"Epoch {epoch}/{worker_self._epochs} — "
                            f"val_loss: {loss:.4f}  acc: {acc:.2%}  f1: {f1:.3f}"
                            f"  [{_fmt_dur(epoch_s)}]"
                        )
                        worker_self.log.emit(msg)
                        print(f"[SensDS] {msg}", flush=True)
                        worker_self.epoch_done.emit(epoch, loss, acc, f1)

                def on_log(self_, args, state, control, logs=None, **kwargs):
                    if not worker_self._running:
                        control.should_training_stop = True

            fp16 = device == "cuda"

            out_path = Path(self._output_dir)
            ta_kwargs = dict(
                output_dir=str(out_path / "checkpoints"),
                save_strategy="epoch",
                learning_rate=self._lr,
                per_device_train_batch_size=self._batch_size,
                per_device_eval_batch_size=self._batch_size,
                num_train_epochs=self._epochs,
                weight_decay=0.01,
                logging_dir=str(out_path / "logs"),
                logging_steps=20,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                remove_unused_columns=False,
                report_to="none",
                fp16=fp16,
                seed=self._seed,
            )
            # transformers renamed evaluation_strategy → eval_strategy in 4.41;
            # support both so older installs on school machines still work.
            try:
                training_args = TrainingArguments(eval_strategy="epoch", **ta_kwargs)
            except TypeError:
                training_args = TrainingArguments(evaluation_strategy="epoch", **ta_kwargs)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
                callbacks=[_UICallback()],
            )

            setup_s = time.monotonic() - t_begin
            self.log.emit(f"Setup took {_fmt_dur(setup_s)}. Training...")
            print(f"[SensDS] Setup took {_fmt_dur(setup_s)}. Training "
                  f"{self._epochs} epochs on {device}...", flush=True)

            self._t_train_start = time.monotonic()
            self._in_training = True
            try:
                trainer.train()
            finally:
                self._in_training = False
            train_s = time.monotonic() - self._t_train_start

            per_epoch = train_s / max(1, self._epochs)
            self.log.emit(
                f"Training finished in {_fmt_dur(train_s)} "
                f"({_fmt_dur(per_epoch)}/epoch avg)."
            )
            print(f"[SensDS] Training finished in {_fmt_dur(train_s)} "
                  f"({_fmt_dur(per_epoch)}/epoch avg)", flush=True)

            self.log.emit("Evaluating best model...")
            metrics = trainer.evaluate()
            acc = metrics.get("eval_accuracy", 0.0)
            f1  = metrics.get("eval_f1_macro", 0.0)
            self.log.emit(f"Final — acc: {acc:.2%}  f1: {f1:.3f}")

            # --- Save HF model + processor + labels ---
            final_dir = out_path / "model"
            final_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(final_dir))
            processor.save_pretrained(str(final_dir))

            import json
            labels_json = out_path / "labels.json"
            with labels_json.open("w") as f:
                json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

            total_s = time.monotonic() - t_begin
            self.log.emit(f"Model saved to: {final_dir}")
            self.log.emit(f"Total time (setup + training + save): {_fmt_dur(total_s)}")
            print(f"[SensDS] Total time: {_fmt_dur(total_s)}  ->  {final_dir}",
                  flush=True)
            self.finished.emit(str(final_dir))

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def stop(self):
        self._running = False


class TrainTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._c = app_colors()
        self.setObjectName("train_root")
        self.setStyleSheet(_train_style(self._c))
        self._worker = None
        self._thread = None
        self._loss_data = []
        self._acc_data = []
        self._f1_data = []
        # Wall-clock tracking for the on-screen elapsed / ETA readout
        self._train_start: float | None = None
        self._epochs_total = 0
        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left_panel())
        outer.addWidget(self._build_right_panel())

    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Train Model")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Dataset"))

        self._radio_all = QtWidgets.QRadioButton("All students")
        self._radio_all.setChecked(True)
        self._radio_all.toggled.connect(self._on_student_mode_changed)
        layout.addWidget(self._radio_all)

        self._radio_select = QtWidgets.QRadioButton("Select students")
        self._radio_select.toggled.connect(self._on_student_mode_changed)
        layout.addWidget(self._radio_select)

        self._student_list = QtWidgets.QWidget()
        self._student_list_layout = QtWidgets.QVBoxLayout(self._student_list)
        self._student_list_layout.setContentsMargins(16, 4, 0, 4)
        self._student_list_layout.setSpacing(4)
        self._student_list.setVisible(False)
        layout.addWidget(self._student_list)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Dataset Status"))
        self._status_classes = QtWidgets.QLabel("")
        self._status_classes.setWordWrap(True)
        layout.addWidget(self._status_classes)
        self._status_samples = QtWidgets.QLabel("")
        layout.addWidget(self._status_samples)
        self._status_ready = QtWidgets.QLabel("")
        layout.addWidget(self._status_ready)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Config"))

        row_epochs = QtWidgets.QHBoxLayout()
        row_epochs.addWidget(self._lbl("Epochs"))
        self._epochs = QtWidgets.QSpinBox()
        self._epochs.setRange(1, 50)
        self._epochs.setValue(15)
        self._epochs.setToolTip(
            "An epoch is one full run through all your samples.\n"
            "Think of it like rereading your notes before a test — the more\n"
            "times you review, the more the model remembers (up to a point).\n"
            "15 is a solid starting number for most gesture projects."
        )
        row_epochs.addWidget(self._epochs)
        layout.addLayout(row_epochs)

        row_batch = QtWidgets.QHBoxLayout()
        row_batch.addWidget(self._lbl("Batch size"))
        self._batch_size = QtWidgets.QSpinBox()
        self._batch_size.setRange(1, 64)
        self._batch_size.setValue(8)
        self._batch_size.setToolTip(
            "Batch size = how many samples the model studies at once\n"
            "before it adjusts itself. It's like grading 8 quizzes at a time\n"
            "instead of one at a time — faster but uses more memory.\n"
            "Keep it at 8 for small datasets; increase it if you have 200+ samples."
        )
        row_batch.addWidget(self._batch_size)
        layout.addLayout(row_batch)

        row_lr = QtWidgets.QHBoxLayout()
        row_lr.addWidget(self._lbl("Learning rate"))
        self._lr = QtWidgets.QDoubleSpinBox()
        self._lr.setDecimals(6)
        self._lr.setRange(0.000001, 0.01)
        self._lr.setSingleStep(0.00001)
        self._lr.setValue(0.00002)
        self._lr.setToolTip(
            "Learning rate controls how fast the model adjusts after a mistake.\n"
            "Too high → it overcorrects and goes haywire (loss shoots up).\n"
            "Too low → it barely changes, like studying 1 word per hour.\n"
            "0.00002 is the sweet spot that works for almost every gesture project."
        )
        row_lr.addWidget(self._lr)
        layout.addLayout(row_lr)

        row_val = QtWidgets.QHBoxLayout()
        row_val.addWidget(self._lbl("Val subjects"))
        self._val_subjects = QtWidgets.QSpinBox()
        self._val_subjects.setRange(1, 10)
        self._val_subjects.setValue(1)
        self._val_subjects.setToolTip(
            "Val subjects = how many students are kept secret from the model\n"
            "during training and used only for the final accuracy test.\n"
            "It's like having a classmate quiz you on new questions — not\n"
            "the ones you already practiced. Keeps the score fair and honest."
        )
        row_val.addWidget(self._val_subjects)
        layout.addLayout(row_val)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Model Size"))
        self._model_size = QtWidgets.QComboBox()
        self._model_size.addItem("Small (vit-small-patch16-224, faster)")
        self._model_size.addItem("Base (vit-base-patch16-224, more accurate)")
        self._model_size.setCurrentIndex(0)   # default: Small
        self._model_size.setToolTip(
            "Small: ~22 M parameters — trains and runs significantly faster.\n"
            "Recommended for Surface Pro and other CPU-only devices.\n\n"
            "Base: ~86 M parameters — higher accuracy but slower to train.\n"
            "Use if you have a machine with a dedicated GPU or more time."
        )
        self._model_size.currentIndexChanged.connect(self._on_model_size_changed)
        layout.addWidget(self._model_size)

        layout.addWidget(self._lbl("Base Model"))
        self._model_status = QtWidgets.QLabel("")
        self._model_status.setWordWrap(True)
        layout.addWidget(self._model_status)

        self._download_btn = QtWidgets.QPushButton("⬇  Download Model (once)")
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self._c['panel']};
                border: 1px solid {self._c['border']};
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 12px;
                color: {self._c['text']};
            }}
            QPushButton:hover {{ background: {self._c['tab_hover']}; }}
            QPushButton:disabled {{ color: {self._c['faint']}; }}
        """)
        self._download_btn.clicked.connect(self._start_download)
        layout.addWidget(self._download_btn)

        layout.addStretch()

        layout.addWidget(HintCard([
            "Epochs: one full run through all your samples. "
            "More = more practice for the model. Watch the chart — "
            "if the orange line stops climbing, it's done learning.",
            "Batch size: how many samples the model sees at once before updating. "
            "8 is great for small datasets. Think of it like studying in groups of 8.",
            "Learning rate: how big a step the model takes when it makes a mistake. "
            "0.00002 is the sweet spot — don't change it unless things go wrong.",
            "Val subjects: classmates kept secret from the model during training. "
            "Their data is used only to check if the model actually learned — not to cheat.",
            "Green accuracy line rising = model is getting smarter. "
            "Orange F1 line is more trustworthy when you have unequal numbers of each gesture.",
            "Training can take several minutes — let it run! "
            "The chart updates after each epoch so you can watch progress live.",
            "The model learns from spectrogram images — "
            "it's basically learning to read radar pictures of your hand movements.",
        ], c=self._c))

        self._train_btn = QtWidgets.QPushButton("▶  Start Training")
        self._train_btn.setObjectName("train_btn")
        self._train_btn.clicked.connect(self._start_training)
        self._train_btn.setEnabled(False)
        layout.addWidget(self._train_btn)

        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._stop_training)
        self._stop_btn.setVisible(False)
        layout.addWidget(self._stop_btn)

        return _scrollable_left(panel, width=300)

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(f"background: {self._c['bg']};")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header row: section title on the left, live elapsed/ETA on the right.
        hdr = QtWidgets.QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(self._lbl_section("Training Progress  (per epoch)"))
        hdr.addStretch()
        self._timer_lbl = QtWidgets.QLabel("")
        self._timer_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {self._c['subtext']};"
            " font-family: monospace;"
        )
        hdr.addWidget(self._timer_lbl)
        layout.addLayout(hdr)

        self._chart_bg   = '#1a1a2e' if self._c['panel'] != '#ffffff' else '#f7f8fc'
        self._axis_pen   = pg.mkPen('w') if self._c['panel'] != '#ffffff' else pg.mkPen('#333')
        self._axis_color = '#ccc' if self._c['panel'] != '#ffffff' else '#333'

        # Three separate charts instead of one shared axis.  Accuracy and F1 are
        # fractions in [0, 1] but validation loss is unbounded and typically
        # starts near ln(n_classes) ~ 1.1, so on a shared 0-1.05 axis the loss
        # curve was clipped off the top and effectively invisible.  Separate
        # axes let loss auto-scale to its own range.
        charts_row = QtWidgets.QWidget()
        charts_layout = QtWidgets.QHBoxLayout(charts_row)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(10)

        self._acc_chart, self._acc_curve = self._make_metric_chart(
            "Accuracy", '#2ecc71', y_range=(0, 1.05),
        )
        self._f1_chart, self._f1_curve = self._make_metric_chart(
            "F1 (macro)", '#f39c12', y_range=(0, 1.05),
        )
        # No y_range -> pyqtgraph auto-scales to the observed loss values.
        self._loss_chart, self._loss_curve = self._make_metric_chart(
            "Val Loss", '#e74c3c',
        )

        for chart in (self._acc_chart, self._f1_chart, self._loss_chart):
            charts_layout.addWidget(chart, 1)

        layout.addWidget(charts_row, 3)   # takes 3/4 of flexible space

        layout.addWidget(self._lbl_section("Training Log"))

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(60)
        layout.addWidget(self._log, 1)     # takes 1/4 of flexible space

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addStretch()
        self._open_model_btn = QtWidgets.QPushButton("📂  Open Models Folder")
        self._open_model_btn.setStyleSheet(f"""
            QPushButton {{
                background: {self._c['panel']};
                border: 1px solid {self._c['border']};
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 12px;
                color: {self._c['text']};
            }}
            QPushButton:hover {{ background: {self._c['tab_hover']}; }}
        """)
        self._open_model_btn.clicked.connect(self._open_models_folder)
        bottom_row.addWidget(self._open_model_btn)
        layout.addLayout(bottom_row)

        return panel

    def _make_metric_chart(self, title: str, color: str, y_range=None):
        """
        Build one metric chart and return (chart, curve).

        y_range=None leaves auto-scaling on, which is what unbounded metrics
        like validation loss need.
        """
        chart = pg.PlotWidget()
        chart.setBackground(self._chart_bg)
        chart.setTitle(title, color=color, size='10pt', bold=True)
        chart.setLabel('bottom', 'Epoch', color=self._axis_color)
        for ax in ('left', 'bottom'):
            chart.getAxis(ax).setPen(self._axis_pen)
            chart.getAxis(ax).setTextPen(self._axis_pen)
        chart.showGrid(x=True, y=True, alpha=0.3)
        if y_range is not None:
            chart.setYRange(*y_range, padding=0)
        chart.setMinimumHeight(120)
        chart.setMinimumWidth(160)

        # Symbols matter here: with a line-only plot the very first epoch
        # renders as nothing at all (a single point has no segment to draw).
        curve = chart.plot(
            [], [],
            pen=pg.mkPen(color, width=2),
            symbol='o', symbolSize=6,
            symbolBrush=color, symbolPen=None,
        )
        return chart, curve

    def _set_chart_title(self, chart, base: str, color: str, value: str | None):
        """Show the latest value in the chart title so small plots stay readable."""
        text = base if value is None else f"{base}   {value}"
        chart.setTitle(text, color=color, size='10pt', bold=True)

    def _lbl(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("field_label")
        return l

    def _lbl_section(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("section_heading")
        return l

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {self._c['divider']}; margin: 2px 0;")
        return line

    def _selected_model_id(self) -> str:
        """Return the HuggingFace model ID matching the current dropdown selection."""
        key = self._model_size.currentText().split()[0]   # "Small" or "Base"
        return _MODEL_OPTIONS.get(key, _HF_MODEL_ID)

    def _on_model_size_changed(self):
        """Refresh the model status label when the user changes the size dropdown."""
        self._refresh_model_status()

    def _refresh_model_status(self):
        mid = self._selected_model_id()
        if model_is_available_offline(mid):
            _, label = _resolve_model(mid)
            self._model_status.setObjectName("status_ok")
            self._model_status.setText(f"✓  {label}")
            self._download_btn.setEnabled(False)
            self._download_btn.setText("✓  Model ready")
        else:
            self._model_status.setObjectName("status_warn")
            self._model_status.setText("⚠  Not downloaded — training requires internet without this")
            self._download_btn.setEnabled(True)
            self._download_btn.setText("⬇  Download Model (once)")
        self._model_status.style().unpolish(self._model_status)
        self._model_status.style().polish(self._model_status)

    def _start_download(self):
        self._download_btn.setEnabled(False)
        self._download_btn.setText("Downloading…")
        self._model_status.setObjectName("status_warn")
        self._model_status.setText("Downloading model weights, please wait…")

        dl_worker = DownloadWorker(self._selected_model_id())
        dl_thread = QtCore.QThread(self)
        dl_worker.moveToThread(dl_thread)
        dl_thread.started.connect(dl_worker.run)
        dl_worker.progress.connect(lambda msg: self._model_status.setText(msg))
        dl_worker.finished.connect(lambda: (dl_thread.quit(), self._refresh_model_status()))
        dl_worker.error.connect(lambda err: (
            dl_thread.quit(),
            self._model_status.setText(f"✗  Download failed: {err}"),
            self._download_btn.setEnabled(True),
            self._download_btn.setText("⬇  Download Model (once)"),
        ))
        dl_thread.finished.connect(dl_thread.deleteLater)
        dl_thread.start()

    def refresh(self):
        self._refresh_model_status()
        class_counts, students = scan_dataset()

        while self._student_list_layout.count():
            item = self._student_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._student_checkboxes = {}
        for student in sorted(students):
            cb = QtWidgets.QCheckBox(student)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_status)
            self._student_list_layout.addWidget(cb)
            self._student_checkboxes[student] = cb

        self._all_students = students
        self._update_status()

    def _on_student_mode_changed(self):
        self._student_list.setVisible(self._radio_select.isChecked())
        self._update_status()

    def _get_selected_students(self):
        if self._radio_all.isChecked():
            return None
        return [s for s, cb in self._student_checkboxes.items() if cb.isChecked()]

    def _update_status(self):
        student_filter = self._get_selected_students()
        class_counts, _ = scan_dataset(student_filter)

        n_classes = len(class_counts)
        n_samples = sum(class_counts.values())
        ready = n_classes >= MIN_CLASSES and n_samples >= MIN_SAMPLES

        if n_classes >= MIN_CLASSES:
            self._status_classes.setObjectName("status_ok")
            self._status_classes.setText(
                f"✓  {n_classes} gesture classes: {', '.join(sorted(class_counts.keys()))}"
            )
        else:
            self._status_classes.setObjectName("status_err")
            self._status_classes.setText(
                f"✗  {n_classes} gesture classes (need {MIN_CLASSES})"
            )

        if n_samples >= MIN_SAMPLES:
            self._status_samples.setObjectName("status_ok")
            self._status_samples.setText(f"✓  {n_samples} total samples")
        else:
            self._status_samples.setObjectName("status_err")
            self._status_samples.setText(
                f"✗  {n_samples} samples (need {MIN_SAMPLES})"
            )

        if ready:
            self._status_ready.setObjectName("status_ok")
            self._status_ready.setText("✓  Ready to train")
        else:
            self._status_ready.setObjectName("status_warn")
            self._status_ready.setText("Collect more data to unlock training")

        for lbl in (self._status_classes, self._status_samples, self._status_ready):
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

        self._train_btn.setEnabled(ready)

    def _start_training(self):
        student_filter = self._get_selected_students()

        if student_filter is None:
            _, all_students = scan_dataset()
            subjects = sorted(all_students)
        else:
            subjects = sorted(student_filter)
        base_name = "_".join(subjects) if subjects else "all_students"
        if len(base_name) > 60:
            base_name = base_name[:60]
        os.makedirs(MODELS_DIR, exist_ok=True)
        version = 1
        while os.path.isdir(os.path.join(MODELS_DIR, f"{base_name}_v{version}")):
            version += 1
        output_dir = os.path.join(MODELS_DIR, f"{base_name}_v{version}")

        self._loss_data = []
        self._acc_data = []
        self._f1_data = []
        self._loss_curve.setData([], [])
        self._acc_curve.setData([], [])
        self._f1_curve.setData([], [])
        self._set_chart_title(self._acc_chart,  "Accuracy",   '#2ecc71', None)
        self._set_chart_title(self._f1_chart,   "F1 (macro)", '#f39c12', None)
        self._set_chart_title(self._loss_chart, "Val Loss",   '#e74c3c', None)
        self._log.clear()
        self._log.appendPlainText("Starting training...")

        # Start the on-screen clock
        self._train_start = time.monotonic()
        self._epochs_total = self._epochs.value()
        self._timer_lbl.setText("⏱  0s elapsed   ·   starting…")
        self._elapsed_timer.start()

        self._worker = TrainWorker(
            student_filter=student_filter,
            epochs=self._epochs.value(),
            batch_size=self._batch_size.value(),
            lr=self._lr.value(),
            val_subjects=self._val_subjects.value(),
            seed=42,
            output_dir=output_dir,
            model_id=self._selected_model_id(),
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.epoch_done.connect(self._on_epoch_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

        self._train_btn.setVisible(False)
        self._stop_btn.setVisible(True)

    def _stop_training(self):
        if self._worker:
            self._worker.stop()

    # ── elapsed / ETA readout ────────────────────────────────────────────────

    def _tick_elapsed(self):
        """Update the on-screen timer once a second while training runs."""
        if self._train_start is None:
            return
        elapsed = time.monotonic() - self._train_start
        done = len(self._acc_data)
        if done > 0 and self._epochs_total > 0:
            # ETA from the average epoch time so far.  Only meaningful once at
            # least one epoch has completed; before that we just show elapsed.
            per_epoch = elapsed / done
            remaining = max(0.0, per_epoch * (self._epochs_total - done))
            self._timer_lbl.setText(
                f"⏱  {_fmt_dur(elapsed)} elapsed   ·   ~{_fmt_dur(remaining)} left"
                f"   ·   epoch {done}/{self._epochs_total}"
            )
        else:
            self._timer_lbl.setText(f"⏱  {_fmt_dur(elapsed)} elapsed   ·   starting…")

    def _stop_elapsed(self, final_note: str = ""):
        self._elapsed_timer.stop()
        if self._train_start is not None:
            total = time.monotonic() - self._train_start
            self._timer_lbl.setText(f"{final_note}{_fmt_dur(total)}")
        self._train_start = None

    def _on_log(self, msg):
        self._log.appendPlainText(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_epoch_done(self, epoch, loss, acc, f1):
        self._loss_data.append(loss)
        self._acc_data.append(acc)
        self._f1_data.append(f1)
        epochs = list(range(1, len(self._loss_data) + 1))
        self._loss_curve.setData(epochs, self._loss_data)
        self._acc_curve.setData(epochs, self._acc_data)
        self._f1_curve.setData(epochs, self._f1_data)

        # Surface the latest value in each title — the three charts are narrow,
        # so reading exact values off the axes is awkward.
        self._set_chart_title(self._acc_chart,  "Accuracy",   '#2ecc71', f"{acc:.1%}")
        self._set_chart_title(self._f1_chart,   "F1 (macro)", '#f39c12', f"{f1:.3f}")
        self._set_chart_title(self._loss_chart, "Val Loss",   '#e74c3c', f"{loss:.4f}")

    def _on_finished(self, model_path):
        self._cleanup_thread()
        self._stop_elapsed("✓  Trained in ")
        self._log.appendPlainText(f"\n✓ Training complete. Model saved to:\n{model_path}")
        self._train_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    def _on_error(self, msg):
        self._cleanup_thread()
        self._stop_elapsed("✗  Failed after ")
        self._log.appendPlainText(f"\n✗ Error: {msg}")
        self._train_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def _open_models_folder(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", MODELS_DIR])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", MODELS_DIR])
        else:
            subprocess.Popen(["xdg-open", MODELS_DIR])
