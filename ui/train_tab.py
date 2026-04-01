import os
import glob
import datetime
import numpy as np
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from ui import app_colors, HintCard


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

# Look for a bundled ViT model next to the project root (mirrors phygo layout)
_HERE = Path(__file__).resolve().parent
_BUNDLED_MODEL = _HERE.parent / "models" / "vit-base-patch16-224"
_HF_MODEL_ID = "google/vit-base-patch16-224"


def _hf_cache_snapshot() -> Path | None:
    """Return the path to the locally cached HF snapshot if it exists."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = cache_root / "hub" / "models--google--vit-base-patch16-224" / "snapshots"
    if snapshots.exists():
        candidates = sorted(snapshots.iterdir())
        for snap in reversed(candidates):          # prefer newest
            if (snap / "config.json").exists():
                return snap
    return None


def _resolve_model() -> tuple[str, str]:
    """Return (model_path_or_id, human_label) for the best available source."""
    if _BUNDLED_MODEL.exists() and (_BUNDLED_MODEL / "config.json").exists():
        return str(_BUNDLED_MODEL), f"local bundle ({_BUNDLED_MODEL.name})"
    snap = _hf_cache_snapshot()
    if snap:
        return str(snap), f"HF cache ({snap.parent.parent.name[:20]}…)"
    return _HF_MODEL_ID, "HuggingFace (requires internet)"


def model_is_available_offline() -> bool:
    if _BUNDLED_MODEL.exists() and (_BUNDLED_MODEL / "config.json").exists():
        return True
    return _hf_cache_snapshot() is not None


def scan_dataset(student_filter=None):
    if not os.path.exists(DATA_ROOT):
        return {}, []

    students = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d != "models"
    ]

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
            class_counts[gesture] = class_counts.get(gesture, 0) + len(pngs)

    return class_counts, students


def _split_subjects(root_dir, val_subjects=1, seed=42):
    """Subject-wise train/val split — mirrors phygo split_subjects()."""
    root = Path(root_dir)
    EXCLUDE = {'models', 'predict_temp', 'temp', 'test'}
    subjects = sorted(d.name for d in root.iterdir()
                      if d.is_dir() and d.name not in EXCLUDE)
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
    """Downloads google/vit-base-patch16-224 into the local models/ directory."""
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from huggingface_hub import snapshot_download
            dest = str(_BUNDLED_MODEL)
            self.progress.emit("Downloading ViT model weights from HuggingFace…")
            snapshot_download(
                repo_id=_HF_MODEL_ID,
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

    def __init__(self, student_filter, epochs, batch_size, lr, val_subjects, seed, output_dir):
        super().__init__()
        self._student_filter = student_filter
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._val_subjects = val_subjects
        self._seed = seed
        self._output_dir = output_dir
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        try:
            import torch
            import torchvision.transforms as T
            from torch.utils.data import Dataset
            from transformers import (
                AutoImageProcessor, AutoModelForImageClassification,
                TrainingArguments, Trainer,
            )
            import evaluate as hf_evaluate
            import json
            from PIL import Image

            # --- Device ---
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.log.emit(f"Device: {device}")

            # --- Model source ---
            model_src, model_label = _resolve_model()
            if model_src == _HF_MODEL_ID:
                self.log.emit(
                    "⚠  No local model found — attempting HuggingFace download.\n"
                    "   Connect to the internet or click 'Download Model' first."
                )
            else:
                self.log.emit(f"Model: {model_label}")

            processor = AutoImageProcessor.from_pretrained(model_src)

            # --- Subject split ---
            root = Path(DATA_ROOT)
            EXCLUDE = {'models', 'predict_temp', 'temp', 'test'}
            all_subjects = sorted(d.name for d in root.iterdir()
                                  if d.is_dir() and d.name not in EXCLUDE)
            if self._student_filter:
                all_subjects = [s for s in all_subjects if s in self._student_filter]

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
            gestures = set()
            for subj_dir in root.iterdir():
                if subj_dir.is_dir() and subj_dir.name not in EXCLUDE:
                    for g in subj_dir.iterdir():
                        if g.is_dir():
                            gestures.add(g.name)
            label_names = sorted(gestures)
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
                raise ValueError("Train or val dataset is empty. Check folder structure.")

            # --- Model ---
            self.log.emit("Loading ViT model...")
            model = AutoModelForImageClassification.from_pretrained(
                model_src,
                num_labels=len(label_names),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            # --- Metrics ---
            acc_metric = hf_evaluate.load("accuracy")
            f1_metric  = hf_evaluate.load("f1")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=1)
                acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
                f1  = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
                return {"accuracy": acc, "f1_macro": f1}

            def collate_fn(batch):
                import torch
                return {
                    "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                    "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
                }

            # Callback to emit epoch signals back to the UI thread
            worker_self = self

            from transformers import TrainerCallback

            class _UICallback(TrainerCallback):
                def on_evaluate(self_, args, state, control, metrics=None, **kwargs):
                    if not worker_self._running:
                        control.should_training_stop = True
                        return
                    if metrics and state.epoch is not None:
                        epoch = int(round(state.epoch))
                        loss = metrics.get("eval_loss", 0.0)
                        acc  = metrics.get("eval_accuracy", 0.0)
                        f1   = metrics.get("eval_f1_macro", 0.0)
                        worker_self.log.emit(
                            f"Epoch {epoch}/{worker_self._epochs} — "
                            f"val_loss: {loss:.4f}  acc: {acc:.2%}  f1: {f1:.3f}"
                        )
                        worker_self.epoch_done.emit(epoch, loss, acc, f1)

                def on_log(self_, args, state, control, logs=None, **kwargs):
                    if not worker_self._running:
                        control.should_training_stop = True

            fp16 = device == "cuda"

            out_path = Path(self._output_dir)
            training_args = TrainingArguments(
                output_dir=str(out_path / "checkpoints"),
                eval_strategy="epoch",
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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
                callbacks=[_UICallback()],
            )

            self.log.emit("Training...")
            trainer.train()

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

            self.log.emit(f"Model saved to: {final_dir}")
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
        panel.setFixedWidth(300)
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

        return panel

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(f"background: {self._c['bg']};")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        layout.addWidget(self._lbl_section("Training Progress  (val accuracy & F1 per epoch)"))

        chart_bg = '#1a1a2e' if self._c['panel'] != '#ffffff' else '#f7f8fc'
        axis_pen = pg.mkPen('w') if self._c['panel'] != '#ffffff' else pg.mkPen('#333')
        axis_color = '#ccc' if self._c['panel'] != '#ffffff' else '#333'

        self._chart = pg.PlotWidget()
        self._chart.setBackground(chart_bg)
        self._chart.setLabel('left', 'Value', color=axis_color)
        self._chart.setLabel('bottom', 'Epoch', color=axis_color)
        for ax in ('left', 'bottom'):
            self._chart.getAxis(ax).setPen(axis_pen)
            self._chart.getAxis(ax).setTextPen(axis_pen)
        self._chart.addLegend()
        self._chart.setYRange(0, 1.05, padding=0)
        self._chart.showGrid(x=True, y=True, alpha=0.3)
        self._loss_curve = self._chart.plot(
            [], [], name='Val Loss', pen=pg.mkPen('#e74c3c', width=3)
        )
        self._acc_curve = self._chart.plot(
            [], [], name='Accuracy', pen=pg.mkPen('#2ecc71', width=3)
        )
        self._f1_curve = self._chart.plot(
            [], [], name='F1 (macro)', pen=pg.mkPen('#f39c12', width=3, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._chart.setMinimumHeight(260)
        layout.addWidget(self._chart)

        layout.addWidget(self._lbl_section("Training Log"))

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(160)
        layout.addWidget(self._log)

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

    def _refresh_model_status(self):
        if model_is_available_offline():
            _, label = _resolve_model()
            self._model_status.setObjectName("status_ok")
            self._model_status.setText(f"✓  {label}")
            self._download_btn.setEnabled(False)
            self._download_btn.setText("✓  Model ready")
        else:
            self._model_status.setObjectName("status_warn")
            self._model_status.setText("⚠  Not downloaded — training requires internet without this")
            self._download_btn.setEnabled(True)
        self._model_status.style().unpolish(self._model_status)
        self._model_status.style().polish(self._model_status)

    def _start_download(self):
        self._download_btn.setEnabled(False)
        self._download_btn.setText("Downloading…")
        self._model_status.setObjectName("status_warn")
        self._model_status.setText("Downloading model weights, please wait…")

        dl_worker = DownloadWorker()
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
        self._log.clear()
        self._log.appendPlainText("Starting training...")

        self._worker = TrainWorker(
            student_filter=student_filter,
            epochs=self._epochs.value(),
            batch_size=self._batch_size.value(),
            lr=self._lr.value(),
            val_subjects=self._val_subjects.value(),
            seed=42,
            output_dir=output_dir,
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

    def _on_finished(self, model_path):
        self._cleanup_thread()
        self._log.appendPlainText(f"\n✓ Training complete. Model saved to:\n{model_path}")
        self._train_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    def _on_error(self, msg):
        self._cleanup_thread()
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
