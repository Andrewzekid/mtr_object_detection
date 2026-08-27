"""Shared pytest harness for the label-review application tests.

CRITICAL: ``QT_QPA_PLATFORM=offscreen`` must be set before PyQt6 is
imported, and a QApplication must exist before the app package
(``gui.label_review``) is imported — both happen here, in this
order. No test ever runs real SAM3/torch inference; worker classes are
monkeypatched with the fakes below.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # BEFORE PyQt6 import

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt6 import QtWidgets

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPLIT_COMBINE_PATH = ROOT / "scripts" / "09b_split_and_combine.py"


# ---------------------------------------------------------------------------
# Qt application + module loading
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """The one QApplication for the whole session (must exist before the
    app module is loaded)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["test"])
    return app


@pytest.fixture(scope="session")
def lr(qapp):
    """The label-review package (``gui.label_review``), imported with the
    session QApplication already in place."""
    import gui.label_review
    return gui.label_review


@pytest.fixture(scope="session")
def workers(lr):
    """The label_review_workers module inside the package."""
    from gui.label_review.workers import label_review_workers
    return label_review_workers


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSig:
    """Stand-in for a pyqtSignal: records connected slots; emit() calls
    them synchronously."""

    def __init__(self):
        self._slots = []

    def connect(self, fn):
        self._slots.append(fn)

    def emit(self, *args):
        for fn in list(self._slots):
            fn(*args)


class FakeWorker:
    """Generic stand-in for any SAM3 QThread worker: records ctor kwargs,
    stays 'running' until stop()/cancel() is called."""

    instances = []

    def __init__(self, *a, **kw):
        self.finished_signal = FakeSig()
        self.failed_signal = FakeSig()
        self.progress_signal = FakeSig()
        self.cancelled_signal = FakeSig()
        self.frame_done_signal = FakeSig()
        self._running = False
        self._cancelled = False
        self._terminated = False
        self.args = a
        self.kw = kw
        FakeWorker.instances.append(self)

    def start(self):
        self._running = True

    def cancel(self):
        # Real workers stop shortly after cancel(); model that.
        self._cancelled = True
        self._running = False

    def stop(self):
        self._running = False

    def isRunning(self):
        return self._running

    def wait(self, ms):
        return not self._running

    def terminate(self):
        self._terminated = True
        self._running = False


class FakeAutolabelSingle(FakeWorker):
    """SAM3AutolabelWorker has NO cancel() — model that faithfully (it is
    hidden even from hasattr, like the real class). Keeps its own
    ``instances`` list so tests can tell box workers from autolabel
    workers."""

    instances = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        FakeWorker.instances.pop()  # undo the base-class registration
        FakeAutolabelSingle.instances.append(self)

    def __getattribute__(self, name):
        if name == "cancel":
            raise AttributeError(name)
        return super().__getattribute__(name)


class FakePropagateWorker:
    """Stand-in for SAM3PropagateWorker with the real positional ctor
    signature."""

    instances = []

    def __init__(self, frame_index, start_frame_idx, seeds, tmp_dir,
                 model_path, device, conf, method="memory",
                 min_iou=0.3, min_seed_iou=0.2, end_frame_idx=None,
                 parent=None):
        self.kw = dict(start_frame_idx=start_frame_idx, seeds=seeds,
                       method=method, min_iou=min_iou,
                       min_seed_iou=min_seed_iou, end_frame_idx=end_frame_idx)
        self.frame_done_signal = FakeSig()
        self.progress_signal = FakeSig()
        self.stage_signal = FakeSig()
        self.finished_signal = FakeSig()
        self.failed_signal = FakeSig()
        self.cancelled_signal = FakeSig()
        self._running = False
        self._cancelled = False
        self._meta = {}
        FakePropagateWorker.instances.append(self)

    def start(self):
        self._running = True

    def cancel(self):
        self._cancelled = True
        self._running = False

    def stop(self):
        self._running = False

    def isRunning(self):
        return self._running

    def wait(self, ms):
        return not self._running

    def terminate(self):
        self._terminated = True
        self._running = False


class FakeIdx:
    """In-memory frame index with the ImageFolderIndex interface."""

    timestamps_real = False

    def __init__(self, n=4):
        self.n = n
        self.files = [f"/img/f{i}.png" for i in range(n)]

    def __len__(self):
        return self.n

    def frame_at(self, idx):
        return {"frame_idx": idx, "timestamp_ns": 1000 + idx,
                "log_time_ns": 1000 + idx, "existing_boxes": [],
                "file_path": f"/img/f{idx}.png", "file_name": f"f{idx}.png"}

    def decode_image(self, idx):
        return np.zeros((80, 100, 3), dtype=np.uint8)

    def find_idx_by_timestamp(self, ts):
        return -1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sam3(lr, monkeypatch):
    """Replace every SAM3 worker class with fakes and pretend SAM3 is
    available. Returns a namespace with the fake classes (instances lists
    are reset per test).

    The names are patched on ``gui.label_review.ui.main_window`` — the
    module where ReviewWindow looks them up (patching the package's
    re-exports would have no effect)."""
    from gui.label_review.ui import main_window as mw
    FakeWorker.instances.clear()
    FakeAutolabelSingle.instances.clear()
    FakePropagateWorker.instances.clear()
    monkeypatch.setattr(mw, "SAM3Worker", FakeWorker)
    monkeypatch.setattr(mw, "SAM3BatchWorker", FakeWorker)
    monkeypatch.setattr(mw, "SAM3AutolabelWorker", FakeAutolabelSingle)
    monkeypatch.setattr(mw, "SAM3AutolabelBatchWorker", FakeWorker)
    monkeypatch.setattr(mw, "SAM3PropagateWorker", FakePropagateWorker)
    monkeypatch.setattr(mw, "SAM3PointWorker", FakeWorker)
    monkeypatch.setattr(mw, "_SAM3_AVAILABLE", True)

    class NS:
        pass
    ns = NS()
    ns.any = FakeWorker
    ns.autolabel_single = FakeAutolabelSingle
    ns.propagate = FakePropagateWorker
    return ns


@pytest.fixture
def auto_yes(lr, monkeypatch):
    """Modal question dialogs auto-accept (Yes)."""
    monkeypatch.setattr(
        lr.QMessageBox, "question",
        staticmethod(lambda *a, **k: lr.QMessageBox.StandardButton.Yes))


@pytest.fixture
def make_coco(lr, tmp_path):
    """Factory: CocoState writing into tmp_path."""
    counter = itertools.count()

    def _make(categories=None, name=None):
        name = name or f"out_{next(counter)}.json"
        return lr.CocoState(str(tmp_path / name),
                            categories if categories is not None else [])
    return _make


@pytest.fixture
def make_window(lr, qapp, tmp_path):
    """Factory: cheap ReviewWindow (loaded + shown). Windows are closed
    (skipping the quit confirmation) at test teardown. The persisted
    UI-state file is redirected into tmp_path so tests never write to
    ~/.config."""
    windows = []

    def _make(index=None, coco=None, load=True, show=True, **kw):
        index = index if index is not None else lr.EmptyIndex()
        if coco is None:
            raise ValueError("pass a CocoState from make_coco")
        win = lr.ReviewWindow(index, coco, **kw)
        win._ui_state_path = tmp_path / "ui_state.json"
        if load:
            win._load_current()
        if show:
            win.show()
        windows.append(win)
        return win

    yield _make
    for w in windows:
        w._quit_confirmed = True  # don't pop the unsaved-changes dialog
        w.close()


def make_image_folder(path, names, size=(10, 12), value=128):
    """Create a folder of small solid-color PNG/JPG images. `names` may be
    a list of file names or (name, gray value) pairs."""
    from PIL import Image
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for item in names:
        name, v = item if isinstance(item, tuple) else (item, value)
        arr = np.full((size[0], size[1], 3), v, np.uint8)
        Image.fromarray(arr).save(path / name)
    return path
