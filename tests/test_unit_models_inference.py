"""Unit tests for core/models_inference.sam3_video_propagate with a faked
ultralytics SAM3VideoPredictor — no real model, GPU, or video decoding."""

import numpy as np
import pytest


class _FakeDataset:
    """Minimal video dataset: yields one batch per frame and advances
    `frame` after each read, like ultralytics' LoadImagesAndVideos."""

    def __init__(self, frames=4, shape=(80, 100, 3)):
        self.frames = frames
        self.frame = 0
        self._img = np.zeros(shape, np.uint8)

    def __iter__(self):
        for _ in range(self.frames):
            self.frame += 1  # loader increments after each read (1-based)
            yield (["f0"], [self._img.copy()], ["video"])


class _FakeVideoPredictor:
    """Stands in for ultralytics.models.sam.SAM3VideoPredictor.

    Tracks 2 objects; object 1's mask is blank from (1-based) frame
    `blank_from` on."""

    instances = []

    def __init__(self, overrides=None):
        self.overrides = overrides
        self.dataset = _FakeDataset()
        self.inference_state = {}
        self.batch = None
        self.model = None  # engine reads model.mask_threshold via getattr
        self.seeded = []
        self.blank_from = 3
        _FakeVideoPredictor.instances.append(self)

    def setup_model(self):
        pass

    def setup_source(self, source):
        self.source = str(source)

    def init_state(self, predictor):
        predictor.inference_state["ready"] = True

    def preprocess(self, ims):
        return np.zeros((1, 3, 80, 100), np.float32)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None,
                         points=None, labels=None, masks=None):
        n = len(bboxes)
        return np.zeros((n, 1, 2)), np.ones((n, 1)), None

    def add_new_prompts(self, obj_id, points=None, labels=None,
                        frame_idx=None, **kw):
        self.seeded.append((obj_id, frame_idx))

    def propagate_in_video_preflight(self):
        pass

    def propagate_in_video(self, state, frame_idx):
        import torch
        masks = torch.ones((2, 80, 100))
        if frame_idx >= self.blank_from:
            masks[1] = 0.0
        return [0, 1], masks, torch.zeros(2)


@pytest.fixture
def fake_video_predictor(monkeypatch):
    _FakeVideoPredictor.instances.clear()
    import ultralytics.models.sam as sam_mod
    monkeypatch.setattr(sam_mod, "SAM3VideoPredictor", _FakeVideoPredictor)
    return _FakeVideoPredictor


def test_video_propagate_obj_id_alignment(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    frames = list(sam3_video_propagate("/tmp/clip.mp4",
                                       [[1, 1, 9, 9], [20, 20, 30, 30]],
                                       model_path="m.pt", device="cpu"))
    assert [f for f, _ in frames] == [0, 1, 2, 3]
    # seed 0 found on every frame (all-ones 80x100 mask → full-frame box)
    assert all(per[0][1] == [0.0, 0.0, 99.0, 79.0] for _, per in frames)
    # seed 1 lost from frame 2 on — slot stays index 1, not shifted
    assert frames[0][1][1][1] is not None
    assert frames[1][1][1][1] is not None
    assert frames[2][1][1] == (None, None, 0.0)
    assert frames[3][1][1] == (None, None, 0.0)
    # both seeds registered as prompts on the first (1-based) frame
    assert fake_video_predictor.instances[0].seeded == [(0, 1), (1, 1)]


def test_video_propagate_cancel_stops_stream(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    # one seed, but the fake returns obj 1 too — the out-of-range guard
    # must skip it without crashing
    frames = list(sam3_video_propagate("/tmp/clip.mp4", [[1, 1, 9, 9]],
                                       model_path="m.pt", device="cpu",
                                       is_cancelled=cancel))
    assert len(frames) == 1  # frame 0 yielded, frame 1 cancelled


def test_video_propagate_default_model_path(fake_video_predictor):
    from core.models_inference import sam3_video_propagate
    list(sam3_video_propagate("/tmp/clip.mp4", [[1, 1, 9, 9]], device="cpu"))
    pred = fake_video_predictor.instances[0]
    assert pred.overrides["model"].endswith("sam3.pt")
    assert pred.overrides["imgsz"] == 1024
    assert pred.overrides["device"] == "cpu"
    assert pred.source == "/tmp/clip.mp4"
