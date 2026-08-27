"""Unit tests for gui/label_review/workers/label_review_workers.py: the pure helpers
(_segment_concepts, _autolabel_frame) with a monkeypatched
run_sam3 / _load_sam3_semantic, and the QThread workers driven
synchronously (worker.run() instead of start()) so no real threads,
models, or inference are involved."""

import os

import numpy as np
import pytest

from conftest import FakeIdx


# ---------------------------------------------------------------------------
# _segment_concepts
# ---------------------------------------------------------------------------

def _ok_result(dets, masks):
    return {"success": True, "detections": dets, "masks": masks}


def test_segment_concepts_iou_pairing(workers, monkeypatch):
    """Each input box is paired with the detection whose bbox has the
    highest IoU with it."""
    m_left = np.zeros((10, 10), bool)
    m_left[1:3, 1:3] = True
    m_right = np.zeros((10, 10), bool)
    m_right[7:9, 7:9] = True

    def fake_run_sam3(image_path, bboxes, concepts, model_path, device,
                      conf):
        # deliberately return detections in reversed order
        return _ok_result(
            [{"bbox": [7, 7, 9, 9]}, {"bbox": [1, 1, 3, 3]}],
            [m_right, m_left])

    monkeypatch.setattr(workers, "run_sam3", fake_run_sam3)
    results, device, cancelled = workers._segment_concepts(
        "/img/x.png", [[0, 0, 4, 4], [6, 6, 10, 10]], ["a", "a"], [11, 12],
        None, "cpu", 0.25)
    assert not cancelled and device == "cpu"
    assert len(results) == 2
    by_ann = {r["ann_id"]: r for r in results}
    assert by_ann[11]["mask"] is m_left
    assert by_ann[12]["mask"] is m_right
    assert all(r["success"] for r in results)


def test_segment_concepts_fallback_to_kth_mask(workers, monkeypatch):
    """No detections at all → fall back to the k-th mask by position."""
    m0 = np.ones((4, 4), bool)
    m1 = np.zeros((4, 4), bool)
    m1[2:, 2:] = True
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([], [m0, m1]))
    results, _, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1], [1, 1, 2, 2]], ["a", "a"], [1, 2],
        None, "cpu", 0.25)
    by_ann = {r["ann_id"]: r for r in results}
    assert by_ann[1]["mask"] is m0 and by_ann[2]["mask"] is m1
    assert all(r["success"] for r in results)


def test_segment_concepts_empty_mask_is_failure(workers, monkeypatch):
    """An all-empty mask counts as a failure (mask=None): it would show
    nothing, save nothing, and block later SAM3-ALL re-runs on the box."""
    m_empty = np.zeros((4, 4), bool)
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([], [m_empty]))
    results, _, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cpu", 0.25)
    assert results[0]["success"] is False
    assert results[0]["mask"] is None


def test_segment_concepts_no_mask_is_failure(workers, monkeypatch):
    monkeypatch.setattr(workers, "run_sam3",
                        lambda **kw: _ok_result([], []))
    results, _, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cpu", 0.25)
    assert results[0]["success"] is False
    assert results[0]["error"] == "no matching mask"


def test_segment_concepts_run_failure_reported(workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: {"success": False, "error": "boom"})
    results, _, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1], [1, 1, 2, 2]], ["a", "a"], [1, 2],
        None, "cpu", 0.25)
    assert [r["success"] for r in results] == [False, False]
    assert all(r["error"] == "boom" for r in results)


def test_segment_concepts_exception_reported(workers, monkeypatch):
    calls = []

    def boom(**kw):
        calls.append(kw["device"])
        raise RuntimeError("not an oom")

    monkeypatch.setattr(workers, "run_sam3", boom)
    results, device, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cuda", 0.25)
    assert results[0]["success"] is False and "not an oom" in results[0]["error"]
    # ANY cuda failure (not just OOM) falls back to CPU once; the error
    # still surfaces when the CPU attempt fails too.
    assert calls == ["cuda", "cpu"]
    assert device == "cpu"


def test_segment_concepts_cuda_oom_retries_on_cpu(workers, monkeypatch):
    calls = []

    def fake_run_sam3(image_path, bboxes, concepts, model_path, device,
                      conf):
        calls.append(device)
        if device != "cpu":
            raise RuntimeError("CUDA out of memory")
        return _ok_result([{"bbox": [0, 0, 1, 1]}], [np.ones((2, 2), bool)])

    monkeypatch.setattr(workers, "run_sam3", fake_run_sam3)
    results, device, cancelled = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cuda", 0.25)
    assert calls == ["cuda", "cpu"]
    assert device == "cpu" and not cancelled
    assert results[0]["success"] is True


def test_segment_concepts_oom_cpu_retry_failure(workers, monkeypatch):
    def always_oom(**kw):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(workers, "run_sam3", always_oom)
    results, device, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cuda", 0.25)
    assert device == "cpu"
    assert results[0]["success"] is False


def test_segment_concepts_oom_in_returned_error_retries(workers,
                                                        monkeypatch):
    """run_sam3 swallows CUDA OOM into a failure dict (never raises) — the
    CPU fallback is keyed off the returned error string."""
    calls = []

    def fake_run_sam3(image_path, bboxes, concepts, model_path, device,
                      conf):
        calls.append(device)
        if device != "cpu":
            return {"success": False, "error": "CUDA out of memory"}
        return _ok_result([{"bbox": [0, 0, 1, 1]}], [np.ones((2, 2), bool)])

    monkeypatch.setattr(workers, "run_sam3", fake_run_sam3)
    results, device, cancelled = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cuda", 0.25)
    assert calls == ["cuda", "cpu"]
    assert device == "cpu" and not cancelled
    assert results[0]["success"] is True


def test_segment_concepts_cancel_between_concepts(workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 1, 1]}],
                                [np.ones((2, 2), bool)]))
    checks = iter([False, True])  # first concept runs, then cancel fires
    results, _, cancelled = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1], [1, 1, 2, 2]], ["a", "b"], [1, 2],
        None, "cpu", 0.25, cancel_check=lambda: next(checks))
    assert cancelled is True
    assert len(results) == 1  # partial results for the completed concept
    assert results[0]["label"] == "a"


def test_segment_concepts_progress_callback(workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 1, 1]}],
                                [np.ones((2, 2), bool)]))
    progress = []
    workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1], [1, 1, 2, 2]], ["a", "b"], [1, 2],
        None, "cpu", 0.25,
        progress_cb=lambda d, t, c: progress.append((d, t, c)))
    assert progress == [(1, 2, "a"), (2, 2, "b")]


# ---------------------------------------------------------------------------
# _autolabel_frame (with a fake semantic predictor)
# ---------------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)

    def __len__(self):
        return len(self.xyxy.numpy())


class _FakeMasks:
    def __init__(self, data):
        self.data = _FakeTensor(data)


class _FakeResult:
    def __init__(self, boxes=None, masks=None):
        self.boxes = boxes
        self.masks = masks


class _FakePred:
    def __init__(self, result):
        self.prompts = None
        self.result = result

    def set_prompts(self, p):
        self.prompts = p

    def __call__(self, source):
        return [self.result]


def test_autolabel_frame_det_mapping(workers):
    mask0 = np.zeros((6, 6), np.uint8)
    mask0[1:3, 1:3] = 1
    result = _FakeResult(
        boxes=_FakeBoxes([[0, 0, 4, 4], [1, 1, 5, 5], [2, 2, 3, 3]],
                         cls=[0, 1, 7],          # 7 out of range → "object"
                         conf=[0.9, 0.5, 0.1]),
        masks=_FakeMasks([mask0, mask0, mask0]))
    pred = _FakePred(result)
    dets = workers._autolabel_frame(pred, "/img/x.png", ["cat", "dog"])
    assert pred.prompts == {"text": ["cat", "dog"]}
    assert len(dets) == 3
    assert dets[0]["label"] == "cat" and dets[0]["confidence"] == 0.9
    assert dets[0]["bbox_xyxy"] == [0.0, 0.0, 4.0, 4.0]
    assert dets[0]["mask"].dtype == bool and dets[0]["mask"][1, 1]
    assert dets[1]["label"] == "dog"
    assert dets[2]["label"] == "object"  # cls id beyond the concept list


def test_autolabel_frame_no_boxes(workers):
    assert workers._autolabel_frame(_FakePred(_FakeResult()), "/i.png",
                                    ["a"]) == []
    empty = _FakeResult(boxes=_FakeBoxes(np.zeros((0, 4)), [], []))
    assert workers._autolabel_frame(_FakePred(empty), "/i.png", ["a"]) == []


def test_autolabel_frame_without_masks(workers):
    result = _FakeResult(boxes=_FakeBoxes([[0, 0, 4, 4]], [0], [0.7]),
                         masks=None)
    dets = workers._autolabel_frame(_FakePred(result), "/i.png", ["a"])
    assert dets[0]["mask"] is None


# ---------------------------------------------------------------------------
# QThread workers driven synchronously
# ---------------------------------------------------------------------------

@pytest.fixture
def sam3_on(workers, monkeypatch):
    monkeypatch.setattr(workers, "_SAM3_AVAILABLE", True)


def test_sam3_worker_run_finished(sam3_on, workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 4, 4]}],
                                [np.ones((4, 4), bool)]))
    w = workers.SAM3Worker("/img/x.png", [[0, 0, 4, 4]], ["a"], [7],
                           None, "cpu", 0.25)
    got = {}
    w.finished_signal.connect(lambda r: got.setdefault("results", r))
    w.failed_signal.connect(lambda e: got.setdefault("error", e))
    w.run()  # synchronous, no event loop needed
    assert "error" not in got
    assert got["results"][0]["ann_id"] == 7
    assert got["results"][0]["success"] is True


def test_sam3_worker_run_cancelled(sam3_on, workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 4, 4]}],
                                [np.ones((4, 4), bool)]))
    w = workers.SAM3Worker("/img/x.png", [[0, 0, 4, 4]], ["a"], [7],
                           None, "cpu", 0.25)
    got = []
    w.cancelled_signal.connect(lambda: got.append("cancelled"))
    w.finished_signal.connect(lambda r: got.append("finished"))
    w.cancel()  # cancel before run → fires at the first concept check
    w.run()
    assert got == ["cancelled"]


def test_sam3_worker_unavailable_fails(workers, monkeypatch):
    monkeypatch.setattr(workers, "_SAM3_AVAILABLE", False)
    w = workers.SAM3Worker("/img/x.png", [[0, 0, 4, 4]], ["a"], [7],
                           None, "cpu", 0.25)
    got = []
    w.failed_signal.connect(lambda e: got.append(e))
    w.run()
    assert got and "not installed" in got[0]


def test_batch_worker_cancel_between_frames(sam3_on, workers, monkeypatch,
                                            tmp_path):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 4, 4]}],
                                [np.ones((4, 4), bool)]))
    idx = FakeIdx(3)
    jobs = [{"frame_idx": i, "bboxes_xyxy": [[0, 0, 4, 4]],
             "concepts": ["a"], "ann_ids": [i + 1]} for i in range(3)]
    w = workers.SAM3BatchWorker(idx, jobs, str(tmp_path / "batch"),
                                None, "cpu", 0.25)
    events = []
    w.frame_done_signal.connect(
        lambda fidx, results: (events.append(("frame", fidx)), w.cancel()))
    w.finished_signal.connect(lambda ok, fail: events.append(("finished",)))
    w.cancelled_signal.connect(lambda: events.append(("cancelled",)))
    w.run()
    assert ("frame", 0) in events
    assert ("frame", 1) not in events  # stopped after the first frame
    assert events[-1] == ("cancelled",)


def test_batch_worker_all_frames(sam3_on, workers, monkeypatch, tmp_path):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [0, 0, 4, 4]}],
                                [np.ones((4, 4), bool)]))
    idx = FakeIdx(2)
    jobs = [{"frame_idx": i, "bboxes_xyxy": [[0, 0, 4, 4]],
             "concepts": ["a"], "ann_ids": [i + 1]} for i in range(2)]
    w = workers.SAM3BatchWorker(idx, jobs, str(tmp_path / "batch2"),
                                None, "cpu", 0.25)
    events = []
    w.frame_done_signal.connect(lambda f, r: events.append(f))
    done = []
    w.finished_signal.connect(lambda ok, fail: done.append((ok, fail)))
    w.run()
    assert events == [0, 1]
    assert done == [(2, 0)]


def test_autolabel_worker_maps_cat_ids(sam3_on, workers, monkeypatch):
    def fake_fallback(image_path, concepts, model_path, device, conf):
        return ([{"label": "monitor", "bbox_xyxy": [0, 0, 4, 4],
                  "mask": None, "confidence": 0.9},
                 {"label": "???", "bbox_xyxy": [5, 5, 9, 9],
                  "mask": None, "confidence": 0.1}], device, object())
    monkeypatch.setattr(workers, "_autolabel_with_fallback", fake_fallback)
    w = workers.SAM3AutolabelWorker("/img/x.png", ["ceiling light",
                                                   "monitor"], [0, 1], 42,
                                    None, "cpu", 0.25)
    got = {}
    w.finished_signal.connect(
        lambda image_id, dets: got.update(image_id=image_id, dets=dets))
    w.failed_signal.connect(lambda e: got.setdefault("error", e))
    w.run()
    assert got["image_id"] == 42
    assert [d["cat_id"] for d in got["dets"]] == [1, None]


def test_autolabel_batch_worker_cancel(sam3_on, workers, monkeypatch,
                                       tmp_path):
    # _autolabel_with_fallback returns (dets, device, predictor) — the
    # predictor is reused across frames.
    monkeypatch.setattr(
        workers, "_autolabel_with_fallback",
        lambda *a, **k: ([{"label": "a", "bbox_xyxy": [0, 0, 4, 4],
                           "mask": None, "confidence": 0.5}], "cpu", None))
    w = workers.SAM3AutolabelBatchWorker(FakeIdx(3), [0, 1, 2], ["a"], [0],
                                         str(tmp_path / "al"), None, "cpu",
                                         0.25)
    events = []
    w.frame_done_signal.connect(
        lambda f, dets: (events.append(f), w.cancel()))
    w.cancelled_signal.connect(lambda: events.append("cancelled"))
    w.run()
    assert events == [0, "cancelled"]


# SAM3PropagateWorker (memory-bank engine, monkeypatched)
# ---------------------------------------------------------------------------

def _fake_engine(scripted):
    """Fake sam3_video_propagate: scripted is one per_seed list per video
    frame. Records the video path + seeds it was called with."""

    def fake(video_path, seed_bboxes_xyxy, is_cancelled=None, **kw):
        fake.video_path = video_path
        fake.seeds = [list(b) for b in seed_bboxes_xyxy]
        assert os.path.exists(video_path)  # the mp4 was really built
        for k, per_seed in enumerate(scripted):
            if is_cancelled is not None and is_cancelled():
                return
            yield k, per_seed

    return fake


def test_propagate_worker_streams_multi_seed(sam3_on, workers, monkeypatch,
                                             tmp_path):
    m = np.ones((80, 100), bool)
    scripted = [
        [(m, [5, 5, 15, 15], 1.0), (m, [40, 40, 50, 50], 1.0)],  # seed frame
        [(m, [10, 10, 20, 20], 0.9), (m, [41, 41, 51, 51], 0.8)],
        [(m, [11, 11, 21, 21], 0.9), (None, None, 0.0)],  # seed 1 lost
        [(m, [12, 12, 22, 22], 0.9), (None, None, 0.0)],
    ]
    fake = _fake_engine(scripted)
    monkeypatch.setattr(workers, "sam3_video_propagate", fake)
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0},
             {"bbox_xyxy": [40, 40, 50, 50], "track_id": 2, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(4), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    frames, fin = [], []
    w.frame_done_signal.connect(lambda f, dets: frames.append((f, dets)))
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert [f for f, _ in frames] == [1, 2, 3]  # seed frame not emitted
    assert frames[0][1][0]["bbox_xyxy"] == [10, 10, 20, 20]
    assert frames[0][1][1]["bbox_xyxy"] == [41, 41, 51, 51]
    assert frames[1][1][1] is None              # lost seed → None slot
    assert fin == [(4, {1: 2})]  # 4 boxes found; seed 1 lost from frame 2
    assert fake.seeds == [[5, 5, 15, 15], [40, 40, 50, 50]]
    assert not os.path.exists(fake.video_path)  # temp mp4 cleaned up


def test_propagate_worker_cancel_mid_stream(sam3_on, workers, monkeypatch,
                                            tmp_path):
    m = np.ones((80, 100), bool)
    scripted = [[(m, [0, 0, 5, 5], 1.0)] for _ in range(5)]
    monkeypatch.setattr(workers, "sam3_video_propagate",
                        _fake_engine(scripted))
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(5), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    events = []

    def on_frame(f, dets):
        events.append(f)
        w.cancel()

    w.frame_done_signal.connect(on_frame)
    w.cancelled_signal.connect(lambda: events.append("cancelled"))
    w.run()
    assert events == [1, "cancelled"]
    assert not os.path.exists(
        os.path.join(str(tmp_path / "prop"), "propagate_clip.mp4"))


def test_propagate_worker_noop_at_last_frame(sam3_on, workers, tmp_path):
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(1), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    fin = []
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert fin == [(0, {})]


def test_iou_xyxy(workers):
    assert workers._iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert workers._iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    iou = workers._iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 25 / 175) < 1e-9
    assert workers._iou_xyxy([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_propagate_worker_reports_clip_build_progress(sam3_on, workers,
                                                      monkeypatch, tmp_path):
    """The clip-build phase (silent before) emits stage lines, and a
    'loading model' stage precedes the streaming phase."""
    m = np.ones((80, 100), bool)
    scripted = [[(m, [0, 0, 5, 5], 1.0)] for _ in range(4)]
    monkeypatch.setattr(workers, "sam3_video_propagate",
                        _fake_engine(scripted))
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0}]
    w = workers.SAM3PropagateWorker(FakeIdx(4), 0, seeds,
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    stages = []
    w.stage_signal.connect(stages.append)
    w.run()
    assert stages[0] == "building clip 1/4…"
    assert "building clip 4/4…" in stages
    assert stages[-1] == "loading model — propagation starts shortly…"


# ---------------------------------------------------------------------------
# SAM3PropagateWorker (chain mode)
# ---------------------------------------------------------------------------

def test_propagate_worker_chain_mode(sam3_on, workers, monkeypatch, tmp_path):
    """Chain mode calls _propagate_step per alive seed per frame, emits the
    same frame_done contract, and never builds an mp4."""
    def fake_step(img_path, prev, concept, model_path, device, conf,
                  min_iou, seed_bbox_xyxy, min_seed_iou):
        # Drift 1px right per frame so we can verify the chain advances.
        x1, y1, x2, y2 = prev
        return {
            "bbox_xyxy": [x1 + 1, y1, x2 + 1, y2],
            "mask": None,
            "confidence": 0.9,
        }, device

    monkeypatch.setattr(workers, "_propagate_step", fake_step)
    seeds = [
        {"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0,
         "concept": "a"},
        {"bbox_xyxy": [40, 40, 50, 50], "track_id": 2, "cat_id": 0,
         "concept": "b"},
    ]
    w = workers.SAM3PropagateWorker(
        FakeIdx(4), 0, seeds, str(tmp_path / "prop"), None, "cpu", 0.25,
        method="chain")
    frames, fin = [], []
    w.frame_done_signal.connect(lambda f, dets: frames.append((f, dets)))
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert [f for f, _ in frames] == [1, 2, 3]
    assert frames[0][1][0]["bbox_xyxy"] == [6, 5, 16, 15]
    assert frames[0][1][1]["bbox_xyxy"] == [41, 40, 51, 50]
    assert frames[2][1][0]["bbox_xyxy"] == [8, 5, 18, 15]
    assert fin == [(6, {})]  # 3 frames * 2 seeds
    assert not os.path.exists(
        os.path.join(str(tmp_path / "prop"), "propagate_clip.mp4"))


def test_propagate_worker_chain_lost_seed_stops(sam3_on, workers, monkeypatch,
                                                tmp_path):
    """In chain mode a seed that returns None is permanently lost; other
    seeds continue."""
    calls = []

    def fake_step(img_path, prev, concept, model_path, device, conf,
                  min_iou, seed_bbox_xyxy, min_seed_iou):
        calls.append((concept, list(prev)))
        # First call (seed 0, frame 1) reports lost.
        if len(calls) == 1:
            return None, device
        x1, y1, x2, y2 = prev
        return {"bbox_xyxy": [x1 + 1, y1, x2 + 1, y2],
                "mask": None, "confidence": 0.9}, device

    monkeypatch.setattr(workers, "_propagate_step", fake_step)
    seeds = [
        {"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0,
         "concept": "lost"},
        {"bbox_xyxy": [40, 40, 50, 50], "track_id": 2, "cat_id": 0,
         "concept": "kept"},
    ]
    w = workers.SAM3PropagateWorker(
        FakeIdx(4), 0, seeds, str(tmp_path / "prop"), None, "cpu", 0.25,
        method="chain")
    frames, fin = [], []
    w.frame_done_signal.connect(lambda f, dets: frames.append((f, dets)))
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert all(f[1][0] is None for f in frames)  # seed 0 never reappears
    assert frames[0][1][1] is not None
    assert fin == [(3, {0: 1})]  # 3 kept detections; seed 0 lost at frame 1


def test_propagate_worker_chain_cancel(sam3_on, workers, monkeypatch,
                                       tmp_path):
    """Cancel is honored between frames in chain mode."""
    def fake_step(img_path, prev, concept, model_path, device, conf,
                  min_iou, seed_bbox_xyxy, min_seed_iou):
        x1, y1, x2, y2 = prev
        return {"bbox_xyxy": [x1 + 1, y1, x2 + 1, y2],
                "mask": None, "confidence": 0.9}, device

    monkeypatch.setattr(workers, "_propagate_step", fake_step)
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0,
              "concept": "a"}]
    w = workers.SAM3PropagateWorker(
        FakeIdx(5), 0, seeds, str(tmp_path / "prop"), None, "cpu", 0.25,
        method="chain")
    events = []

    def on_frame(f, dets):
        events.append(f)
        w.cancel()

    w.frame_done_signal.connect(on_frame)
    w.cancelled_signal.connect(lambda: events.append("cancelled"))
    w.run()
    assert events == [1, "cancelled"]


def test_propagate_worker_chain_passes_iou_thresholds(sam3_on, workers,
                                                      monkeypatch, tmp_path):
    """The worker forwards min_iou / min_seed_iou to _propagate_step."""
    seen = {}

    def fake_step(img_path, prev, concept, model_path, device, conf,
                  min_iou, seed_bbox_xyxy, min_seed_iou):
        seen["min_iou"] = min_iou
        seen["min_seed_iou"] = min_seed_iou
        return None, device

    monkeypatch.setattr(workers, "_propagate_step", fake_step)
    seeds = [{"bbox_xyxy": [5, 5, 15, 15], "track_id": 1, "cat_id": 0,
              "concept": "a"}]
    w = workers.SAM3PropagateWorker(
        FakeIdx(2), 0, seeds, str(tmp_path / "prop"), None, "cpu", 0.25,
        method="chain", min_iou=0.5, min_seed_iou=0.4)
    w.run()
    assert seen == {"min_iou": 0.5, "min_seed_iou": 0.4}


# ---------------------------------------------------------------------------
# Generic open-set autolabel backends (Grounding DINO / Falcon)
# and OWLv2 exemplar workers — detector fns monkeypatched, no HF downloads.
# ---------------------------------------------------------------------------

def test_build_grounding_dino_prompt():
    from core.detectors import build_grounding_dino_prompt
    # lowercased, " . "-joined, trailing period, existing periods stripped
    assert build_grounding_dino_prompt(["Door", "Exit Sign."]) == \
        "door . exit sign ."


def test_generic_detect_dispatch_grounding_dino(workers, monkeypatch):
    import core.detectors as gd
    calls = {}

    def fake(img, concepts, model_id=None, device="cuda",
             box_threshold=0.35, text_threshold=0.25, _state=None):
        calls.update(img=img, concepts=concepts, model_id=model_id,
                     device=device, box_threshold=box_threshold,
                     state=_state)
        return [{"label": "door", "bbox_xyxy": [0, 0, 4, 4], "mask": None,
                 "confidence": 0.9}]

    monkeypatch.setattr(gd, "grounding_dino_detect", fake)
    state = {}
    dets = workers._generic_detect("grounding_dino", "/img/x.png", ["door"],
                                   "m/gd", "cpu", 0.5, state)
    assert len(dets) == 1
    # conf maps to Grounding DINO's box_threshold; state is passed through
    assert calls["box_threshold"] == 0.5
    assert calls["state"] is state and calls["model_id"] == "m/gd"


def test_generic_detect_dispatch_falcon(workers, monkeypatch):
    import core.detectors as fa
    monkeypatch.setattr(
        fa, "falcon_detect",
        lambda *a, **k: [{"label": "y", "bbox_xyxy": [0, 0, 2, 2],
                          "mask": None, "confidence": 1.0}])
    assert workers._generic_detect(
        "falcon", "/i.png", ["y"], None, "cpu", 0.3, None
    )[0]["label"] == "y"
    with pytest.raises(ValueError, match="Unknown autolabel detector"):
        workers._generic_detect("nope", "/i.png", ["x"], None, "cpu", 0.3,
                                None)


def test_generic_autolabel_worker_maps_cat_ids(workers, monkeypatch):
    monkeypatch.setattr(
        workers, "_generic_detect",
        lambda detector, img, concepts, model_id, device, conf, state: [
            {"label": "monitor", "bbox_xyxy": [0, 0, 4, 4], "mask": None,
             "confidence": 0.9},
            {"label": "???", "bbox_xyxy": [5, 5, 9, 9], "mask": None,
             "confidence": 0.1}])
    w = workers.GenericAutolabelWorker("falcon", "/img/x.png",
                                       ["ceiling light", "monitor"], [0, 1],
                                       42, None, "cpu", 0.35)
    got = {}
    w.finished_signal.connect(lambda i, d: got.update(image_id=i, dets=d))
    w.failed_signal.connect(lambda e: got.setdefault("error", e))
    w.run()
    assert got["image_id"] == 42
    assert [d["cat_id"] for d in got["dets"]] == [1, None]


def test_generic_autolabel_worker_failure_reported(workers, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model missing")

    monkeypatch.setattr(workers, "_generic_detect", boom)
    w = workers.GenericAutolabelWorker("grounding_dino", "/img/x.png",
                                       ["a"], [0], 7, None, "cpu", 0.3)
    got = {}
    w.failed_signal.connect(lambda e: got.update(error=e))
    w.run()
    assert "grounding_dino" in got["error"] and "model missing" in got["error"]


def test_generic_autolabel_batch_worker(workers, monkeypatch, tmp_path):
    monkeypatch.setattr(
        workers, "_generic_detect",
        lambda detector, img, concepts, model_id, device, conf, state: [
            {"label": "a", "bbox_xyxy": [0, 0, 4, 4], "mask": None,
             "confidence": 0.5}])
    w = workers.GenericAutolabelBatchWorker(
        "grounding_dino", FakeIdx(3), [0, 1, 2], ["a"], [0],
        str(tmp_path / "al"), None, "cpu", 0.35)
    events = []
    w.frame_done_signal.connect(lambda f, dets: events.append((f, dets)))
    w.finished_signal.connect(lambda total: events.append(("done", total)))
    w.run()
    assert [e[0] for e in events[:3]] == [0, 1, 2]
    assert events[3] == ("done", 3)
    assert all(e[1][0]["cat_id"] == 0 for e in events[:3])


def test_generic_autolabel_batch_worker_cancel(workers, monkeypatch,
                                               tmp_path):
    monkeypatch.setattr(
        workers, "_generic_detect",
        lambda *a, **k: [{"label": "a", "bbox_xyxy": [0, 0, 4, 4],
                          "mask": None, "confidence": 0.5}])
    w = workers.GenericAutolabelBatchWorker(
        "falcon", FakeIdx(3), [0, 1, 2], ["a"], [0], str(tmp_path / "al"),
        None, "cpu", 0.3)
    events = []
    w.frame_done_signal.connect(
        lambda f, dets: (events.append(f), w.cancel()))
    w.cancelled_signal.connect(lambda: events.append("cancelled"))
    w.run()
    assert events == [0, "cancelled"]


def test_owlv2_exemplar_worker(workers, monkeypatch):
    import core.detectors as od
    assert workers._OWLV2_AVAILABLE
    calls = {}

    def fake(img, exemplar, label, model_id=None, device="cuda", conf=0.3,
             _state=None):
        calls.update(img=img, label=label,
                     exemplar_shape=tuple(exemplar.shape), conf=conf)
        return [{"label": label, "bbox_xyxy": [1, 1, 5, 5], "mask": None,
                 "confidence": 0.8}]

    monkeypatch.setattr(od, "owlv2_detect_exemplar", fake)
    crop = np.zeros((10, 12, 3), np.uint8)
    w = workers.Owlv2ExemplarWorker("/img/x.png", crop, "door", 3, 42,
                                    "m/ow", "cpu", 0.4)
    got = {}
    w.finished_signal.connect(lambda i, d: got.update(image_id=i, dets=d))
    w.run()
    assert calls["label"] == "door"
    assert calls["exemplar_shape"] == (10, 12, 3)
    assert calls["conf"] == 0.4 and calls["img"] == "/img/x.png"
    assert got["image_id"] == 42 and got["dets"][0]["cat_id"] == 3


def test_owlv2_exemplar_batch_worker(workers, monkeypatch, tmp_path):
    import core.detectors as od
    monkeypatch.setattr(
        od, "owlv2_detect_exemplar",
        lambda *a, **k: [{"label": k.get("label") or a[2],
                          "bbox_xyxy": [0, 0, 4, 4], "mask": None,
                          "confidence": 0.8}])
    crop = np.zeros((8, 8, 3), np.uint8)
    w = workers.Owlv2ExemplarBatchWorker(
        FakeIdx(2), [0, 1], crop, "door", 5, str(tmp_path / "ex"), None,
        "cpu", 0.3)
    events = []
    w.frame_done_signal.connect(lambda f, dets: events.append((f, dets)))
    w.finished_signal.connect(lambda total: events.append(("done", total)))
    w.run()
    assert [e[0] for e in events[:2]] == [0, 1]
    assert events[2] == ("done", 2)
    assert all(e[1][0]["cat_id"] == 5 for e in events[:2])


# ---------------------------------------------------------------------------
# core.detectors falcon helpers (no HF download — model/load monkeypatched)
# ---------------------------------------------------------------------------

def test_falcon_bbox_conversion(monkeypatch):
    """Normalized centre+size -> absolute xyxy using the image size."""
    import core.detectors as fd

    class FakeModel:
        def generate(self, img, query, compile=False):
            return [[{"xy": {"x": 0.5, "y": 0.5},
                      "hw": {"h": 0.2, "w": 0.4},
                      "mask_rle": {"counts": "", "size": [100, 200]}}]]

    monkeypatch.setattr(fd, "load_falcon", lambda *a, **k: FakeModel())
    img = np.zeros((100, 200, 3), np.uint8)  # H=100, W=200
    dets = fd.falcon_detect(img, ["door"], device="cpu")
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "door"
    assert d["bbox_xyxy"] == pytest.approx([60, 40, 140, 60])
    assert d["mask"] is None  # empty counts decode to None
    assert d["confidence"] == 1.0


def test_falcon_decode_mask_roundtrip():
    pytest.importorskip("pycocotools")
    from pycocotools import mask as mask_utils
    from core.detectors import _decode_mask
    m = np.zeros((10, 12), dtype=np.uint8, order="F")
    m[2:5, 3:8] = 1
    rle = mask_utils.encode(m)
    out = _decode_mask({"size": list(rle["size"]),
                        "counts": rle["counts"].decode("utf-8")})
    assert out is not None and out.shape == (10, 12)
    assert np.array_equal(out, m.astype(bool))


def test_falcon_decode_mask_empty():
    from core.detectors import _decode_mask
    assert _decode_mask({"counts": "", "size": [10, 10]}) is None
    assert _decode_mask({}) is None
