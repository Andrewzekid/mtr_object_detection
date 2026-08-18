"""Unit tests for scripts/label_review_workers.py: the pure helpers
(_segment_concepts, _propagate_step, _autolabel_frame) with a monkeypatched
run_sam3 / _load_sam3_semantic, and the QThread workers driven
synchronously (worker.run() instead of start()) so no real threads,
models, or inference are involved."""

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
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([], [m0, m1]))
    results, _, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1], [1, 1, 2, 2]], ["a", "a"], [1, 2],
        None, "cpu", 0.25)
    by_ann = {r["ann_id"]: r for r in results}
    assert by_ann[1]["mask"] is m0 and by_ann[2]["mask"] is m1
    assert all(r["success"] for r in results)


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
    def boom(**kw):
        raise RuntimeError("not an oom")
    monkeypatch.setattr(workers, "run_sam3", boom)
    results, device, _ = workers._segment_concepts(
        "/img/x.png", [[0, 0, 1, 1]], ["a"], [1], None, "cuda", 0.25)
    assert results[0]["success"] is False and "not an oom" in results[0]["error"]
    assert device == "cuda"  # no retry for non-OOM errors


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
# _propagate_step
# ---------------------------------------------------------------------------

def test_propagate_step_picks_best_iou(workers, monkeypatch):
    m_near = np.ones((4, 4), bool)
    m_far = np.zeros((4, 4), bool)
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result(
            [{"bbox": [50, 50, 60, 60], "confidence": 0.4},
             {"bbox": [10, 10, 20, 20], "confidence": 0.9}],
            [m_far, m_near]))
    det, device = workers._propagate_step("/img/x.png", [9, 9, 21, 21],
                                          "obj", None, "cpu", 0.25)
    assert device == "cpu"
    assert det["bbox_xyxy"] == [10, 10, 20, 20]
    assert det["mask"] is m_near
    assert det["confidence"] == 0.9


def test_propagate_step_lost_on_empty(workers, monkeypatch):
    monkeypatch.setattr(workers, "run_sam3", lambda **kw: _ok_result([], []))
    det, device = workers._propagate_step("/img/x.png", [0, 0, 5, 5],
                                          "obj", None, "cpu", 0.25)
    assert det is None


def test_propagate_step_lost_below_min_iou(workers, monkeypatch):
    """A detection that doesn't overlap the previous box is not 'the same
    object' — treat the track as lost."""
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: _ok_result([{"bbox": [90, 90, 99, 99]}], []))
    det, _ = workers._propagate_step("/img/x.png", [0, 0, 5, 5],
                                     "obj", None, "cpu", 0.25)
    assert det is None


def test_propagate_step_failure_raises(workers, monkeypatch):
    monkeypatch.setattr(
        workers, "run_sam3",
        lambda **kw: {"success": False, "error": "nope"})
    with pytest.raises(RuntimeError, match="nope"):
        workers._propagate_step("/img/x.png", [0, 0, 5, 5], "obj", None,
                                "cpu", 0.25)


def test_propagate_step_cuda_oom_retries_on_cpu(workers, monkeypatch):
    calls = []

    def fake_run_sam3(image_path, bboxes, concepts, model_path, device,
                      conf):
        calls.append(device)
        if device != "cpu":
            raise RuntimeError("CUDA out of memory")
        return _ok_result([{"bbox": [0, 0, 5, 5]}], [])

    monkeypatch.setattr(workers, "run_sam3", fake_run_sam3)
    det, device = workers._propagate_step("/img/x.png", [0, 0, 5, 5],
                                          "obj", None, "cuda", 0.25)
    assert calls == ["cuda", "cpu"]
    assert device == "cpu"
    assert det["bbox_xyxy"] == [0, 0, 5, 5]
    assert det["mask"] is None  # dets without masks still fine


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


def test_propagate_worker_chains_and_stops_when_lost(sam3_on, workers,
                                                     monkeypatch, tmp_path):
    det1 = {"bbox_xyxy": [10, 10, 20, 20], "mask": None, "confidence": 0.9}
    scripted = {1: det1, 2: None}  # found on frame 1, lost on frame 2
    calls = []

    def fake_step(image_path, prev_bbox, concept, model_path, device, conf):
        calls.append(list(prev_bbox))
        frame_idx = int(image_path.split("_")[-1].split(".")[0])
        return scripted[frame_idx], device

    monkeypatch.setattr(workers, "_propagate_step", fake_step)
    idx = FakeIdx(4)
    w = workers.SAM3PropagateWorker(idx, 0, [5, 5, 15, 15], "obj",
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    frames = []
    w.frame_done_signal.connect(lambda f, det: frames.append((f, det)))
    fin = []
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert [f for f, _ in frames] == [1, 2]
    assert frames[0][1] == det1 and frames[1][1] is None
    assert fin == [(1, 2)]  # 1 box found, lost at frame 2
    # chained: frame 2 was prompted with frame 1's detected box
    assert calls == [[5, 5, 15, 15], [10, 10, 20, 20]]


def test_propagate_worker_noop_at_last_frame(sam3_on, workers, tmp_path):
    w = workers.SAM3PropagateWorker(FakeIdx(1), 0, [5, 5, 15, 15], "obj",
                                    str(tmp_path / "prop"), None, "cpu",
                                    0.25)
    fin = []
    w.finished_signal.connect(lambda n, lost: fin.append((n, lost)))
    w.run()
    assert fin == [(0, -1)]


def test_iou_xyxy(workers):
    assert workers._iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert workers._iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    iou = workers._iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 25 / 175) < 1e-9
    assert workers._iou_xyxy([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0
