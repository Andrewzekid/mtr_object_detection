"""Unified open-set detectors for GUI auto-labelling.

Consolidates the three zero-shot / open-vocabulary detector backends that
previously lived in separate scripts:

* **OWLv2**        - ``owlv2_detect`` (text-prompted) and
  ``owlv2_detect_exemplar`` (1-shot image-guided).
* **Grounding DINO** - ``grounding_dino_detect``.

All detect functions share one contract. Inputs are an image (path or RGB
numpy array), a list of text queries, optional model override, device, a
confidence/box threshold where applicable, and an optional ``_state`` dict
that lets callers (the batch worker) reuse the loaded model across frames.

Detections come back in the same dict shape the SAM3 autolabel path
produces::

    {label, bbox_xyxy, mask: None-or-HxW-bool, confidence, cat_id (added
    by caller)}

Models load lazily on first use, are cached per (model_id, device), and a
CUDA OOM falls back to CPU once with the fallback remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

DEFAULT_OWLV2_MODEL = "google/owlv2-large-patch14-ensemble"
DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"

_model_cache: Dict[tuple, Any] = {}


def _to_pil(image) -> "PILImage.Image":
    """Accept a path or RGB(HWC uint8) array; return an RGB PIL image."""
    from PIL import Image as PILImage

    if isinstance(image, (str, Path)):
        return PILImage.open(image).convert("RGB")
    return PILImage.fromarray(np.asarray(image)[..., :3])


def _load_cached(key: tuple, loader: Callable[[], Any]):
    """Module-level cache shared by all backends."""
    if key not in _model_cache:
        _model_cache[key] = loader()
    return _model_cache[key]


# --------------------------------------------------------------------------- #
# OWLv2
# --------------------------------------------------------------------------- #

def load_owlv2(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the OWLv2 processor + model for one device."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = model_id or DEFAULT_OWLV2_MODEL

    def _load():
        proc = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
        return proc, model.to(device).eval()

    return _load_cached((model_id, device), _load)


def _owlv2_prepare(state: Optional[dict], model_id: str):
    """Return the OWLv2 (proc, model) pair for this call, handling OOM."""
    state = state if state is not None else {}
    device = state.get("device", "cuda")
    key = (DEFAULT_OWLV2_MODEL if model_id is None else model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_owlv2(key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_owlv2(key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    return state["proc"], state["model"], device


def _owlv2_postprocess(results, queries: List[str],
                       label_override: Optional[str] = None,
                       img_size: Optional[tuple] = None
                       ) -> List[Dict[str, Any]]:
    """OWLv2 result dict -> standard detection dicts.

    OWLv2's post-processing rescales boxes to the target size but never
    clips them, so predictions regularly stick out past the image edges;
    when ``img_size`` (w, h) is given, boxes are clamped to the image and
    degenerate ones are dropped."""
    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    labels = results.get("labels")
    for i, box in enumerate(boxes.tolist()):
        x1, y1, x2, y2 = (float(v) for v in box)
        if img_size is not None:
            w_img, h_img = img_size
            x1, x2 = max(0.0, min(x1, w_img)), max(0.0, min(x2, w_img))
            y1, y2 = max(0.0, min(y1, h_img)), max(0.0, min(y2, h_img))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue  # fully (or almost) outside the image
        dets.append({
            "label": label_override if label_override is not None else (
                queries[int(labels[i])] if labels is not None and
                int(labels[i]) < len(queries) else "object"),
            "bbox_xyxy": [x1, y1, x2, y2],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets


def owlv2_detect(image, queries: List[str],
                 model_id: Optional[str] = None, device: str = "cuda",
                 conf: float = 0.3,
                 _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run text-prompted OWLv2 detection on one image."""
    import torch

    proc, model, device = _owlv2_prepare(_state, model_id)
    pil = _to_pil(image)

    try:
        inputs = proc(text=[list(queries)], images=pil, return_tensors="pt")
    except TypeError:  # older signature without nested query lists
        inputs = proc(text=list(queries), images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = proc.post_process_grounded_object_detection(
        outputs, threshold=conf, target_sizes=target_sizes)[0]
    return _owlv2_postprocess(results, queries, img_size=pil.size)


def owlv2_detect_exemplar(image, exemplar, label: str,
                          model_id: Optional[str] = None,
                          device: str = "cuda", conf: float = 0.3,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """1-shot image-guided OWLv2 detection against an exemplar crop.

    Image-guided scores run much hotter than text-prompt scores (random
    patches routinely score 0.3+), so ``conf`` here should be ~0.6 (HF's
    image-guided recipe), not the lower text-query threshold. Boxes are
    clamped to the image bounds."""
    import torch

    proc, model, device = _owlv2_prepare(_state, model_id)
    pil, pil_q = _to_pil(image), _to_pil(exemplar)

    inputs = proc(images=pil, query_images=pil_q, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.image_guided_detection(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = proc.post_process_image_guided_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=conf)[0]
    return _owlv2_postprocess(results, [label], label_override=label,
                              img_size=pil.size)


# --------------------------------------------------------------------------- #
# Grounding DINO
# --------------------------------------------------------------------------- #

def load_grounding_dino(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Grounding DINO processor + model for one device."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = model_id or DEFAULT_GDINO_MODEL

    def _load():
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        return proc, model.to(device).eval()

    return _load_cached((model_id, device), _load)


def build_grounding_dino_prompt(queries: List[str]) -> str:
    """Grounding DINO text prompt: labels joined by " . ", trailing period."""
    return " ".join(f"{q.strip().rstrip('.').lower()} ." for q in queries)


def grounding_dino_detect(image, queries: List[str],
                          model_id: Optional[str] = None,
                          device: str = "cuda",
                          box_threshold: float = 0.35,
                          text_threshold: float = 0.25,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run text-prompted Grounding DINO detection on one image.

    Returned labels are Grounding DINO's matched text spans; they are
    mapped back to the closest query (case-insensitive substring match),
    so the caller's query->cat_id mapping still applies.
    """
    import torch

    model_id = model_id or DEFAULT_GDINO_MODEL
    state = _state if _state is not None else {}
    device = state.get("device", device)
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_grounding_dino(
                key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_grounding_dino(
                    key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    pil = _to_pil(image)

    text = build_grounding_dino_prompt(queries)
    inputs = proc(images=pil, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)

    try:  # transformers 5.x: threshold=, optional input_ids
        results = proc.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]])[0]
    except TypeError:  # older: box_threshold=, input_ids positional
        results = proc.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]])[0]

    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    # transformers >= 4.51 warns that `labels` will become integer ids —
    # prefer the string `text_labels` when present.
    labels = results.get("text_labels") or results.get("labels")
    lowered = [q.lower() for q in queries]
    for i, box in enumerate(boxes.tolist()):
        raw = str(labels[i]).strip() if labels is not None else ""
        # Map the matched text span back to the session's query: exact
        # (case-insensitive) first, then substring containment either way.
        label = next((q for q in queries if q.lower() == raw.lower()), None)
        if label is None:
            label = next((q for q, ql in zip(queries, lowered)
                          if ql in raw.lower() or raw.lower() in ql), None)
        if label is None:
            label = raw or "object"
        dets.append({
            "label": label,
            "bbox_xyxy": [float(v) for v in box],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets


