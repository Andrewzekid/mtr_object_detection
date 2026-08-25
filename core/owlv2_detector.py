"""OWLv2 zero-shot detector for GUI auto-labelling.

Wraps HuggingFace ``Owlv2Processor`` + ``Owlv2ForObjectDetection``
(default checkpoint: ``google/owlv2-large-patch14-ensemble`` - the largest
OWLv2 release). Text queries come straight from the session's category
names; detections come back in the same dict shape the SAM3 autolabel
path produces:

    {label, bbox_xyxy, mask: None, confidence, cat_id (added by caller)}

The model is loaded lazily on first use and cached per (model_id, device);
a CUDA OOM falls back to CPU once and the fallback is remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_OWLV2_MODEL = "google/owlv2-large-patch14-ensemble"

_model_cache: Dict[tuple, Any] = {}


def load_owlv2(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the OWLv2 processor + model for one device."""
    model_id = model_id or DEFAULT_OWLV2_MODEL
    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)
    model = model.to(device).eval()
    _model_cache[key] = (proc, model)
    return proc, model


def owlv2_detect(image, queries: List[str],
                 model_id: Optional[str] = None, device: str = "cuda",
                 conf: float = 0.3,
                 _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run text-prompted OWLv2 detection on one image.

    ``image`` is a path or an RGB numpy array. ``_state`` lets callers
    (the batch worker) reuse the loaded model across frames and mirrors
    the device fallback; pass a fresh ``{}`` to opt in.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    state = _state if _state is not None else {}
    device = state.get("device", device)
    model_id = model_id or DEFAULT_OWLV2_MODEL
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_owlv2(model_id, device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_owlv2(model_id, device)
                state["model_key"] = (model_id, device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    if isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
    else:
        pil = PILImage.fromarray(np.asarray(image)[..., :3])

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

    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    labels = results.get("labels")
    for i, box in enumerate(boxes.tolist()):
        dets.append({
            "label": queries[int(labels[i])] if labels is not None and
            int(labels[i]) < len(queries) else "object",
            "bbox_xyxy": [float(v) for v in box],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets


def owlv2_detect_exemplar(image, exemplar, label: str,
                          model_id: Optional[str] = None,
                          device: str = "cuda", conf: float = 0.3,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """1-shot image-guided OWLv2 detection: find objects in ``image`` that
    look like the ``exemplar`` crop.

    The exemplar (a crop of an annotated object) goes through the same ViT
    backbone; its visual feature embeddings replace the text-encoder
    embeddings as the query (``model.image_guided_detection``). Every
    returned detection gets ``label`` (the exemplar's category name) — the
    caller maps that to a cat_id.

    ``image`` / ``exemplar``: path or RGB numpy array. ``_state`` mirrors
    ``owlv2_detect`` (model reuse + device fallback).
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    state = _state if _state is not None else {}
    device = state.get("device", device)
    model_id = model_id or DEFAULT_OWLV2_MODEL
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_owlv2(model_id, device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_owlv2(model_id, device)
                state["model_key"] = (model_id, device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    def _pil(img):
        if isinstance(img, (str, Path)):
            return PILImage.open(img).convert("RGB")
        return PILImage.fromarray(np.asarray(img)[..., :3])

    pil, pil_q = _pil(image), _pil(exemplar)
    inputs = proc(images=pil, query_images=pil_q, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.image_guided_detection(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = proc.post_process_image_guided_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=conf)[0]

    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    for i, box in enumerate(boxes.tolist()):
        dets.append({
            "label": label,
            "bbox_xyxy": [float(v) for v in box],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets
