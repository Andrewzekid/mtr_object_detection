"""Grounding DINO zero-shot detector for GUI auto-labelling.

Wraps HuggingFace ``AutoProcessor`` + ``AutoModelForZeroShotObjectDetection``
(default checkpoint: ``IDEA-Research/grounding-dino-base`` — the largest
Grounding DINO available on the HF hub in transformers format;
``IDEA-Research/grounding-dino-large`` does not exist there). Text prompts
follow Grounding DINO's required syntax — labels lowercased, separated by
" . " and ending with a period, e.g. ``"a dog . a laptop ."``.

Detections come back in the same dict shape the SAM3/OWLv2 autolabel paths
produce:

    {label, bbox_xyxy, mask: None, confidence, cat_id (added by caller)}

The model is loaded lazily on first use and cached per (model_id, device);
a CUDA OOM falls back to CPU once and the fallback is remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"

_model_cache: Dict[tuple, Any] = {}


def load_grounding_dino(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Grounding DINO processor + model for one device."""
    model_id = model_id or DEFAULT_GDINO_MODEL
    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device).eval()
    _model_cache[key] = (proc, model)
    return proc, model


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

    ``image`` is a path or an RGB numpy array. ``_state`` lets callers
    (the batch worker) reuse the loaded model across frames and mirrors
    the device fallback; pass a fresh ``{}`` to opt in.

    Returned labels are Grounding DINO's matched text spans; they are
    mapped back to the closest query (case-insensitive substring match),
    so the caller's query->cat_id mapping still applies.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    state = _state if _state is not None else {}
    device = state.get("device", device)
    model_id = model_id or DEFAULT_GDINO_MODEL
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_grounding_dino(model_id, device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_grounding_dino(
                    model_id, device)
                state["model_key"] = (model_id, device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    if isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
    else:
        pil = PILImage.fromarray(np.asarray(image)[..., :3])

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
