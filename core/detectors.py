"""Unified open-set detectors for GUI auto-labelling.

Consolidates the four zero-shot / open-vocabulary detector backends that
previously lived in separate scripts:

* **OWLv2**        - ``owlv2_detect`` (text-prompted) and
  ``owlv2_detect_exemplar`` (1-shot image-guided).
* **Grounding DINO** - ``grounding_dino_detect``.
* **Florence-2**   - ``florence2_detect`` (phrase grounding).
* **Falcon Perception** - ``falcon_detect`` (grounding + instance masks).

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
DEFAULT_FLORENCE2_MODEL = "microsoft/Florence-2-large"
GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
DEFAULT_FALCON_MODEL = "tiiuae/Falcon-Perception"

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
                       label_override: Optional[str] = None
                       ) -> List[Dict[str, Any]]:
    """OWLv2 result dict -> standard detection dicts."""
    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    labels = results.get("labels")
    for i, box in enumerate(boxes.tolist()):
        dets.append({
            "label": label_override if label_override is not None else (
                queries[int(labels[i])] if labels is not None and
                int(labels[i]) < len(queries) else "object"),
            "bbox_xyxy": [float(v) for v in box],
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
    return _owlv2_postprocess(results, queries)


def owlv2_detect_exemplar(image, exemplar, label: str,
                          model_id: Optional[str] = None,
                          device: str = "cuda", conf: float = 0.3,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """1-shot image-guided OWLv2 detection against an exemplar crop."""
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
    return _owlv2_postprocess(results, [label], label_override=label)


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


# --------------------------------------------------------------------------- #
# Florence-2
# --------------------------------------------------------------------------- #

def _patch_transformers5_legacy() -> None:
    """Shims so Florence-2's 2023-era remote code loads under transformers 5.x.

    - ``forced_bos_token_id`` / ``forced_eos_token_id`` were removed from
      ``PretrainedConfig``; the remote config class reads them -> restore
      class-level ``None`` defaults.
    """
    from transformers import PretrainedConfig

    for attr in ("forced_bos_token_id", "forced_eos_token_id"):
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, None)


def load_florence2(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Florence-2 processor + model for one device.

    The native transformers Florence2 classes cannot consume the original
    ``microsoft/Florence-2-large`` checkpoint (weight keys differ), so this
    uses the repo's remote code with a few transformers-5.x compatibility
    shims (see _patch_transformers5_legacy and below).
    """
    model_id = model_id or DEFAULT_FLORENCE2_MODEL

    def _load():
        import torch
        import torch.nn as nn
        from transformers import AutoImageProcessor, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E501

        _patch_transformers5_legacy()

        # The remote processor's __init__ reads tokenizer.additional_special_tokens
        # (removed from slow tokenizers in transformers 5.x) — default it.
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if not hasattr(tok, "additional_special_tokens"):
            tok.additional_special_tokens = []
        img_proc = AutoImageProcessor.from_pretrained(model_id)
        proc_cls = get_class_from_dynamic_module(
            "processing_florence2.Florence2Processor", model_id)
        proc = proc_cls(image_processor=img_proc, tokenizer=tok)

        # The remote wrapper defines _supports_sdpa / _supports_flash_attn_2 as
        # properties that dereference self.language_model — which does not exist
        # yet when transformers 5.x checks them during __init__. Replace the
        # properties with plain False (eager attention) before instantiation.
        model_cls = get_class_from_dynamic_module(
            "modeling_florence2.Florence2ForConditionalGeneration", model_id)
        for klass in model_cls.__mro__:
            for attr in ("_supports_sdpa", "_supports_flash_attn_2"):
                if isinstance(getattr(klass, "__dict__", {}).get(attr),
                              property):
                    setattr(klass, attr, False)

        torch_dtype = torch.float16 if device != "cpu" \
            and torch.cuda.is_available() else torch.float32
        model = model_cls.from_pretrained(
            model_id, dtype=torch_dtype, trust_remote_code=True)
        # transformers 5.x no longer ties the Florence-2 language model's
        # embeddings/lm_head to the shared embedding (the checkpoint only
        # stores language_model.model.shared.weight, and the legacy tie
        # mechanism is gone), leaving them randomly initialized — tie them.
        lm = model.language_model
        shared = lm.model.shared.weight
        for emb in (lm.model.encoder.embed_tokens,
                    lm.model.decoder.embed_tokens, lm.lm_head):
            if emb.weight.data_ptr() != shared.data_ptr():
                emb.weight = nn.Parameter(shared, requires_grad=False)
        return proc, model.to(device).eval()

    return _load_cached((model_id, device), _load)


def _beam_generate(model, input_ids, pixel_values, max_new_tokens: int = 256,
                   num_beams: int = 3, length_penalty: float = 1.0):
    """Minimal beam search for the Florence-2 remote code under transformers 5.x.

    ``model.generate`` is unusable here: with ``use_cache=True`` the remote
    decoder subscripts the 5.x ``EncoderDecoderCache`` (TypeError), and with
    ``use_cache=False`` the 5.x beam-search loop degenerates (repeats ``<s>``
    forever). This replicates the reference recipe (3 beams, length-normalized
    scoring) with a full-prefix forward each step — Florence-2 grounding
    outputs are short, so skipping the KV cache is cheap.

    The checkpoint's language-model generation config carries
    ``no_repeat_ngram_size=3``, which is load-bearing: the raw model puts
    ``<s>`` on top at every ``<s>``-only prefix, and only the trigram ban
    (the third ``<s>`` repeat becomes illegal) lets real label tokens win.
    Returns the best finished token-id list (including the decoder start
    token), or the best unfinished one if nothing finished.
    """
    import torch

    ngram_size = 3

    def banned_tokens(tokens):
        """Tokens that would repeat an already-seen ngram (HF no-repeat rule)."""
        if len(tokens) < ngram_size:
            return ()
        prefix = tuple(tokens[-(ngram_size - 1):])
        banned = set()
        for i in range(len(tokens) - ngram_size + 1):
            if tuple(tokens[i:i + ngram_size - 1]) == prefix:
                banned.add(tokens[i + ngram_size - 1])
        return tuple(banned)

    device = input_ids.device
    dt = next(model.parameters()).dtype
    if pixel_values.dtype != dt:
        pixel_values = pixel_values.to(dt)

    with torch.inference_mode():
        feats = model._encode_image(pixel_values)
        emb = model.get_input_embeddings()(input_ids)
        merged, attn = model._merge_input_ids_with_image_features(feats, emb)
        enc = model.language_model.model.encoder(
            inputs_embeds=merged, attention_mask=attn, return_dict=True)
        enc_hidden = enc.last_hidden_state

        lm_cfg = model.language_model.config
        eos_id = int(getattr(model.config, "eos_token_id", None)
                     or getattr(lm_cfg, "eos_token_id", 2))
        start_id = int(getattr(model.config, "decoder_start_token_id", None)
                       or getattr(lm_cfg, "decoder_start_token_id", 2))
        beams = [([start_id], 0.0)]          # (tokens, sum logprob), alive
        finished = []                        # (tokens, normalized score)

        for _ in range(max_new_tokens):
            if not beams:
                break
            ids = torch.tensor([b[0] for b in beams], device=device)
            if enc_hidden.shape[0] != ids.shape[0]:
                from transformers.modeling_outputs import BaseModelOutput
                enc = BaseModelOutput(
                    last_hidden_state=enc_hidden.repeat(ids.shape[0], 1, 1))
            logits = model.language_model(
                decoder_input_ids=ids,
                decoder_attention_mask=torch.ones_like(ids),
                encoder_outputs=enc, use_cache=False,
                return_dict=True).logits[:, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            for b, (tokens, _) in enumerate(beams):
                banned = banned_tokens(tokens)
                if banned:
                    logp[b, list(banned)] = float("-inf")
            top = torch.topk(logp, num_beams * 2, dim=-1)
            candidates = []
            for b, (tokens, score) in enumerate(beams):
                for val, tok in zip(top.values[b], top.indices[b]):
                    tok = int(tok)
                    new_score = score + float(val)
                    if tok == eos_id:
                        norm = new_score / (len(tokens) ** length_penalty)
                        finished.append((tokens, norm))
                    else:
                        candidates.append((tokens + [tok], new_score))
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = candidates[:num_beams]

    if finished:
        finished.sort(key=lambda f: f[1], reverse=True)
        return finished[0][0]
    beams.sort(key=lambda b: b[1], reverse=True)
    return beams[0][0]


def florence2_detect(image, queries: List[str],
                     model_id: Optional[str] = None, device: str = "cuda",
                     _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run Florence-2 phrase grounding on one image, one prompt per query.

    Every box returned for a query gets that query as its label, so the
    caller's query->cat_id mapping applies directly.
    """
    import torch
    from PIL import Image as PILImage

    model_id = model_id or DEFAULT_FLORENCE2_MODEL
    state = _state if _state is not None else {}
    device = state.get("device", device)
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_florence2(key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_florence2(key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    pil = _to_pil(image)

    # Florence-2's DaViT tower only accepts a square input (768x768 for the
    # -large checkpoint; the image is aspect-distorted, loc coords are
    # relative so post-processing rescales to the original size). Under
    # transformers 5.x the remote processor silently skips both resize and
    # mean/std normalization (it forwards do_resize=None/do_normalize=None,
    # which the 5.x image processor treats as "off"), so resize here and
    # force normalization explicitly below.
    size_cfg = getattr(proc.image_processor, "size", None) or {}
    input_size = int(size_cfg.get("height", 768))
    pil_sq = pil.resize((input_size, input_size),
                        resample=PILImage.BICUBIC)

    torch_dtype = next(model.parameters()).dtype
    dets: List[Dict[str, Any]] = []
    for query in queries:
        prompt = f"{GROUNDING_TASK} {query}"
        inputs = proc(text=prompt, images=pil_sq, return_tensors="pt",
                      do_resize=False, do_normalize=True,
                      image_mean=list(proc.image_processor.image_mean),
                      image_std=list(proc.image_processor.image_std))
        inputs = {k: v.to(device, torch_dtype) if v.is_floating_point()
                  else v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            # model.generate is unusable under transformers 5.x (see
            # _beam_generate docstring) — decode with the local beam search.
            generated_ids = _beam_generate(
                model, inputs["input_ids"], inputs["pixel_values"])
        generated_text = proc.tokenizer.decode(
            generated_ids, skip_special_tokens=False)
        parsed = proc.post_process_generation(
            generated_text, task=GROUNDING_TASK,
            image_size=(pil.width, pil.height))
        result = parsed.get(GROUNDING_TASK) or {}
        for box in result.get("bboxes") or []:
            dets.append({
                "label": query,
                "bbox_xyxy": [float(v) for v in box],
                "mask": None,
                "confidence": 1.0,
            })
    return dets


# --------------------------------------------------------------------------- #
# Falcon Perception
# --------------------------------------------------------------------------- #

def _patch_flex_attention_for_smem(model) -> bool:
    """Work around Falcon's attention on consumer GPUs.

    The model's layers always call ``torch.compile``'d flex_attention whose
    default triton config needs ~146KB of shared memory per block — more
    than consumer GPUs have (RTX 4090: ~99KB), so kernel compilation fails
    with "No valid triton configs". When the GPU's limit is below that,
    swap the module-level compiled variants for eager flex_attention
    (slower, but runs). Returns True when the patch was applied.
    """
    import sys
    import torch

    if not torch.cuda.is_available():
        return False
    smem = torch.cuda.get_device_properties(
        torch.cuda.current_device()).shared_memory_per_block_optin
    if smem >= 150_000:
        return False  # datacenter GPU — the default kernels fit
    # Eager flex_attention is NOT an option (it materializes the full
    # scores matrix — the AnyUp upsampler's cross-attention alone would
    # need ~64GB), so recompile with small blocks that fit the smem limit.
    from torch.nn.attention.flex_attention import flex_attention

    def _flex_small_blocks(*args, **kwargs):
        ko = dict(kwargs.get("kernel_options") or {})
        ko.setdefault("BLOCK_M", 64)
        ko.setdefault("BLOCK_N", 64)
        ko.setdefault("num_stages", 1)
        kwargs["kernel_options"] = ko
        return flex_attention(*args, **kwargs)

    pkg_root = type(model).__module__.split(".modeling_")[0]
    patched = False
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(pkg_root):
            continue
        g = getattr(mod, "__dict__", {})
        for attr, kw in (("compiled_flex_attn_decode",
                          {"fullgraph": True}),
                         ("compiled_flex_attn_prefill",
                          {"dynamic": True})):
            if attr in g:
                g[attr] = torch.compile(_flex_small_blocks, **kw)
                patched = True
    return patched


def load_falcon(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Falcon Perception model for one device."""

    def _load():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map={"": device})
        model.eval()
        if device != "cpu" and _patch_flex_attention_for_smem(model):
            print("⚠️ Falcon: GPU shared-memory limit too small for the "
                  "default flex-attention kernels — recompiled with smaller "
                  "blocks (somewhat slower).")
        return model

    return _load_cached((model_id, device), _load)


def _decode_mask(mask_rle: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decode Falcon's COCO RLE dict into an HxW bool array (None on failure)."""
    counts = mask_rle.get("counts")
    size = mask_rle.get("size")
    if not counts or not size:
        return None
    try:
        from pycocotools import mask as mask_utils
        rle = {"size": list(size),
               "counts": counts.encode("utf-8") if isinstance(counts, str)
               else counts}
        return mask_utils.decode(rle).astype(bool)
    except Exception:
        return None


def falcon_detect(image, queries: List[str],
                  model_id: Optional[str] = None, device: str = "cuda",
                  _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run Falcon Perception on one image, one query per category.

    Every detection returned for a query gets that query as its label, so
    the caller's query->cat_id mapping applies directly. Unlike the other
    backends, Falcon also produces real instance masks (RLE-decoded).
    """
    model_id = model_id or DEFAULT_FALCON_MODEL
    state = _state if _state is not None else {}
    device = state.get("device", device)
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["model"] = load_falcon(key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["model"] = load_falcon(key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    model = state["model"]

    pil = _to_pil(image)
    width, height = pil.size

    dets: List[Dict[str, Any]] = []
    for query in queries:
        preds = model.generate(pil, query, compile=False)[0]
        for pred in preds or []:
            xy, hw = pred.get("xy") or {}, pred.get("hw") or {}
            cx, cy = float(xy.get("x", 0.0)), float(xy.get("y", 0.0))
            bh, bw = float(hw.get("h", 0.0)), float(hw.get("w", 0.0))
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            dets.append({
                "label": query,
                "bbox_xyxy": [x1, y1, x2, y2],
                "mask": _decode_mask(pred.get("mask_rle") or {}),
                "confidence": 1.0,
            })
    return dets
