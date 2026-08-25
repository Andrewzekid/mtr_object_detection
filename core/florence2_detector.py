"""Florence-2 open-set detector for GUI auto-labelling.

Wraps HuggingFace ``AutoProcessor`` + ``AutoModelForCausalLM``
(default checkpoint: ``microsoft/Florence-2-large``). Each session category
is grounded with a ``<CAPTION_TO_PHRASE_GROUNDING>`` prompt; the autoregressive
output is decoded via ``processor.post_process_generation`` into absolute
``[x1, y1, x2, y2]`` boxes.

Florence-2 does not emit confidence scores, so every detection reports
``confidence: 1.0``. Detections come back in the same dict shape the
SAM3/OWLv2 autolabel paths produce:

    {label, bbox_xyxy, mask: None, confidence, cat_id (added by caller)}

The model is loaded lazily on first use and cached per (model_id, device);
a CUDA OOM falls back to CPU once and the fallback is remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_FLORENCE2_MODEL = "microsoft/Florence-2-large"
GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"

_model_cache: Dict[tuple, Any] = {}


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
    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

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
            if isinstance(getattr(klass, "__dict__", {}).get(attr), property):
                setattr(klass, attr, False)

    torch_dtype = torch.float16 if device != "cpu" and torch.cuda.is_available() \
        else torch.float32
    model = model_cls.from_pretrained(
        model_id, dtype=torch_dtype, trust_remote_code=True)
    # transformers 5.x no longer ties the Florence-2 language model's
    # embeddings/lm_head to the shared embedding (the checkpoint only stores
    # language_model.model.shared.weight, and the legacy tie mechanism is
    # gone), leaving them randomly initialized — tie them manually.
    import torch.nn as nn
    lm = model.language_model
    shared = lm.model.shared.weight
    for emb in (lm.model.encoder.embed_tokens, lm.model.decoder.embed_tokens,
                lm.lm_head):
        if emb.weight.data_ptr() != shared.data_ptr():
            emb.weight = nn.Parameter(shared, requires_grad=False)
    model = model.to(device).eval()
    _model_cache[key] = (proc, model)
    return proc, model


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

    ``image`` is a path or an RGB numpy array. ``_state`` lets callers
    (the batch worker) reuse the loaded model across frames and mirrors
    the device fallback; pass a fresh ``{}`` to opt in.

    Every box returned for a query gets that query as its label, so the
    caller's query->cat_id mapping applies directly.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    state = _state if _state is not None else {}
    device = state.get("device", device)
    model_id = model_id or DEFAULT_FLORENCE2_MODEL
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_florence2(model_id, device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_florence2(model_id, device)
                state["model_key"] = (model_id, device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    if isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
    else:
        pil = PILImage.fromarray(np.asarray(image)[..., :3])

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
