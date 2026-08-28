# Object Detection Application

An end-to-end pipeline that turns raw robot footage (Metacam rosbags or
plain image folders) into a trained instance-segmentation model and
tracked output videos. Frames are undistorted, sampled and keyframed; a
Qwen VLM server seeds the initial bounding boxes; then a PyQt6 review
GUI — the one human step — where you fix boxes, segment them with SAM3
(bbox prompts). The reviewed labels are assembled into a YOLO-seg dataset
(augment + split), trained with Ultralytics YOLO, evaluated per class,
and run through DeepOCSort tracking.

---

## Quickstart — one complete run, start to finish

Everything below is paste-able top to bottom. It takes raw footage to a
trained segmentation model + tracking results in a single pipeline run.

### Step 0 — install

See **[Installation](#installation)** below for the one-command setup
(`./install.sh`), the Docker image, or manual steps. The short version:

```bash
./install.sh                # venv + pip + SAM3 weights (GPU auto-detected)
# or with the Qwen VLM server too:
./install.sh --llamacpp --hf-token hf_xxx
```

Then start the Qwen VLM server (seed labels): 


### Step 1 — start the Qwen VLM server (seed labels)
Note: you must download the qwen3.8 gguf from huggingface and check that the path is correct before running.
```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
    --mmproj Qwen3.8-mmproj-F16.gguf \
    --image-min-tokens 2048 --port 8089
```

### Step 2 — run the orchestrator (blocks at the GUI)

From a Metacam rosbag, stereo:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
  --camera both \
  --sample-size 1000 --keyframe-stride 10 \
  --llamacpp-url http://127.0.0.1:8089 \
  --ratios 0.7 0.15 0.15 --split-seed 42 \
  --augmentations flip_horizontal rotate brightness --multiplier 2 \
  --epochs 100 --batch-size 16 --model-type yolo26n --task segment --imgsz 768 \
  --eval-splits test train --tracker deepocsort --device 0
```

Or from an existing image folder (mono; for stereo point `--images` at the
parent containing `left/` + `right/` and pass `--camera both`):

```bash
python scripts/orchestrate_pipeline.py --images Datasets/HKU_GH/HKU_GH_left
```

The pipeline runs automatically through:
`undistort → sample → keyframes → stats → qwen → qwen_coco` and then opens
the **label-review GUI** on the keyframes.

### Step 3 — review labels in the GUI (the only human step)

Fix the seeded boxes (delete/draw/adjust), segment them from the box
prompts with SAM3 (`Re-segment`; adjust the box and re-segment if the
mask is off, or use the 🎯 point prompt when boxes don't segment well),
propagate across frames, discard bad frames, then **save** and close.
The pipeline resumes by itself.

If you closed without saving, re-run just the GUI stage:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag .../2026-08-20_22-06-52 --camera both --stage gui
```

After a review session you can also re-run only everything downstream:

```bash
python scripts/orchestrate_pipeline.py \
  --rosbag .../2026-08-20_22-06-52 --camera both \
  --stage yolo --stage split --stage augment --stage assemble \
  --stage train --stage evaluate --stage tracking
```

### Step 4 — collect the outputs

Under `<input>_pipeline/`:

| Path | Contents |
|------|----------|
| `reviewed/labels_coco.json` | human-reviewed COCO annotations |
| `dataset/final/` (+ `dataset.yaml`) | final train/val/test dataset |
| `dataset/dataset_statistics.csv` | per-class instance statistics |
| `training/yolo_training/weights/best.pt` | trained model |
| `evaluation/metrics.csv` | per-class P/R/F1/AP metrics |
| `tracking/<cam>/` | tracked frames, `results.json`, video |

---

## Overview
Label review and segmentation:
```
python -m gui.label_review.main
# stereo dual-view (left/right folders, frames paired by timestamp filename;
# frames without a matching timestamp on the other side are skipped).
# The saved COCO JSON lists every timestamp-synced image (sorted earliest
# first) — not just the frames you viewed or annotated.
python -m gui.label_review.main --images /path/to/left --images-right /path/to/right
```
Launches interactive GUI for complete label review and segmentation. Point it
at the keyframe folder of a pipeline run to review the combined Qwen output
(it auto-loads a `labels_coco.json` found next to the images):

```bash
python -m gui.label_review.main --images output/<run>/keyframes/left
```

### Autolabel backends

The GUI can pre-fill boxes/masks for the session categories
(**Autolabel frame** / **Autolabel ALL frames**). Pick the backend under
Settings → Autolabel (persisted as `autolabel.detector` in the GUI config):

| Backend | Output | Notes |
|---|---|---|
| `sam3` | boxes + masks | text prompts, local Ultralytics SAM3 checkpoint |
| `owlv2` | boxes | zero-shot, `google/owlv2-large-patch14-ensemble` (HF) |
| `owlv2_exemplar` | boxes | 1-shot: select an existing box first — its crop becomes the visual query (`image_guided_detection`) |
| `grounding_dino` | boxes | zero-shot; default `IDEA-Research/grounding-dino-base` (the `-large` checkpoint is not published in transformers format) |
| `falcon` | boxes + masks | `tiiuae/Falcon-Perception`; free-form text query per category |

Per-backend config keys (Settings dialog or config JSON): `owlv2_model` /
`owlv2_conf` (default 0.3), `gdino_model` / `gdino_conf` (0.35, maps to the
box threshold), `falcon_model`. First use downloads the
checkpoint from Hugging Face; models are cached per session.

Two compatibility notes for the HF backends (handled automatically in
`core/`): Falcon's pre-compiled flex-attention kernels exceed the shared
memory of consumer GPUs (RTX 4090), so they are recompiled with smaller
blocks at load (BLOCK=128, 1 stage; tunable via `FALCON_FLEX_BLOCK_M` /
`FALCON_FLEX_BLOCK_N` / `FALCON_FLEX_STAGES`), and per-category queries are
batched into one `generate` call.

### Labelling assist features (when and how to use them)

The review GUI is built so a human never labels every frame by hand. Typical
session: let Qwen seed boxes on the keyframes, open the GUI, then combine
the assistants below depending on the footage.

- **Autolabel (frame / ALL frames)** — open-set detectors fill in boxes (+
  masks for SAM3/Falcon) for the session categories. Highlighting 2+
  categories in the list restricts the run to just those. Use it to add a
  category you forgot to prompt Qwen for, or to re-label with a stronger
  open-set model. `owlv2_exemplar` is the 1-shot variant: draw/fix ONE good
  box by hand, select it, then autolabel — best for odd objects text
  prompts miss.
- **Keyframes** — for near-static cameras (corridors, platforms) frames are
  near-duplicates, so label every Nth (`--keyframe-stride N`, N≈10) and let
  interpolation/propagation cover the gaps. Small N for fast motion, large
  N for static scenes. The GUI's ★ Keyframe toggle (K) marks frames that
  anchor interpolation.
- **Interpolate (I)** — linearly/optical-flow interpolates boxes of the
  same track between two labelled frames. Use when motion is **smooth and
  in a constant direction** (walking past a static camera, slow pans).
  Workflow: label a frame, jump ~10–30 frames, label the same objects again,
  select the boxes, Interpolate. Bad under occlusion, direction reversals,
  or objects appearing/disappearing — check the in-between frames after
  running it.
- **Propagate → (SAM3 tracking)** — select a box and let SAM3's video
  memory bank follow that instance forward frame by frame, producing masks.
  Use for deformable/rotating objects or longer ranges where interpolation
  drifts. Two methods in Settings → SAM3: `memory` (default; one video
  session, can recover after temporary loss) and `chain` (re-detect each
  frame from the previous box, IoU-gated; stops permanently at the first
  miss). Stops being reliable under large camera jumps — re-seed from a
  later frame. **Keyframe-bounded:** when a later ★ keyframe (K) is marked,
  the run only covers the frames between the keyframes — it stops at the
  next keyframe instead of running to the end of the dataset, so you can
  annotate keyframe by keyframe and let SAM3 fill each segment.
- **Re-segment selected** — re-runs SAM3 on a box you moved/resized to get
  a fresh mask. Use after every manual box edit if masks matter.
- **Point segment (🎯 Add points + ▶ Segment points)** — SAM2-style point
  prompting in two steps. With **Add points** toggled ON, left-click adds a
  positive point, right-click a negative point — clicks only accumulate,
  nothing runs yet, so you can place as many points as you want. Press
  **▶ Segment points** to run SAM3 once with the full point set; press it
  again after adding more points to refine. Enter accepts the object,
  Esc cancels it. The selected category is sent to SAM3 as a **text
  prompt**: it detects all instances of that category and your points pick
  which one to keep (pure point prompting is the fallback when the text
  finds nothing there). Pick a category row first — points added before a
  category is picked are kept and used when you press Segment. Use for
  precise single-object masks where a drag-box would include too much
  background.
- **Discard image (🚫)** — drops the frame from the saved COCO entirely.
  Use for blurry/irrelevant frames so they never reach training.
- **Stereo sync** — left/right frames pair by timestamp filename; unmatched
  frames are skipped everywhere (display and save), so a saved stereo COCO
  only contains synced pairs, sorted earliest first.

End-to-end pipeline that takes **raw images** and produces a **trained YOLO
detection/segmentation model** plus **tracking results**, with a
human-in-the-loop review step in the middle. It targets Hong Kong MTR and
industrial (IW) station imagery but is dataset-agnostic.
For a quick segmentation run for the YOLO seg  + Tracking, try the following command:
```
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```
There are two labeling flows:

**Default pipeline** (label keyframes in the GUI with Qwen seed
labels, use the GUI's labelling assistants to cover the rest):

```
data folder (or rosbag) → undistort → 00 sample → 12 keyframes → 01a stats
   → 07 Qwen seed labels on keyframes → 08c combine to COCO
   → GUI review (fix boxes, SAM3-segment, autolabel, interpolate/propagate,
     discard bad frames) → 01b COCO→YOLO-seg → 03 train/test/val split
   → 02 augment (train split only, mask-aware) → 04 YOLO seg train
   → 05 evaluate (per-class P/R/F1 CSV) → 11 tracking on raw frames
```

The whole chain runs from one command — see "Orchestrated pipeline" below.

**Full pipeline** (dense labeling — every frame goes through the VLM; slower,
no keyframe step):

```
raw images → 07 Qwen VLM auto-label → 08c combine per-image results into
            labels_coco.json → GUI review (same as above)
            → 01b → 03 split → 02 augment train → 04 train → 05 evaluate
            → 11 tracking
```

**GUI segmentation pipeline** (annotate + segment in the label-review GUI, no
Roboflow round-trip — masks come from SAM3 in the GUI):

```
GUI annotate (SAM3 masks) → COCO labels_coco.json (discard unwanted frames
   in the GUI) → 01b COCO→YOLO-seg → 02 augment → 03 train/test/val split
   → 04 YOLO seg train
```

See "Segmentation dataset pipeline (GUI → YOLO seg)" below for commands, or
run all post-GUI steps at once with
`python scripts/orchestrate_pipeline.py --coco-json ... --images-dir ...`.

**Orchestrated pipeline** — the whole keyframe workflow with one command.
Input is either a Metacam rosbag (adds the fisheye undistortion stage) or a
plain image folder:

```bash
# 1. Start the llama.cpp server with the Qwen GGUF + mmproj:
#    llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj Qwen3.8-mmproj-F16.gguf --port 8089

# 2. Run the orchestrator (blocks at the GUI stage; resumes when you close
#    the GUI after saving):
python scripts/orchestrate_pipeline.py \
  --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
  --camera both \
  --sample-size 1000 --keyframe-stride 10 \
  --llamacpp-url http://127.0.0.1:8089 \
  --qwen-model Qwen3.8-27B-Q4_K_M.gguf --qwen-mmproj Qwen3.8-mmproj-F16.gguf \
  --ratios 0.7 0.15 0.15 --split-seed 42 \
  --augmentations flip_horizontal rotate brightness --multiplier 2 \
  --epochs 100 --batch-size 16 --model-type yolo26n --task segment --imgsz 768 \
  --eval-splits test train --tracker deepocsort

# From an existing image folder instead of a rosbag (mono; for stereo point
# --images at the parent of left/ + right/ and pass --camera both):
python scripts/orchestrate_pipeline.py --images Datasets/HKU_GH --camera both

# Or via the shell wrapper (env overrides: LLAMACPP_URL, QWEN_MODEL,
# QWEN_MMPROJ; all extra args pass through):
./scripts/run_pipeline.sh 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 --camera both
```

Stages: `undistort`, `sample`, `keyframes`, `stats`, `qwen`, `qwen_coco`,
`gui`, `yolo`, `split`, `augment`, `assemble`, `train`, `evaluate`,
`tracking`. Run a subset with `--stage <name>` (repeatable) and skip with
`--skip-stage <name>`; completed stages write `stage_completed.json` markers
so re-runs resume where they stopped (`--force` ignores markers). The `gui`
stage launches the label-review GUI on the keyframes (`--gui-on all` loads
every sampled frame instead) and waits — the pipeline continues after you
save and close. Useful per-stage knobs:

- llama.cpp / Qwen: `--llamacpp-url`, `--qwen-model`, `--qwen-mmproj`,
  `--prompt` (class list is parsed from it; or `--classes`), `--resume-from`
- sampling/keyframes: `--sample-size` (stereo: right side is synced to the
  sampled left timestamps), `--sample-seed`, `--keyframe-stride`
- dataset: `--ratios TRAIN TEST VAL`, `--split-seed`, `--bbox-as-rect`
- augmentation (train split only): `--augmentations` (flip_horizontal,
  flip_vertical, rotate, brightness, contrast, hue, blur, resize, mosaic),
  `--multiplier`, `--rotation-range`, `--brightness-range`, `--hue-range`,
  `--blur-range`, `--resize W H`, `--skip-augment`
- training: `--epochs`, `--batch-size`, `--model-type`, `--task`,
  `--imgsz`, `--pretrained`, `--lr0`, `--device`
- evaluation: `--eval-splits`, `--eval-conf`, `--eval-iou`
- tracking: `--tracker`, `--track-conf`, `--track-iou`, `--track-imgsz`,
  `--track-fps`, `--track-max-frames`

Outputs under `<input>_pipeline/`: `undistorted/`, `sampled/`, `keyframes/`,
`qwen/`, `reviewed/labels_coco.json`, `dataset/{yolo_flat,split,
train_augmented,final}` (+ `dataset.yaml` and `dataset_statistics.csv`),
`training/yolo_training/weights/best.pt`, `evaluation/metrics.csv`,
`tracking/<camera>/`.


### Directory layout

| Path | Contents |
|------|----------|
| `scripts/` | Numbered CLI entry points (pipeline stages) + helpers. |
| `core/` | Reusable OOP classes / headless inference wrappers the scripts build on. |
The numbered `scripts/` are thin CLI wrappers over `core/` classes
(`ModelTrainer`, `ModelEvaluator`, `DataProcessor`, `DatasetCreator`,
`ModelVisualizer`) and inference wrappers (`run_qwen`, `run_qwen_api`,
`run_sam3`, `run_gemini` in `core/models_inference.py`). A full per-script
reference with inputs/outputs is at the bottom of this file.

---

## Installation

The app needs **Python 3.10–3.13**, a Linux box (Ubuntu 20.04+/Debian 12
tested), and **optionally an NVIDIA GPU** (strongly recommended — SAM3,
YOLO training and tracking are 10–50× slower on CPU). There are three
ways to install; pick one.

### Option A — one-command installer (`install.sh`)

`install.sh` (repo root) sets up a Python venv, installs system Qt/X
libraries, installs all pip dependencies (CUDA torch auto-detected), and
downloads the SAM3 + SAM3.1 weights from HuggingFace.

```bash
# default: venv at ./.venv, auto-detect GPU, fetch SAM3 weights
./install.sh --hf-token hf_xxx          # token for the gated facebook/sam3 repo

# use a conda env instead of a venv
./install.sh --conda objdet --hf-token hf_xxx

# also build llama.cpp for the Qwen seed-label VLM server
./install.sh --llamacpp --hf-token hf_xxx

# force CPU torch (no NVIDIA GPU)
./install.sh --cpu --hf-token hf_xxx
```

Flags:

| Flag | Effect |
|------|--------|
| `--gpu` / `--cpu` | Force CUDA / CPU torch (default: auto-detect via `nvidia-smi`) |
| `--conda NAME` | Use/create a conda env `NAME` instead of `./.venv` |
| `--llamacpp` | Also clone + build llama.cpp (`~/code/llama/llama.cpp`) for the Qwen server |
| `--skip-models` | Skip SAM3 weight download (GUI still starts, SAM3 features off until you place the weights) |
| `--skip-apt` | Skip the `apt-get` system-lib step (you've already installed them) |
| `--hf-token TOKEN` | HuggingFace token for the gated `facebook/sam3` repo (or `export HF_TOKEN=…`) |
| `-h` / `--help` | Show usage |

**SAM3 weights are gated.** Before the first run: visit
<https://huggingface.co/facebook/sam3>, accept the license, create a token
at <https://huggingface.co/settings/tokens>, and pass it via `--hf-token`
or `HF_TOKEN`. The script downloads:

| File | Destination | Size |
|------|-------------|------|
| `sam3.pt` (SAM3) | `core/sam3/models/sam3-model/sam3.pt` | ~3.4 GB |
| `sam3.1_multiplex.pt` (SAM3.1, Propagate→) | `core/sam3/models/sam3.1-model/sam3.1_multiplex.pt` | ~3.4 GB |

**Qwen VLM weights are NOT bundled** — they're ~18 GB and user-supplied.
With `--llamacpp` the script builds `llama-server`; you then download a
Qwen3.8 vision GGUF + its `mmproj` (e.g. `Qwen3.8-27B-Q4_K_M.gguf` +
`Qwen3.8-mmproj-F16.gguf`) into `~/code/llama/llama.cpp/` and start it
yourself (see Step 1 of the Quickstart).

**HF autolabel backends** (`owlv2`, `grounding-dino`,
`falcon`) download automatically on first use in the GUI's
Settings → Autolabel, into `~/.cache/huggingface/`. YOLO pretrained base
weights (`yolo26n.pt`, …) are fetched by Ultralytics on first training
run into `models/`. Neither needs an install step.

### Option B — Docker (GPU + GUI + full pipeline)

The `Dockerfile` (repo root) builds an image with the CUDA runtime, all
Python deps, the label-review GUI, and (optionally) a built `llama-server`
for the Qwen VLM. GPU + X11 are forwarded to the host so the GUI shows
on your desktop.

```bash
# build (fetch SAM3 weights at build time with a token):
docker build -t object-detection-app --build-arg HF_TOKEN=hf_xxx .

# run the GUI (Linux host; forwards display + GPU + HF cache):
xhost +local:docker
docker run --gpus all --rm -it --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/work -w /work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -p 8089:8089 \
  object-detection-app \
  --images /work/Datasets/YourData
```

Run the **Qwen VLM server** in a second container (mount your GGUFs):

```bash
docker run --gpus all --rm -it --net=host \
  -v "$HOME/code/llama/llama.cpp":/llama.cpp \
  -v "$HOME/models/qwen":/models/qwen:ro \
  object-detection-app \
  /llama.cpp/build/bin/llama-server \
    -m /models/qwen/Qwen3.8-27B-Q4_K_M.gguf \
    --mmproj /models/qwen/Qwen3.8-mmproj-F16.gguf \
    --port 8089 --host 0.0.0.0
```

Build-args:

| Arg | Effect |
|-----|--------|
| `HF_TOKEN=hf_xxx` | Fetch SAM3 + SAM3.1 weights at build time (gated repo) |
| `BUILD_LLAMACPP=1` | Also build `llama-server` inside the image (off by default — most users mount a prebuilt binary) |

Without `HF_TOKEN` the image builds fine but the SAM3 weights are absent;
mount them at runtime (`-v "$PWD/core/sam3/models":/app/core/sam3/models`).

### Option C — manual install

```bash
# 1. Python env (3.10–3.13)
python3 -m venv .venv && source .venv/bin/activate

# 2. System libs for PyQt6 (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
  libgl1 libegl1 libxkbcommon0 libdbus-1-3 \
  libglib2.0-0 libfontconfig1 libxcb-cursor0 ffmpeg

# 3. CUDA torch (NVIDIA GPU) — pick the index matching your CUDA:
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision
# CPU-only:
#   pip install torch torchvision

# 4. Python deps
pip install -r requirements.txt
pip install "git+https://github.com/ultralytics/CLIP.git"   # SAM3 text encoder

# 5. SAM3 weights (gated HF repo — accept license + create token first)
pip install huggingface_hub
export HF_TOKEN=hf_xxx
huggingface-cli download facebook/sam3     sam3.pt             \
  --local-dir core/sam3/models/sam3-model      --token "$HF_TOKEN"
huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt  \
  --local-dir core/sam3/models/sam3.1-model    --token "$HF_TOKEN"

# 6. (optional) Qwen VLM server — build llama.cpp + supply GGUFs
git clone https://github.com/ggerganov/llama.cpp ~/code/llama/llama.cpp
cd ~/code/llama/llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_CUDA=on -DLLAMA_BUILD_SERVER=on && cmake --build . --config Release -j
# download Qwen3.8-27B-Q4_K_M.gguf + Qwen3.8-mmproj-F16.gguf into ~/code/llama/llama.cpp/
```

Verify:

```bash
python -c "from ultralytics.models.sam import SAM3SemanticPredictor; print('ok')"
python -m gui.label_review.main --help
```

### Where each model lives

| Model | Source | Destination | Downloaded by |
|-------|--------|-------------|---------------|
| SAM3 (`sam3.pt`) | `facebook/sam3` (gated) | `core/sam3/models/sam3-model/sam3.pt` | `install.sh` / manual / Docker build |
| SAM3.1 (`sam3.1_multiplex.pt`) | `facebook/sam3.1` (gated) | `core/sam3/models/sam3.1-model/sam3.1_multiplex.pt` | same |
| YOLO base weights (`yolo26n.pt` …) | Ultralytics | `models/` | Ultralytics on first training run |
| OWLv2 / Grounding-DINO / Falcon | HuggingFace | `~/.cache/huggingface/` | `transformers.from_pretrained` on first GUI use |
| Qwen VLM (GGUF) | user-supplied | `~/code/llama/llama.cpp/` | you (install.sh `--llamacpp` builds the server only) |

### Standalone bundle (no Python needed on the target)

The label review app can be frozen into a single-folder executable with
PyInstaller (Linux "exe"):

```bash
pyinstaller scripts/label_review.spec --distpath dist --workpath build --noconfirm
# result: dist/label-review/label-review
```

> **Note:** the spec is stale — it still targets the old monolithic
> `scripts/09_label_review.py`. It needs updating for the new
> `gui/label_review/` package layout (entry point
> `python -m gui.label_review.main`) before this works again.

Copy the `dist/label-review/` folder to the target machine (e.g.
`tar czf label-review.tar.gz -C dist label-review`) and run:

```bash
./label-review --images /path/to/frames --output_json out/coco.json \
    --sam3-model /path/to/sam3.pt
```

Notes:

- SAM3 weights are not bundled (3.4 GB) — pass `--sam3-model` or place
  `core/sam3/models/sam3-model/sam3.pt` under the working directory.
- The bundle inherits this machine's torch build: a CUDA torch bundle needs
  a compatible NVIDIA driver on the target; build in a CPU-torch env for
  CPU-only targets.
- On a fresh Linux install PyQt6 needs system GL/X libraries:
  `sudo apt install libgl1 libegl1 libxkbcommon0 libdbus-1-3`.


- **Qwen3.6 labeling** can run locally with Ollama or via DashScope API.
  The examples below use the DashScope API. Set your key first:

  ```bash
  export API_KEY=your_dashscope_key
  ```

- **SAM3 segmentation** needs the Ultralytics SAM checkpoint. Place it at
  `core/sam3/models/sam3-model/sam3.pt` (or pass `--model` explicitly).

---

## Complete end-to-end example: rosbag → trained model → tracking

One command chain runs the entire default pipeline. This example uses a
Metacam rosbag; for a plain image folder use `--images <dir>` instead of
`--rosbag` (mono, or the parent of `left/` + `right/` with
`--camera both`). Paste each block into the terminal from top to bottom.

```bash
# 0. Install (once) — see Installation for Docker / manual options
./install.sh --llamacpp --hf-token hf_xxx

# 1. Start the Qwen VLM server (seeds the initial boxes)
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
    --mmproj Qwen3.8-mmproj-F16.gguf \
    --image-min-tokens 2048 --port 8089

# 2. Run the whole pipeline (blocks at the GUI review stage and resumes
#    automatically once you save and close the GUI)
python scripts/orchestrate_pipeline.py \
  --rosbag Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag \
  --camera both \
  --sample-size 1000 --keyframe-stride 10 \
  --llamacpp-url http://127.0.0.1:8089 \
  --ratios 0.7 0.15 0.15 --split-seed 42 \
  --augmentations flip_horizontal rotate brightness --multiplier 2 \
  --epochs 100 --batch-size 16 --model-type yolo26n --task segment --imgsz 768 \
  --eval-splits test train --tracker deepocsort --device 0
```

That single command runs every stage of the default pipeline:

```
undistort → sample → keyframes → stats → qwen seed labels → qwen_coco
→ GUI review (the only human step: fix boxes, SAM3-segment, autolabel,
  interpolate/propagate, discard bad frames, save)
→ COCO→YOLO-seg → split → augment (train split) → assemble → train
→ evaluate (per-class P/R/F1 CSV) → tracking on the raw frames
```

Outputs land under `<rosbag>_pipeline/`:

| Path | Contents |
|------|----------|
| `reviewed/labels_coco.json` | human-reviewed COCO annotations |
| `dataset/final/` (+ `dataset.yaml`) | final train/val/test dataset |
| `dataset/dataset_statistics.csv` | per-class instance statistics |
| `training/yolo_training/weights/best.pt` | trained model |
| `evaluation/metrics.csv` | per-class P/R/F1/AP metrics |
| `tracking/<cam>/` | tracked frames, `results.json`, video |

Completed stages write `stage_completed.json` markers, so re-runs resume
where they stopped (`--force` ignores the markers). To re-run individual
stages later:

```bash
# reopen the GUI review only:
python scripts/orchestrate_pipeline.py \
  --rosbag Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag \
  --camera both --stage gui

# redo everything after the review:
python scripts/orchestrate_pipeline.py \
  --rosbag Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag \
  --camera both \
  --stage yolo --stage split --stage augment --stage assemble \
  --stage train --stage evaluate --stage tracking
```

To run tracking standalone on new footage with the trained model
(undistort fisheye footage first — see `scripts/undistort_rosbag.py`):

```bash
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model ..._pipeline/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/tracking/MTR_left_undistorted \
  --conf 0.5 --device 0 \
  --output output/tracking/MTR/run1
```

---


## Segmentation dataset pipeline (GUI → YOLO seg)

Post-annotation chain that turns the label-review GUI's COCO output into a
YOLO segmentation dataset ready for training — no Roboflow round-trip:

```
GUI annotate (SAM3 masks on every frame, discard bad frames with the
   discard button) → labels_coco.json
   → 01b_coco_to_yolo_seg.py   COCO → flat YOLO-seg (images/ + labels/ + classes.txt)
   → 02_augment_data.py        augmented flat YOLO-seg dataset
   → 03_split_dataset.py       train/test/val split + dataset.yaml
   → 04_train_model.py --task segment
```

### One-command runner

Runs all three post-GUI steps and validates the dataset format after each
step (image/label pairing, label syntax, coordinate range, class ids), so a
format mismatch fails at the step that introduced it. This is the
orchestrator's dataset-only mode (`--coco-json`):

```bash
python scripts/orchestrate_pipeline.py \
  --coco-json /data/run/labels_coco.json \
  --images-dir /data/run/camera \
  --output-root output/my_dataset \
  --augmentations flip_horizontal rotate brightness hue blur \
  --multiplier 2 --ratios 0.7 0.15 0.15 --split-seed 42

# skip augmentation entirely:
python scripts/orchestrate_pipeline.py \
  --coco-json /data/run/labels_coco.json --images-dir /data/run/camera \
  --output-root output/my_dataset --skip-augment
```

`--images-dir` is the plain image folder for mono sessions, or the parent
folder containing `left/` + `right/` for stereo sessions (the GUI writes a
`side` field per image; output filenames get `left_`/`right_` prefixes so the
identical timestamp names don't collide). The dataset-only mode is also
reachable via the shell wrapper: `./scripts/run_pipeline.sh --coco-json <coco>
--images-dir <dir> --output-root <out>`. The runner prints the exact
`04_train_model.py --config .../dataset.yaml --task segment` command at the end.

Output layout:

```
output/my_dataset/
├── yolo_flat/      # images/ + labels/ + classes.txt + conversion_summary.json
├── augmented/      # same layout, originals + augmented copies
└── dataset/        # train/ test/ val/ (each images/ + labels/) + dataset.yaml
```

### Step by step

```bash
# 1. COCO → flat YOLO-seg (mask-less boxes are skipped by default;
#    --bbox-as-rect emits them as rectangle polygons)
python scripts/01b_coco_to_yolo_seg.py \
  --coco-json /data/run/labels_coco.json \
  --images-dir /data/run/camera \
  --output-dir output/my_dataset/yolo_flat

# 2. Augment (masks are transformed with the image: flip/rotate/mosaic move
#    every polygon point; brightness/contrast/hue/blur/resize leave the
#    normalized labels untouched; classes.txt is passed through)
python scripts/02_augment_data.py \
  --input-dir output/my_dataset/yolo_flat \
  --output-dir output/my_dataset/augmented \
  --augmentations flip_horizontal rotate brightness hue blur \
  --multiplier 2 --hue-range -15 15 --blur-range 3 9
# add the 'resize' augmentation with: --resize WIDTH HEIGHT

# 3. Split + dataset.yaml (class names are read from classes.txt when
#    --class-names is not given)
python scripts/03_split_dataset.py \
  --input-dir output/my_dataset/augmented \
  --output-dir output/my_dataset/dataset \
  --ratios 0.7 0.15 0.15 --seed 42 --generate-yaml

# 4. Train
python scripts/04_train_model.py \
  --config output/my_dataset/dataset/dataset.yaml --task segment \
  --epochs 100 --device 0
```

> Caveat: because augmentation runs before the split, augmented copies of one
> source image can land in both train and val. Use `--skip-augment` (or split
> first and augment the splits separately) if you need strict separation.

---

## Script reference

Every entry point in `scripts/`. "Key inputs" lists the important CLI flags
(not exhaustive — run `python scripts/<name>.py --help` for the full set).

### Pipeline stages

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `00_sample_from_dataset.py` | Pick N images from a source folder that aren't already in the labeled YOLO set, ready for labeling. | `--source-dir`, `--labeled-dir`, `--out-dir`, `-n`, `--copy`/`--symlink`, `--dry-run` | New image folder (copied or symlinked) |
| `07_run_qwen.py` | Auto-label images with a VLM: Qwen via Ollama or DashScope API, or Gemma/Gemini. Supports conditioning refs, per-class, per-image labels. | `--prompt`, `--image`/`--image-folder`, `--template`, `--model`, `--conditioning-images`, `--per-class`, `--per-image-labels`, `--bbox-order`, `--dedup-iou`, `--use-api` | `<stem>_result.json` per image, `summary.json`, optional `--vis-output` |
| `08c_qwen_results_to_coco.py` | Combine per-image Qwen `*_result.json` files into one COCO `labels_coco.json` the label review GUI can open (COCO xywh bboxes, categories from Qwen labels, `annotated_image_ids`). | `--qwen-results-dir`, `--output`, `--side` | `labels_coco.json` |
| `04_train_model.py` | Train a YOLO detect/segment model (Ultralytics). | `--config` (data.yaml), `--model-type`, `--task`, `--epochs`, `--batch-size`, `--device`, `--loss-type` | `runs/.../weights/best.pt` |
| `05_evaluate_model.py` | Evaluate a trained model on one or more splits (default `test train`) with per-class precision/recall/F1/AP50/AP50-95 (box + mask), or compare pred vs GT COCO JSON. | `--model`, `--data`, `--split`, `--conf`, `--iou`, `--csv` (or `--pred-json`/`--gt-json`) | Metrics report (stdout) + per-class CSV |
| `11_run_tracking.py` | YOLO tracking (ByteTrack / BoT-SORT / Deep OC-SORT / detect-then-SAM3). Pre-tracker class-agnostic NMS; | `--tracker`, `--model`, `--data`, `--output`, `--conf`, `--device`, `--warmup-frames`, `--nms-iou`, `--trt`, `--no-masks`, `--no-vis`, `--postprocess-workers` | `tracked_*.jpg`, `results.json`, `tracking_result.mp4` |
| `12_extract_keyframes.py` | Select every Nth frame as a keyframe + write a manifest for the interpolator. | `--image-folder`/`--video`, `--output-dir`, `--every`, `--mode` | Keyframe images + `keyframe_manifest.json` |
| `13_interpolate_tracks.py` | Propagate reviewed keyframe boxes to every frame via anchored optical flow. An optional per-frame RANSAC camera model (`--camera-model global`) can absorb non-linear camera shake (tracking/anchoring on the residual, KLT dropouts re-seeded); off by default — accuracy testing on the re-reviewed MTR 4k frames showed no net gain on this fisheye camera. New objects at a keyframe are back-tracked. Each output box carries a `source`/`confidence` provenance field. | `--keyframes-coco`, `--manifest`, `--image-folder`, `--output-coco`, `--match-max-dist`, `--flow-method`, `--interp-method`, `--camera-model` | COCO annotations for every frame (+ optional vis) |
| `orchestrate_pipeline.py` | End-to-end keyframe pipeline: undistort → sample → keyframes → stats → Qwen → COCO combine → GUI review → 01b COCO→YOLO-seg → 03 split → 02 augment (train only) → assemble final dataset + stats CSV → 04 train → 05 evaluate → 11 tracking. Stage markers make it resumable. Also supports a dataset-only mode (`--coco-json` + `--images-dir`) that runs just 01b convert → 02 augment → 03 split on an already-reviewed COCO file. | `--rosbag`/`--images`/`--coco-json`, `--camera`, `--stage`, `--skip-stage`, `--force`, per-stage args (`--sample-size`, `--ratios`, `--augmentations`, `--epochs`, `--tracker`, ...) | `<input>_pipeline/` output tree (or `{yolo_flat,augmented,dataset}/` in dataset-only mode) |
| `run_pipeline.sh` | Shell wrapper around the orchestrator with env-var overrides (`LLAMACPP_URL`, `QWEN_MODEL`, `QWEN_MMPROJ`). Pass `--coco-json ...` for the dataset-only mode. | `<rosbag_path>` (or `--coco-json <coco> --images-dir <dir>`) + any orchestrator args | same as orchestrator |

### Data prep & conversion

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `01_verify_labels.py` | Validate YOLO labels: missing files, bad formats, class stats. | `--input-dir`, `--images-subdir`, `--labels-subdir`, `--class-names`, `--fix` | Report (stdout / `--output`); optional in-place fixes |
| `01a_dataset_statistics.py` | Dataset statistics for a YOLO dataset (single `images/`+`labels/`, `<split>/{images,labels}` or `images/<split>`+`labels/<split>` layouts): image/label/background counts, per-class instances, % of instances, % of images, avg instances/img. Class names from `classes.txt` or `dataset.yaml`. | `--input-dir`, `--csv` | Report (stdout) + optional per-class CSV |
| `01b_coco_to_yolo_seg.py` | Convert label-review GUI COCO output into a flat YOLO segmentation dataset (input for `02`). Stereo-aware (`left_`/`right_` prefixes); mask-less boxes skipped unless `--bbox-as-rect`. | `--coco-json`, `--images-dir`, `--output-dir`, `--bbox-as-rect`, `--symlink` | `images/`, `labels/`, `classes.txt`, `conversion_summary.json` |
| `02_augment_data.py` | Augment a labeled YOLO dataset (flip/rotate/brightness/contrast/hue/blur/resize/mosaic); polygon labels are transformed with the image. | `--input-dir`, `--output-dir`, `--augmentations`, `--multiplier`, `--hue-range`, `--blur-range`, `--resize` | Augmented images + labels (+ `classes.txt` passthrough) |
| `03_split_dataset.py` | Split a labeled YOLO dataset into train/val/test + `data.yaml`. Class names fall back to `classes.txt` when `--class-names` is omitted. | `--input-dir`, `--output-dir`, `--ratios`, `--generate-yaml` | `images/labels/{train,val,test}/`, `data.yaml` |
| `10_qwen_json_to_yolo.py` | Convert `07 --split-by-class` JSON into a YOLO detection dataset. | `--annotations-dir`, `--image-folder`, `--output-dir`, `--data-yaml` | YOLO detect dataset (`images/`, `labels/`, `data.yaml`) |

### Standalone tools

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `06_run_sam3.py` | Run SAM3 segmentation on an image/folder, optionally with bbox exemplars. | `--image`/`--image-folder`, `--bbox`/`--bbox-json`, `--concept`, `--model` | Masks / overlay images (`--output`, `--save-overlay`) |
| `undistort_rosbag.py` | Fisheye-undistort a folder of images using a calibration JSON. | `--images-root`, `--output-root`, `--calibration`, `--camera-name` | Undistorted images |
| `visualize.py` | Visualize Qwen annotations / YOLO detect / YOLO seg / model predictions. | `--mode`, `--dataset`/`--annotations-folder`, `--output`, `--model` | Annotated images |
| `tracking_utils.py` | Shared helpers (tracker YAML, IoU, mask→polygon, summary video). | — (library, not a CLI) | — |

### `core/` modules

These back the scripts above; not normally invoked directly.

| Module | Role |
|--------|------|
| `models_inference.py` | Headless VLM/SAM3 inference wrappers: `run_qwen` (Ollama), `run_qwen_api` (DashScope), `run_gemini` (HKU proxy), `run_sam3`; prompt templates + output parsing. |
| `model_trainer.py` | `ModelTrainer` — YOLO training pipeline (backs `04`). |
| `model_evaluator.py` | `ModelEvaluator` — evaluation + pred/GT comparison (backs `05`). |
| `data_processor.py` | `DataProcessor` — augmentation + dataset stats (backs `02`). |
| `dataset_creator.py` | `DatasetCreator` — undistortion, random selection, split (backs `00`/`03`). |
| `model_visualizer.py` / `visualizer.py` / `visualization.py` | Visualization helpers (back `visualize.py`). |
| `hyperparameter_search.py` | Search-space + grid/random search strategies for training. |
| `sam3_test.py` | SAM3 scratch/test harness (not a pipeline stage). |



---

## Label-review GUI — complete user guide

The interactive review app (`gui/label_review/`) is where the human step of
the pipeline happens: correct the seeded boxes, create masks with SAM3, and
discard bad frames. Launch it standalone or let the orchestrator open it for
you.

### Launching

```bash
# mono session
python -m gui.label_review.main --images /path/to/left --output_json out/coco.json

# stereo dual-view (frames paired by timestamp filename; unsynced frames skipped)
python -m gui.label_review.main \
    --images /path/to/left --images-right /path/to/right

# seed from an existing COCO (e.g. the qwen_coco stage output) + SAM3 options
python -m gui.label_review.main --images /path/to/keyframes \
    --json output/<run>/reviewed/labels_coco.json \
    --sam3-model core/sam3/models/sam3-model/sam3.pt --sam3-device cuda \
    --auto-segment
```

All flags: `--images`, `--images-right`, `--output_json`, `--json` (seed
COCO), `--sam3-model/--sam3-device/--sam3-conf`, `--sam3-imgsz` (inference
size; smaller = faster, e.g. 770), `--sam3-quantize {8,16,32}` (16 = FP16,
~1.5-2x faster on GPU — set once in the config's `sam3` block),
`--auto-segment`
(SAM3 after every drawn box), `--interp-flow-method {dis,klt,farneback}`,
`--interp-camera-model {none,global}`, `--output-yolo-dir` (also export YOLO
on exit), `--pose-db` (Clio poses for the Rerun map view),
`--data-yaml` (class order), `--config FILE`. Omitting `--images` starts in
idle mode — pick a source from the File menu. If a `labels_coco.json` sits
next to the images it is auto-loaded.

### File menu

| Action | Shortcut | Notes |
|---|---|---|
| Open image file(s)… | `Ctrl+O` | |
| Open folder… | `Ctrl+Shift+O` | |
| Open stereo folders… | — | pick left then right folder |
| Load annotations file… | `Ctrl+I` | import COCO JSON |
| **Open rerun file…** | — | open a Rerun recording (`.rrd`) in the rerun viewer (see below) |
| **Open pose database…** | — | Clio inspection SQLite with per-timestamp poses for the map view |
| Save / Save as… | `Ctrl+S` / `Ctrl+Shift+S` | |
| Config settings… | `Ctrl+G` | same as the ⚙ button |

The **View** menu switches the UI theme at runtime (**dark**, **light**, or
**pastel** lime); the choice persists across sessions. It also carries
**Switch to Rerun waypoint view**, which replaces the image labeling view
with the embedded rerun waypoint map (the menu item becomes **Switch back
to image view** while the map is shown; see below).

### Rerun viewer & point-cloud map

The GUI can place your labels on a 3D map with the [Rerun](https://rerun.io)
viewer:

1. **Open rerun file…** (File menu or the sidebar's **🎬 Rerun viewer / map…**
   button) opens a `.rrd` recording — it carries the colored point-cloud
   map, the camera images and their timestamps — embedded in the app: the
   native rerun viewer runs as a subprocess and its X11 window is
   reparented into the waypoint view, which replaces the image labeling
   view (switch back with View → **Switch back to image view**). This is
   the full-speed native renderer, so even maps with 100M+ points stay
   smooth. Embedding needs an X11 session (xcb) and the `xwininfo` tool —
   standard on Linux desktops; over SSH/Wayland the recording opens in a
   standalone viewer window instead. The viewer opens on the **3D map
   view** by default — the camera image/depth views are left out; add
   them back from the viewer's blueprint menu if needed. The camera-body
   subtree (`body/camera_left`, `body/camera_right`, `body/axes`,
   `body/cloud_body` scans) is hidden by default — re-include it from the
   streams panel. If the recording's ground plane comes out tilted (some
   pipelines write a "leveled" ancestor transform that doesn't actually
   level the map), the GUI fits the map cloud's ground plane on open and
   streams a corrective static transform, so the map renders flat — the
   `.rrd` itself is never modified. Clicking the embedded map hands it the
   keyboard focus,
   so the viewer's WASD camera controls work as usual. Switching to the
   waypoint view with a recording already open in a standalone window
   moves it into the app (the extra window is closed). The GUI then
   streams into that recording.
2. **Open pose database…** (or **📍 Pose database…**) loads a Clio
   inspection SQLite DB whose `images` table holds per-timestamp
   camera/lidar poses (`--pose-db` does the same at launch).
3. **🗺 Show annotated in Rerun** plots every frame marked
   **✔ Mark as annotated** as a labeled camera position on the map — a
   visual coverage overview of what you labeled along the route. Markers
   are labeled with their waypoint ordinal in route order, e.g.
   `Waypoint #2 (frame 1042)` (both stereo sides of a frame share one
   waypoint number). **✖ Clear waypoints** removes all markers from the
   map again (the annotated marks themselves are kept).

How a frame is matched to a DB row is configurable under **Settings →
Rerun map / pose DB → Pose DB match** (config key `pose_db.match`):

| Mode | Matching rule |
|------|---------------|
| `auto` (default) | exact match on the DB `filename` column, then numeric filename stem as the DB `id`, then nearest `timestamp_ns` |
| `filename` | image file name = DB `filename` column only |
| `filename_id` | numeric filename stem (`1042.jpg` → 1042) = DB `id` column only |
| `timestamp` | filename stem / frame timestamp = DB `timestamp_ns` (nearest, within 10 s) only |

Use `filename_id` when your image folder is named by the images-table
`id` (e.g. `Datasets/complete3/outputs/images/1000.jpg`), `timestamp`
when files are named by nanosecond timestamps. The timestamp rule rejects
matches further than 10 s away, so sequential file names can no longer
silently pin every frame to the first pose.

CLI equivalent: `--pose-db` preloads the pose DB at launch.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `N` / `B` (or `→` / `←`) | next / previous frame (forward nav marks reviewed) |
| `U` | jump to next unlabeled frame |
| `Space` | play/pause playback |
| `S` | save and quit · `Q` quit (progress kept) |
| `A` | toggle draw-box mode |
| `D` / `Del` | delete selected box(es) |
| `X` | discard all boxes on current frame, mark reviewed, advance |
| `0–9` | assign category to a pending new box |
| `R` | re-segment selected box(es) with SAM3 (fresh masks) |
| `I` | interpolate between anchors |
| `K` | toggle ★ keyframe on current frame |
| `M` | show/hide mask overlays |
| `Z` | zoom to selected box · `F` / `0` fit view · `+` / `-` zoom steps |
| `Ctrl+A` | select all boxes on frame |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo (every operation is undoable) |
| `Esc` | clear selection / cancel point-segment or pending box |

### Mouse on the canvas

- **Left-drag in draw mode** (`A`) draws a box; release adds it if a category
  is preselected, otherwise pick one with digit keys or a category click.
- **Left-click** a box to select + drag-move; drag **corner handles** to
  resize. **Shift-click** toggles multi-selection.
- **Middle-drag** pans; **wheel** zooms around the cursor.
- **Point-segment mode (🎯 Add points)**: left-click = positive point (+),
  right-click = negative point (−); clicks only accumulate. Press
  **▶ Segment points** to run SAM3 with all placed points; add more and
  press again to refine. `Enter` accepts the object, `Esc` cancels it.

### Side panel buttons

| Button | What it does |
|---|---|
| **Add / Rename / Delete** (+ name field) | manage categories |
| **▶ Play / ⏸ Pause** + speed combo | playback at 0.25x–10x |
| **−10 −5 +5 +10** | jump buttons |
| **✔ Mark as annotated** | count frame as done without boxes (Viewpoint Selection section) |
| **🚫 Discard image** | drop the frame from the saved COCO entirely (blurry/irrelevant frames never reach training) |
| **🗺 Show annotated in Rerun** | plot all annotated frames' camera positions on the Rerun map (needs a pose DB + opened `.rrd`) |
| **✖ Clear waypoints** | remove all waypoint markers from the Rerun map (annotated marks are kept) |
| **🎬 Rerun viewer / map…** | open a `.rrd` recording in the rerun viewer — see *Rerun viewer & point-cloud map* above |
| **📍 Pose database…** | load a Clio pose DB to place annotated frames on the map |
| **★ Keyframe** (`K`) | mark anchor frame for interpolation |
| **Interpolate (I)** + **Stop** | optical-flow-fill boxes between two labeled/keyframe anchors; use for smooth, constant-direction motion |
| **Run SAM3 (all)** / **Re-seg sel (R)** / **Cancel** | segment all boxes on the frame / re-mask the selection |
| **SAM3 ALL frames** | background segmentation of every frame (checkpoints every 10 frames) |
| **🎯 Add points** + **▶ Segment points** | point-prompt mode: accumulate +/− points on clicks, then run SAM3 once with all of them (see mouse section) |
| **Propagate →** | follow the selected instance(s) forward with SAM3 video tracking, producing masks per frame (`memory` or `chain` method, Settings → SAM3); use for deformable objects / longer ranges than interpolation |
| **Autolabel frame** / **Autolabel ALL frames** | open-vocabulary pre-labeling with the configured backend; highlight 2+ categories to restrict the run |
| **Masks ON/OFF** + opacity slider | mask overlay visibility/transparency |

The **Boxes on this frame** list selects boxes by clicking rows;
**Cat of selected** / **Track of selected** fields reassign them by typing an
id + Enter (`C` / `T` focus the fields). The progress bar counts annotated
frames; the slider scrubs the sequence (progress saved on release).

### Recommended labeling workflow

The GUI loop is **box first, then segment**:

1. Qwen seeds boxes on keyframes → GUI opens on them.
2. Per frame, fix the boxes first: delete wrong ones (`D`), draw missed
   ones (`A`), move/resize the ones that are off.
3. Segment from the bbox prompts: **Run SAM3 (all)** segments every box
   on the frame; **Re-seg sel (`R`)** re-segments just the selected
   box(es).
4. Mask not right? Adjust the bbox and segment again (`R`) — the box is
   the prompt, so a tighter box gives a tighter mask.
5. Box segment still not working (thin or irregular objects)? Switch to
   **🎯 Add points**: click +/− points on the object, then **▶ Segment
   points** to run SAM3 once with all of them.
6. Sparse footage? Label every ~10th keyframe, then `I` interpolate
   between anchors, or select a good box and `Propagate →` for masks
   over long gaps (with a ★ keyframe marked, propagation stops there).
7. Missing a whole category? Highlight it in the category list and
   **Autolabel ALL frames**.
8. 🚫 Discard blurry/broken frames; ✔ mark empty-but-checked ones.
9. `Ctrl+S` save (or `S` save-and-quit). Discarded frames are excluded
   from the final COCO; stereo sessions save only timestamp-synced pairs.

### Settings dialog (⚙ Config / Ctrl+G)

- **Appearance** — UI theme: `dark`, `light`, or `pastel` lime (same as the
  View menu; applied on Apply and persisted, config key `ui.theme`).
- **Hide UI elements** — hide panel groups you don't use (`ui.hide`).
- **Advanced settings** — enables Interpolation/Tracking sections.
  - Interpolation: flow method (`dis`/`klt`/`farneback`), camera model,
    match distance, mismatch confirmation (`interpolation.*`).
  - Tracking: sticky track ids, show ids (`tracking.*`).
- **SAM3**: device, model path, confidence, auto-segment-on-add, min polygon
  area, autolabel NMS IoU, propagate method (`memory`/`chain`) + thresholds
  (`sam3.*`).
- **Autolabel detector**: `sam3`, `owlv2`, `owlv2_exemplar` (1-shot — select
  an existing box first; its crop becomes the visual query),
  `grounding_dino`, `falcon`, plus per-backend model/conf keys
  (`autolabel.*`). First use downloads HF checkpoints.
- **Masks / Display**: overlay opacity (`ui.mask_opacity`), max image size
  (`display.max_image_dim`, 0 = original).

Load/Apply/Save buttons live at the bottom; example config:
`scripts/config/label_review.example.json`.

### Files written by the GUI

- `<output_json>` — the reviewed COCO: polygon `segmentation` masks,
  `"side"` + `timestamp_ns` per image, `annotated_image_ids`, and
  `annotated_timestamps` (unique `timestamp_ns` of annotated frames, stereo
  pair collapsed to one entry); discarded frames excluded. Stereo saves
  only synced pairs, sorted earliest first.
- `<output>.progress` sidecar — current index, reviewed/annotated/discard
  marks, keyframes (so you can quit and resume anytime).

Rerun recordings (`.rrd`) are not written by the GUI — open an existing
recording via **Open rerun file…** and the GUI streams annotation markers
into it.
