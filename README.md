# Object Detection Application

A desktop application and CLI pipeline for object detection / segmentation
training and inference, built with PyQt6 and Ultralytics YOLO.

---

## Overview
Label review and segmentation:
```
python -m gui.label_review.main
# stereo dual-view (left/right folders, frames paired positionally):
python -m gui.label_review.main --images /path/to/left --images-right /path/to/right
```
Launches interactive GUI for complete label review and segmentation.

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

**Full pipeline** (dense labeling — every frame is labeled. RECOMMENDED):

```
raw images → 07 Qwen VLM auto-label → 08 human review/clean boxes
            → 08b train/val/test split → 09 SAM3 box→mask seg dataset → Review the masks on roboflow and augment data
            → 04 YOLO train → 05 evaluate → 11 tracking on raw frames
```
**Keyframe pipeline** (sparse labeling for near-static cameras — label every
Nth frame, interpolate the rest.):

```
12 extract keyframes → 07 Qwen seed boxes on keyframes → 08 review keyframes
   → 13 interpolation to all frames → 08 review all
   → 09 SAM3 segmentation dataset → Review the masks on roboflow and augment data → 04 YOLO train → 05 evaluate → 11 tracking on raw frames
```


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

## Prerequisites

```bash
pip install -r requirements.txt        # loose bounds
# or exactly reproduce the dev machine:
pip install -r requirements-lock.txt
```

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

  ```

---

## Complete pipeline example: `Datasets/MTR/MTR_new_1k`

The commands below run the whole workflow from raw images in
`Datasets/MTR/MTR_new_1k` to a trained model and tracked results.
You can paste each block into the terminal from top to bottom.

### 0. (Optional) select new unlabeled images

Skip this if you already have a folder of images to label.

```bash
python scripts/00_sample_from_dataset.py \
  --out-dir Datasets/MTR/MTR_new_1k

#Example with only 10 images
python scripts/00_sample_from_dataset.py \
  --source-dir Datasets/MTR/MTR_new_1k --out-dir Datasets/MTR/MTR_new_10_images -n 10 --copy
```

### 1. Auto-label with Qwen3.6

```bash
python scripts/07_run_qwen.py \
  --use-api --api-key-env API_KEY --api-model qwen3.6-flash \
  --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions." \
  --template object_detection --format json \
  --image-folder Datasets/MTR/MTR_new_1k \
  --output Datasets/MTR/MTR_new_1k/qwen_results \
  --vis-output output/MTR_new_1k/qwen_vis

#Example with only 10 images
python scripts/07_run_qwen.py \
  --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions." \
  --template object_detection --format json \
  --image-folder Datasets/MTR/MTR_new_10_images \
  --output Datasets/MTR/MTR_new_10_images_annotations \
  --vis-output output/MTR_new_10_images/qwen_vis

#Per image labels
python scripts/07_run_qwen.py   --prompt "Detect all: Ceiling light, Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions."   --template object_detection --format json   --image-folder Datasets/MTR/MTR_new_10_images   --output Datasets/MTR/MTR_new_10_images_annotations   --vis-output output/MTR_new_10_images/qwen_vis --per-class --conditioning-images ./ref_images --per-image-labels 

python scripts/07_run_qwen.py   --prompt "Detect all: Exit Sign, Advertisement Board, Ticket Gate, Map, TV. Ceiling lights are flat, horizontal rectangular strips on the ceiling. Exit signs are hanging LCD screens showing directions. Advertisement boards are flat LCD screens on the green wall showing commercial content. Maps are posters showing MTR directions. Ticket gates are turnstiles. TVs are hanging LCD screens showing general content, not directions."   --template object_detection --format json   --image-folder Datasets/MTR/MTR_new_10_images   --output Datasets/MTR/MTR_new_10_images_annotations   --vis-output output/MTR_new_10_images/qwen_vis --per-class --conditioning-images ./ref_images --per-image-labels 

```

Output:

- `Datasets/MTR/MTR_new_1k/qwen_results/<image_stem>_result.json`
- `Datasets/MTR/MTR_new_1k/qwen_results/summary.json`
- `output/MTR_new_1k/qwen_vis/`

> `08_click_review_coco.py` auto-detects both the default flat `*_result.json`
> layout and the `--split-by-class` per-image/per-class layout, so either Qwen
> output format works in the review step.

### 2. Review and clean the boxes with `08_click_review_coco.py`

```
python scripts/08_click_review_coco.py \
  --qwen-annotations-dir output/annotations/MTR_4k \
  --img_dir Datasets/MTR/MTR_4k_dataset_exit_signs \
  --output_json output/MTR_4k/reviewed/coco_reviewed.json \
  --output-yolo-dir output/MTR_4k/reviewed/yolo_reviewed \
  --data-yaml Datasets/MTR/MTR_4k/detect/train_yolo_detection/data.yaml

python scripts/08_click_review_coco.py \
  --qwen-annotations-dir Datasets/MTR/MTR_new_10_images_annotations \
  --img_dir Datasets/MTR/MTR_new_10_images \
  --output_json output/MTR_new_10_images/reviewed/coco_reviewed.json \
  --output-yolo-dir output/MTR_new_10_images/reviewed/yolo_reviewed \
  --data-yaml Datasets/MTR/detect/train_yolo_detection/data.yaml

```

Interactive controls in the matplotlib window:

| Key | Action |
|-----|--------|
| click | select a box |
| `D` | delete selected box |
| `A` | add a new box (draw, then press digit for class) |
| `N` | next image |
| `B` | previous image |
| `S` | save and quit |
| `Q` | quit (progress is saved in `.progress`) |

Output:

- `output/MTR_new_1k/reviewed/coco_reviewed.json`
- `output/MTR_new_1k/reviewed/yolo_reviewed/` (images, labels, data.yaml)


### 3. Split the reviewed dataset into train/val/test

`09_create_seg_dataset.py` expects a split layout (`images/{train,val,test}/`,
`labels/{train,val,test}/`), but `08_click_review_coco.py` emits a flat
`yolo_reviewed/` dataset. This intermediate step splits it and writes a
`data.yaml` carrying the class names from the reviewed dataset.

```bash
python scripts/08b_split_reviewed_dataset.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_reviewed \
  --output-dir output/MTR_new_1k/reviewed/yolo_split \
  --ratios 0.7 0.15 0.15 --seed 42

python scripts/08b_split_reviewed_dataset.py \
  --input-dir output/MTR_new_10_images/reviewed/yolo_reviewed \
  --output-dir output/MTR_new_10_images/reviewed/yolo_split \
  --ratios 0.7 0.15 0.15 --seed 42
```

Output layout:

```
output/MTR_new_10_images/reviewed/yolo_split/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml
```

### 4. Convert detection boxes to SAM3 segmentation masks

```bash
python scripts/09_create_seg_dataset.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_split \
  --output-dir output/MTR_new_1k/reviewed/yolo_seg \
  --model core/sam3/models/sam3-model/sam3.pt \
  --conf 0.4 --device cuda

python scripts/09_create_seg_dataset.py \
  --input-dir output/MTR_new_10_images/reviewed/yolo_split \
  --output-dir output/MTR_new_10_images/reviewed/yolo_seg \
  --model core/sam3/models/sam3-model/sam3.pt \
  --conf 0.4 --device cuda
```

Output layout:

```
output/MTR_new_1k/reviewed/yolo_seg/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
└── creation_summary.json
```

### 5. Upload to Roboflow (optional)

Upload to roboflow and clean the segmentation masks. Afterwards, create a new dataset on roboflow and do data augmentation. Download the dataset and use for training model

### 6. Train a YOLO segmentation model

```bash
python scripts/04_train_model.py \
  --config output/MTR_new_10_images/reviewed/yolo_seg/data.yaml \
  --model-type yolo26l --task segment --loss-type focal \
  --epochs 1000 --batch-size 32 --device 0 --imgsz 640
```

Trained weights are written to
`runs/segment/output/training/yolo_training/weights/best.pt` by default.

### 7. Evaluate the trained model

```bash
python scripts/05_evaluate_model.py \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data output/MTR_new_1k/reviewed/yolo_seg/data.yaml \
  --split val --conf 0.5 --iou 0.5
```

### 8. Run tracking on the original images
If data is from metcam, need to first undistort the result. Undistortion example:
```bash
python scripts/undistort_rosbag.py --images-root Datasets/iw/tracking/IW_run2/camera/left --output-root Datasets/iw/tracking/IW_run2_left_undistorted --calibration Datasets/iw/tracking/IW_run2/info/calibration.json --camera-name left

python scripts/undistort_rosbag.py --images-root Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag/camera/left --output-root Datasets/MTR/tracking/MTR_left_undistorted --calibration Datasets/MTR/rosbags/2026-06-11_16-50-08_rosbag/info/calibration.json --camera-name left

```
```bash
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/MTR_new_1k \
  --output output/MTR_new_1k/tracking \
  --conf 0.5 --device 0

#Examples:
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/MTR_segmentation/weights/best.pt \
  --data Datasets/MTR/tracking/MTR_left_undistorted \
  --conf 0.4 --device 0 \
  --warmup-frames 3 \
  --output output/tracking/MTR/metacam/left/tracking1

python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```

Tracking output:

- `output/MTR_new_1k/tracking/tracked_*.jpg`
- `output/MTR_new_1k/tracking/results.json` — COCO-style tracking results
- `output/MTR_new_1k/tracking/tracking_result.mp4` — summary video

---

### Benchmark mode (no output files, reports timing)

Run YOLO tracking + segmentation without saving images, JSON, or video, and
measure the time split between the model+tracker call and the result
post-processing.

```bash
python scripts/11_run_tracking.py \
  --tracker benchmark \
  --benchmark-tracker botsort \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/MTR_new_1k \
  --conf 0.5 --device 0 \
  --warmup-frames 3

#Examples:
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/MTR/tracking/MTR_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/MTR/metacam/left/tracking1

python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/iw_segmentation/weights/best.pt \
  --data Datasets/iw/tracking/IWrun2/IW_run2_left_undistorted \
  --conf 0.5 --device 0 \
  --warmup-frames 3
  --output output/tracking/iw/IWrun2
```

It prints mean / min / max / p50 / p95 / p99 latencies and FPS for:

- `YOLO model + tracker time`
- `Tracking result processing time`
- `Combined per-frame time`

Speed options for benchmark mode:

- `--half` — FP16 inference on CUDA.
- `--no-masks` — skip reading segmentation masks from results.
- `--no-cmc` — disable BoT-SORT camera-motion compensation.
- `--imgsz 320` — smaller input resolution (trades accuracy for speed).

---

## Script reference

Every entry point in `scripts/`. "Key inputs" lists the important CLI flags
(not exhaustive — run `python scripts/<name>.py --help` for the full set).

### Pipeline stages

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `00_sample_from_dataset.py` | Pick N images from a source folder that aren't already in the labeled YOLO set, ready for labeling. | `--source-dir`, `--labeled-dir`, `--out-dir`, `-n`, `--copy`/`--symlink`, `--dry-run` | New image folder (copied or symlinked) |
| `07_run_qwen.py` | Auto-label images with a VLM: Qwen via Ollama or DashScope API, or Gemma/Gemini. Supports conditioning refs, per-class, per-image labels. | `--prompt`, `--image`/`--image-folder`, `--template`, `--model`, `--conditioning-images`, `--per-class`, `--per-image-labels`, `--bbox-order`, `--dedup-iou`, `--use-api` | `<stem>_result.json` per image, `summary.json`, optional `--vis-output` |
| `08_click_review_coco.py` | Interactive matplotlib reviewer to clean Qwen boxes; exports COCO + YOLO detection labels. | `--qwen-annotations-dir`, `--img_dir`, `--output_json`, `--output-yolo-dir`, `--data-yaml` | `coco_reviewed.json`, `yolo_reviewed/` (images, labels, `data.yaml`) |
| `08b_split_reviewed_dataset.py` | Split `08`'s flat `yolo_reviewed/` into train/val/test for `09`. | `--input-dir`, `--output-dir`, `--ratios`, `--seed`, `--symlink-images` | `images/labels/{train,val,test}/`, `data.yaml` |
| `09_create_seg_dataset.py` | Convert detection boxes → SAM3 masks → YOLO seg polygons, per class. | `--input-dir` (split), `--output-dir`, `--model` (sam3), `--conf`, `--device` | YOLO seg dataset + `creation_summary.json` |
| `04_train_model.py` | Train a YOLO detect/segment model (Ultralytics). | `--config` (data.yaml), `--model-type`, `--task`, `--epochs`, `--batch-size`, `--device`, `--loss-type` | `runs/.../weights/best.pt` |
| `05_evaluate_model.py` | Evaluate a trained model on a split, or compare pred vs GT COCO JSON. | `--model`, `--data`, `--split`, `--conf`, `--iou` (or `--pred-json`/`--gt-json`) | Metrics report (stdout) |
| `11_run_tracking.py` | YOLO tracking (ByteTrack / BoT-SORT / detect-then-SAM3) + a no-output benchmark mode. | `--tracker`, `--model`, `--data`, `--output`, `--conf`, `--device`, `--warmup-frames` | `tracked_*.jpg`, `results.json`, `tracking_result.mp4` |

### Keyframe pipeline

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `12_extract_keyframes.py` | Select every Nth frame as a keyframe + write a manifest for the interpolator. | `--image-folder`/`--video`, `--output-dir`, `--every`, `--mode` | Keyframe images + `keyframe_manifest.json` |
| `13_interpolate_tracks.py` | Propagate reviewed keyframe boxes to every frame via anchored optical flow. An optional per-frame RANSAC camera model (`--camera-model global`) can absorb non-linear camera shake (tracking/anchoring on the residual, KLT dropouts re-seeded); off by default — accuracy testing on the re-reviewed MTR 4k frames showed no net gain on this fisheye camera. New objects at a keyframe are back-tracked. Each output box carries a `source`/`confidence` provenance field. | `--keyframes-coco`, `--manifest`, `--image-folder`, `--output-coco`, `--match-max-dist`, `--flow-method`, `--interp-method`, `--camera-model` | COCO annotations for every frame (+ optional vis) |

### Data prep & conversion

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `01_verify_labels.py` | Validate YOLO labels: missing files, bad formats, class stats. | `--input-dir`, `--images-subdir`, `--labels-subdir`, `--class-names`, `--fix` | Report (stdout / `--output`); optional in-place fixes |
| `02_augment_data.py` | Augment a labeled YOLO dataset (flip/rotate/brightness/contrast/mosaic). | `--input-dir`, `--output-dir`, `--augmentations`, `--multiplier` | Augmented images + labels |
| `03_split_dataset.py` | Split a labeled YOLO dataset into train/val/test + `data.yaml`. | `--input-dir`, `--output-dir`, `--ratios`, `--generate-yaml` | `images/labels/{train,val,test}/`, `data.yaml` |
| `10_qwen_json_to_yolo.py` | Convert `07 --split-by-class` JSON into a YOLO detection dataset. | `--annotations-dir`, `--image-folder`, `--output-dir`, `--data-yaml` | YOLO detect dataset (`images/`, `labels/`, `data.yaml`) |

### Standalone tools

| Script | Purpose | Key inputs | Outputs |
|-------|---------|------------|---------|
| `06_run_sam3.py` | Run SAM3 segmentation on an image/folder, optionally with bbox exemplars. | `--image`/`--image-folder`, `--bbox`/`--bbox-json`, `--concept`, `--model` | Masks / overlay images (`--output`, `--save-overlay`) |
| `track_sam3_video.py` | SAM3VideoPredictor video segmentation (single / reseed / chunks modes). | `--mode`, `--yolo-model`, `--sam3-model`, `--data`, `--output` | Per-frame masks / video |
| `undistort_rosbag.py` | Fisheye-undistort a folder of images using a calibration JSON. | `--images-root`, `--output-root`, `--calibration`, `--camera-name` | Undistorted images |
| `visualize.py` | Visualize Qwen annotations / YOLO detect / YOLO seg / model predictions. | `--mode`, `--dataset`/`--annotations-folder`, `--output`, `--model` | Annotated images |
| `tracking_utils.py` | Shared helpers (tracker YAML, IoU, mask→polygon, summary video). | — (library, not a CLI) | — |

> `scripts/12_upload_to_roboflow.py` is referenced in step 5 but is not present
> in the repo (only a stale `.pyc`); the `12_*` slot is now `12_extract_keyframes.py`.

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


