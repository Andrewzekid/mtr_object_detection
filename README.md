# Object Detection Application

A desktop application and CLI pipeline for object detection / segmentation
training and inference, built with PyQt6 and Ultralytics YOLO.

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install roboflow                 # only needed for the Roboflow upload step
```

- **Qwen3.6 labeling** can run locally with Ollama or via DashScope API.
  The examples below use the DashScope API. Set your key first:

  ```bash
  export API_KEY=your_dashscope_key
  ```

- **SAM3 segmentation** needs the Ultralytics SAM checkpoint. Place it at
  `core/sam3/models/sam3-model/sam3.pt` (or pass `--model` explicitly).

- **Roboflow upload** needs a project already created in your workspace and
  `ROBOFLOW_API_KEY` exported:

  ```bash
  export ROBOFLOW_API_KEY=your_roboflow_key
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
  --qwen-annotations-dir Datasets/MTR/mtr_new_1k_annotations \
  --img_dir Datasets/MTR/MTR_new_1k \
  --output_json output/MTR_new_1k/reviewed/coco_reviewed.json \
  --output-yolo-dir output/MTR_new_1k/reviewed/yolo_reviewed \
  --data-yaml Datasets/MTR/detect/train_yolo_detection/data.yamlbash

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


### 3. Convert detection boxes to SAM3 segmentation masks

```bash
python scripts/09_create_seg_dataset.py \
  --input-dir output/MTR_new_1k/reviewed/yolo_split \
  --output-dir output/MTR_new_1k/reviewed/yolo_seg \
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

### 4. Upload to Roboflow
Data augmentation and train test split is done by roboflow
```bash
python scripts/12_upload_to_roboflow.py \
  --dataset-dir output/MTR_new_1k/reviewed/yolo_seg \
  --workspace my-workspace \
  --project my-project \
  --api-key-env ROBOFLOW_API_KEY
```

Use `--dry-run` to preview the upload plan, or `--max-images-per-split N` to
limit the number of images sent per split.

### 5. Train a YOLO segmentation model

```bash
python scripts/04_train_model.py \
  --config output/MTR_new_1k/reviewed/yolo_seg/data.yaml \
  --model-type yolo26l --task segment --loss-type focal \
  --epochs 1000 --batch-size 32 --device 0 --imgsz 640
```

Trained weights are written to
`runs/segment/output/training/yolo_training/weights/best.pt` by default.

### 6. Evaluate the trained model

```bash
python scripts/05_evaluate_model.py \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data output/MTR_new_1k/reviewed/yolo_seg/data.yaml \
  --split val --conf 0.5 --iou 0.5
```

### 7. Run tracking on the original images

```bash
python scripts/11_run_tracking.py \
  --tracker botsort \
  --model runs/segment/output/training/yolo_training/weights/best.pt \
  --data Datasets/MTR/MTR_new_1k \
  --output output/MTR_new_1k/tracking \
  --conf 0.5 --device 0
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

## Notes

- The pipeline assumes a standard YOLO layout:
  `images/{train,val,test}/` and `labels/{train,val,test}/`. The augmentation
  script also uses `images/` and `labels/` directly under its `--input-dir`.
- To skip the interactive review step and convert Qwen JSON directly to YOLO,
  use `scripts/10_qwen_json_to_yolo.py`.
- For detection-only training, replace `--task segment` with `--task detect`
  and point `04_train_model.py` at a detection `data.yaml`.
- Launch the GUI any time with `python app.py`.
