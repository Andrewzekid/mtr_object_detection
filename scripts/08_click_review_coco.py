#!/usr/bin/env python3
"""
Interactive matplotlib COCO reviewer for cleaning Qwen-seeded bounding boxes.

Loads Qwen annotations (flat ``*_result.json`` or ``--split-by-class`` per-image
layout) or an existing reviewed COCO (``--json``), lets you delete / add boxes
with number-key category selection, and exports the cleaned COCO + an optional
YOLO detection dataset. Progress auto-saves to ``<output>.progress`` and
``<output>_tmp.json`` so a session can be resumed after Ctrl-C / quit.

USAGE:
    # Review Qwen-seeded boxes on the 400 NEW keyframes only (stride-5 set,
    # the other 400 already-reviewed keyframes are kept out so the reviewer
    # only sees unannotated frames). After review, merge this 400 with the
    # old 400 into the final 800-keyframe COCO for the interpolator.
    python scripts/08_click_review_coco.py \
        --qwen-annotations-dir output/annotations/MTR_4k \
        --img_dir /tmp/opencode/new_keyframes_only \
        --output_json output/MTR_4k/MTR_4k_keyframes/coco_reviewed_new400.json \
        --output-yolo-dir output/MTR_4k/MTR_4k_keyframes/yolo_export_new400 \
        --data-yaml Datasets/MTR/detect/train_yolo_detection/data.yaml
"""

import json
import os
import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="tkinter")  # 屏蔽字体警告


# ---------------------------------------------------------------------------
# Helpers for building a COCO dataset from Qwen split-by-class annotations and
# for exporting the reviewed COCO dataset back to YOLO format.
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def normalize_label(raw):
    """Map common Qwen label strings onto the canonical dataset names."""
    if raw is None:
        return None
    aliases = {
        "advertisement board": "Advertisement Board",
        "advertisement": "Advertisement Board",
        "ad board": "Advertisement Board",
        "ad": "Advertisement Board",
        "exit sign": "Exit Sign",
        "exit": "Exit Sign",
        "sign": "Exit Sign",
        "ceiling light": "Lights",
        "ceiling lights": "Lights",
        "light": "Lights",
        "lights": "Lights",
        "map": "Map",
        "maps": "Map",
        "tv": "TV",
        "tvs": "TV",
        "television": "TV",
        "ticket gate": "Ticket Gate",
        "ticket gates": "Ticket Gate",
        "gate": "Ticket Gate",
        "turnstile": "Ticket Gate",
        "turnstiles": "Ticket Gate",
    }
    key = str(raw).strip().lower()
    if key in aliases:
        return aliases[key]
    for alias, canon in aliases.items():
        if alias in key or key in alias:
            return canon
    return raw.strip()


def read_image_size(image_path):
    """Return (W, H) using PIL."""
    try:
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None


def _find_image_file(stem: str, img_dir: Path):
    """Return the first matching image file for ``stem`` under ``img_dir``.

    Checks ``img_dir`` first, then ``img_dir/images/``. Returns ``None`` if no
    supported image is found.
    """
    candidates = [img_dir, img_dir / "images"]
    for base in candidates:
        if not base.exists():
            continue
        for ext in IMAGE_EXTS:
            candidate = base / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _parse_split_by_class_json(json_file, stem, class_names, stem_boxes):
    """Parse one per-class JSON from ``--split-by-class`` output."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {json_file}: {e}")
        return
    class_name = data.get("class_name")
    if not class_name:
        return
    for bbox in data.get("bboxes", []) or []:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            stem_boxes[stem].append((class_name, [float(v) for v in bbox]))
            class_names.add(class_name)


def _parse_flat_result_json(json_file, class_names, stem_boxes):
    """Parse a flat ``<image_stem>_result.json`` file.

    The expected format is the default Qwen output without
    ``--split-by-class``::

        {
          "image": ".../1781167841090437000.jpg",
          "parsed_output": [
            {"bbox_2d": [x1, y1, x2, y2], "label": "Ceiling light"},
            ...
          ]
        }
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {json_file}: {e}")
        return

    # Prefer the image path embedded in the JSON to derive the stem.
    image_path = data.get("image", "")
    if image_path:
        stem = Path(image_path).stem
    else:
        # Fallback: strip the "_result" suffix from the JSON filename.
        stem = json_file.stem.replace("_result", "")

    parsed_output = data.get("parsed_output", [])
    if not isinstance(parsed_output, list):
        # Some result files contain a dict with a parse_error instead of boxes.
        return
    for obj in parsed_output:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        bbox = obj.get("bbox_2d") or obj.get("bbox")
        if not label or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        class_name = normalize_label(label)
        if not class_name:
            continue
        stem_boxes[stem].append((class_name, [float(v) for v in bbox]))
        class_names.add(class_name)


def build_coco_from_qwen(qwen_dir, img_dir):
    """Convert Qwen detection output into COCO format.

    Supports two layouts:

    1. Default flat output (no ``--split-by-class``)::

           qwen_dir/<image_stem>_result.json

       Each JSON contains ``image`` and ``parsed_output`` with ``bbox_2d`` and
       ``label``.

    2. ``--split-by-class`` output::

           qwen_dir/<image_stem>/<safe_class_name>.json

       Each JSON contains ``class_name`` and ``bboxes``.
    """
    qwen_dir = Path(qwen_dir)
    img_dir = Path(img_dir)

    if not qwen_dir.exists():
        raise FileNotFoundError(f"Qwen annotations dir not found: {qwen_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Collect unique class names and assign COCO category ids 0..N-1.
    class_names = set()
    stem_boxes = defaultdict(list)  # stem -> [(class_name, bbox)]

    flat_jsons = sorted(qwen_dir.glob("*_result.json"))
    if flat_jsons:
        print(f"  Detected flat Qwen result files ({len(flat_jsons)} files)")
        for json_file in flat_jsons:
            _parse_flat_result_json(json_file, class_names, stem_boxes)
    else:
        print("  Detected --split-by-class Qwen output")
        for stem_dir in sorted(d for d in qwen_dir.iterdir() if d.is_dir()):
            for json_file in sorted(stem_dir.glob("*.json")):
                _parse_split_by_class_json(
                    json_file, stem_dir.name, class_names, stem_boxes
                )

    if not class_names:
        raise ValueError(f"No valid annotations found in {qwen_dir}")

    sorted_class_names = sorted(class_names)
    name_to_cat_id = {name: i for i, name in enumerate(sorted_class_names)}
    categories = [{"id": i, "name": name} for i, name in enumerate(sorted_class_names)]

    # Build images and annotations. Include ALL images in img_dir (even
    # those with no Qwen annotations) so the reviewer can navigate the full
    # sequence and add boxes where Qwen missed.
    images = []
    annotations = []
    ann_id = 1

    # First, collect all image files in the folder (sorted chronologically).
    all_img_files = sorted(
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not all_img_files:
        raise ValueError(f"No images found in {img_dir}")

    # Build a stem -> file map for quick lookup.
    img_by_stem = {f.stem: f for f in all_img_files}

    # Process images in folder order (not annotation order) so the reviewer
    # sees the chronological sequence.
    for idx, img_file in enumerate(all_img_files):
        stem = img_file.stem
        size = read_image_size(img_file)
        if size is None:
            print(f"  Warning: could not read image size for {img_file}, skipping")
            continue
        w, h = size

        image_id = idx + 1
        images.append({
            "id": image_id,
            "file_name": img_file.name,
            "width": w,
            "height": h,
        })

        # Add any Qwen annotations for this image.
        for class_name, (x1, y1, x2, y2) in stem_boxes.get(stem, []):
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            x1 = max(0.0, min(x1, w))
            x2 = max(0.0, min(x2, w))
            y1 = max(0.0, min(y1, h))
            y2 = max(0.0, min(y2, h))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": name_to_cat_id[class_name],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def load_class_names(data_yaml: Path):
    """Read the ``names`` list from a data.yaml."""
    try:
        import yaml
    except ImportError:
        return None
    if not data_yaml.exists():
        return None
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, list) and names:
        return names
    return None


def export_yolo_from_coco(
    coco_data,
    img_dir,
    output_dir,
    data_yaml=None,
    copy_images=True,
    symlink_images=False,
):
    """Write a YOLO detection dataset from a COCO dict.

    Args:
        coco_data: dict with ``images``, ``annotations``, ``categories``.
        img_dir: directory containing the source image files.
        output_dir: directory to create with ``images/``, ``labels/`` and ``data.yaml``.
        data_yaml: optional reference data.yaml whose ``names`` order to mirror.
        copy_images: copy source images into the output dataset (default).
        symlink_images: symlink source images instead of copying.
    """
    output_dir = Path(output_dir)
    img_dir = Path(img_dir)
    if data_yaml is not None:
        data_yaml = Path(data_yaml)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    # Determine YOLO class id mapping.
    ref_names = load_class_names(data_yaml) if data_yaml else None
    if ref_names:
        ref_normalized = {normalize_label(n): i for i, n in enumerate(ref_names)}
        yolo_id_map = {}
        used_ref_ids = set()
        for cat_id, cat_name in categories.items():
            norm = normalize_label(cat_name)
            if norm in ref_normalized:
                yolo_id_map[cat_id] = ref_normalized[norm]
                used_ref_ids.add(ref_normalized[norm])
            else:
                # Fallback: append to the end using a fresh id.
                next_id = len(yolo_id_map)
                while next_id in used_ref_ids:
                    next_id += 1
                yolo_id_map[cat_id] = next_id
                used_ref_ids.add(next_id)
        final_names = list(ref_names)
        # Add any categories that were not in the reference names.
        for cat_id, cat_name in categories.items():
            if yolo_id_map[cat_id] >= len(final_names):
                if yolo_id_map[cat_id] == len(final_names):
                    final_names.append(cat_name)
                else:
                    # Pad if ids are sparse (rare).
                    while len(final_names) <= yolo_id_map[cat_id]:
                        final_names.append("")
                    final_names[yolo_id_map[cat_id]] = cat_name
    else:
        sorted_cat_ids = sorted(categories.keys())
        yolo_id_map = {cat_id: i for i, cat_id in enumerate(sorted_cat_ids)}
        final_names = [categories[cat_id] for cat_id in sorted_cat_ids]

    # Group annotations by image.
    anns_by_image = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    total_boxes = 0

    for img in coco_data.get("images", []):
        file_name = img["file_name"]
        img_w = img.get("width")
        img_h = img.get("height")

        stem = Path(file_name).stem
        src_img = _find_image_file(stem, img_dir)
        if src_img is None:
            print(f"  Warning: source image not found for '{file_name}' in {img_dir}")
            continue
        if img_w is None or img_h is None:
            size = read_image_size(src_img)
            if size is None:
                print(f"  Warning: could not read image size: {src_img}")
                continue
            img_w, img_h = size

        dst_img = images_out / file_name
        if not dst_img.exists():
            if symlink_images:
                dst_img.symlink_to(src_img.resolve())
            elif copy_images:
                shutil.copy2(src_img, dst_img)

        stem = Path(file_name).stem
        label_path = labels_out / f"{stem}.txt"
        lines = []
        for ann in anns_by_image.get(img["id"], []):
            cat_id = ann["category_id"]
            if cat_id not in yolo_id_map:
                continue
            x, y, w, h = ann["bbox"]
            xc = ((x + w / 2.0) / img_w)
            yc = ((y + h / 2.0) / img_h)
            nw = w / img_w
            nh = h / img_h
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{yolo_id_map[cat_id]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            total_boxes += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    # Write data.yaml.
    yaml_block = (
        f"train: ./images\n"
        f"val: ./images\n"
        f"test: ./images\n\n"
        f"nc: {len(final_names)}\n"
        f"names: {final_names}\n"
    )
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_block)

    print(f"\nExported YOLO dataset to {output_dir}")
    print(f"  Images: {len(coco_data.get('images', []))}")
    print(f"  Boxes written: {total_boxes}")
    print(f"  Classes: {final_names}")


class ProReviewer:
    def __init__(self, json_path=None, img_dir=None, output_json=None, original_coco=None):
        self.img_dir = img_dir
        self.output_json = output_json
        self.progress_file = output_json.replace('.json', '.progress')

        if original_coco is not None:
            self.original_coco = original_coco
        elif json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.original_coco = json.load(f)
        else:
            raise ValueError("Either --json or --qwen-annotations-dir must be provided")

        self.coco = self.original_coco.copy()

        # 如果输出文件已存在，则加载其中的标注（包含已删除的框）
        if os.path.exists(output_json):
            with open(output_json, 'r', encoding='utf-8') as f:
                self.coco = json.load(f)
            print(f"📂 加载已有标注文件 {output_json}，继续上次进度")
        else:
            print("📂 使用原始标注文件")

        # 构建类别映射
        self.cat_map = {cat['id']: cat['name']
                        for cat in self.coco.get('categories', [])}
        self.cat_name_to_id = {name: id for id, name in self.cat_map.items()}

        # 已删除的ID集合（从原始标注中推导）
        original_ids = {ann['id'] for ann in self.original_coco['annotations']}
        current_ids = {ann['id'] for ann in self.coco['annotations']}
        self.removed_ids = original_ids - current_ids

        # 图片列表（按原始顺序）
        self.img_list = self.original_coco['images']
        self.total = len(self.img_list)

        # 进度恢复：读取上次索引
        self.current_idx = 0
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                idx = data.get('last_index', 0)
                if 0 <= idx < self.total:
                    self.current_idx = idx
                    print(f"⏳ 从第 {idx+1}/{self.total} 张图片继续")
                else:
                    print("⚠️ 进度文件索引无效，从头开始")

        # 状态变量
        self.fig, self.ax = None, None
        self.img_array = None
        self.current_img_id = None
        self.current_anns = []
        self.selected_idx = -1

        # 绘制模式
        self.drawing = False
        self.rect_start = None
        self.rect_artist = None
        self.waiting_category = False
        self.pending_rect = None

        self.processed_count = 0

    def save_progress(self, is_final=False):
        """保存当前标注状态和进度索引"""
        # 保存标注（过滤掉删除的框）
        final_anns = [ann for ann in self.coco['annotations']
                      if ann['id'] not in self.removed_ids]
        coco_copy = self.coco.copy()
        coco_copy['annotations'] = final_anns
        save_path = self.output_json if is_final else self.output_json.replace(
            '.json', '_tmp.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_copy, f, indent=4, ensure_ascii=False)

        # 保存进度索引
        with open(self.progress_file, 'w') as f:
            json.dump({'last_index': self.current_idx + 1}, f)

        print(f"✅ 进度已保存至 {save_path}，当前索引 {self.current_idx+1}/{self.total}")

    def add_bbox(self, x, y, w, h, cat_id):
        if w <= 0 or h <= 0:
            return
        max_id = max([ann['id']
                     for ann in self.coco['annotations']], default=0)
        new_id = max_id + 1
        new_ann = {
            'id': new_id,
            'image_id': self.current_img_id,
            'category_id': cat_id,
            'bbox': [float(x), float(y), float(w), float(h)],
            'area': float(w * h),
            'iscrowd': 0
        }
        self.coco['annotations'].append(new_ann)
        # 更新当前显示列表（加到末尾）
        self.current_anns.append(new_ann)
        print(f"✅ 添加新框 ID={new_id}, 类别={self.cat_map[cat_id]}")

    def load_image(self, idx):
        """加载第 idx 张图片并更新界面"""
        if idx < 0 or idx >= self.total:
            return False
        img_info = self.img_list[idx]
        self.current_img_id = img_info['id']
        self.current_idx = idx
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        try:
            self.img_array = np.array(Image.open(img_path))
        except Exception as e:
            print(f"⚠️ 无法读取 {img_path}，跳过")
            return False

        # ***** 修正点：直接从 self.coco['annotations'] 动态过滤 ****
        all_anns = [ann for ann in self.coco['annotations']
                    if ann['image_id'] == self.current_img_id]
        self.current_anns = [
            ann for ann in all_anns if ann['id'] not in self.removed_ids]

        self.selected_idx = -1
        self.waiting_category = False
        self.drawing = False

        # 创建或重用图形窗口
        first_time = self.fig is None
        if first_time:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            try:
                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()
            except:
                pass
        else:
            self.ax.clear()

        self.redraw()
        if first_time:
            # 仅在首次加载时启动 GUI；此时还在 run() 调用
            # plt.show(block=True) 之前，进入阻塞事件循环是安全的。
            plt.show(block=False)
            plt.pause(0.1)
        else:
            # 已经处在阻塞的 plt.show(block=True) 事件循环中（由 n/b 键触发）。
            # 用 draw_idle + flush_events 保证重绘不阻塞事件循环。
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        return True

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img_array)

        # 绘制所有框（黄色，选中红色）
        for idx, ann in enumerate(self.current_anns):
            x, y, w, h = ann['bbox']
            color = 'red' if idx == self.selected_idx else 'yellow'
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                 linewidth=2 if idx == self.selected_idx else 1)
            self.ax.add_patch(rect)
            cat_name = self.cat_map.get(
                ann['category_id'], str(ann['category_id']))
            label = f"{cat_name} (id:{ann['id']})"
            self.ax.text(x, y-8, label, color=color, fontsize=10, weight='bold',
                         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        info = f"Image {self.current_idx+1}/{self.total} | Boxes: {len(self.current_anns)} | Selected: {self.selected_idx+1 if self.selected_idx>=0 else 0}"
        self.ax.set_title(info)
        if self.waiting_category:
            help_text = "Choose category: press number key (0-9) or ESC to cancel"
        else:
            help_text = "[D]elete  [A]dd  [N]ext  [B]ack  [X]Discard all  [S]ave & quit  [Q]uit"
        self.ax.set_xlabel(help_text)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def on_key(self, event):
        if self.waiting_category:
            if event.key == 'escape':
                self.waiting_category = False
                self.pending_rect = None
                print("Cancel adding box")
                self.redraw()
                return
            if event.key.isdigit():
                num = int(event.key)
                if num in self.cat_map:
                    cat_id = num
                    if self.pending_rect:
                        x, y, w, h = self.pending_rect
                        self.add_bbox(x, y, w, h, cat_id)
                        self.pending_rect = None
                        self.waiting_category = False
                        self.redraw()
                        print(f"Selected category: {self.cat_map[cat_id]}")
                    else:
                        print("Error: no pending rectangle")
                else:
                    print(f"⚠️ Category ID {num} not exist")
            return

        # 普通模式按键
        if event.key in ('d', 'D'):
            if self.selected_idx >= 0:
                ann = self.current_anns[self.selected_idx]
                self.removed_ids.add(ann['id'])
                print(f"🗑️ Deleted box ID: {ann['id']}")
                self.current_anns.pop(self.selected_idx)
                self.selected_idx = -1
                self.redraw()
            else:
                print("⚠️ No box selected")

        elif event.key in ('a', 'A'):
            if not self.drawing:
                self.drawing = True
                self.rect_start = None
                print("✏️ Drawing mode: click top-left, drag to bottom-right")
                self.fig.canvas.mpl_connect(
                    'button_press_event', self.on_draw_start)
                self.fig.canvas.mpl_connect(
                    'motion_notify_event', self.on_draw_move)
                self.fig.canvas.mpl_connect(
                    'button_release_event', self.on_draw_end)
            else:
                print("Already in drawing mode")

        elif event.key in ('x', 'X'):
            # Discard ALL annotations on the current image and move to next.
            if self.current_anns:
                for ann in self.current_anns:
                    self.removed_ids.add(ann['id'])
                n = len(self.current_anns)
                self.current_anns = []
                self.selected_idx = -1
                print(f"🗑️ Discarded {n} box(es) on this image")
            else:
                print("ℹ️ No boxes to discard on this image")
            self.save_progress(is_final=False)
            self.current_idx += 1
            if self.current_idx >= self.total:
                print("🏁 Reached last image, finishing")
                self.save_progress(is_final=True)
                plt.close()
                exit(0)
            else:
                self.load_image(self.current_idx)

        elif event.key in ('n', 'N'):
            self.save_progress(is_final=False)
            self.current_idx += 1
            if self.current_idx >= self.total:
                print("🏁 Reached last image, finishing")
                self.save_progress(is_final=True)
                plt.close()
                exit(0)
            else:
                self.load_image(self.current_idx)

        elif event.key in ('b', 'B'):
            if self.current_idx > 0:
                self.save_progress(is_final=False)
                self.current_idx -= 1
                self.load_image(self.current_idx)
            else:
                print("⛔ Already first image")

        elif event.key in ('s', 'S'):
            self.save_progress(is_final=True)
            print("✅ Saved final result and exit")
            plt.close()
            exit(0)

        elif event.key in ('q', 'Q'):
            print("⏹️ Exit without saving (progress saved in temp file)")
            plt.close()
            exit(0)

    def on_click(self, event):
        if self.drawing or self.waiting_category:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        for idx, ann in enumerate(reversed(self.current_anns)):
            bx, by, bw, bh = ann['bbox']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.selected_idx = len(self.current_anns) - 1 - idx
                self.redraw()
                break
        else:
            self.selected_idx = -1
            self.redraw()

    def on_draw_start(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        self.rect_start = (event.xdata, event.ydata)
        if self.rect_artist:
            self.rect_artist.remove()
            self.rect_artist = None

    def on_draw_move(self, event):
        if not self.drawing or self.rect_start is None or event.inaxes != self.ax:
            return
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        if self.rect_artist:
            self.rect_artist.remove()
        from matplotlib.patches import Rectangle
        rect = Rectangle((min(x0, x1), min(y0, y1)), abs(x1-x0), abs(y1-y0),
                         fill=False, edgecolor='blue', linestyle='dashed', linewidth=2)
        self.rect_artist = self.ax.add_patch(rect)
        self.fig.canvas.draw_idle()

    def on_draw_end(self, event):
        if not self.drawing or self.rect_start is None or event.inaxes != self.ax:
            return
        x0, y0 = self.rect_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        self.drawing = False
        if self.rect_artist:
            self.rect_artist.remove()
            self.rect_artist = None
        self.fig.canvas.draw_idle()
        if w <= 0 or h <= 0:
            print("Invalid rectangle, cancelled")
            return
        print(f"📐 Drawn rectangle: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
        self.pending_rect = (x, y, w, h)
        self.waiting_category = True
        print("📋 Available categories (press number key):")
        for cat_id, name in self.cat_map.items():
            print(f"  {cat_id} -> {name}")
        print("Press ESC to cancel")
        self.redraw()

    def run(self):
        if self.load_image(self.current_idx):
            plt.show(block=True)
        else:
            print("Failed to load starting image")


def main():
    parser = argparse.ArgumentParser(
        description='COCO标注高级审查：数字键选类别，B返回，自动续传。'
                    '也支持直接加载 Qwen 输出（默认 flat 的 *_result.json 或 '
                    '--split-by-class 的 per-image/per-class 目录）并导出 YOLO 数据集。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--json', help='原始COCO JSON文件（与 --qwen-annotations-dir 二选一）')
    parser.add_argument('--img_dir', required=True, help='图片文件夹')
    parser.add_argument('--output_json', default='coco_filtered.json', help='输出JSON文件')
    parser.add_argument(
        '--qwen-annotations-dir',
        help='Qwen 标注目录。支持两种布局：'
             '(1) 默认 flat 输出 qwen_dir/<image_stem>_result.json；'
             '(2) --split-by-class 输出 qwen_dir/<image_stem>/<class>.json。'
             '提供此参数时无需 --json。',
    )
    parser.add_argument(
        '--output-yolo-dir',
        help='退出时将审查后的 COCO 同时导出为 YOLO 检测数据集（包含 images/ labels/ data.yaml）。',
    )
    parser.add_argument(
        '--data-yaml',
        help='参考 data.yaml，用于确定 YOLO 导出时的类别顺序（可选）。',
    )
    parser.add_argument(
        '--copy-images', action='store_true', default=True,
        help='导出 YOLO 时复制图片（默认）。',
    )
    parser.add_argument(
        '--symlink-images', action='store_true',
        help='导出 YOLO 时用软链接代替复制。',
    )
    args = parser.parse_args()

    if not args.json and not args.qwen_annotations_dir:
        parser.error("必须提供 --json 或 --qwen-annotations-dir 之一")

    if args.qwen_annotations_dir:
        print(f"正在从 Qwen 标注构建 COCO: {args.qwen_annotations_dir}")
        original_coco = build_coco_from_qwen(args.qwen_annotations_dir, args.img_dir)
        print(f"  图片: {len(original_coco['images'])}, 标注: {len(original_coco['annotations'])}")
        print(f"  类别: {[c['name'] for c in original_coco['categories']]}")
        reviewer = ProReviewer(
            img_dir=args.img_dir,
            output_json=args.output_json,
            original_coco=original_coco,
        )
    else:
        reviewer = ProReviewer(args.json, args.img_dir, args.output_json)

    try:
        reviewer.run()
    finally:
        # Always export YOLO if requested, even when the user quits early.
        if args.output_yolo_dir:
            print(f"\n正在导出 YOLO 数据集到: {args.output_yolo_dir}")
            # Build the latest COCO view excluding removed boxes.
            final_coco = reviewer.coco.copy()
            final_coco['annotations'] = [
                ann for ann in reviewer.coco['annotations']
                if ann['id'] not in reviewer.removed_ids
            ]
            export_yolo_from_coco(
                final_coco,
                img_dir=args.img_dir,
                output_dir=args.output_yolo_dir,
                data_yaml=args.data_yaml,
                copy_images=args.copy_images,
                symlink_images=args.symlink_images,
            )


if __name__ == '__main__':
    main()
