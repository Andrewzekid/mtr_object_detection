#!/usr/bin/env python3
"""
Interactive 2D bbox reviewer for image files and folders.

Code layout: this package holds the UI (``ui/``: canvas, side panel,
dialogs, main window) and the COCO/undo state (``state/``). The
background worker threads (SAM3 segmentation, optical-flow
interpolation) live in ``workers/label_review_workers.py`` and are
imported by the UI modules.

Run it
------
    # With an image source:
    python -m gui.label_review.main \
        --images /path/to/folder_or_image.jpg \
        --output_json output/my_labels/coco.json

    # Stereo pair (left + right folders, frames paired positionally):
    python -m gui.label_review.main \
        --images /path/to/left --images-right /path/to/right

    # Idle mode (default when no source is given) — pick a source from the
    # File menu once the window is up:
    python -m gui.label_review.main

What this does
---------------
* Opens a set of plain image files (``--images`` with one or more
  files/folders; folders are scanned for jpg/jpeg/png/bmp/webp/tif, sorted
  by name). If every file stem is a bare integer
  (e.g. ``1712345678901234567.jpg``), the filenames are treated as
  nanosecond timestamps: frames sort by timestamp and the slider/info show
  them; otherwise the UI shows the plain image index. Without ``--images``
  the app starts idle until you load a source from the File menu.
* New categories can be added from the side panel at any time (name field +
  Add button under the category list); the new category gets the next free
  id and is preselected for the next draw. The Rename / Delete buttons next
  to it act on the category selected in the list: rename only changes the
  name (boxes keep their category), delete asks for confirmation and removes
  the boxes using that category too.
* The File menu switches the frame source at runtime (the current session
  is saved first, categories are kept): ``Ctrl+O`` open image files,
  ``Ctrl+Shift+O`` open folder, ``Ctrl+I`` import annotations from a COCO
  json (images matched by file name, categories merged by name, masks
  restored from polygon ``segmentation``; duplicates are skipped).
  ``Ctrl+G`` opens the
  Config settings dialog. ``Ctrl+S`` saves the COCO JSON to the current
  output path without quitting; ``Ctrl+Shift+S`` (Save as…) picks a new
  location and keeps saving there. Image sessions write
  ``labels_coco.json`` next to the chosen images unless ``--output_json``
  is given.
* Renders a 2D canvas of the *current frame's* image, plus any bboxes you
  draw. Number keys pick the category (same UX as
  ``08_click_review_coco.py``).
* Persists edits to a COCO json + ``.progress`` file in the same on-disk
  format that ``08_click_review_coco.py`` produces, so downstream scripts
  (08b, 13_interpolate_tracks.py, ...) keep working unchanged.
* Navigate frames with N/B keys, the arrow keys, or the slider; a play
  button auto-advances at 0.25x–10x speed (1x = 30 fps, index-based —
  one frame per tick, independent of any timestamps).

Stereo dual-view
----------------
* ``--images`` is the left (or mono) source; ``--images-right`` adds the
  right folder (or use File → Open stereo folders…). Frames are paired
  positionally — matching filenames in the same order are expected;
  shorter folder wins (a warning is printed on a length mismatch).
* Both sides render side by side (LEFT/RIGHT HUD tag) and navigate
  together: the slider, N/B, ±5/±10 jumps and playback move both canvases
  in lockstep.
* The last-clicked canvas is the ACTIVE side: edit shortcuts (D/A/X/R,
  digit category picks), discard-all, draw/pending-cat assignment,
  SAM3-this-frame, re-segment and autolabel-frame all act on it. Batch ops
  (Interpolate, SAM3 ALL frames, Autolabel ALL, Propagate) run on the
  active side only.
* Annotations distinguish sides: each image record in the COCO output
  carries ``"side": "left"|"right"`` (old files without the field count
  as left), and ``annotated_image_idxs`` unions frames annotated on either
  side. The default output in stereo is ``<left folder>/labels_coco.json``.
  Annotation import (Ctrl+I) matches images by (file name, side).

SAM3 segmentation
-----------------
* ``M`` toggles mask overlay visibility.
* ``Run SAM3 (all)`` button (or ``Shift+M``) runs Ultralytics SAM3 on every
  bbox on the current frame and overlays the resulting masks. Masks are
  stored per-annotation in the COCO output as polygon ``segmentation``
  (same shape as ``scripts/results.json``; the legacy base64-PNG ``mask``
  field is still read on load) so they round-trip with the json file.
  Requesting another run while one is in flight queues it (FIFO); the
  status line shows the pending count, e.g. ``SAM3 (1 in queue): done —
  4 mask(s), 0 failed``. Cancel drops the running job and the queue.
* ``R`` re-segments the *selected* bbox only — useful after you've redrawn a
  bbox to fix a bad SAM3 mask: delete the old box, draw a new one, then press
  R while it's selected to regenerate the mask.
* ``--auto-segment`` runs SAM3 automatically right after each new bbox is
  drawn, so you don't need to press anything.
  While SAM3 runs, the side panel shows per-concept progress and a Cancel
  button (cooperative — the in-flight concept finishes first). The mask
  overlay opacity is adjustable via the slider next to the mask toggle.
  "SAM3 ALL frames" runs SAM3 in the background over every box without a
  mask on every frame; the progress bar above the frame slider shows how
  many frames have at least one box.

SAM3 autolabel (text prompts)
-----------------------------
The "Autolabel frame" / "Autolabel ALL frames" buttons (hidden by
default — unhide via the Config dialog's "Autolabel buttons" checkbox or
``ui.hide``) run SAM3's open-vocabulary detection
(``SAM3SemanticPredictor``) using the category *names* as text prompts —
no boxes needed first. Highlighting categories in the side list
(Ctrl/Shift multi-select) restricts the run to those categories; with no
highlight every category is prompted. Detections become regular, editable
bboxes with masks attached: class-aware NMS (``sam3.autolabel_nms_iou``,
default 0.7) deduplicates the overlapping masks SAM3 tends to return for
one object, duplicates of an existing same-category box (IoU > 0.7) are
skipped, and the whole batch is one undo step. Single-frame autolabels
join the same FIFO queue as box segmentation; the all-frames variant is a
background batch with progress in the status line and checkpoint saves
every 10 frames.

SAM3 track propagation
----------------------
With one or more boxes selected, "Propagate →" tracks every selected
object through the following frames with SAM3's video memory bank
(SAM3VideoPredictor): the selected boxes seed ONE video session per
side (in stereo the left and right selections run as two back-to-back
jobs through the SAM3 queue), and each object keeps its seed's track id
and category. Object identity comes from the memory bank — there is no
per-frame re-detection or IoU chaining. A track that yields an empty
mask is reported as lost from that frame on (the memory bank may still
recover it on later frames); frames that already have a box with that
track id are skipped. Propagated boxes are ordinary editable boxes with
masks plus ``propagated: true`` / ``confidence`` provenance fields; each
side's run is one Ctrl+Z step, and edits you make while it runs stay
separate undo entries. Propagation shares the SAM3 queue and the Cancel
button.

Categories start empty; they are created from the side panel's Add
field, come back with a resumed labels file, or are seeded from a COCO
json via ``--json``.

Timestamps are stored in the COCO output as the ``timestamp_ns`` field on
each image, and a side-table ``timestamp_ns → image_id`` is written into the
JSON so the result can be joined back to external databases by timestamp.
The JSON also carries ``annotated_image_idxs``: the sorted 0-based frame
indices (matching each image's ``frame_idx``) that count as annotated —
images with at least one annotation, plus frames explicitly marked with
the "Mark as annotated" button in the side panel.

USAGE
-----
    python -m gui.label_review.main \
        --images <folder | image file> [more paths ...] \
        [--images-right <right folder> [more paths ...]] \
        --output_json output/my_labels/coco.json \
        [--json <seed coco>] \
        [--output-yolo-dir ...] [--data-yaml ...]

    # or with no source at all (idle; pick one later from the File menu):
    python -m gui.label_review.main

Key bindings (in the 2D canvas, when it has focus — click it once):
    D / del / backspace : delete selected box(es)
    shift+click : toggle a box in/out of the multi-selection (delete
               applies to all selected; the last clicked box stays the
               primary for move/resize/recat/track)
    A        : toggle draw mode (click-drag a new box)
    N / →    : next frame
    B / ←    : previous frame
    X        : discard ALL boxes on this frame and advance
    S        : save final result and quit
    Q        : quit (progress saved in tmp file)
    ESC      : clear the current selection; quit when nothing is selected
    Ctrl+A   : select all boxes on the current frame
    0..9     : when drawing, assign category id to the pending rectangle
               (use the buttons in the side panel for ids > 9). After the
               first box, new draws auto-reuse the previous box's category
               (preselect another cat in the side panel to change it)
    M        : toggle mask overlay visibility
    R        : re-segment the selected bbox with SAM3 (replaces its mask)
    + / =    : zoom in   |  -  : zoom out  |  0 : reset zoom
    arrows   : pan (when zoomed)
    Ctrl+Z   : undo last edit (add/delete/move/resize/mask/discard-all)
    Ctrl+Shift+Z / Ctrl+Y : redo
    U        : jump to the next unlabeled frame (no boxes, not yet reviewed)
    C        : focus the "Cat of selected" field — type any category id
               (e.g. 13) and press Enter to reassign the selected box
    T        : focus the "Track of selected" field — type a track id and
               press Enter to set it (clear the field to unset). Track ids
                are shown by default (T<id> canvas labels, box-list suffix,
                "Track of selected" row); hide them by unchecking "Show track
                ids" in Config → Advanced settings → Interpolation
                (tracking.show_ids). Track ids
               are auto-assigned by inheritance: the k-th box drawn on a
               frame gets the k-th track id from the nearest earlier
               annotated frame (box 1 on frame 2 → track 1, etc.), falling
               back to a global counter (1, 2, 3, ...) when there is no
               earlier frame or it has fewer boxes
    K        : toggle keyframe on the current frame (anchors for
               interpolation; keyframes with boxes take priority over
               other labeled frames)
    I        : interpolate — fill the gap between the nearest labeled
               frames with optical-flow boxes (frames that already have
               boxes are skipped; Ctrl+Z undoes the whole fill)

Reviewed frames (marked on N forward-nav and X discard) are tracked in the
.progress sidecar and shown in the status bar; X, S and quitting with
unsaved changes ask for confirmation first.

Config file (--config)
----------------------
A JSON file can adjust interpolation, SAM3, and UI settings. Values in the
config override the corresponding CLI flags. Example:
``scripts/config/label_review.example.json``.

    {
      "interpolation": {
        "flow_method": "dis",         // dis | klt | farneback
        "camera_model": "none",       // none | global
        "match_max_dist_frac": 0.2,   // anchor box pairing distance,
                                      // fraction of min(frame w, h)
        "confirm_mismatch": true      // false = skip the track-id mismatch
                                      // dialog (mismatches are logged)
      },
      "sam3": {
        "device": "auto",             // auto (default): cuda when torch
                                      // reports CUDA available, else cpu.
                                      // Or force "cuda" / "cpu".
        "conf": 0.25,
        "auto_segment": false,
        "min_polygon_area": 100,      // drop saved mask contours smaller
                                      // than this (px²); filters the speck
                                      // polygons SAM3 occasionally spawns.
                                      // The largest contour is always kept.
                                      // 0 keeps every contour.
        "autolabel_nms_iou": 0.7      // class-aware NMS on autolabel
                                      // detections (dedup overlapping masks
                                      // of the same category). 1.0 disables.
      },
      "ui": {
        "advanced": false,            // false (default): the interpolation
                                      // and tracking settings are hidden in
                                      // the Config dialog, and the
                                      // keyframe/interpolate button groups
                                      // stay hidden in the side panel.
                                      // true: those sections appear and
                                      // "hide" fully controls them.
        "hide": ["sam3_all_frames"],  // button groups to hide; known groups:
                                      // keyframe, interpolate, jump,
                                      // sam3_run, sam3_all_frames, autolabel,
                                      // masks, play. Without "advanced",
                                      // keyframe and interpolate are always
                                      // hidden. When no config is given,
                                      // "autolabel" is hidden by default;
                                      // a config "hide" list is authoritative
                                      // (omit "autolabel" to show it).
        "mask_opacity": 47            // initial mask overlay opacity (0-100)
      },
      "display": {
        "max_image_dim": 0            // max displayed image dimension (px);
                                      // larger frames are downscaled for
                                      // display only (box/mask coordinates
                                      // stay in original pixels).
                                      // 0 (default) = original resolution.
      },
      "tracking": {
        "sticky_ids": false,          // false (default): every new box gets
                                      // a fresh auto-increment track id;
                                      // true: the k-th box on a frame
                                      // inherits the k-th track id from the
                                      // nearest earlier annotated frame
                                      // (what interpolation pairing expects)
        "show_ids": true              // true (default): track ids are
                                       // displayed ("T<id>" canvas labels,
                                       // box-list suffix, "Track of selected"
                                       // row); false hides them. Editable
                                       // under the dialog's advanced
                                       // Interpolation section.
      }
    }

The same settings can be edited at runtime from the ⚙ Config button in the
menu bar (or File → Config settings…, Ctrl+G): a dialog with checkboxes for
the UI groups plus SAM3 / opacity fields (the interpolation and tracking
sections appear once "Advanced settings" is checked), with buttons to
Load from file…, Apply, Save… (writes the JSON above), and Close.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    # Direct execution (``python gui/label_review/main.py``): put the repo
    # root on sys.path and set __package__ so the relative imports below
    # resolve through the gui.label_review package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "gui.label_review"

from .qt_compat import QApplication  # noqa: E402
from .state.index import EmptyIndex, ImageFolderIndex, StereoIndex  # noqa: E402
from .state.coco_state import CocoState  # noqa: E402
from .config import _resolve_device, _seed_categories  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 2D bbox reviewer for image files/folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", nargs="+", metavar="PATH",
                        help="Image files and/or folders to review (folders "
                             "are scanned for jpg/jpeg/png/bmp/webp/tif, "
                             "sorted by name). Omit to start idle and pick "
                             "a source from the File menu.")
    parser.add_argument("--images-right", nargs="+", metavar="PATH",
                        help="Right-side image folder(s) for stereo review. "
                             "Frames pair positionally with --images "
                             "(matching filenames expected) and both sides "
                             "show side by side, navigated together; "
                             "annotations are saved per side. Requires "
                             "--images (the left side).")
    parser.add_argument("--output_json",
                        help="Output COCO JSON path (progress file is "
                             "auto-saved). Default: <source>/labels_coco.json "
                             "for --images, ./untitled_labels_coco.json in "
                             "idle mode.")
    parser.add_argument("--json", help="Seed COCO JSON for categories / existing labels.")
    parser.add_argument("--db", help="Deprecated/unused: categories are no longer read "
                        "from a SQLite DB. Kept so old commands still parse.")
    parser.add_argument("--output-yolo-dir", help="Also export YOLO dataset on exit.")
    parser.add_argument("--data-yaml", help="Reference data.yaml for YOLO class order.")
    # SAM3 options
    parser.add_argument("--sam3-model", default=None,
                        help="Path to SAM3 weights (default: "
                             "core/sam3/models/sam3-model/sam3.pt).")
    parser.add_argument("--sam3-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for SAM3 inference (default: auto — "
                             "cuda when torch reports CUDA available, else "
                             "cpu).")
    parser.add_argument("--sam3-conf", type=float, default=0.25,
                        help="SAM3 confidence threshold (default: 0.25).")
    parser.add_argument("--auto-segment", action="store_true",
                        help="Automatically run SAM3 after each new bbox is drawn.")
    # Interpolation options (I key / Interpolate button)
    parser.add_argument("--interp-flow-method",
                        choices=["dis", "klt", "farneback"], default="dis",
                        help="Optical flow method for gap interpolation "
                             "(default: dis — dense inverse search, best for "
                             "handheld camera shake).")
    parser.add_argument("--interp-camera-model",
                        choices=["none", "global"], default="none",
                        help="Camera motion model for gap interpolation "
                             "(default: none; 'global' fits a per-frame "
                             "RANSAC similarity transform).")
    parser.add_argument("--config",
                        help="JSON config file (interpolation/sam3/ui "
                             "settings; values override the corresponding "
                             "CLI flags). See scripts/config/"
                             "label_review.example.json.")
    args = parser.parse_args()

    # No source given → start idle; the user picks a source from the
    # File menu (Open image file(s) / Open folder).

    # Optional JSON config; its values override the corresponding CLI flags.
    cfg: Dict[str, Any] = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"⚙️  Loaded config: {args.config}")
        except Exception as e:
            print(f"❌ Could not read config {args.config}: {e}")
            sys.exit(1)
    interp_cfg = cfg.get("interpolation", {})
    sam3_cfg = cfg.get("sam3", {})
    ui_cfg = cfg.get("ui", {})
    tracking_cfg = cfg.get("tracking", {})
    display_cfg = cfg.get("display", {})

    flow_method = interp_cfg.get("flow_method", args.interp_flow_method)
    if flow_method not in ("dis", "klt", "farneback"):
        print(f"⚠️ config interpolation.flow_method {flow_method!r} "
              "invalid; using 'dis'")
        flow_method = "dis"
    camera_model = interp_cfg.get("camera_model", args.interp_camera_model)
    if camera_model not in ("none", "global"):
        print(f"⚠️ config interpolation.camera_model {camera_model!r} "
              "invalid; using 'none'")
        camera_model = "none"
    sam3_device = sam3_cfg.get("device", args.sam3_device)
    if sam3_device not in ("auto", "cuda", "cpu"):
        print(f"⚠️ config sam3.device {sam3_device!r} invalid; "
              f"using '{args.sam3_device}'")
        sam3_device = args.sam3_device
    sam3_device = _resolve_device(sam3_device)
    print(f"🖥️  SAM3 device: {sam3_device}")
    propagate_method = sam3_cfg.get("propagate_method", "memory")
    if propagate_method not in ("memory", "chain"):
        print(f"⚠️ config sam3.propagate_method {propagate_method!r} "
              "invalid; using 'memory'")
        propagate_method = "memory"

    # ---------- 1. Index the frame source ----------
    if args.images_right and not args.images:
        print("❌ --images-right requires --images (the left side).")
        sys.exit(1)
    if args.images:
        print(f"🖼️  Loading images from: {args.images}")
        if args.images_right:
            print(f"🖼️  Loading right-side images from: {args.images_right}")
            frame_index = StereoIndex(args.images, args.images_right)
            print(f"✅ Indexed {len(frame_index)} stereo pair(s) "
                  f"(left={len(frame_index.left)}, "
                  f"right={len(frame_index.right)})")
            if frame_index.pairing_warning:
                print(f"⚠️  {frame_index.pairing_warning}")
        else:
            frame_index = ImageFolderIndex(args.images)
            print(f"✅ Indexed {len(frame_index)} images")
        if frame_index.timestamps_real:
            print("🕒 Filenames look like timestamps — "
                  "slider/info will show them.")
        if not args.output_json:
            base = (args.images[0] if os.path.isdir(args.images[0])
                    else os.path.dirname(os.path.abspath(args.images[0])))
            args.output_json = os.path.join(base, "labels_coco.json")
    else:
        # Idle start — no source; pick one from the File menu.
        frame_index = EmptyIndex()
        if not args.output_json:
            args.output_json = os.path.abspath("untitled_labels_coco.json")
        print("💤 No source given — idle mode. Use File → Open image "
              "file(s) / Open folder to begin.")
    if len(frame_index) == 0 and args.images:
        print("❌ No image frames found.")
        sys.exit(1)

    # ---------- 2. Categories / COCO state ----------
    categories = _seed_categories(args.json)
    print(f"🏷️  Categories: {[c['name'] for c in categories]}")
    coco = CocoState(args.output_json, categories)
    # Default: fresh auto-increment track ids; sticky inheritance is opt-in.
    coco.sticky_track_ids = bool(tracking_cfg.get("sticky_ids", False))
    coco.min_polygon_area = max(
        0.0, float(sam3_cfg.get("min_polygon_area", 100)))
    coco.load_existing()
    coco.current_idx = coco.load_progress(len(frame_index))

    # ---------- 3. Qt app ----------
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Computer Vision Label Review Tool")

    signal.signal(signal.SIGINT, lambda *a: app.quit())

    # Resolve ReviewWindow through the package at call time (not a module
    # global) so tests can monkeypatch gui.label_review.ReviewWindow.
    ReviewWindow = sys.modules[__package__].ReviewWindow
    win = ReviewWindow(frame_index, coco,
                       sam3_model=sam3_cfg.get("model") or args.sam3_model,
                       sam3_device=sam3_device,
                       sam3_conf=float(sam3_cfg.get("conf", args.sam3_conf)),
                       propagate_method=propagate_method,
                       propagate_min_iou=float(sam3_cfg.get(
                           "propagate_min_iou", 0.3)),
                       propagate_min_seed_iou=float(sam3_cfg.get(
                           "propagate_min_seed_iou", 0.2)),
                       auto_segment=bool(sam3_cfg.get("auto_segment",
                                                      args.auto_segment)),
                        interp_flow_method=flow_method,
                        interp_camera_model=camera_model,
                        interp_match_frac=float(interp_cfg.get(
                            "match_max_dist_frac", 0.2)),
                        interp_confirm_mismatch=bool(interp_cfg.get(
                            "confirm_mismatch", True)),
                        # Interpolation + keyframe controls are hidden unless
                        # "ui": {"advanced": true} (the Config dialog exposes
                        # the same toggle); "ui": {"hide": [...]} hides
                        # additional groups on top.
                        # Autolabel buttons are hidden by default; a config
                        # "hide" list fully controls them.
                        ui_hide=ui_cfg.get("hide", ["autolabel"]),
                        mask_opacity=ui_cfg.get("mask_opacity"),
                        advanced_ui=bool(ui_cfg.get("advanced", False)),
                         show_track_ids=bool(tracking_cfg.get("show_ids",
                                                              True)),
                         display_max_dim=int(
                             display_cfg.get("max_image_dim", 0)))
    win.show()
    exit_code = app.exec()

    # ---------- 4. Optional YOLO export ----------
    if args.output_yolo_dir:
        print(f"\nExporting YOLO dataset to {args.output_yolo_dir}")
        # Reuse helpers from 08_click_review_coco.py if importable.
        try:
            from importlib import util
            here = Path(__file__).resolve().parents[2] / "scripts"
            mod_path = here / "08_click_review_coco.py"
            if mod_path.exists():
                spec = util.spec_from_file_location("click_review_coco", mod_path)
                mod = util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                final_coco = {
                    "images": coco.images,
                    "annotations": [
                        ann for ann in coco.annotations
                        if ann["id"] not in coco.removed_ids
                    ],
                    "categories": coco.categories,
                }
                # Write images to a temp dir: copy the source files directly.
                tmp_img_dir = Path(args.output_yolo_dir) / "_src_images"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                for img in coco.images:
                    frame = frame_index.frame_at(img["frame_idx"])
                    if frame.get("file_path"):
                        with open(frame["file_path"], "rb") as src, \
                                open(tmp_img_dir / img["file_name"], "wb") as dst:
                            dst.write(src.read())
                mod.export_yolo_from_coco(
                    final_coco, img_dir=tmp_img_dir,
                    output_dir=args.output_yolo_dir,
                    data_yaml=args.data_yaml,
                    copy_images=True,
                )
            else:
                print("⚠️ 08_click_review_coco.py not found; skipping YOLO export.")
        except Exception as e:
            print(f"⚠️ YOLO export failed: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
