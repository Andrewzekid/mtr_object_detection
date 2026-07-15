"""
SAM3 Vision Tool page.
Provides interface for Ultralytics SAM3 segmentation with multi-bbox exemplars,
JSON bbox loading, and configurable model path.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QGridLayout, QSplitter, QSpinBox, QDoubleSpinBox, QFileDialog,
    QCheckBox, QListWidget, QListWidgetItem, QMessageBox,
)
from PyQt6.QtCore import Qt, QPoint, QRect, QRectF, QPointF
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QPainterPath

from utils.config import config
from utils.workers import TaskWorker
import numpy as np
import cv2
import json
from typing import List, Dict, Tuple, Optional


# ----------------------------------------------------------------------------
# A canvas that supports drawing MULTIPLE bounding-box exemplars.
# Each click-drag adds a new rectangle to self.bboxes.
# ----------------------------------------------------------------------------

class ImageCanvas(QLabel):
    """Canvas that supports drawing multiple bounding boxes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(500, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        """)

        self.original_image = None
        self.display_image = None
        self.drawing = False
        self.start_point: Optional[QPoint] = None
        self.current_rect: Optional[QRect] = None
        self.scale_factor: float = 1.0

        # List of [x1, y1, x2, y2] in IMAGE coordinates.
        self.bboxes: List[List[float]] = []

        # Drawing colours rotated per-box.
        self._bbox_colors = [
            QColor(255, 0, 0),
            QColor(0, 128, 255),
            QColor(0, 200, 0),
            QColor(255, 128, 0),
            QColor(180, 0, 255),
        ]

        self.setText("Click 'Load Image' to start")

    # ----- image loading / drawing helpers -----

    def load_image(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is not None:
            self.bboxes = []  # reset exemplars when a new image is loaded
            self._display_image(self.original_image)

    def add_bbox_from_json(self, bbox: List[float]):
        """Add a bbox in image coordinates (clamped to canvas)."""
        if self.original_image is None:
            return
        h, w = self.original_image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1
        self.bboxes.append([x1, y1, x2, y2])
        self._redraw_with_all_boxes()

    def clear_all_bboxes(self):
        self.bboxes = []
        if self.original_image is not None:
            self._display_image(self.original_image)

    def get_bboxes(self) -> List[List[float]]:
        return [list(b) for b in self.bboxes]

    def _display_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        canvas_w, canvas_h = self.width(), self.height()
        self.scale_factor = min(canvas_w / w, canvas_h / h)
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        q_img = QImage(img_rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        self.display_image = QPixmap.fromImage(q_img)
        self._redraw_with_all_boxes()

    def _redraw_with_all_boxes(self):
        """Draw image + every committed bbox + the in-progress rect."""
        if self.display_image is None:
            return
        pixmap = self.display_image.copy()
        painter = QPainter(pixmap)

        for idx, bbox in enumerate(self.bboxes):
            color = self._bbox_colors[idx % len(self._bbox_colors)]
            pen = QPen(color, 2)
            painter.setPen(pen)
            x1, y1, x2, y2 = bbox
            rx1 = int(x1 * self.scale_factor)
            ry1 = int(y1 * self.scale_factor)
            rx2 = int(x2 * self.scale_factor)
            ry2 = int(y2 * self.scale_factor)
            painter.drawRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1))
            # Label
            painter.setPen(QPen(color))
            painter.drawText(rx1 + 4, ry1 + 14, f"#{idx + 1}")

        if self.drawing and self.current_rect:
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        painter.end()
        self.setPixmap(pixmap)

    # ----- mouse events -----

    def mousePressEvent(self, event):
        if self.original_image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.drawing = True
        self.start_point = QPoint(int(event.position().x()), int(event.position().y()))

    def mouseMoveEvent(self, event):
        if not self.drawing or self.start_point is None:
            return
        end_point = QPoint(int(event.position().x()), int(event.position().y()))
        self.current_rect = QRect(self.start_point, end_point).normalized()
        self._redraw_with_all_boxes()

    def mouseReleaseEvent(self, event):
        if not self.drawing or self.start_point is None:
            return
        self.drawing = False
        end_point = QPoint(int(event.position().x()), int(event.position().y()))
        self.current_rect = QRect(self.start_point, end_point).normalized()

        if self.current_rect.width() > 10 and self.current_rect.height() > 10:
            # Convert canvas pixels → image pixels, store as image-coordinate bbox.
            x1 = int(self.current_rect.x() / self.scale_factor)
            y1 = int(self.current_rect.y() / self.scale_factor)
            x2 = int((self.current_rect.x() + self.current_rect.width()) / self.scale_factor)
            y2 = int((self.current_rect.y() + self.current_rect.height()) / self.scale_factor)
            self.bboxes.append([x1, y1, x2, y2])

        self.current_rect = None
        self._redraw_with_all_boxes()

    def set_overlay(self, mask: np.ndarray, original_image: np.ndarray):
        overlay = original_image.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.5, color_mask, 0.5, 0)
        self._display_image(overlay)


# ----------------------------------------------------------------------------
# SAM3 page
# ----------------------------------------------------------------------------

class SAMPage(QWidget):
    """SAM3 Vision Tool page (Ultralytics SAM3SemanticPredictor)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_worker = None
        self.current_image = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        title = QLabel("SAM3 Vision Tool")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        layout.addWidget(title)

        desc = QLabel(
            "Draw one or more exemplar bounding boxes, optionally load boxes from a JSON file, "
            "then click 'Run Segmentation'. SAM3 finds all similar objects in the image."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 13px;")
        layout.addWidget(desc)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------- LEFT: input ----------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # image path
        image_group = QGroupBox("Image Input")
        image_layout = QVBoxLayout(image_group)
        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("Image Path:"))
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Select an image...")
        image_row.addWidget(self.image_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_image)
        image_row.addWidget(browse_btn)
        image_layout.addLayout(image_row)

        self.input_canvas = ImageCanvas()
        image_layout.addWidget(self.input_canvas)

        instructions = QLabel(
            "Tip: click-and-drag on the image to add an exemplar bounding box. "
            "Multiple boxes help SAM3 understand the visual concept."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        image_layout.addWidget(instructions)

        # bbox list + loader controls
        bbox_toolbar = QHBoxLayout()
        bbox_toolbar.setSpacing(8)
        load_json_btn = QPushButton("Load BBoxes JSON")
        load_json_btn.clicked.connect(self._load_bboxes_from_json)
        clear_btn = QPushButton("Clear BBoxes")
        clear_btn.clicked.connect(self._clear_bboxes)
        bbox_toolbar.addWidget(load_json_btn)
        bbox_toolbar.addWidget(clear_btn)
        bbox_toolbar.addStretch()
        image_layout.addLayout(bbox_toolbar)

        self.bbox_list = QListWidget()
        self.bbox_list.setMaximumHeight(90)
        self.bbox_list.setStyleSheet("font-family: monospace;")
        image_layout.addWidget(self.bbox_list)

        left_layout.addWidget(image_group)

        # model path & quantize
        model_group = QGroupBox("SAM3 Model")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(str(config.sam3_models_dir / "sam3.pt"))
        self.model_path_edit.setPlaceholderText("Path to sam3.pt")
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(model_browse_btn, 0, 2)

        model_layout.addWidget(QLabel("Quantize:"), 1, 0)
        self.quantize_combo = QComboBox()
        self.quantize_combo.addItems(["none (full precision)", "INT8", "INT16"])
        model_layout.addWidget(self.quantize_combo, 1, 1)
        left_layout.addWidget(model_group)

        # inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QGridLayout(settings_group)
        settings_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CUDA (GPU)", "CPU"])
        settings_layout.addWidget(self.device_combo, 0, 1)
        settings_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        settings_layout.addWidget(self.conf_spin, 1, 1)

        # Concepts textbox — user can type one or more concept labels
        settings_layout.addWidget(QLabel("Concepts:"), 2, 0)
        self.concepts_edit = QLineEdit()
        self.concepts_edit.setPlaceholderText("e.g., car, person, dog (comma-separated)")
        settings_layout.addWidget(self.concepts_edit, 2, 1)

        left_layout.addWidget(settings_group)

        # Run / clear
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run SAM3 Segmentation")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4; color: white; border: none;
                border-radius: 6px; padding: 12px 24px; font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.run_btn.clicked.connect(self._run_segmentation)
        btn_layout.addWidget(self.run_btn)

        clear_btn2 = QPushButton("Clear")
        clear_btn2.clicked.connect(self._clear)
        btn_layout.addWidget(clear_btn2)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        # ---------- RIGHT: output ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)

        output_group = QGroupBox("Segmentation Output")
        output_layout = QVBoxLayout(output_group)
        self.output_canvas = ImageCanvas()
        output_layout.addWidget(self.output_canvas)
        right_layout.addWidget(output_group)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(180)
        self.results_text.setPlaceholderText("Segmentation results will appear here...")
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa; border: 1px solid #ccc;
                border-radius: 5px; font-size: 12px;
            }
        """)
        results_layout.addWidget(self.results_text)
        right_layout.addWidget(results_group)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        right_layout.addWidget(self.status_label)
        right_layout.addStretch()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([550, 550])
        layout.addWidget(splitter)

    # ---------- slot helpers ----------

    def _browse_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file:
            self.image_path_edit.setText(file)
            self.input_canvas.load_image(file)
            self.current_image = cv2.imread(file)
            self.bbox_list.clear()
            self._refresh_bbox_list()

    def _browse_model(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select SAM3 Model", str(Path(str(config.sam3_models_dir))),
            "PyTorch / ONNX Weights (*.pt *.pth *.onnx);;All Files (*)"
        )
        if file:
            self.model_path_edit.setText(file)

    def _load_bboxes_from_json(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        file, _ = QFileDialog.getOpenFileName(
            self, "Load BBoxes JSON",
            "", "JSON Files (*.json);;All Files (*)"
        )
        if not file:
            return
        try:
            with open(file, "r") as f:
                data = json.load(f)
            bboxes = data.get("bboxes") if isinstance(data, dict) else data
            if not isinstance(bboxes, list) or not all(isinstance(b, list) and len(b) == 4 for b in bboxes):
                raise ValueError("Expected a list of 4-element lists [x1, y1, x2, y2] under 'bboxes'.")

            self.input_canvas.bboxes = []
            for b in bboxes:
                self.input_canvas.add_bbox_from_json(b)

            self.status_label.setText(f"Loaded {len(self.input_canvas.bboxes)} bbox(es) from JSON.")
            self._refresh_bbox_list()
        except Exception as e:
            QMessageBox.critical(self, "Bad JSON", f"Could not load bbox JSON:\n{e}")

    def _clear_bboxes(self):
        self.input_canvas.clear_all_bboxes()
        self.bbox_list.clear()
        self._refresh_bbox_list()

    def _refresh_bbox_list(self):
        self.bbox_list.clear()
        for i, b in enumerate(self.input_canvas.bboxes):
            self.bbox_list.addItem(QListWidgetItem(f"#{i + 1}: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]"))

    def _get_device(self) -> str:
        return "cuda" if self.device_combo.currentIndex() == 0 else "cpu"

    def _get_quantize(self):
        idx = self.quantize_combo.currentIndex()
        return None if idx == 0 else (8 if idx == 1 else 16)

    def _run_segmentation(self):
        if self.current_image is None:
            self.status_label.setText("Please load an image first")
            return
        image_path = self.image_path_edit.text()
        if not image_path:
            self.status_label.setText("Please select an image")
            return

        bboxes = self.input_canvas.get_bboxes()
        concepts_text = self.concepts_edit.text().strip()
        concepts = [c.strip() for c in concepts_text.split(",") if c.strip()] if concepts_text else None

        if not bboxes and not concepts:
            QMessageBox.information(
                self, "No Input",
                "Draw at least one exemplar bounding box on the image, "
                "load them via 'Load BBoxes JSON', or type concept labels."
            )
            return

        model_path = self.model_path_edit.text() or None
        self._refresh_bbox_list()

        self.run_btn.setEnabled(False)
        desc_parts = []
        if bboxes:
            desc_parts.append(f"{len(bboxes)} bbox(es)")
        if concepts:
            desc_parts.append(f"concepts: {', '.join(concepts)}")
        self.status_label.setText(f"Running SAM3 with {' and '.join(desc_parts)}...")

        def run_task(progress_callback=None, status_callback=None,
                    log_callback=None, is_cancelled=None):
            from core.models_inference import run_sam3
            return run_sam3(
                image_path=image_path,
                bboxes=bboxes,
                concepts=concepts,
                model_path=model_path,
                device=self._get_device(),
                conf=self.conf_spin.value(),
                quantize=self._get_quantize(),
                save=False,
                progress_callback=progress_callback,
                status_callback=status_callback,
                log_callback=log_callback,
                is_cancelled=is_cancelled,
            )

        self.current_worker = TaskWorker(run_task)
        self.current_worker.status.connect(self._on_status)
        self.current_worker.finished.connect(self._on_finished)
        self.current_worker.error.connect(self._on_error)
        self.current_worker.log.connect(self._on_results_log)
        self.current_worker.start()

    def _on_status(self, message):
        self.status_label.setText(message)

    def _on_results_log(self, message):
        # append into the results text for extra insight
        self.results_text.append(message)

    def _on_finished(self, result):
        self.run_btn.setEnabled(True)
        if not isinstance(result, dict):
            return
        if result.get("success"):
            self.status_label.setText("SAM3 segmentation completed")
            mask_overlay = result.get("mask_overlay")
            if mask_overlay is not None and self.current_image is not None:
                self.output_canvas.load_image(self.image_path_edit.text())
                self.output_canvas.set_overlay(mask_overlay, self.current_image)

            detections = result.get("detections", [])
            regions = result.get("segmented_regions", [])
            concepts_used = result.get("concepts", [])
            
            info = (
                f"SAM3 found {len(detections)} detection(s)\n"
                f"Concepts queried: {', '.join(concepts_used) if concepts_used else 'N/A'}\n\n"
            )
            
            # Group detections by label
            detections_by_label = {}
            for det in detections:
                label = det.get("label", "unknown")
                if label not in detections_by_label:
                    detections_by_label[label] = []
                detections_by_label[label].append(det)
            
            # Display summary by label
            for label, dets in detections_by_label.items():
                info += f"  {label}: {len(dets)} instance(s)\n"
            
            info += "\n"
            
            # Display individual detections
            for i, det in enumerate(detections):
                bbox = det.get("bbox", [])
                conf = det.get("confidence", 0)
                info += (
                    f"Detection {i + 1} [{det.get('label', 'unknown')}]:\n"
                    f"  BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n"
                    f"  Confidence: {conf:.2f}\n"
                    f"  Area: {det.get('area', 0):.0f} pixels\n\n"
                )
            
            if not detections:
                info += "No detections found. Try different concepts or lower confidence.\n"
            
            self.results_text.setText(info)
        else:
            self.status_label.setText(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
            self.results_text.setText(
                f"Error: {result.get('error', 'Unknown error')}"
            )

    def _on_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Error")
        self.results_text.setText(f"Error: {error_msg}")

    def _clear(self):
        self.input_canvas.bboxes = []
        self.bbox_list.clear()
        if self.current_image is not None:
            self.input_canvas._display_image(self.current_image)
        self.results_text.clear()
        self.status_label.setText("Ready")


# Helper for: import Path inside method
from pathlib import Path
