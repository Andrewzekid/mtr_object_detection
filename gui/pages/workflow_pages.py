"""
Workflow pages for the 6-step training pipeline.
Pages 1-6: Labeling, Augment, Split, Train, Eval, Viz
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QTextEdit, QProgressBar, QCheckBox, QGroupBox,
    QScrollArea, QGridLayout, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal

from utils.config import config
from utils.workers import TaskWorker


class BaseWorkflowPage(QWidget):
    """Base class for workflow pages with common elements."""
    
    task_completed = pyqtSignal(str, bool)  # step_name, success
    
    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.current_worker = None
        
        self._setup_base_ui()
    
    def _setup_base_ui(self):
        """Setup base UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #333;")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setStyleSheet("color: #666; font-size: 13px;")
        layout.addWidget(desc_label)
        
        # Content area (to be overridden)
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        
        # Status/Progress area
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                border-radius: 5px;
            }
        """)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)
        
        layout.addStretch()
    
    def log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
        self.log_text.setVisible(True)
    
    def set_progress(self, value: int):
        """Set progress bar value."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
    
    def set_status(self, message: str):
        """Set status message."""
        self.status_label.setText(message)
    
    def start_task(self, task_func, *args, **kwargs):
        """Start a background task."""
        self.current_worker = TaskWorker(task_func, *args, **kwargs)
        self.current_worker.progress.connect(self.set_progress)
        self.current_worker.status.connect(self.set_status)
        self.current_worker.log.connect(self.log)
        self.current_worker.finished.connect(self._on_task_finished)
        self.current_worker.error.connect(self._on_task_error)
        self.current_worker.start()
    
    def _on_task_finished(self, result):
        """Handle task completion."""
        self.progress_bar.setVisible(False)
        if isinstance(result, dict):
            success = result.get("success", False)
            self.log(f"Task completed: {'Success' if success else 'Failed'}")
            self.task_completed.emit(self.title, success)
    
    def _on_task_error(self, error_msg):
        """Handle task error."""
        self.progress_bar.setVisible(False)
        self.log(f"ERROR: {error_msg}")
        self.set_status("Task failed")


# =============================================================================
# Page 1: Label Data
# =============================================================================

class LabelDataPage(BaseWorkflowPage):
    """Page 1: Label Data - Launch external labeling tool or embedded canvas."""
    
    def __init__(self, parent=None):
        super().__init__(
            "🏷️ Step 1: Label Data",
            "Annotate images with bounding boxes for object detection training.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Dataset path selection
        path_group = QGroupBox("Dataset Configuration")
        path_layout = QVBoxLayout(path_group)
        
        # Current dataset path
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Dataset Path:"))
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Select dataset directory...")
        path_row.addWidget(self.dataset_path_edit)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_dataset)
        path_row.addWidget(browse_btn)
        
        path_layout.addLayout(path_row)
        
        # Dataset info
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("color: #666;")
        path_layout.addWidget(self.dataset_info_label)
        
        self.content_layout.addWidget(path_group)
        
        # Labeling options
        options_group = QGroupBox("Labeling Options")
        options_layout = QVBoxLayout(options_group)
        
        # Tool selection
        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("Labeling Tool:"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["External (labelImg)", "External (CVAT)", "Embedded Canvas (Coming Soon)"])
        tool_row.addWidget(self.tool_combo)
        tool_row.addStretch()
        options_layout.addLayout(tool_row)
        
        # Class names
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class Names (comma-separated):"))
        self.class_names_edit = QLineEdit()
        self.class_names_edit.setPlaceholderText("e.g., car, person, dog")
        class_row.addWidget(self.class_names_edit)
        options_layout.addLayout(class_row)
        
        self.content_layout.addWidget(options_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        launch_btn = QPushButton("🚀 Launch Labeling Tool")
        launch_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        launch_btn.clicked.connect(self._launch_labeling_tool)
        btn_layout.addWidget(launch_btn)
        
        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        btn_layout.addWidget(mark_done_btn)
        
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)
    
    def _browse_dataset(self):
        """Browse for dataset directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.dataset_path_edit.setText(directory)
            config.set_dataset_path(directory)
            self._update_dataset_info()
    
    def _update_dataset_info(self):
        """Update dataset info display."""
        if config.current_dataset_path and config.current_dataset_path.exists():
            images = config.get_image_files()
            dtype = getattr(config, "dataset_type", "unknown")
            type_label = {
                "unlabelled": "unlabelled folder",
                "yolo_flat": "YOLO dataset (images/ + labels/)",
                "yolo_nested": "YOLO dataset (nested train/val/test)",
                "yolo_split": "YOLO dataset (train/test/val split)",
            }.get(dtype, dtype)
            self.dataset_info_label.setText(
                f"Found {len(images)} image(s) — {type_label}"
            )
        else:
            self.dataset_info_label.setText("No dataset loaded")
    
    def _launch_labeling_tool(self):
        """Launch the selected labeling tool."""
        import subprocess
        import sys
        
        
        if tool == 0:  # labelImg
            try:
                subprocess.Popen(["labelImg"])
                self.log("Launched labelImg")
            except Exception as e:
                self.log(f"Could not launch labelImg: {e}")
                self.log("Install with: pip install labelImg")
        elif tool == 1:  # CVAT
            self.log("CVAT requires Docker. Visit: https://github.com/opencv/cvat")
        else:
            self.log("Embedded canvas coming soon!")
    
    def _mark_complete(self):
        """Mark labeling step as complete."""
        config.update_pipeline_state("labeling_complete", True)
        self.set_status("Labeling marked as complete")
        self.task_completed.emit("labeling_complete", True)


# =============================================================================
# Page 2: Data Augmentation
# =============================================================================

class AugmentationPage(BaseWorkflowPage):
    """Page 2: Data Augmentation & Statistics."""
    
    def __init__(self, parent=None):
        super().__init__(
            "📊 Step 2: Data Augmentation",
            "Apply transformations to expand your dataset and view statistics.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Augmentation settings
        aug_group = QGroupBox("Augmentation Settings")
        aug_layout = QGridLayout(aug_group)
        
        # Augmentation types
        self.flip_h_check = QCheckBox("Horizontal Flip")
        self.flip_h_check.setChecked(True)
        aug_layout.addWidget(self.flip_h_check, 0, 0)
        
        self.flip_v_check = QCheckBox("Vertical Flip")
        aug_layout.addWidget(self.flip_v_check, 0, 1)
        
        self.rotate_check = QCheckBox("Rotation")
        self.rotate_check.setChecked(True)
        aug_layout.addWidget(self.rotate_check, 1, 0)
        
        self.brightness_check = QCheckBox("Brightness")
        self.brightness_check.setChecked(True)
        aug_layout.addWidget(self.brightness_check, 1, 1)
        
        self.contrast_check = QCheckBox("Contrast")
        aug_layout.addWidget(self.contrast_check, 2, 0)

        self.mosaic_check = QCheckBox("Mosaic")
        aug_layout.addWidget(self.mosaic_check, 2, 1)

        self.hue_check = QCheckBox("Hue Shift")
        aug_layout.addWidget(self.hue_check, 3, 0)

        self.blur_check = QCheckBox("Blur")
        aug_layout.addWidget(self.blur_check, 3, 1)

        # Multiplier
        aug_layout.addWidget(QLabel("Augmentations per image:"), 4, 0)
        self.multiplier_spin = QSpinBox()
        self.multiplier_spin.setRange(1, 10)
        self.multiplier_spin.setValue(3)
        aug_layout.addWidget(self.multiplier_spin, 4, 1)
        
        # Rotation range
        aug_layout.addWidget(QLabel("Rotation range (±degrees):"), 5, 0)
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(1, 90)
        self.rotation_spin.setValue(15)
        aug_layout.addWidget(self.rotation_spin, 5, 1)

        # Hue shift range
        aug_layout.addWidget(QLabel("Hue shift range (±degrees):"), 6, 0)
        self.hue_spin = QSpinBox()
        self.hue_spin.setRange(1, 90)
        self.hue_spin.setValue(18)
        aug_layout.addWidget(self.hue_spin, 6, 1)

        # Blur kernel size
        aug_layout.addWidget(QLabel("Blur kernel size (odd):"), 7, 0)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(3, 15)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(5)
        aug_layout.addWidget(self.blur_spin, 7, 1)
        
        self.content_layout.addWidget(aug_group)
        
        # Statistics button
        stats_btn = QPushButton("📊 Get Dataset Statistics")
        stats_btn.clicked.connect(self._get_statistics)
        self.content_layout.addWidget(stats_btn)
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setVisible(False)
        self.content_layout.addWidget(self.stats_text)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        run_btn = QPushButton("🚀 Run Augmentation")
        run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        run_btn.clicked.connect(self._run_augmentation)
        btn_layout.addWidget(run_btn)
        
        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        btn_layout.addWidget(mark_done_btn)
        
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)
    
    def _get_augmentation_types(self):
        """Get selected augmentation types."""
        types = []
        if self.flip_h_check.isChecked():
            types.append("flip_horizontal")
        if self.flip_v_check.isChecked():
            types.append("flip_vertical")
        if self.rotate_check.isChecked():
            types.append("rotate")
        if self.brightness_check.isChecked():
            types.append("brightness")
        if self.contrast_check.isChecked():
            types.append("contrast")
        if self.mosaic_check.isChecked():
            types.append("mosaic")
        if self.hue_check.isChecked():
            types.append("hue")
        if self.blur_check.isChecked():
            types.append("blur")
        return types
    
    def _get_statistics(self):
        """Get dataset statistics."""
        if not config.current_dataset_path:
            self.log("No dataset path configured")
            return

        from core.data_processor import DataProcessor

        self.log("Calculating statistics...")

        def run_stats(progress_callback=None, status_callback=None, log_callback=None, is_cancelled=None):
            processor = DataProcessor({"input_dir": str(config.current_dataset_path)})
            return processor.get_statistics(None, progress_callback, status_callback, log_callback, is_cancelled)

        self.start_task(run_stats)

    def _on_task_finished(self, result):
        """Handle task completion with enhanced statistics display."""
        self.progress_bar.setVisible(False)
        if not isinstance(result, dict):
            return

        success = result.get("success", False)
        if success and result.get("total_images") is not None and "class_distribution" in result:
            self._display_statistics(result)
            self.log("Statistics calculated successfully")
            return

        self.log(f"Task completed: {'Success' if success else 'Failed'}")
        self.task_completed.emit(self.title, success)

    def _display_statistics(self, result):
        """Display statistics in enhanced format."""
        stats_text = "=== Dataset Statistics ===\n\n"
        stats_text += f"Total Images: {result.get('total_images', 0)}\n"
        stats_text += f"Total Annotations: {result.get('total_annotations', 0)}\n"
        stats_text += f"Images With Labels: {result.get('images_with_labels', 0)}\n"
        stats_text += f"Images Without Labels: {result.get('images_without_labels', 0)}\n"

        if result.get('avg_width'):
            stats_text += f"Average Dimensions: {result['avg_width']:.0f} x {result['avg_height']:.0f}\n"

        stats_text += "\n=== Class Distribution ===\n"
        class_dist = result.get('class_distribution', {})
        if class_dist:
            max_count = max(class_dist.values()) if class_dist else 1
            for cls_id, count in sorted(class_dist.items()):
                bar_len = int((count / max_count) * 30)
                bar = '#' * bar_len
                stats_text += f"  Class {cls_id}: {count:5d} {bar}\n"
        else:
            stats_text += "  No annotations found\n"

        stats_text += "\n=== Train/Test/Val Split Counts ===\n"
        for split_name in ['train', 'test', 'val']:
            split_dir = config.split_dir / split_name / 'images'
            if split_dir.exists():
                count = len([f for f in split_dir.iterdir()
                              if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}])
                stats_text += f"  {split_name.capitalize():10s}: {count} images\n"
            else:
                stats_text += f"  {split_name.capitalize():10s}: Not yet split\n"

        self.stats_text.setVisible(True)
        self.stats_text.setText(stats_text)
    
    def _run_augmentation(self):
        """Run data augmentation."""
        if not config.current_dataset_path:
            self.log("No dataset path configured")
            return
        
        from core.data_processor import augment_dataset
        
        aug_config = {
            "input_dir": config.current_images_path,
            "output_dir": config.augmented_dir,
            "augmentation_types": self._get_augmentation_types(),
            "multiplier": self.multiplier_spin.value(),
            "rotation_range": (-self.rotation_spin.value(), self.rotation_spin.value()),
            "brightness_range": (0.7, 1.3),
            "hue_range": (-self.hue_spin.value(), self.hue_spin.value()),
            "blur_kernel": self.blur_spin.value(),
        }
        
        self.log(f"Starting augmentation with {self._get_augmentation_types()}")
        self.start_task(augment_dataset, aug_config)
    
    def _mark_complete(self):
        """Mark augmentation step as complete."""
        config.update_pipeline_state("augmentation_complete", True)
        self.set_status("Augmentation marked as complete")
        self.task_completed.emit("augmentation_complete", True)


# =============================================================================
# Page 3: Train/Test Split
# =============================================================================

class SplitPage(BaseWorkflowPage):
    """Page 3: Train/Test/Val Split."""
    
    def __init__(self, parent=None):
        super().__init__(
            "📁 Step 3: Train/Test/Val Split",
            "Divide your dataset into training, validation, and test sets.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Split ratios
        ratio_group = QGroupBox("Split Ratios")
        ratio_layout = QGridLayout(ratio_group)
        
        ratio_layout.addWidget(QLabel("Train:"), 0, 0)
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.1, 0.9)
        self.train_spin.setValue(0.7)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.valueChanged.connect(self._update_ratios)
        ratio_layout.addWidget(self.train_spin, 0, 1)
        
        ratio_layout.addWidget(QLabel("Test:"), 1, 0)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.05, 0.5)
        self.test_spin.setValue(0.15)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.valueChanged.connect(self._update_ratios)
        ratio_layout.addWidget(self.test_spin, 1, 1)
        
        ratio_layout.addWidget(QLabel("Validation:"), 2, 0)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.05, 0.5)
        self.val_spin.setValue(0.15)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.valueChanged.connect(self._update_ratios)
        ratio_layout.addWidget(self.val_spin, 2, 1)
        
        self.sum_label = QLabel("Sum: 1.00")
        self.sum_label.setStyleSheet("color: #28a745; font-weight: bold;")
        ratio_layout.addWidget(self.sum_label, 3, 0, 1, 2)
        
        self.content_layout.addWidget(ratio_group)
        
        # Random seed
        seed_group = QGroupBox("Random Seed")
        seed_layout = QHBoxLayout(seed_group)
        seed_layout.addWidget(QLabel("Seed (optional):"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        seed_layout.addWidget(self.seed_spin)
        self.use_seed_check = QCheckBox("Use fixed seed")
        self.use_seed_check.setChecked(True)
        seed_layout.addWidget(self.use_seed_check)
        seed_layout.addStretch()
        self.content_layout.addWidget(seed_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        run_btn = QPushButton("🚀 Run Split")
        run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        run_btn.clicked.connect(self._run_split)
        btn_layout.addWidget(run_btn)
        
        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        btn_layout.addWidget(mark_done_btn)
        
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)
    
    def _update_ratios(self):
        """Update ratio sum display."""
        total = self.train_spin.value() + self.test_spin.value() + self.val_spin.value()
        self.sum_label.setText(f"Sum: {total:.2f}")
        if abs(total - 1.0) < 0.01:
            self.sum_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.sum_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def _run_split(self):
        """Run dataset split."""
        if not config.current_dataset_path:
            self.log("No dataset path configured")
            return
        
        from core.dataset_creator import split_dataset
        
        ratios = [self.train_spin.value(), self.test_spin.value(), self.val_spin.value()]
        
        if abs(sum(ratios) - 1.0) > 0.01:
            self.log("Error: Ratios must sum to 1.0")
            return
        
        seed = self.seed_spin.value() if self.use_seed_check.isChecked() else None
        
        self.log(f"Splitting dataset with ratios: train={ratios[0]}, test={ratios[1]}, val={ratios[2]}")
        self.start_task(split_dataset, config.current_images_path, config.split_dir, ratios, seed)
    
    def _mark_complete(self):
        """Mark split step as complete."""
        config.update_pipeline_state("split_complete", True)
        self.set_status("Split marked as complete")
        self.task_completed.emit("split_complete", True)


# =============================================================================
# Page 4: Train Model
# =============================================================================

class TrainModelPage(BaseWorkflowPage):
    """Page 4: Train Model - YOLO training with progress."""
    
    def __init__(self, parent=None):
        super().__init__(
            "🚀 Step 4: Train Model",
            "Train a YOLO model on your prepared dataset.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Scrollable area for long training-settings form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QWidget()
        scroll_layout = QVBoxLayout(scroll_body)
        scroll_layout.setContentsMargins(0, 0, 10, 0)
        scroll_layout.setSpacing(15)
        scroll.setWidget(scroll_body)

        # ---- Model & Basic Training ----
        basic_group = QGroupBox("Model & Basic Training")
        basic_layout = QGridLayout(basic_group)

        basic_layout.addWidget(QLabel("Model Family:"), 0, 0)
        self.model_combo = QComboBox()
        # Built-in pretrained model family
        self.model_combo.addItems([
            "yolov8n (Nano)",
            "yolov8s (Small)",
            "yolov8m (Medium)",
            "yolov8l (Large)",
            "yolov8x (XLarge)",
            "yolo11n (YOLO11 Nano)",
            "yolo11s (YOLO11 Small)",
            "yolo11m (YOLO11 Medium)",
            "yolo11l (YOLO11 Large)",
            "yolo11x (YOLO11 XLarge)",
            "yolo12n (YOLO12 Nano)",
            "yolo12s (YOLO12 Small)",
            "yolo12m (YOLO12 Medium)",
            "yolo12l (YOLO12 Large)",
            "yolo12x (YOLO12 XLarge)",
            "yolo26n (YOLO26 Nano)",
            "yolo26s (YOLO26 Small)",
            "yolo26m (YOLO26 Medium)",
            "yolo26l (YOLO26 Large)",
            "yolo26x (YOLO26 XLarge)",
            "Custom Checkpoint (.pt) ...",
        ])
        self.model_combo.currentIndexChanged.connect(self._on_model_selection_changed)
        basic_layout.addWidget(self.model_combo, 0, 1)

        # Custom-checkpoint path field (only visible when "Custom Checkpoint" is chosen)
        self.checkpoint_label = QLabel("Checkpoint Path:")
        basic_layout.addWidget(self.checkpoint_label, 1, 0)
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("Path to your .pt checkpoint")
        basic_layout.addWidget(self.checkpoint_path_edit, 1, 1)
        self.checkpoint_browse_btn = QPushButton("Browse")
        self.checkpoint_browse_btn.clicked.connect(self._browse_checkpoint)
        basic_layout.addWidget(self.checkpoint_browse_btn, 1, 2)

        basic_layout.addWidget(QLabel("Device:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU (0)", "CPU"])
        basic_layout.addWidget(self.device_combo, 2, 1)

        basic_layout.addWidget(QLabel("Epochs:"), 3, 0)
        self.epochs_spin = QSpinBox()
        # Epoch count is effectively unlimited: QSpinBox's hard ceiling is 2^31-1,
        # but we use a very large practical cap (10 million) so users can run
        # any long training schedule they need.
        self.epochs_spin.setRange(1, 10_000_000)
        self.epochs_spin.setValue(100)
        basic_layout.addWidget(self.epochs_spin, 3, 1)

        basic_layout.addWidget(QLabel("Batch Size:"), 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        basic_layout.addWidget(self.batch_spin, 4, 1)

        basic_layout.addWidget(QLabel("Image Size:"), 5, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        basic_layout.addWidget(self.imgsz_spin, 5, 1)

        basic_layout.addWidget(QLabel("Task:"), 6, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Detection", "Segmentation"])
        basic_layout.addWidget(self.task_combo, 6, 1)

        scroll_layout.addWidget(basic_group)

        # Custom-checkpoint widgets start hidden until "Custom Checkpoint" is selected
        self._on_model_selection_changed()

        # ---- Optimizer & Learning Rate ----
        opt_group = QGroupBox("Optimizer & Learning Rate")
        opt_layout = QGridLayout(opt_group)

        opt_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW", "RMSProp", "auto"])
        opt_layout.addWidget(self.optimizer_combo, 0, 1)

        opt_layout.addWidget(QLabel("Initial Learning Rate (lr0):"), 1, 0)
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.0001, 1.0)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setDecimals(4)
        self.lr0_spin.setValue(0.01)
        opt_layout.addWidget(self.lr0_spin, 1, 1)

        opt_layout.addWidget(QLabel("Final LR Fraction (lrf):"), 2, 0)
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0.0, 1.0)
        self.lrf_spin.setSingleStep(0.01)
        self.lrf_spin.setDecimals(3)
        self.lrf_spin.setValue(0.01)
        opt_layout.addWidget(self.lrf_spin, 2, 1)

        opt_layout.addWidget(QLabel("LR Scheduler:"), 3, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["cosine (cos_lr=True)", "linear (cos_lr=False)"])
        opt_layout.addWidget(self.scheduler_combo, 3, 1)

        opt_layout.addWidget(QLabel("Momentum:"), 4, 0)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setSingleStep(0.01)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setValue(0.937)
        opt_layout.addWidget(self.momentum_spin, 4, 1)

        opt_layout.addWidget(QLabel("Weight Decay (L2):"), 5, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setValue(0.0005)
        opt_layout.addWidget(self.weight_decay_spin, 5, 1)

        opt_layout.addWidget(QLabel("Warmup Epochs:"), 6, 0)
        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 50.0)
        self.warmup_epochs_spin.setSingleStep(0.5)
        self.warmup_epochs_spin.setValue(3.0)
        opt_layout.addWidget(self.warmup_epochs_spin, 6, 1)

        opt_layout.addWidget(QLabel("Warmup Momentum:"), 7, 0)
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setSingleStep(0.05)
        self.warmup_momentum_spin.setValue(0.8)
        opt_layout.addWidget(self.warmup_momentum_spin, 7, 1)

        scroll_layout.addWidget(opt_group)

        # ---- Loss Function ----
        loss_group = QGroupBox("Loss Function")
        loss_layout = QGridLayout(loss_group)

        loss_layout.addWidget(QLabel("Loss Type:"), 0, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["auto (Ultralytics default)", "focal (Focal Loss)", "VFL (Varifocal)", "BCE", "DFL"])
        loss_layout.addWidget(self.loss_combo, 0, 1)

        loss_layout.addWidget(QLabel("Box Loss Weight:"), 1, 0)
        self.box_spin = QDoubleSpinBox()
        self.box_spin.setRange(0.1, 20.0)
        self.box_spin.setSingleStep(0.1)
        self.box_spin.setValue(7.5)
        loss_layout.addWidget(self.box_spin, 1, 1)

        loss_layout.addWidget(QLabel("Class Loss Weight:"), 2, 0)
        self.cls_spin = QDoubleSpinBox()
        self.cls_spin.setRange(0.1, 20.0)
        self.cls_spin.setSingleStep(0.1)
        self.cls_spin.setValue(0.5)
        loss_layout.addWidget(self.cls_spin, 2, 1)

        loss_layout.addWidget(QLabel("DFL Loss Weight:"), 3, 0)
        self.dfl_spin = QDoubleSpinBox()
        self.dfl_spin.setRange(0.1, 10.0)
        self.dfl_spin.setSingleStep(0.1)
        self.dfl_spin.setValue(1.5)
        loss_layout.addWidget(self.dfl_spin, 3, 1)

        loss_layout.addWidget(QLabel("Focal Loss Gamma (γ):"), 4, 0)
        self.focal_gamma_spin = QDoubleSpinBox()
        self.focal_gamma_spin.setRange(0.5, 5.0)
        self.focal_gamma_spin.setSingleStep(0.1)
        self.focal_gamma_spin.setDecimals(2)
        self.focal_gamma_spin.setValue(2.0)
        loss_layout.addWidget(self.focal_gamma_spin, 4, 1)

        loss_layout.addWidget(QLabel("Focal Loss Alpha (α):"), 5, 0)
        self.focal_alpha_spin = QDoubleSpinBox()
        self.focal_alpha_spin.setRange(0.0, 1.0)
        self.focal_alpha_spin.setSingleStep(0.05)
        self.focal_alpha_spin.setDecimals(2)
        self.focal_alpha_spin.setValue(0.25)
        loss_layout.addWidget(self.focal_alpha_spin, 5, 1)

        scroll_layout.addWidget(loss_group)

        # ---- Augmentation Settings ----
        aug_group = QGroupBox("Augmentation Settings")
        aug_layout = QGridLayout(aug_group)

        # Mosaic / Mixup / Copy-paste row
        self.mosaic_check = QCheckBox("Mosaic")
        self.mosaic_check.setChecked(True)
        aug_layout.addWidget(self.mosaic_check, 0, 0)

        self.mixup_check = QCheckBox("MixUp")
        self.mixup_check.setChecked(False)
        aug_layout.addWidget(self.mixup_check, 0, 1)

        self.copy_paste_check = QCheckBox("Copy-Paste")
        self.copy_paste_check.setChecked(False)
        aug_layout.addWidget(self.copy_paste_check, 1, 0)

        # HSV & Flip
        aug_layout.addWidget(QLabel("HSV-Hue:"), 2, 0)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 0.1)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setValue(0.015)
        aug_layout.addWidget(self.hsv_h_spin, 2, 1)

        aug_layout.addWidget(QLabel("HSV-Saturation:"), 3, 0)
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.05)
        self.hsv_s_spin.setValue(0.7)
        aug_layout.addWidget(self.hsv_s_spin, 3, 1)

        aug_layout.addWidget(QLabel("HSV-Value:"), 4, 0)
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.05)
        self.hsv_v_spin.setValue(0.4)
        aug_layout.addWidget(self.hsv_v_spin, 4, 1)

        aug_layout.addWidget(QLabel("Horizontal Flip Prob:"), 5, 0)
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.05)
        self.fliplr_spin.setValue(0.5)
        aug_layout.addWidget(self.fliplr_spin, 5, 1)

        aug_layout.addWidget(QLabel("Vertical Flip Prob:"), 6, 0)
        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setSingleStep(0.05)
        self.flipud_spin.setValue(0.0)
        aug_layout.addWidget(self.flipud_spin, 6, 1)

        # Geometric
        aug_layout.addWidget(QLabel("Rotation (±degrees):"), 7, 0)
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setValue(0.0)
        aug_layout.addWidget(self.degrees_spin, 7, 1)

        aug_layout.addWidget(QLabel("Translation:"), 8, 0)
        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setSingleStep(0.05)
        self.translate_spin.setValue(0.1)
        aug_layout.addWidget(self.translate_spin, 8, 1)

        aug_layout.addWidget(QLabel("Scale:"), 9, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 2.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(0.5)
        aug_layout.addWidget(self.scale_spin, 9, 1)

        aug_layout.addWidget(QLabel("Shear:"), 10, 0)
        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 30.0)
        self.shear_spin.setSingleStep(0.5)
        self.shear_spin.setValue(0.0)
        aug_layout.addWidget(self.shear_spin, 10, 1)

        scroll_layout.addWidget(aug_group)

        # ---- Early stopping & checkpointing ----
        es_group = QGroupBox("Early Stopping & Checkpointing")
        es_layout = QGridLayout(es_group)

        es_layout.addWidget(QLabel("Patience (epochs, 0=disabled):"), 0, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(50)
        es_layout.addWidget(self.patience_spin, 0, 1)

        es_layout.addWidget(QLabel("Save Period (-1=off):"), 1, 0)
        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(-1, 100)
        self.save_period_spin.setValue(-1)
        es_layout.addWidget(self.save_period_spin, 1, 1)

        scroll_layout.addWidget(es_group)

        # ---- Dataset config ----
        config_group = QGroupBox("Dataset Configuration")
        config_layout = QVBoxLayout(config_group)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Dataset YAML:"))
        self.yaml_path_edit = QLineEdit()
        self.yaml_path_edit.setPlaceholderText("Path to dataset.yaml...")
        config_row.addWidget(self.yaml_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_yaml)
        config_row.addWidget(browse_btn)
        config_layout.addLayout(config_row)

        # Class names for YAML creation
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class Names:"))
        self.class_names_edit = QLineEdit()
        self.class_names_edit.setPlaceholderText("comma-separated class names")
        class_row.addWidget(self.class_names_edit)
        config_layout.addLayout(class_row)

        create_yaml_btn = QPushButton("Create dataset.yaml")
        create_yaml_btn.clicked.connect(self._create_yaml)
        config_layout.addWidget(create_yaml_btn)

        scroll_layout.addWidget(config_group)
        scroll_layout.addStretch()

        # Place the scroll area into the page content
        self.content_layout.addWidget(scroll)

        # ---- Action buttons (fixed at the bottom, outside the scroll area) ----
        btn_layout = QHBoxLayout()

        train_btn = QPushButton("🚀 Start Training")
        train_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(train_btn)

        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_training)
        btn_layout.addWidget(self.cancel_btn)

        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        btn_layout.addWidget(mark_done_btn)

        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)

    def _on_model_selection_changed(self):
        """Show / hide the custom-checkpoint widgets based on the combo choice."""
        is_custom = self.model_combo.currentText().startswith("Custom Checkpoint")
        self.checkpoint_label.setVisible(is_custom)
        self.checkpoint_path_edit.setVisible(is_custom)
        self.checkpoint_browse_btn.setVisible(is_custom)
        if is_custom and not self.checkpoint_path_edit.text():
            # Preselect the most recent best.pt if it exists
            candidate = config.training_dir / "yolo_training" / "weights" / "best.pt"
            if candidate.exists():
                self.checkpoint_path_edit.setText(str(candidate))

    def _browse_checkpoint(self):
        """Browse for a custom-trained .pt checkpoint."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Trained Checkpoint", str(config.project_root),
            "PyTorch Weights (*.pt);;All Files (*)"
        )
        if file:
            self.checkpoint_path_edit.setText(file)

    def _browse_yaml(self):
        """Browse for dataset YAML file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML", "", "YAML Files (*.yaml *.yml)")
        if file:
            self.yaml_path_edit.setText(file)

    def _create_yaml(self):
        """Create dataset YAML file."""
        if not config.split_dir.exists():
            self.log("Split directory not found. Run split first.")
            return

        class_names = [c.strip() for c in self.class_names_edit.text().split(",") if c.strip()]
        if not class_names:
            self.log("Please enter class names")
            return

        from core.model_trainer import create_dataset_yaml

        yaml_path = create_dataset_yaml(config.split_dir, class_names)
        self.yaml_path_edit.setText(str(yaml_path))
        self.log(f"Created dataset.yaml at: {yaml_path}")

    def _get_model_type(self):
        """Return the pretrained model name, or "" for "Custom Checkpoint"."""
        text = self.model_combo.currentText()
        if text.startswith("Custom Checkpoint"):
            return ""
        # Combo entries are written as "<name> (<suffix>)" — extract the model name.
        return text.split(" ", 1)[0]

    def _get_loss_type(self):
        """Convert the loss combo selection to a literal name."""
        # The combo also has descriptive text; map to a canonical token.
        idx = self.loss_combo.currentIndex()
        mapping = {0: "auto", 1: "focal", 2: "VFL", 3: "BCE", 4: "DFL"}
        return mapping.get(idx, "auto")

    def _start_training(self):
        """Start YOLO training with all user-selected hyperparameters."""
        yaml_path = self.yaml_path_edit.text()
        if not yaml_path:
            self.log("Please specify dataset YAML path")
            return

        device = "0" if self.device_combo.currentIndex() == 0 else "cpu"
        cos_lr = (self.scheduler_combo.currentIndex() == 0)  # 0 = cosine, 1 = linear
        mosaic = 1.0 if self.mosaic_check.isChecked() else 0.0
        mixup = 1.0 if self.mixup_check.isChecked() else 0.0
        copy_paste = 1.0 if self.copy_paste_check.isChecked() else 0.0

        model_type = self._get_model_type()
        checkpoint_path = self.checkpoint_path_edit.text() if model_type == "" else None
        if model_type == "" and not checkpoint_path:
            self.log("Please choose a custom checkpoint .pt file.")
            return

        task = "detect" if self.task_combo.currentIndex() == 0 else "segment"

        self.log(
            f"Starting training: model={model_type or 'custom checkpoint'}, "
            f"task={task}, epochs={self.epochs_spin.value()}, "
            f"batch={self.batch_spin.value()}, imgsz={self.imgsz_spin.value()}, "
            f"lr0={self.lr0_spin.value()}, optimizer={self.optimizer_combo.currentText()}"
        )

        def run_training(progress_callback=None, status_callback=None,
                        log_callback=None, is_cancelled=None):
            from core.model_trainer import ModelTrainer
            trainer = ModelTrainer(config_path=yaml_path, output_dir=config.training_dir)
            return trainer.train_yolo(
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_spin.value(),
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device,
                lr0=self.lr0_spin.value(),
                lrf=self.lrf_spin.value(),
                cos_lr=cos_lr,
                warmup_epochs=self.warmup_epochs_spin.value(),
                warmup_momentum=self.warmup_momentum_spin.value(),
                imgsz=self.imgsz_spin.value(),
                optimizer=self.optimizer_combo.currentText(),
                momentum=self.momentum_spin.value(),
                weight_decay=self.weight_decay_spin.value(),
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                hsv_h=self.hsv_h_spin.value(),
                hsv_s=self.hsv_s_spin.value(),
                hsv_v=self.hsv_v_spin.value(),
                fliplr=self.fliplr_spin.value(),
                flipud=self.flipud_spin.value(),
                degrees=self.degrees_spin.value(),
                translate=self.translate_spin.value(),
                scale=self.scale_spin.value(),
                shear=self.shear_spin.value(),
                loss_type=self._get_loss_type(),
                focal_gamma=self.focal_gamma_spin.value(),
                focal_alpha=self.focal_alpha_spin.value(),
                box=self.box_spin.value(),
                cls=self.cls_spin.value(),
                dfl=self.dfl_spin.value(),
                patience=self.patience_spin.value(),
                save_period=self.save_period_spin.value(),
                task=task,
                progress_callback=progress_callback,
                status_callback=status_callback,
                log_callback=log_callback,
                is_cancelled=is_cancelled,
            )

        self.cancel_btn.setEnabled(True)
        self.start_task(run_training)

    def _cancel_training(self):
        """Cancel training."""
        if self.current_worker:
            self.current_worker.cancel()
            self.log("Training cancellation requested...")
        self.cancel_btn.setEnabled(False)

    def _mark_complete(self):
        """Mark training step as complete."""
        config.update_pipeline_state("training_complete", True)
        self.set_status("Training marked as complete")
        self.task_completed.emit("training_complete", True)


# =============================================================================
# Page 5: Evaluate Model
# =============================================================================

class EvaluateModelPage(BaseWorkflowPage):
    """Page 5: Evaluate Model - Metrics, charts, and GT comparisons."""
    
    def __init__(self, parent=None):
        super().__init__(
            "📈 Step 5: Evaluate Model",
            "Assess model performance with metrics and ground truth comparisons.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)
        
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to trained model (.pt)...")
        model_row.addWidget(self.model_path_edit)
        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(browse_model_btn)
        model_layout.addLayout(model_row)
        
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test Data:"))
        self.test_path_edit = QLineEdit()
        self.test_path_edit.setPlaceholderText("Path to test dataset or YAML...")
        test_row.addWidget(self.test_path_edit)
        browse_test_btn = QPushButton("Browse")
        browse_test_btn.clicked.connect(self._browse_test)
        test_row.addWidget(browse_test_btn)
        model_layout.addLayout(test_row)
        
        # Thresholds
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Confidence Threshold:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        thresh_row.addWidget(self.conf_spin)
        thresh_row.addWidget(QLabel("IoU Threshold:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setSingleStep(0.05)
        thresh_row.addWidget(self.iou_spin)
        model_layout.addLayout(thresh_row)
        
        self.content_layout.addWidget(model_group)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setVisible(False)
        self.content_layout.addWidget(self.results_text)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        eval_btn = QPushButton("🚀 Run Evaluation")
        eval_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        eval_btn.clicked.connect(self._run_evaluation)
        btn_layout.addWidget(eval_btn)
        
        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        btn_layout.addWidget(mark_done_btn)
        
        btn_layout.addStretch()
        self.content_layout.addLayout(btn_layout)
    
    def _browse_model(self):
        """Browse for model file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Models (*.pt)")
        if file:
            self.model_path_edit.setText(file)
            config.set_trained_model_path(file)
    
    def _browse_test(self):
        """Browse for test data."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Test Data", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if file:
            self.test_path_edit.setText(file)
        else:
            directory = QFileDialog.getExistingDirectory(self, "Select Test Directory")
            if directory:
                self.test_path_edit.setText(directory)
    
    def _run_evaluation(self):
        """Run model evaluation."""
        model_path = self.model_path_edit.text()
        test_path = self.test_path_edit.text()
        
        if not model_path or not test_path:
            self.log("Please specify model and test data paths")
            return
        
        from core.model_evaluator import evaluate_unseen
        
        self.log(f"Evaluating model: {model_path}")
        self.start_task(
            evaluate_unseen,
            model_path,
            test_path,
            self.conf_spin.value(),
            self.iou_spin.value()
        )
    
    def _mark_complete(self):
        """Mark evaluation step as complete."""
        config.update_pipeline_state("evaluation_complete", True)
        self.set_status("Evaluation marked as complete")
        self.task_completed.emit("evaluation_complete", True)


# =============================================================================
# Page 6: Visualizations
# =============================================================================

class VisualizationsPage(BaseWorkflowPage):
    """Page 6: Timeline Visualizations - Chronological run output browser."""
    
    def __init__(self, parent=None):
        super().__init__(
            "🎬 Step 6: Visualizations",
            "View prediction results and timeline visualizations.",
            parent
        )
        self._setup_content()
    
    def _setup_content(self):
        """Setup page content."""
        # Prediction visualization
        pred_group = QGroupBox("Prediction Visualizations")
        pred_layout = QVBoxLayout(pred_group)
        
        pred_row = QHBoxLayout()
        pred_row.addWidget(QLabel("Images Directory:"))
        self.images_path_edit = QLineEdit()
        pred_row.addWidget(self.images_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_images)
        pred_row.addWidget(browse_btn)
        pred_layout.addLayout(pred_row)
        
        pred_btn = QPushButton("🎨 Generate Prediction Visualizations")
        pred_btn.clicked.connect(self._generate_predictions)
        pred_layout.addWidget(pred_btn)
        
        self.content_layout.addWidget(pred_group)
        
        # Timeline visualization
        timeline_group = QGroupBox("Timeline Visualization")
        timeline_layout = QVBoxLayout(timeline_group)
        
        timeline_btn = QPushButton("📅 Generate Timeline")
        timeline_btn.clicked.connect(self._generate_timeline)
        timeline_layout.addWidget(timeline_btn)
        
        self.content_layout.addWidget(timeline_group)
        
        # Output browser
        browser_group = QGroupBox("Output Browser")
        browser_layout = QVBoxLayout(browser_group)
        
        self.output_list = QTextEdit()
        self.output_list.setReadOnly(True)
        self.output_list.setMaximumHeight(150)
        browser_layout.addWidget(self.output_list)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_outputs)
        browser_layout.addWidget(refresh_btn)
        
        self.content_layout.addWidget(browser_group)
        
        # Mark complete button
        mark_done_btn = QPushButton("✓ Mark as Complete")
        mark_done_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        mark_done_btn.clicked.connect(self._mark_complete)
        self.content_layout.addWidget(mark_done_btn)
    
    def _browse_images(self):
        """Browse for images directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if directory:
            self.images_path_edit.setText(directory)
    
    def _generate_predictions(self):
        """Generate prediction visualizations."""
        model_path = config.trained_model_path
        images_path = self.images_path_edit.text()
        
        if not model_path:
            self.log("No trained model path configured")
            return
        if not images_path:
            self.log("Please specify images directory")
            return
        
        from core.visualizer import generate_prediction_visualizations
        
        self.log("Generating prediction visualizations...")
        self.start_task(
            generate_prediction_visualizations,
            model_path,
            images_path,
            config.visualizations_dir
        )
    
    def _generate_timeline(self):
        """Generate timeline visualization."""
        from core.visualizer import create_timeline_visualization
        
        # Create sample pipeline runs from config
        pipeline_runs = []
        for step, completed in config.pipeline_state.items():
            pipeline_runs.append({
                "timestamp": "2024-01-01 12:00:00",  # Would be actual timestamps
                "step": step.replace("_", " ").title(),
                "status": "success" if completed else "pending",
                "duration": 0,
            })
        
        self.log("Generating timeline...")
        self.start_task(create_timeline_visualization, pipeline_runs, config.visualizations_dir)
    
    def _refresh_outputs(self):
        """Refresh output browser."""
        from core.visualizer import browse_timeline_outputs
        
        result = browse_timeline_outputs(config.visualizations_dir)
        if result.get("success"):
            self.output_list.clear()
            self.output_list.setVisible(True)
            self.output_list.append("Timeline Images:")
            for f in result.get("timeline_images", []):
                self.output_list.append(f"  - {f}")
            self.output_list.append("\nPrediction Summaries:")
            for f in result.get("prediction_summaries", []):
                self.output_list.append(f"  - {f}")
        else:
            self.output_list.setText("No outputs found")
    
    def _mark_complete(self):
        """Mark visualization step as complete."""
        config.update_pipeline_state("visualization_complete", True)
        self.set_status("Visualizations marked as complete")
        self.task_completed.emit("visualization_complete", True)