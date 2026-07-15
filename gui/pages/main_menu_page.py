"""
Main menu dashboard page with timeline progress overview.
Shows the 6 workflow steps with visual timeline widget.
"""

    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from utils.config import config


class TimelineStepWidget(QFrame):
    """Widget representing a single step in the pipeline timeline."""

    clicked = pyqtSignal(int)

    def __init__(self, step_index, title, description, icon="", parent=None):
        super().__init__(parent)
        self.step_index = step_index
        self.title = title
        self.description = description
        self.icon = icon
        self.completed = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            TimelineStepWidget {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 15px;
            }
            TimelineStepWidget:hover {
                border: 2px solid #0078d4;
                background-color: #f8f9fa;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(120)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        self.circle_label = QLabel(self.icon if self.icon else str(self.step_index + 1))
        self.circle_label.setFixedSize(50, 50)
        self.circle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.circle_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.circle_label.setStyleSheet("""
            QLabel {
                background-color: #0078d4;
                color: white;
                border-radius: 25px;
                font-size: 24px;
            }
        """)
        header_layout.addWidget(self.circle_label)

        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.title_label.setStyleSheet("margin-left: 10px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self.desc_label = QLabel(self.description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #666; font-size: 12px; margin-top: 5px;")
        layout.addWidget(self.desc_label)

    def set_completed(self, completed):
        self.completed = completed
        if completed:
            self.circle_label.setStyleSheet("""
                QLabel {
                    background-color: #28a745;
                    color: white;
                    border-radius: 25px;
                    font-size: 24px;
                }
            """)
        else:
            self.circle_label.setStyleSheet("""
                QLabel {
                    background-color: #0078d4;
                    color: white;
                    border-radius: 25px;
                    font-size: 24px;
                }
            """)

    def mousePressEvent(self, event):
        self.clicked.emit(self.step_index)
        super().mousePressEvent(event)


class MainMenuPage(QWidget):
    """Main dashboard page with timeline overview."""

    navigate_to_step = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._update_status()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)

        title = QLabel("Dashboard Overview")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        layout.addWidget(title)

        subtitle = QLabel("Welcome to the Object Detection Application. Click on any step to begin.")
        subtitle.setStyleSheet("color: #666; font-size: 14px; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        steps_layout = QGridLayout()
        steps_layout.setSpacing(25)

        self.steps = [
            ("Label Data", "Annotate images with bounding boxes", "[1]"),
            ("Data Augmentation", "Apply transformations to expand dataset", "[2]"),
            ("Train/Test Split", "Divide data into training and validation sets", "[3]"),
            ("Train Model", "Train YOLO model on prepared dataset", "[4]"),
            ("Evaluate Model", "Assess model performance with metrics", "[5]"),
            ("Visualizations", "View predictions and timeline results", "[6]"),
        ]

        self.step_widgets = []
        for i, (t, d, ic) in enumerate(self.steps):
            step_widget = TimelineStepWidget(i, t, d, ic)
            step_widget.clicked.connect(self._on_step_clicked)
            self.step_widgets.append(step_widget)
            steps_layout.addWidget(step_widget, i // 2, i % 2)

        layout.addLayout(steps_layout)

        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
        """)
        actions_layout = QVBoxLayout(actions_frame)

        actions_title = QLabel("Quick Actions")
        actions_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        actions_layout.addWidget(actions_title)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)

        btn_open_dataset = QPushButton("Open Dataset")
        btn_open_dataset.setMinimumHeight(40)
        btn_open_dataset.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        btn_open_dataset.clicked.connect(self._on_open_dataset)
        buttons_layout.addWidget(btn_open_dataset)

        btn_load_config = QPushButton("Load Configuration")
        btn_load_config.setMinimumHeight(40)
        btn_load_config.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #545b62; }
        """)
        btn_load_config.clicked.connect(self._on_load_config)
        buttons_layout.addWidget(btn_load_config)

        btn_save_config = QPushButton("Save Configuration")
        btn_save_config.setMinimumHeight(40)
        btn_save_config.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #1e7e34; }
        """)
        btn_save_config.clicked.connect(self._on_save_config)
        buttons_layout.addWidget(btn_save_config)

        buttons_layout.addStretch()
        actions_layout.addLayout(buttons_layout)
        layout.addWidget(actions_frame)
        layout.addStretch()

    def _on_step_clicked(self, step_index):
        self.navigate_to_step.emit(step_index + 1)

    def _on_open_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            config.set_dataset_path(directory)

            images = config.get_image_files()
            images_count = len(images)

            labels_count = 0
            if config.current_labels_path and config.current_labels_path.exists():
                # For nested YOLO structure, count labels from all split subdirectories
                if getattr(config, 'dataset_type', None) == 'yolo_nested':
                    for split_dir in ['train', 'val', 'test']:
                        split_path = config.current_labels_path / split_dir
                        if split_path.exists():
                            labels_count += len(list(split_path.glob("*.txt")))
                else:
                    labels_count = len(list(config.current_labels_path.glob("*.txt")))

            dtype = getattr(config, "dataset_type", "unknown")
            type_label = {
                "unlabelled": "Unlabelled folder",
                "yolo_flat": "YOLO dataset (images/ + labels/)",
                "yolo_nested": "YOLO dataset (nested train/val/test)",
                "yolo_split": "YOLO dataset (train/test/val split)",
            }.get(dtype, dtype)

            self._update_status()

            QMessageBox.information(
                self,
                "Dataset Loaded",
                f"Dataset loaded successfully!\n\n"
                f"Path: {directory}\n"
                f"Type: {type_label}\n"
                f"Images: {images_count}\n"
                f"Labels: {labels_count}"
            )

    def _on_load_config(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", str(config.project_root), "JSON Files (*.json)"
        )
        if filepath:
            config.load_config(filepath)
            self._update_status()
            QMessageBox.information(
                self, "Configuration Loaded",
                f"Configuration loaded from:\n{filepath}"
            )

    def _on_save_config(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", str(config.project_root / "config.json"), "JSON Files (*.json)"
        )
        if filepath:
            config.save_config(filepath)
            QMessageBox.information(
                self, "Configuration Saved",
                f"Configuration saved to:\n{filepath}"
            )

    def _update_status(self):
        step_keys = list(config.pipeline_state.keys())
        for i, (key, widget) in enumerate(zip(step_keys, self.step_widgets)):
            widget.set_completed(config.pipeline_state[key])

    def refresh(self):
        self._update_status()
