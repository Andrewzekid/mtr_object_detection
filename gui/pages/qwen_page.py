"""
Qwen3.6 Multi-modal Tool page.
Provides interface for Qwen3.6 inference via Ollama API.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QGridLayout, QSplitter, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from utils.config import config
from utils.workers import TaskWorker


class QwenPage(QWidget):
    """Qwen3.6 Multi-modal Tool page."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_worker = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the Qwen page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🧠 Qwen3.6 Multi-modal Tool")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Use Qwen3.6 via Ollama for image analysis, object detection, and scene understanding.")
        desc.setStyleSheet("color: #666; font-size: 13px;")
        layout.addWidget(desc)
        
        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # Template selection
        template_group = QGroupBox("Prompt Template")
        template_layout = QVBoxLayout(template_group)
        
        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Custom",
            "Object Detection",
            "Image Captioning",
            "Scene Understanding",
            "Counting",
            "Spatial Reasoning",
            "Advertisement Board Detection",
        ])
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        template_row.addWidget(self.template_combo)
        template_layout.addLayout(template_row)
        
        # Template description
        self.template_desc_label = QLabel("Enter your custom prompt below.")
        self.template_desc_label.setStyleSheet("color: #666; font-size: 11px;")
        self.template_desc_label.setWordWrap(True)
        template_layout.addWidget(self.template_desc_label)
        
        left_layout.addWidget(template_group)
        
        # Prompt input
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        self.prompt_edit.setMinimumHeight(150)
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
            }
        """)
        prompt_layout.addWidget(self.prompt_edit)
        
        left_layout.addWidget(prompt_group)
        
        # Image input
        image_group = QGroupBox("Image Input (Optional)")
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
        
        # Image preview
        self.image_preview_label = QLabel("No image selected")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumHeight(150)
        self.image_preview_label.setStyleSheet("""
            QLabel {
                border: 1px dashed #ccc;
                border-radius: 5px;
                background-color: #f8f9fa;
                color: #999;
            }
        """)
        image_layout.addWidget(self.image_preview_label)
        
        left_layout.addWidget(image_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Output format
        settings_layout.addWidget(QLabel("Output Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JSON", "YAML", "Bounding Box", "Plain Text"])
        settings_layout.addWidget(self.format_combo, 0, 1)
        
        # Model selection
        settings_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["qwen3.6:27b", "qwen3.6", "qwen2.5", "qwen2"])
        self.model_combo.setEditable(True)
        settings_layout.addWidget(self.model_combo, 1, 1)
        
        # Ollama URL
        settings_layout.addWidget(QLabel("Ollama URL:"), 2, 0)
        self.ollama_url_edit = QLineEdit()
        self.ollama_url_edit.setText(config.ollama_config["base_url"])
        settings_layout.addWidget(self.ollama_url_edit, 2, 1)
        
        left_layout.addWidget(settings_group)
        
        # Run button
        self.run_btn = QPushButton("🚀 Run Qwen3.6")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.run_btn.clicked.connect(self._run_qwen)
        left_layout.addWidget(self.run_btn)
        
        left_layout.addStretch()
        
        # Right panel - Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        # Output preview
        output_group = QGroupBox("Output Preview")
        output_layout = QVBoxLayout(output_group)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        output_layout.addWidget(self.status_label)
        
        # Raw response
        self.response_edit = QTextEdit()
        self.response_edit.setReadOnly(True)
        self.response_edit.setPlaceholderText("Model response will appear here...")
        self.response_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                border-radius: 5px;
            }
        """)
        output_layout.addWidget(QLabel("Raw Response:"))
        output_layout.addWidget(self.response_edit)
        
        # Parsed output
        self.parsed_edit = QTextEdit()
        self.parsed_edit.setReadOnly(True)
        self.parsed_edit.setPlaceholderText("Parsed output will appear here...")
        self.parsed_edit.setMaximumHeight(200)
        self.parsed_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        output_layout.addWidget(QLabel("Parsed Output:"))
        output_layout.addWidget(self.parsed_edit)
        
        right_layout.addWidget(output_group)
        
        # History
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setMaximumHeight(150)
        history_layout.addWidget(self.history_list)
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self._clear_history)
        history_layout.addWidget(clear_history_btn)
        
        right_layout.addWidget(history_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        # Progress bar (hidden by default)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
    
    def _on_template_changed(self, index: int):
        """Handle template selection change."""
        templates = {
            0: ("Custom", "Enter your custom prompt below."),
            1: ("Object Detection", "Analyze this image and detect all objects. For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2] format (normalized 0-1000). Results are returned as x, y, w, h."),
            2: ("Image Captioning", "Describe this image in detail, including all visible objects, their positions, and any actions occurring."),
            3: ("Scene Understanding", "Analyze the scene in this image. What is happening? What are the main elements? What is the context?"),
            4: ("Counting", "Count all objects in this image by category. Provide a detailed breakdown."),
            5: ("Spatial Reasoning", "Analyze the spatial relationships between objects in this image. Describe relative positions and distances."),
            6: ("Advertisement Board Detection", "Analyze this image and detect all Advertisement Boards. For each object, provide the class name and bounding box coordinates. Follow the format below:\n\nYou are an object detection model. Identify and locate the following objects in the image: Advertisement Boards.\n\nFor each detected object, return a JSON array with elements like:\n[{\"label\": \"object_name\", \"bbox_2d\": [x1, y1, x2, y2]}]\n\nONLY OUTPUT THE JSON ARRAY. YOU MUST PRODUCE AN OUTPUT. PRODUCE AN EMPTY ARRAY IF NOTHING IS DETECTED."),
        }
        
        if index in templates:
            name, desc = templates[index]
            self.template_desc_label.setText(desc)
            if index > 0 and not self.prompt_edit.toPlainText():
                self.prompt_edit.setPlainText(desc)
    
    def _browse_image(self):
        """Browse for image file."""
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtGui import QPixmap
        
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file:
            self.image_path_edit.setText(file)
            
            # Show preview
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                scaled = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
                self.image_preview_label.setPixmap(scaled)
    
    def _get_format_string(self) -> str:
        """Get output format string."""
        formats = ["json", "yaml", "bbox", "text"]
        return formats[self.format_combo.currentIndex()]
    
    def _run_qwen(self):
        """Run Qwen3.6 inference."""
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            self.status_label.setText("Please enter a prompt")
            return
        
        from core.models_inference import run_qwen
        
        # Get template ID if not custom
        template_id = None
        template_index = self.template_combo.currentIndex()
        if template_index > 0:
            template_ids = [None, "object_detection", "image_captioning", "scene_understanding", "counting", "spatial_reasoning", "advertisement_board_detection"]
            template_id = template_ids[template_index]
        
        # Get image path if provided
        image_path = self.image_path_edit.text() if self.image_path_edit.text() else None
        
        # Disable run button
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running...")
        self.progress_label.setVisible(True)
        self.progress_label.setText("Processing...")
        
        # Start task
        self.current_worker = TaskWorker(
            run_qwen,
            prompt,
            template_id,
            self._get_format_string(),
            image_path,
            self.ollama_url_edit.text(),
            self.model_combo.currentText(),
        )
        self.current_worker.status.connect(self._on_status)
        self.current_worker.finished.connect(self._on_finished)
        self.current_worker.error.connect(self._on_error)
        self.current_worker.start()
    
    def _on_status(self, message: str):
        """Handle status update."""
        self.status_label.setText(message)
        self.progress_label.setText(message)
    
    def _on_finished(self, result):
        """Handle inference completion."""
        self.run_btn.setEnabled(True)
        self.progress_label.setVisible(False)
        
        if isinstance(result, dict):
            if result.get("success"):
                self.status_label.setText("Completed successfully")
                self.response_edit.setText(result.get("response", ""))
                
                # Display parsed output
                parsed = result.get("parsed_output")
                if parsed:
                    import json
                    if isinstance(parsed, (dict, list)):
                        self.parsed_edit.setText(json.dumps(parsed, indent=2))
                    else:
                        self.parsed_edit.setText(str(parsed))
                
                # Add to history
                self._add_to_history(result)
            else:
                self.status_label.setText(f"Failed: {result.get('error', 'Unknown error')}")
                self.response_edit.setText(f"Error: {result.get('error', 'Unknown error')}")
    
    def _on_error(self, error_msg: str):
        """Handle inference error."""
        self.run_btn.setEnabled(True)
        self.progress_label.setVisible(False)
        self.status_label.setText("Error")
        self.response_edit.setText(f"Error: {error_msg}")
    
    def _add_to_history(self, result: dict):
        """Add result to history."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        prompt_preview = self.prompt_edit.toPlainText()[:50]
        
        history_entry = f"[{timestamp}] {prompt_preview}...\n"
        self.history_list.append(history_entry)
    
    def _clear_history(self):
        """Clear history."""
        self.history_list.clear()