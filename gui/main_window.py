"""
Main window with sidebar navigation and dynamic content area.
Acts as the layout skeleton and page router.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from utils.config import config


class SidebarButton(QPushButton):
    """Custom styled sidebar navigation button."""
    
    def __init__(self, text: str, icon_text: str = "", parent=None):
        super().__init__(parent)
        self.setText(f"  {icon_text}  {text}" if icon_text else f"  {text}")
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setMaximumWidth(250)
        
        # Style
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px 15px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                color: #333333;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            QPushButton:checked {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
            }
        """)


class MainWindow(QMainWindow):
    """Main application window with sidebar navigation."""
    
    # Signal emitted when page changes
    page_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection Application")
        self.setMinimumSize(1200, 800)
        
        # Initialize pages dict
        self.pages = {}
        
        # Setup UI
        self._setup_ui()
        
        # Connect to config for state updates
        self._update_pipeline_status()
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = self._create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Content area
        self.content_area = QStackedWidget()
        self.content_area.setStyleSheet("background-color: #f5f5f5;")
        main_layout.addWidget(self.content_area, 1)  # Stretch factor 1
    
    def _create_sidebar(self) -> QFrame:
        """Create the sidebar navigation panel."""
        sidebar = QFrame()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e0e0e0;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(5)
        
        # App title
        title_label = QLabel("🔍 Object Detection")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("padding: 20px 0; color: #333;")
        layout.addWidget(title_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(line)
        
        # Navigation buttons
        self.nav_buttons = []
        
        # Main Menu
        btn_main = SidebarButton("Main Menu", "🏠")
        btn_main.clicked.connect(lambda: self.navigate_to_page(0))
        self.nav_buttons.append(btn_main)
        layout.addWidget(btn_main)
        
        # Training Workflow header
        workflow_label = QLabel("🔄 Training Workflow")
        workflow_label.setStyleSheet("padding: 15px 10px 5px; color: #666; font-size: 12px;")
        layout.addWidget(workflow_label)
        
        # Workflow steps
        workflow_steps = [
            ("1. Label Data", "🏷️"),
            ("2. Augmentation", "📊"),
            ("3. Train/Test Split", "📁"),
            ("4. Train Model", "🚀"),
            ("5. Evaluate Model", "📈"),
            ("6. Visualizations", "🎬"),
        ]
        
        for text, icon in workflow_steps:
            btn = SidebarButton(text, icon)
            btn.clicked.connect(lambda checked, b=btn: self._on_workflow_button_clicked(b))
            self.nav_buttons.append(btn)
            layout.addWidget(btn)
        
        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(line2)
        
        # AI Tools header
        tools_label = QLabel("🤖 AI Tools")
        tools_label.setStyleSheet("padding: 15px 10px 5px; color: #666; font-size: 12px;")
        layout.addWidget(tools_label)
        
        # Qwen3.6
        btn_qwen = SidebarButton("Qwen3.6 Tool", "🧠")
        btn_qwen.clicked.connect(lambda: self.navigate_to_page(7))
        self.nav_buttons.append(btn_qwen)
        layout.addWidget(btn_qwen)
        
        # SAM3
        btn_sam = SidebarButton("SAM3 Tool", "🧬")
        btn_sam.clicked.connect(lambda: self.navigate_to_page(8))
        self.nav_buttons.append(btn_sam)
        layout.addWidget(btn_sam)
        
        # Spacer
        layout.addStretch()
        
        # Pipeline progress indicator
        self.progress_label = QLabel("Pipeline: 0% Complete")
        self.progress_label.setStyleSheet("padding: 10px; color: #666; font-size: 11px;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)
        
        return sidebar
    
    def _on_workflow_button_clicked(self, button: SidebarButton):
        """Handle workflow button click."""
        # Find button index in nav_buttons (offset by 1 for main menu)
        for i, btn in enumerate(self.nav_buttons):
            if btn == button:
                self.navigate_to_page(i)
                break
    
    def navigate_to_page(self, page_index: int):
        """Navigate to a specific page by index."""
        if page_index < self.content_area.count():
            self.content_area.setCurrentIndex(page_index)
            self.page_changed.emit(page_index)
            
            # Update button states
            for i, btn in enumerate(self.nav_buttons):
                btn.setChecked(i == page_index)
    
    def add_page(self, page: QWidget, name: str = ""):
        """Add a page to the content area."""
        self.content_area.addWidget(page)
        self.pages[name] = page
    
    def set_page(self, page_index: int, page: QWidget):
        """Set a page at a specific index."""
        if page_index < self.content_area.count():
            old_widget = self.content_area.widget(page_index)
            self.content_area.removeWidget(old_widget)
            old_widget.deleteLater()
        self.content_area.insertWidget(page_index, page)
    
    def _update_pipeline_status(self):
        """Update the pipeline progress display."""
        progress = config.get_pipeline_progress()
        self.progress_label.setText(f"Pipeline: {progress:.0f}% Complete")
    
    def refresh_pipeline_status(self):
        """Refresh the pipeline status display."""
        self._update_pipeline_status()