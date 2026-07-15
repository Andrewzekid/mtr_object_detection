#    app.py - Main entry point for the Object Detection Application.
#
#    RUN THIS FILE TO LAUNCH THE GUI:
#        python app.py
#
#    PROGRAMMATIC USAGE:
#        import sys
#        from app import ObjectDetectionApp
#        app = ObjectDetectionApp()
#        sys.exit(app.run())
#
#    REQUIREMENTS:
#        pip install PyQt6 ultralytics opencv-python-headless requests PyYAML
#
#    ARGUMENTS:
#        None. The module is the GUI launcher.
#
#    WHAT IT DOES:
#        1. Initializes QApplication with the "Fusion" style.
#        2. Loads (or creates) config.json via utils.config.config.
#        3. Creates the main window with sidebar navigation + 9 pages.
#        4. Opens the GUI event loop with QApplication.exec().
#
#    SEE ALSO:
#        gui/main_window.py    - the layout skeleton + page router
#        utils/config.py       - centralized state used by every page

"""
Main entry point for the Object Detection Application.
Initializes GUI and page router.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from utils.config import config
from gui.main_window import MainWindow
from gui.pages.main_menu_page import MainMenuPage
from gui.pages.workflow_pages import (
    LabelDataPage, AugmentationPage, SplitPage,
    TrainModelPage, EvaluateModelPage, VisualizationsPage
)
from gui.pages.qwen_page import QwenPage
from gui.pages.sam_page import SAMPage


class ObjectDetectionApp:
    """Main application class."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Object Detection Application")
        self.app.setOrganizationName("ObjectDetectionApp")

        # Set application style
        self.app.setStyle("Fusion")

        # Create main window
        self.window = MainWindow()

        # Setup pages
        self._setup_pages()

        # Connect signals
        self._connect_signals()

    def _setup_pages(self):
        """Setup all application pages."""
        # Page 0: Main Menu
        self.main_menu = MainMenuPage()
        self.window.add_page(self.main_menu, "main_menu")

        # Page 1: Label Data
        self.label_page = LabelDataPage()
        self.window.add_page(self.label_page, "label_data")

        # Page 2: Augmentation
        self.augment_page = AugmentationPage()
        self.window.add_page(self.augment_page, "augmentation")

        # Page 3: Split
        self.split_page = SplitPage()
        self.window.add_page(self.split_page, "split")

        # Page 4: Train Model
        self.train_page = TrainModelPage()
        self.window.add_page(self.train_page, "train")

        # Page 5: Evaluate Model
        self.eval_page = EvaluateModelPage()
        self.window.add_page(self.eval_page, "evaluate")

        # Page 6: Visualizations
        self.viz_page = VisualizationsPage()
        self.window.add_page(self.viz_page, "visualizations")

        # Page 7: Qwen3.6
        self.qwen_page = QwenPage()
        self.window.add_page(self.qwen_page, "qwen")

        # Page 8: SAM3
        self.sam_page = SAMPage()
        self.window.add_page(self.sam_page, "sam")

        # Set initial page
        self.window.navigate_to_page(0)

    def _connect_signals(self):
        """Connect signals between components."""
        # Main menu navigation
        self.main_menu.navigate_to_step.connect(self.window.navigate_to_page)

        # Task completion signals
        self.label_page.task_completed.connect(self._on_task_completed)
        self.augment_page.task_completed.connect(self._on_task_completed)
        self.split_page.task_completed.connect(self._on_task_completed)
        self.train_page.task_completed.connect(self._on_task_completed)
        self.eval_page.task_completed.connect(self._on_task_completed)
        self.viz_page.task_completed.connect(self._on_task_completed)

        # Page change signal
        self.window.page_changed.connect(self._on_page_changed)

    def _on_task_completed(self, step_name: str, success: bool):
        """Handle task completion."""
        if success:
            self.window.refresh_pipeline_status()
            self.main_menu.refresh()

    def _on_page_changed(self, page_index: int):
        """Handle page change."""
        # Refresh page content if needed
        if page_index == 0:
            self.main_menu.refresh()

        # Update pipeline status
        self.window.refresh_pipeline_status()

    def run(self) -> int:
        """Run the application."""
        # Load saved configuration
        config.load_config()

        # Show window
        self.window.show()

        # Run application
        return self.app.exec()


def main():
    """Main entry point."""
    app = ObjectDetectionApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
