"""Interactive 2D bbox reviewer (modular package layout).

Re-exports the public API so external code and tests can simply do
``import gui.label_review as lr``.
"""

from .qt_compat import (  # noqa: F401
    Qt, QtCore, QtGui, QtWidgets, QMessageBox, QFileDialog,
)
from .config import _seed_categories, _resolve_device  # noqa: F401
from .state.index import (  # noqa: F401
    ImageFolderIndex, EmptyIndex, StereoIndex, StereoSideIndex,
)
from .state.undo_stack import UndoStack  # noqa: F401
from .state.coco_state import CocoState  # noqa: F401
from .ui.canvas import CanvasWidget  # noqa: F401
from .ui.side_panel import SidePanel  # noqa: F401
from .ui.dialogs import ConfigDialog  # noqa: F401
from .ui.main_window import ReviewWindow  # noqa: F401
from .utils.mask_utils import (  # noqa: F401
    _encode_mask_png, _decode_mask_png, _mask_to_polygons,
    _polygons_to_mask,
)
from .workers.label_review_workers import (  # noqa: F401
    SAM3Worker, SAM3BatchWorker, InterpBatchWorker,
    SAM3AutolabelWorker, SAM3AutolabelBatchWorker, SAM3PropagateWorker,
    _get_interp13, _SAM3_AVAILABLE, run_sam3, _iou_xyxy,
)
from .main import main  # noqa: F401
