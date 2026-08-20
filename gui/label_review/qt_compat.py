"""PyQt6 imports + scoped-enum compatibility aliases.

Every ``ui/`` module imports this module FIRST: it monkeypatches
PyQt5-style flat enum names (``Qt.AlignCenter``, ``QPainter.Antialiasing``
...) onto the PyQt6 classes as a side effect, and all the UI code relies
on those aliases existing.
"""

# Qt
import PyQt6.QtCore as QtCore
import PyQt6.QtGui as QtGui
import PyQt6.QtWidgets as QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent
# PyQt6 uses scoped enums. Provide short aliases for PyQt5-style names used below.
_QT_HORZ = Qt.Orientation.Horizontal
Qt.StrongFocus = Qt.FocusPolicy.StrongFocus  # type: ignore[attr-defined]
Qt.NoFocus = Qt.FocusPolicy.NoFocus  # type: ignore[attr-defined]
Qt.AlignCenter = Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
Qt.AlignLeft = Qt.AlignmentFlag.AlignLeft  # type: ignore[attr-defined]
Qt.DashLine = Qt.PenStyle.DashLine  # type: ignore[attr-defined]
Qt.LeftButton = Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
Qt.MiddleButton = Qt.MouseButton.MiddleButton  # type: ignore[attr-defined]
Qt.RightButton = Qt.MouseButton.RightButton  # type: ignore[attr-defined]
Qt.UserRole = Qt.ItemDataRole.UserRole  # type: ignore[attr-defined]
Qt.Key_Escape = Qt.Key.Key_Escape  # type: ignore[attr-defined]
Qt.Key_D = Qt.Key.Key_D  # type: ignore[attr-defined]
Qt.Key_Delete = Qt.Key.Key_Delete  # type: ignore[attr-defined]
Qt.Key_A = Qt.Key.Key_A  # type: ignore[attr-defined]
Qt.Key_N = Qt.Key.Key_N  # type: ignore[attr-defined]
Qt.Key_Right = Qt.Key.Key_Right  # type: ignore[attr-defined]
Qt.Key_B = Qt.Key.Key_B  # type: ignore[attr-defined]
Qt.Key_Left = Qt.Key.Key_Left  # type: ignore[attr-defined]
Qt.Key_X = Qt.Key.Key_X  # type: ignore[attr-defined]
Qt.Key_S = Qt.Key.Key_S  # type: ignore[attr-defined]
Qt.Key_Q = Qt.Key.Key_Q  # type: ignore[attr-defined]
Qt.Key_Plus = Qt.Key.Key_Plus  # type: ignore[attr-defined]
Qt.Key_Equal = Qt.Key.Key_Equal  # type: ignore[attr-defined]
Qt.Key_Minus = Qt.Key.Key_Minus  # type: ignore[attr-defined]
Qt.Key_0 = Qt.Key.Key_0  # type: ignore[attr-defined]
Qt.Key_M = Qt.Key.Key_M  # type: ignore[attr-defined]
Qt.Key_R = Qt.Key.Key_R  # type: ignore[attr-defined]
Qt.Key_Shift = Qt.Key.Key_Shift  # type: ignore[attr-defined]
Qt.Key_Space = Qt.Key.Key_Space  # type: ignore[attr-defined]
Qt.Key_F = Qt.Key.Key_F  # type: ignore[attr-defined]
Qt.Key_Z = Qt.Key.Key_Z  # type: ignore[attr-defined]
Qt.Key_Control = Qt.Key.Key_Control  # type: ignore[attr-defined]
Qt.Key_Y = Qt.Key.Key_Y  # type: ignore[attr-defined]
Qt.Key_U = Qt.Key.Key_U  # type: ignore[attr-defined]
Qt.Key_C = Qt.Key.Key_C  # type: ignore[attr-defined]
# Generic fallback: alias every remaining Qt.Key.Key_* member as Qt.Key_*,
# so newly added shortcuts (T, K, I, ...) don't each need a line above.
for _qn in dir(Qt.Key):
    if _qn.startswith("Key_") and not hasattr(Qt, _qn):
        setattr(Qt, _qn, getattr(Qt.Key, _qn))
# Cursor shapes (PyQt6 scoped enums)
Qt.SizeFDiagCursor = Qt.CursorShape.SizeFDiagCursor  # type: ignore[attr-defined]
Qt.SizeBDiagCursor = Qt.CursorShape.SizeBDiagCursor  # type: ignore[attr-defined]
Qt.SizeAllCursor = Qt.CursorShape.SizeAllCursor  # type: ignore[attr-defined]
Qt.CrossCursor = Qt.CursorShape.CrossCursor  # type: ignore[attr-defined]
Qt.NoModifier = Qt.KeyboardModifier.NoModifier  # type: ignore[attr-defined]
Qt.ControlModifier = Qt.KeyboardModifier.ControlModifier  # type: ignore[attr-defined]
Qt.ShiftModifier = Qt.KeyboardModifier.ShiftModifier  # type: ignore[attr-defined]
from PyQt6.QtGui import QPen, QColor, QPainter, QPixmap, QFont, QTransform
# PyQt6 scoped-enum shims for QtGui
QPainter.Antialiasing = QPainter.RenderHint.Antialiasing  # type: ignore[attr-defined]
QPainter.SmoothPixmapTransform = QPainter.RenderHint.SmoothPixmapTransform  # type: ignore[attr-defined]
QtGui.QImage.Format_RGB888 = QtGui.QImage.Format.Format_RGB888  # type: ignore[attr-defined]
QtGui.QImage.Format_ARGB32 = QtGui.QImage.Format.Format_ARGB32  # type: ignore[attr-defined]
# QFont doesn't accept "Sans" string in PyQt6; we use QFont() with family name below.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSlider,
    QSplitter, QFrame, QSizePolicy, QMessageBox, QCheckBox,
    QLineEdit, QProgressBar, QFileDialog, QAbstractItemView,
)
# QSizePolicy scoped enum alias
QSizePolicy.Expanding = QSizePolicy.Policy.Expanding  # type: ignore[attr-defined]
QSizePolicy.Fixed = QSizePolicy.Policy.Fixed  # type: ignore[attr-defined]
QSizePolicy.Preferred = QSizePolicy.Policy.Preferred  # type: ignore[attr-defined]
from PyQt6.QtGui import QShortcut, QAction
