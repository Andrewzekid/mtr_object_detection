"""UI themes for the label-review tool: dark / light / pastel (lime).

The active theme is a process-wide singleton: ``apply_theme()`` installs a
full application-wide stylesheet (so dialogs and menus follow along) and
custom-painted widgets (the canvases) read the live palette on every paint
via :func:`color` — a theme switch therefore needs no per-widget plumbing,
only a repaint, which the stylesheet change already triggers.

Palettes are plain dicts of hex colors; every role used by the QSS template
and by custom painting has a key in each palette.
"""

from typing import Dict, Optional

from PyQt6.QtGui import QColor

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        # Surfaces
        "window": "#1e1f24",
        "panel": "#26272e",
        "surface": "#2e3038",
        "surface_hover": "#373a45",
        "input": "#23242b",
        # Text
        "text": "#e8eaf0",
        "muted": "#9aa0ad",
        # Accent
        "accent": "#4f8cff",
        "accent_hover": "#6da0ff",
        "accent_pressed": "#3a6fd8",
        "accent_fg": "#ffffff",
        # Semantic
        "danger": "#ff6b6b",
        "danger_soft": "#28ff6b6b",
        "success": "#43cf7c",
        # Lines / selection
        "border": "#383a45",
        "selection_bg": "#31415f",
        "selection_fg": "#ffffff",
        # Canvas
        "canvas": "#14141a",
        "canvas_fg": "#c8cbd4",
    },
    "light": {
        "window": "#f4f5f7",
        "panel": "#ffffff",
        "surface": "#ffffff",
        "surface_hover": "#eceff4",
        "input": "#ffffff",
        "text": "#23262e",
        "muted": "#6b7280",
        "accent": "#2f6bff",
        "accent_hover": "#2456d6",
        "accent_pressed": "#1c46b0",
        "accent_fg": "#ffffff",
        "danger": "#d64545",
        "danger_soft": "#14d64545",
        "success": "#1fa855",
        "border": "#d7dae1",
        "selection_bg": "#dbe6ff",
        "selection_fg": "#17264d",
        "canvas": "#e7e9ee",
        "canvas_fg": "#4a4f5c",
    },
    # Pastel: soft lime-green base.
    "pastel": {
        "window": "#f2f8dc",
        "panel": "#fafdf0",
        "surface": "#fdfef6",
        "surface_hover": "#eef7d2",
        "input": "#ffffff",
        "text": "#35401f",
        "muted": "#7d875f",
        "accent": "#82b91a",
        "accent_hover": "#93cc2a",
        "accent_pressed": "#6ea012",
        "accent_fg": "#2c350f",
        "danger": "#e06c5b",
        "danger_soft": "#19e06c5b",
        "success": "#5cab3d",
        "border": "#d9e6ae",
        "selection_bg": "#e2f2b6",
        "selection_fg": "#33400f",
        "canvas": "#e6f0c4",
        "canvas_fg": "#55603a",
    },
}

DEFAULT_THEME = "dark"
_current: str = DEFAULT_THEME

_QCOLOR_CACHE: Dict[str, Dict[str, QColor]] = {}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def current_theme() -> str:
    """The active theme name (key of THEMES)."""
    return _current


def palette(name: Optional[str] = None) -> Dict[str, str]:
    """The color dict for `name` (default: the active theme)."""
    return THEMES.get(name or _current, THEMES[DEFAULT_THEME])


def color(role: str, name: Optional[str] = None) -> QColor:
    """A palette role as a QColor (cached; safe to call every paint)."""
    pname = name or _current
    cache = _QCOLOR_CACHE.setdefault(pname, {})
    cq = cache.get(role)
    if cq is None:
        cq = QColor(palette(pname)[role])
        cache[role] = cq
    return cq


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet(name: Optional[str] = None) -> str:
    """The application-wide QSS for theme `name` (default: active)."""
    p = palette(name)
    return f"""
QWidget {{
    background-color: {p['window']};
    color: {p['text']};
    font-size: 13px;
}}

/* ---------- menus ---------- */
QMenuBar {{
    background-color: {p['panel']};
    color: {p['text']};
    border-bottom: 1px solid {p['border']};
    padding: 2px 6px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background: {p['selection_bg']}; }}
QMenuBar::item:pressed {{ background: {p['selection_bg']}; }}
QMenu {{
    background-color: {p['panel']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {p['selection_bg']}; }}
QMenu::item:disabled {{ color: {p['muted']}; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 4px 6px; }}

/* ---------- buttons ---------- */
QPushButton {{
    background-color: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 5px 12px;
}}
QPushButton:hover {{
    border-color: {p['accent']};
    background-color: {p['surface_hover']};
}}
QPushButton:pressed {{ background-color: {p['selection_bg']}; }}
QPushButton:disabled {{
    color: {p['muted']};
    background-color: {p['window']};
    border-color: {p['border']};
}}
QPushButton:checked {{
    background-color: {p['selection_bg']};
    border-color: {p['accent']};
    font-weight: 600;
}}
QPushButton[cssClass="primary"] {{
    background-color: {p['accent']};
    color: {p['accent_fg']};
    border: 1px solid {p['accent']};
    font-weight: 600;
}}
QPushButton[cssClass="primary"]:hover {{ background-color: {p['accent_hover']}; border-color: {p['accent_hover']}; }}
QPushButton[cssClass="primary"]:checked {{ background-color: {p['accent_pressed']}; border-color: {p['accent_pressed']}; }}
QPushButton[cssClass="danger"] {{
    color: {p['danger']};
    border-color: {p['danger']};
    background-color: {p['danger_soft']};
}}
QPushButton[cssClass="danger"]:hover,
QPushButton[cssClass="danger"]:checked {{
    background-color: {p['danger']};
    color: {p['accent_fg']};
}}
QPushButton#configButton {{
    border: none;
    background: transparent;
    color: {p['muted']};
    padding: 4px 10px;
}}
QPushButton#configButton:hover {{ color: {p['accent']}; background: transparent; }}

/* ---------- inputs ---------- */
QLineEdit, QComboBox {{
    background-color: {p['input']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 4px 8px;
}}
QLineEdit:focus {{ border-color: {p['accent']}; }}
QComboBox:hover {{ border-color: {p['accent']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {p['muted']};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {p['panel']};
    color: {p['text']};
    border: 1px solid {p['border']};
    selection-background-color: {p['selection_bg']};
    selection-color: {p['selection_fg']};
    outline: 0;
}}

/* ---------- lists ---------- */
QListWidget {{
    background-color: {p['input']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    padding: 2px;
    outline: 0;
}}
QListWidget::item {{ padding: 3px 6px; border-radius: 4px; }}
QListWidget::item:hover {{ background: {p['surface_hover']}; }}
QListWidget::item:selected {{
    background: {p['selection_bg']};
    color: {p['selection_fg']};
}}

/* ---------- sliders ---------- */
QSlider::groove:horizontal {{
    height: 6px;
    background: {p['border']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {p['accent']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    background: {p['accent']};
}}
QSlider::handle:horizontal:hover {{ background: {p['accent_hover']}; }}

/* ---------- progress ---------- */
QProgressBar {{
    background: {p['input']};
    border: 1px solid {p['border']};
    border-radius: 6px;
    text-align: center;
    color: {p['text']};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {p['accent']};
    border-radius: 5px;
}}

/* ---------- checkbox ---------- */
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {p['border']};
    border-radius: 4px;
    background: {p['input']};
}}
QCheckBox::indicator:checked {{
    background: {p['accent']};
    border-color: {p['accent']};
}}
QCheckBox::indicator:hover {{ border-color: {p['accent']}; }}

/* ---------- chrome ---------- */
QStatusBar {{
    background: {p['panel']};
    color: {p['muted']};
    border-top: 1px solid {p['border']};
}}
QStatusBar QLabel {{ background: transparent; }}
QSplitter::handle {{ background: {p['border']}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QSplitter::handle:hover {{ background: {p['accent']}; }}
QGroupBox {{
    border: 1px solid {p['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px 6px 6px 6px;
    background-color: {p['panel']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {p['accent']};
    font-weight: 600;
}}
QToolTip {{
    background-color: {p['panel']};
    color: {p['text']};
    border: 1px solid {p['border']};
    padding: 4px 8px;
    border-radius: 4px;
}}
QScrollArea {{ background: transparent; border: none; }}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {p['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {p['border']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p['accent']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- side-panel pieces ---------- */
QFrame#sectionSeparator {{
    background-color: {p['border']};
    border: none;
    max-height: 1px;
}}
QLabel#sectionHeader {{
    color: {p['accent']};
    font-weight: 700;
    font-size: 12px;
    border-bottom: 1px solid {p['border']};
    padding: 2px 0 4px 0;
    background: transparent;
}}
QLabel#mutedLabel, QLabel#helpLabel {{
    color: {p['muted']};
    font-size: 11px;
    background: transparent;
}}
"""


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_theme(name: str, app=None) -> str:
    """Switch the active theme and install its stylesheet on `app`.

    Returns the theme actually applied (falls back to the previous theme /
    default for unknown names). Custom-painted widgets pick the new palette
    up on their next repaint, which the stylesheet swap triggers.
    """
    global _current
    if name not in THEMES:
        return _current
    _current = name
    if app is not None:
        app.setStyleSheet(build_stylesheet(name))
    return _current
