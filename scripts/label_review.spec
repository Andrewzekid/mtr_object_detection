# PyInstaller spec for the label review app (scripts/09_rerun_label_review.py).
#
# Build (from the repo root):
#     pyinstaller scripts/label_review.spec --distpath dist --workpath build
#
# Result: dist/label-review/label-review  (onedir bundle — run it from there)
#
# Notes:
# * SAM3 weights are NOT bundled (3.4 GB). On the target machine pass
#   --sam3-model /path/to/sam3.pt (or place core/sam3/models/sam3-model/sam3.pt
#   next to the bundle and run from that directory).
# * GPU: the bundle ships the torch build from THIS machine. A CUDA build of
#   torch only works on target machines with a compatible NVIDIA driver.
#   For CPU-only targets, build in a CPU-torch environment.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# ultralytics reads YAML configs and default settings at runtime.
_d, _b, _h = collect_all("ultralytics")
datas += _d; binaries += _b; hiddenimports += _h
# torch is imported lazily (SAM3 device resolve + ultralytics).
_d, _b, _h = collect_all("torch")
datas += _d; binaries += _b; hiddenimports += _h

# Conda's pyexpat is built against conda's newer libexpat
# (XML_SetAllocTrackerActivationThreshold); PyInstaller otherwise grabs the
# older system libexpat and the frozen app fails to start.
import sysconfig as _sc
_conda_lib = Path(_sc.get_config_var("LIBDIR")) / "libexpat.so.1"
if _conda_lib.exists():
    binaries.append((str(_conda_lib), "."))

a = Analysis(
    ["09_rerun_label_review.py"],
    pathex=["..", "."],
    binaries=binaries,
    datas=datas + [
        # Loaded lazily by path (label_review_workers._get_interp13);
        # lands in the bundle root (sys._MEIPASS) in onedir mode.
        ("13_interpolate_tracks.py", "."),
    ],
    hiddenimports=hiddenimports + ["label_review_workers",
                                   "scripts.tracking_utils"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # not used by this app — keeps the bundle smaller
        "rerun", "rerun_sdk", "pyarrow", "matplotlib", "tkinter",
        "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets",
        # exactly one Qt binding may be frozen; this app uses PyQt6
        "PyQt5", "PySide2", "PySide6",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="label-review",
    console=True,   # status/errors print to the terminal
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="label-review",
)
