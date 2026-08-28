"""Canvas view behavior: auto-fit keeps the whole frame visible across
resizes unless the user zoomed/panned; display.max_image_dim downscales
the displayed pixmap only (logical image coordinates are unaffected)."""

import numpy as np


def _key(lr, k):
    return lr.QtGui.QKeyEvent(
        lr.QtCore.QEvent.Type.KeyPress, k, lr.Qt.NoModifier)


def test_resize_refits_when_not_user_zoomed(lr, qapp):
    c = lr.CanvasWidget()  # min size 480x320 — stay above it
    c.show()  # resize events reach only shown widgets
    c.resize(800, 600)
    c.set_image(np.zeros((1200, 1600, 3), np.uint8))
    s0 = c._scale
    assert 0 < s0 < 1.0  # fitted, not cropped at 1.0
    # Splitter/window shrink → the whole frame must still fit.
    c.resize(500, 400)
    assert c._scale < s0
    assert c._image_size[0] * c._scale <= 500 + 1e-6
    assert c._image_size[1] * c._scale <= 400 + 1e-6
    c.close()


def test_resize_preserves_manual_zoom(lr, qapp):
    c = lr.CanvasWidget()
    c.show()
    c.resize(800, 600)
    c.set_image(np.zeros((1200, 1600, 3), np.uint8))
    c.keyPressEvent(_key(lr, lr.Qt.Key_Plus))  # manual zoom in
    assert c._user_zoomed
    s = c._scale
    c.resize(500, 400)
    assert c._scale == s  # user's zoom survives the resize
    # fit key (0) returns to auto-fit mode
    c.keyPressEvent(_key(lr, lr.Qt.Key_0))
    assert not c._user_zoomed
    c.resize(700, 500)
    assert c._image_size[0] * c._scale <= 700 + 1e-6
    assert c._image_size[1] * c._scale <= 500 + 1e-6
    c.close()


def test_display_max_dim_downscales_pixmap_only(lr, qapp):
    c = lr.CanvasWidget()
    c.display_max_dim = 100
    c.resize(800, 600)
    c.set_image(np.zeros((200, 400, 3), np.uint8))
    # logical size stays original → box/mask coordinates unaffected
    assert c._image_size == (400, 200)
    assert max(c._pixmap.width(), c._pixmap.height()) <= 100
    # 0 (default) keeps the original resolution
    c2 = lr.CanvasWidget()
    c2.set_image(np.zeros((200, 400, 3), np.uint8))
    assert (c2._pixmap.width(), c2._pixmap.height()) == (400, 200)


def test_apply_config_display_max_image_dim(lr, make_coco, make_window):
    win = make_window(coco=make_coco([]))
    assert win.display_max_dim == 0
    win._apply_runtime_config({"display": {"max_image_dim": 64}})
    assert win.display_max_dim == 64
    assert all(c.display_max_dim == 64 for c in win.canvases.values())


def test_apply_config_sam3_imgsz_zero_means_default(lr, make_coco,
                                                    make_window):
    """sam3.imgsz=0 is documented as 'library default' (the dialog
    spinbox's 0) — it must normalize to None, not reach the predictor as
    imgsz=0. Valid values and quantize round-trip unchanged."""
    win = make_window(coco=make_coco([]))
    win.sam3_imgsz = 1024
    win._apply_runtime_config({"sam3": {"imgsz": 0}})
    assert win.sam3_imgsz is None
    win._apply_runtime_config({"sam3": {"imgsz": 768, "quantize": 16}})
    assert win.sam3_imgsz == 768
    assert win.sam3_quantize == 16
    win._apply_runtime_config({"sam3": {"quantize": 0}})
    assert win.sam3_quantize is None  # 0/None = no quantization override


def test_config_dialog_display_roundtrip(lr, make_coco, make_window):
    win = make_window(coco=make_coco([]), display_max_dim=128)
    dlg = lr.ConfigDialog(win)
    assert dlg.spin_max_image_dim.value() == 128  # prefilled from window
    dlg.spin_max_image_dim.setValue(256)
    assert dlg._collect()["display"]["max_image_dim"] == 256
    dlg._prefill_from_config({"display": {"max_image_dim": 512}})
    assert dlg.spin_max_image_dim.value() == 512
    dlg.close()
