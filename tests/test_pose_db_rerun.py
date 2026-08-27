"""Tests for PoseDb pose lookup and RerunLogger marker-entity derivation.

Regression coverage for the "Show annotated in Rerun shows nothing" bug:
image folders named 1000.jpg, 1001.jpg, ... were misparsed as nanosecond
timestamps, so every frame snapped to the first DB pose (or, after the
range guard, to None) - the pose DB's ``filename`` column is the reliable
key. Markers also used to be logged at a fixed ``world/map/...`` path,
which misses the leveling transform the map cloud sits under; the marker
entity is now derived as a sibling of the map point cloud.
"""

import os
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.label_review.map_view import PoseDb  # noqa: E402
from gui.label_review.rerun_logger import (  # noqa: E402
    MARKER_ENTITY, RerunLogger)

BASE_TS = 1_785_987_148_689_014_000


def _make_db(path, with_filename=True):
    """Test DB: ids (101/102/103) deliberately differ from the filename
    numbers (1/2/3.jpg) so id-matching and filename-matching are
    distinguishable."""
    con = sqlite3.connect(path)
    cols = ("id INTEGER PRIMARY KEY, timestamp_ns INTEGER, "
            "tf_translation_x REAL, tf_translation_y REAL, "
            "tf_translation_z REAL, cam_tf_translation_x REAL, "
            "cam_tf_translation_y REAL, cam_tf_translation_z REAL")
    if with_filename:
        cols += ", filename TEXT"
    con.execute(f"CREATE TABLE images ({cols})")
    rows = [
        # (id, timestamp_ns, cam xyz, lidar xyz, filename)
        (101, BASE_TS, (1.0, 2.0, 3.0), (9.0, 9.0, 9.0), "1.jpg"),
        (102, BASE_TS + 100_000_000, (4.0, 5.0, 6.0), (9.0, 9.0, 9.0),
         "2.jpg"),
        (103, BASE_TS + 200_000_000,
         (float("nan"), float("nan"), float("nan")),
         (7.0, 8.0, 9.0), "3.jpg"),
    ]
    for row_id, ts, cam, lidar, name in rows:
        if with_filename:
            con.execute(
                "INSERT INTO images (id, timestamp_ns, "
                "cam_tf_translation_x, cam_tf_translation_y, "
                "cam_tf_translation_z, tf_translation_x, tf_translation_y, "
                "tf_translation_z, filename) VALUES (?,?,?,?,?,?,?,?,?)",
                (row_id, ts, *cam, *lidar, name))
        else:
            con.execute(
                "INSERT INTO images (id, timestamp_ns, "
                "cam_tf_translation_x, cam_tf_translation_y, "
                "cam_tf_translation_z, tf_translation_x, tf_translation_y, "
                "tf_translation_z) VALUES (?,?,?,?,?,?,?,?)",
                (row_id, ts, *cam, *lidar))
    con.commit()
    con.close()


class TestPoseDbFilenameLookup:
    def test_filename_wins_over_bogus_timestamp(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file)
        # Folder images named 1.jpg/2.jpg get misparsed as ts=1/2 ns;
        # the filename must still resolve to the right pose.
        np.testing.assert_allclose(db.pose_for("2.jpg", 2), [4.0, 5.0, 6.0])

    def test_timestamp_fallback_within_window(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file)
        np.testing.assert_allclose(
            db.pose_for("unknown.jpg", BASE_TS + 100_000_100),
            [4.0, 5.0, 6.0])

    def test_timestamp_fallback_rejected_out_of_range(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file)
        assert db.pose_for("unknown.jpg", 1000) is None

    def test_lidar_fallback_when_cam_nan(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file)
        np.testing.assert_allclose(db.pose_for("3.jpg", None), [7.0, 8.0, 9.0])

    def test_db_without_filename_column(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file, with_filename=False)
        db = PoseDb(db_file)
        # No filename column: auto mode falls to the id-column match
        # (id 102 <- '102.jpg'), not to the misparsed timestamp.
        np.testing.assert_allclose(db.pose_for("102.jpg", 102),
                                   [4.0, 5.0, 6.0])
        assert db.pose_for("999.jpg", 999) is None


class TestPoseDbMatchModes:
    def test_invalid_mode_rejected(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        with pytest.raises(ValueError, match="match_mode"):
            PoseDb(db_file, match_mode="bogus")

    def test_filename_id_mode(self, tmp_path):
        """Images named by the images-table id (101.jpg) match the `id`
        column even though the filename column says something else."""
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file, match_mode="filename_id")
        np.testing.assert_allclose(db.pose_for("102.jpg", None),
                                   [4.0, 5.0, 6.0])
        # a name matching only the filename column does NOT match in this
        # mode ('2.jpg' stem 2 is not an id)
        assert db.pose_for("2.jpg", None) is None
        # non-numeric names never match
        assert db.pose_for("frame_a.jpg", None) is None

    def test_filename_mode(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file, match_mode="filename")
        np.testing.assert_allclose(db.pose_for("2.jpg", None),
                                   [4.0, 5.0, 6.0])
        assert db.pose_for("102.jpg", None) is None  # id match disabled
        # timestamp fallback disabled too
        assert db.pose_for("unknown.jpg", BASE_TS) is None

    def test_timestamp_mode(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file, match_mode="timestamp")
        np.testing.assert_allclose(
            db.pose_for("1.jpg", BASE_TS + 100_000_100), [4.0, 5.0, 6.0])
        assert db.pose_for("1.jpg", 1) is None  # filename match disabled

    def test_auto_order_filename_then_id_then_timestamp(self, tmp_path):
        db_file = tmp_path / "poses.db"
        _make_db(db_file)
        db = PoseDb(db_file, match_mode="auto")
        # filename column wins over the id interpretation of the stem
        np.testing.assert_allclose(db.pose_for("2.jpg", 2), [4.0, 5.0, 6.0])
        # stem 101 is no filename, but it is an id
        np.testing.assert_allclose(db.pose_for("101.jpg", 101),
                                   [1.0, 2.0, 3.0])
        # neither filename nor id → guarded timestamp fallback
        np.testing.assert_allclose(
            db.pose_for("unknown.jpg", BASE_TS + 100_000_100),
            [4.0, 5.0, 6.0])
        assert db.pose_for("unknown.jpg", 1000) is None


class TestMarkerEntityDerivation:
    @pytest.fixture()
    def map_rrd(self, tmp_path):
        rr = pytest.importorskip("rerun")
        path = tmp_path / "map.rrd"
        rec = rr.RecordingStream("test_app", recording_id="test_rec")
        rec.save(str(path))
        rec.log("world/leveled/camera_init/colored_map",
                rr.Points3D([[0, 0, 0], [1, 1, 1]]), static=True)
        rec.log("world/leveled/camera_init/body/cloud_body",
                rr.Points3D([[0, 0, 0]]))
        rec.flush()
        return str(path), "test_rec"

    def test_sibling_of_map_cloud(self, map_rrd):
        path, rec_id = map_rrd
        entity = RerunLogger._find_marker_entity(path, rec_id)
        assert entity == "world/leveled/camera_init/annotated_frames"

    def test_fallback_for_missing_recording(self, map_rrd):
        path, _ = map_rrd
        assert RerunLogger._find_marker_entity(path, "nope") == MARKER_ENTITY


class TestMapBlueprint:
    def test_default_blueprint_is_single_3d_map_view(self):
        """The viewer opens on the 3D map: the sent blueprint is one
        Spatial3DView rooted at the map cloud's directory (the markers'
        parent), leaving out the camera image/depth 2D views."""
        pytest.importorskip("rerun")
        logger = RerunLogger()
        logger._marker_entity = "world/leveled/camera_init/annotated_frames"
        sent = []
        logger._stream = types.SimpleNamespace(send_blueprint=sent.append)
        logger._send_map_blueprint()
        (view,) = sent[0].root_container.contents
        assert type(view).__name__ == "Spatial3DView"
        assert view.origin == "/world/leveled/camera_init"
        # the camera-body subtree is excluded from the default map view
        assert list(view.contents) == ["+ $origin/**", "- $origin/body/**"]


class TestAutoLevel:
    """Auto-levelling: on open, the map cloud's ground plane is fitted and
    a corrective static transform is derived for the ancestor carrying the
    recorded (mis-)levelling transform, so the map renders flat."""

    @staticmethod
    def _quat_about_y(deg):
        rad = np.radians(deg)
        return [0.0, float(np.sin(rad / 2)), 0.0, float(np.cos(rad / 2))]

    @staticmethod
    def _quat_to_R(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    @pytest.fixture()
    def tilted_rrd(self, tmp_path):
        """Cloud whose ground plane is tilted -7 deg about Y, under a
        recorded 'leveled' transform of +20 deg (net displayed tilt 13)."""
        rr = pytest.importorskip("rerun")
        path = tmp_path / "tilted.rrd"
        rec = rr.RecordingStream("test_app", recording_id="test_rec")
        rec.save(str(path))
        rec.log("world/leveled",
                rr.Transform3D(rotation=rr.Quaternion(
                    xyzw=self._quat_about_y(20.0))),
                static=True)
        rng = np.random.default_rng(0)
        gx, gy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
        flat = np.stack([gx.ravel(), gy.ravel(),
                         rng.normal(0, 0.02, gx.size)], axis=1)
        tilted = flat @ self._quat_to_R(self._quat_about_y(-7.0)).T
        rec.log("world/leveled/camera_init/colored_map",
                rr.Points3D(tilted), static=True)
        rec.flush()
        return str(path), "test_rec"

    def test_correction_levels_the_map(self, tilted_rrd):
        path, rec_id = tilted_rrd
        entity, level = RerunLogger._inspect_recording(path, rec_id)
        assert entity == "world/leveled/camera_init/annotated_frames"
        assert level is not None
        target, quat, trans = level
        # the override lands on the ancestor carrying the recorded transform
        assert target == "world/leveled"
        # applying the override INSTEAD of the recorded transform (static
        # last-wins) levels the cloud: its ground normal becomes +Z
        r_new = self._quat_to_R(quat)
        n_local = self._quat_to_R(self._quat_about_y(-7.0)) @ [0, 0, 1.0]
        n_disp = r_new @ n_local
        assert np.degrees(np.arccos(np.clip(n_disp[2], -1, 1))) < 0.5
        # and the map stays in place: the displayed position of the
        # ground slice's centroid is preserved by the correction
        rng = np.random.default_rng(0)
        gx, gy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
        flat = np.stack([gx.ravel(), gy.ravel(),
                         rng.normal(0, 0.02, gx.size)], axis=1)
        tilted = flat @ self._quat_to_R(self._quat_about_y(-7.0)).T
        low = tilted[tilted[:, 2] <= np.quantile(tilted[:, 2], 0.15)]
        c = low.mean(axis=0)
        r_old = self._quat_to_R(self._quat_about_y(20.0))
        np.testing.assert_allclose(r_new @ c + np.asarray(trans),
                                   r_old @ c, atol=0.05)

    def test_level_map_is_left_alone(self, tmp_path):
        rr = pytest.importorskip("rerun")
        path = tmp_path / "level.rrd"
        rec = rr.RecordingStream("test_app", recording_id="test_rec")
        rec.save(str(path))
        rng = np.random.default_rng(1)
        gx, gy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
        flat = np.stack([gx.ravel(), gy.ravel(),
                         rng.normal(0, 0.02, gx.size)], axis=1)
        rec.log("world/leveled/camera_init/colored_map",
                rr.Points3D(flat), static=True)
        rec.flush()
        _, level = RerunLogger._inspect_recording(str(path), "test_rec")
        assert level is None

    def test_tilt_without_recorded_transform_targets_cloud_parent(
            self, tmp_path):
        rr = pytest.importorskip("rerun")
        path = tmp_path / "bare.rrd"
        rec = rr.RecordingStream("test_app", recording_id="test_rec")
        rec.save(str(path))
        rng = np.random.default_rng(2)
        gx, gy = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
        flat = np.stack([gx.ravel(), gy.ravel(),
                         rng.normal(0, 0.02, gx.size)], axis=1)
        tilted = flat @ self._quat_to_R(self._quat_about_y(-12.0)).T
        rec.log("world/leveled/camera_init/colored_map",
                rr.Points3D(tilted), static=True)
        rec.flush()
        _, level = RerunLogger._inspect_recording(str(path), "test_rec")
        assert level is not None
        target, quat, _ = level
        assert target == "world/leveled/camera_init"
        n_disp = self._quat_to_R(quat) @ \
            (self._quat_to_R(self._quat_about_y(-12.0)) @ [0, 0, 1.0])
        assert np.degrees(np.arccos(np.clip(n_disp[2], -1, 1))) < 0.5

    def test_apply_logs_static_transform(self):
        pytest.importorskip("rerun")
        logger = RerunLogger()
        logger._level_transform = ("world/leveled", [0, 0, 0, 1], [0, 0, 0])
        logged = []
        logger._stream = types.SimpleNamespace(
            log=lambda *a, **k: logged.append((a, k)), flush=lambda: None)
        logger._apply_level_transform()
        (args, kwargs), = logged
        assert args[0] == "/world/leveled"
        assert kwargs.get("static") is True


class TestEmbeddedReopen:
    def test_embed_reopen_closes_external_viewer(self, monkeypatch):
        """Re-opening a recording with embed=True (the "Rerun map" panel
        was shown after the windowed viewer had already launched)
        terminates the windowed process so both views don't coexist."""
        import subprocess
        pytest.importorskip("rerun")
        logger = RerunLogger()
        spawned = []

        def fake_spawn(embed=False):
            proc = subprocess.Popen(["sleep", "30"])
            spawned.append(proc)
            if embed:
                logger._proc = proc
            else:
                logger._ext_proc = proc
            logger._port = 1
            return True

        monkeypatch.setattr(logger, "_spawn_viewer", fake_spawn)
        monkeypatch.setattr(logger, "_read_store_id", lambda p: ("app", None))
        monkeypatch.setattr(logger, "_find_marker_entity",
                            lambda p, r: MARKER_ENTITY)
        try:
            assert logger.open_recording("map.rrd", embed=False)
            ext = logger._ext_proc
            assert ext.poll() is None  # windowed viewer running
            assert logger.open_recording("map.rrd", embed=True)
            ext.wait(timeout=5)
            assert ext.returncode is not None  # windowed viewer terminated
        finally:
            logger.shutdown()
            for p in spawned:
                if p.poll() is None:
                    p.terminate()
