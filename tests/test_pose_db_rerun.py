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
    con = sqlite3.connect(path)
    cols = ("id INTEGER PRIMARY KEY, timestamp_ns INTEGER, "
            "tf_translation_x REAL, tf_translation_y REAL, "
            "tf_translation_z REAL, cam_tf_translation_x REAL, "
            "cam_tf_translation_y REAL, cam_tf_translation_z REAL")
    if with_filename:
        cols += ", filename TEXT"
    con.execute(f"CREATE TABLE images ({cols})")
    rows = [
        # (timestamp_ns, cam xyz, lidar xyz, filename)
        (BASE_TS, (1.0, 2.0, 3.0), (9.0, 9.0, 9.0), "1.jpg"),
        (BASE_TS + 100_000_000, (4.0, 5.0, 6.0), (9.0, 9.0, 9.0), "2.jpg"),
        (BASE_TS + 200_000_000,
         (float("nan"), float("nan"), float("nan")),
         (7.0, 8.0, 9.0), "3.jpg"),
    ]
    for ts, cam, lidar, name in rows:
        if with_filename:
            con.execute(
                "INSERT INTO images (timestamp_ns, cam_tf_translation_x, "
                "cam_tf_translation_y, cam_tf_translation_z, "
                "tf_translation_x, tf_translation_y, tf_translation_z, "
                "filename) VALUES (?,?,?,?,?,?,?,?)",
                (ts, *cam, *lidar, name))
        else:
            con.execute(
                "INSERT INTO images (timestamp_ns, cam_tf_translation_x, "
                "cam_tf_translation_y, cam_tf_translation_z, "
                "tf_translation_x, tf_translation_y, tf_translation_z) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, *cam, *lidar))
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
        np.testing.assert_allclose(
            db.pose_for("2.jpg", BASE_TS + 100_000_000), [4.0, 5.0, 6.0])
        assert db.pose_for("2.jpg", 1000) is None


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
