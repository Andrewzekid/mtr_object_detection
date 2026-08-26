"""Annotated-viewpoint markers on top of a provided Rerun (.rrd) recording.

The map (colored point cloud), images and timestamps live in an .rrd
produced elsewhere; this app only ADDS annotated-frame camera-position
markers. "File -> Open rerun file…" opens the recording in a windowed
viewer (which hosts a gRPC server); "🗺 Show annotated in Rerun" then
streams the markers into that SAME recording - the stream is initialized
with the .rrd's own application/recording id (read from the file), so the
viewer merges the markers into the loaded store instead of showing a
second recording. Markers are static points (always visible, independent
of the timeline) under

    world/map/annotated_frames

Marker positions come from a pose database (PoseDb in map_view.py), keyed
by each frame's ``timestamp_ns``.
"""

from __future__ import annotations

import socket
import subprocess
import time
from typing import List, Optional, Tuple

try:
    import rerun as rr
    _HAS_RERUN = True
except Exception:  # pragma: no cover - rerun is optional
    rr = None  # type: ignore[assignment]
    _HAS_RERUN = False

# Entity path of the markers inside the recording (sits next to the map,
# conventionally logged at world/map).
MARKER_ENTITY = "world/map/annotated_frames"


class RerunLogger:
    """Opens a provided .rrd in the viewer and streams markers into it.

    A no-op (``enabled`` False) when rerun-sdk is not installed, so call
    sites never need to guard for that.
    """

    def __init__(self):
        self.enabled = _HAS_RERUN
        self.rrd_path: Optional[str] = None
        self._stream = None
        self._port: Optional[int] = None  # set while connected to a viewer

    # ------------------------------------------------------------------ #

    def open_recording(self, rrd_path: str) -> bool:
        """Remember an .rrd as the marker target and open it in a windowed
        rerun viewer."""
        if not self.enabled:
            return False
        app_id, rec_id = self._read_store_id(rrd_path)
        try:
            self._stream = rr.RecordingStream(app_id, recording_id=rec_id)
        except Exception as exc:
            print(f"WARNING: could not init Rerun stream for {rrd_path}: {exc}")
            self._stream = None
            return False
        self.rrd_path = rrd_path
        self._port = None
        return self._spawn_viewer()

    def log_annotated_markers(self, markers: List[Tuple]) -> bool:
        """Replace the annotated-frame markers on the recording's map.

        ``markers`` is a list of ``(position, label)``; an empty list just
        clears the markers. Spawns the viewer first when not connected.
        """
        if not self.enabled or self._stream is None or not self.rrd_path:
            return False
        try:
            if self._port is None and not self._spawn_viewer():
                return False
            # Static clear + static points: previous markers disappear even
            # when frames got un-marked, and markers stay visible at every
            # timeline position.
            self._stream.log(MARKER_ENTITY, rr.Clear(recursive=False),
                             static=True)
            if markers:
                self._stream.log(
                    MARKER_ENTITY,
                    rr.Points3D([m[0] for m in markers], radii=0.15,
                                colors=[(255, 60, 60)] * len(markers),
                                labels=[m[1] for m in markers]),
                    static=True,
                )
            self._stream.flush()
            return True
        except Exception as exc:
            print(f"WARNING: Rerun marker logging failed: {exc}")
            self._port = None  # viewer likely closed — respawn next time
            return False

    # ------------------------------------------------------------------ #

    def _spawn_viewer(self) -> bool:
        """Launch ``rerun <path> --port <free>`` (windowed, with a gRPC
        server) and connect once the server accepts connections."""
        try:
            # Grab a free port instead of using the default (9876): with the
            # default port occupied, the CLI "helpfully" streams into the
            # already-running process instead of opening a new window.
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            subprocess.Popen(["rerun", self.rrd_path, "--port", str(port)])
            # flush() on an unconnected gRPC sink raises after ~6s, so wait
            # for the viewer's server to accept connections before
            # connecting (the viewer takes a moment to boot).
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), 0.5):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                print("WARNING: rerun viewer did not start listening")
                return False
            self._stream.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
            self._port = port
            return True
        except Exception as exc:
            print(f"WARNING: could not spawn the rerun viewer: {exc}")
            return False

    @staticmethod
    def _read_store_id(rrd_path: str) -> Tuple[str, Optional[str]]:
        """(application_id, recording_id) of the recording inside the .rrd,
        so streamed markers merge into the same store in the viewer."""
        try:
            import rerun_bindings
            entries = rerun_bindings.RrdReaderInternal(rrd_path) \
                .store_entries()
            for e in entries:
                if getattr(e, "application_id", None):
                    return e.application_id, getattr(e, "recording_id", None)
        except Exception as exc:
            print(f"WARNING: could not read the store id from {rrd_path}: "
                  f"{exc}")
        return "label_review", None
