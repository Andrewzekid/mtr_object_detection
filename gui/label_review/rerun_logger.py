"""Annotated-viewpoint markers on top of a provided Rerun (.rrd) recording.

The map (colored point cloud), images and timestamps live in an .rrd
produced elsewhere; this app only ADDS annotated-frame camera-position
markers. "File -> Open rerun file…" opens the recording in a viewer —
a windowed one by default, or a headless process hosting the web viewer
when the GUI embeds it in its "Rerun map" dock panel (embed=True); either
viewer hosts a gRPC server. "🗺 Show annotated in Rerun" then
streams the markers into that SAME recording - the stream is initialized
with the .rrd's own application/recording id (read from the file), so the
viewer merges the markers into the loaded store instead of showing a
second recording. Markers are static points (always visible, independent
of the timeline) under

    world/map/annotated_frames

(or, when the recording's map point cloud lives elsewhere, as a sibling of
that cloud - e.g. ``world/leveled/camera_init/annotated_frames`` - so the
markers inherit the same Transform3D chain as the map and land on it).
Marker positions come from a pose database (PoseDb in map_view.py), keyed
by each frame's ``timestamp_ns`` or matched by image filename.
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

# Default entity path of the markers inside the recording; the actual path
# is derived per recording (sibling of the map point cloud, see
# _find_marker_entity) and this is only the fallback.
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
        self._marker_entity = MARKER_ENTITY
        # Embedded mode: URL of the hosted web viewer (set when the
        # recording was opened with embed=True) and the headless viewer
        # process we spawned for it (terminated via shutdown()).
        self.web_url: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._embed = False  # respawn mode once the viewer drops away

    # ------------------------------------------------------------------ #

    def open_recording(self, rrd_path: str, embed: bool = False) -> bool:
        """Remember an .rrd as the marker target and open it in a viewer.

        ``embed=True`` spawns a headless viewer that only hosts the web
        viewer (for the GUI's embedded panel) instead of a native window.
        """
        if not self.enabled:
            return False
        app_id, rec_id = self._read_store_id(rrd_path)
        self._marker_entity = self._find_marker_entity(rrd_path, rec_id)
        try:
            self._stream = rr.RecordingStream(app_id, recording_id=rec_id)
        except Exception as exc:
            print(f"WARNING: could not init Rerun stream for {rrd_path}: {exc}")
            self._stream = None
            return False
        self.rrd_path = rrd_path
        self._port = None
        self.web_url = None
        self._embed = embed
        self.shutdown()  # an earlier embedded viewer process is useless now
        return self._spawn_viewer(embed=embed)

    def shutdown(self) -> None:
        """Terminate the headless viewer process spawned for the embedded
        panel (a windowed external viewer is left alone on purpose)."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def log_annotated_markers(self, markers: List[Tuple]) -> bool:
        """Replace the annotated-frame markers on the recording's map.

        ``markers`` is a list of ``(position, label)``; an empty list just
        clears the markers. Spawns the viewer first when not connected.
        """
        if not self.enabled or self._stream is None or not self.rrd_path:
            return False
        try:
            if self._port is None and not self._spawn_viewer(embed=self._embed):
                return False
            # Static clear + static points: previous markers disappear even
            # when frames got un-marked, and markers stay visible at every
            # timeline position.
            self._stream.log(self._marker_entity,
                             rr.Clear(recursive=False),
                             static=True)
            if markers:
                self._stream.log(
                    self._marker_entity,
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

    def _spawn_viewer(self, embed: bool = False) -> bool:
        """Launch ``rerun <path> --port <free>`` (windowed, with a gRPC
        server) and connect once the server accepts connections.

        With ``embed=True`` the viewer runs headless and additionally
        hosts the web viewer over HTTP (``--serve-web``); ``self.web_url``
        is then set for the GUI's embedded panel. A generous server memory
        limit keeps big maps from being dropped from the proxy buffer.
        """
        try:
            # Grab a free port instead of using the default (9876): with the
            # default port occupied, the CLI "helpfully" streams into the
            # already-running process instead of opening a new window.
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            args = ["rerun", self.rrd_path, "--port", str(port)]
            web_port = None
            if embed:
                with socket.socket() as s2:
                    s2.bind(("127.0.0.1", 0))
                    web_port = s2.getsockname()[1]
                args += ["--serve-web", "--web-viewer-port", str(web_port),
                         "--headless", "--server-memory-limit", "8GB"]
            proc = subprocess.Popen(args)
            if embed:
                # Headless helper only makes sense with the GUI around.
                self._proc = proc
            # flush() on an unconnected gRPC sink raises after ~6s, so wait
            # for the viewer's server to accept connections before
            # connecting (the viewer takes a moment to boot).
            ports = [port] + ([web_port] if embed else [])
            for p in ports:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        with socket.create_connection(("127.0.0.1", p), 0.5):
                            break
                    except OSError:
                        time.sleep(0.2)
                else:
                    print(f"WARNING: rerun viewer did not start listening "
                          f"on port {p}")
                    return False
            self._stream.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
            self._port = port
            self.web_url = (
                f"http://127.0.0.1:{web_port}"
                f"?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A{port}%2Fproxy"
                if embed else None)
            return True
        except Exception as exc:
            print(f"WARNING: could not spawn the rerun viewer: {exc}")
            return False

    @staticmethod
    def _find_marker_entity(rrd_path: str, rec_id: Optional[str]) -> str:
        """Entity path for the markers: sibling of the recording's map
        point cloud, so markers inherit the same Transform3D chain (e.g. a
        leveling rotation on an ancestor) as the map itself.

        Finds Points3D entities in the data store, prefers names containing
        "map" (the aggregated clouds rather than per-frame scans), and
        returns ``<parent>/annotated_frames``. Falls back to MARKER_ENTITY.
        """
        try:
            import rerun_bindings
            reader = rerun_bindings.RrdReaderInternal(rrd_path)
            entry = next(
                (e for e in reader.store_entries()
                 if getattr(e, "application_id", None)
                 and getattr(e, "recording_id", None) == rec_id), None)
            if entry is None:
                return MARKER_ENTITY
            clouds = []
            for chunk in reader.stream(store=entry):
                comps = {f.name for f in chunk.to_record_batch().schema}
                if any(c.startswith("Points3D:") for c in comps):
                    clouds.append(chunk.entity_path.strip("/"))
            if not clouds:
                return MARKER_ENTITY
            clouds.sort(key=lambda p: ("map" not in p.lower(), p))
            parent, _, _ = clouds[0].rpartition("/")
            entity = f"{parent}/annotated_frames" if parent else \
                MARKER_ENTITY
            if entity != MARKER_ENTITY:
                print(f"Rerun markers will be logged at '{entity}' "
                      f"(next to the map '{clouds[0]}')")
            return entity
        except Exception as exc:
            print(f"WARNING: could not inspect {rrd_path} for the map "
                  f"point cloud: {exc}")
            return MARKER_ENTITY

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
