"""Annotated-viewpoint markers on top of a provided Rerun (.rrd) recording.

The map (colored point cloud), images and timestamps live in an .rrd
produced elsewhere; this app only ADDS annotated-frame camera-position
markers. "File -> Open rerun file…" opens the recording in a viewer —
a standalone windowed one by default, or a native viewer process whose
X11 window the GUI reparents into its in-app waypoint view (embed=True);
either viewer hosts a gRPC server. A blueprint is sent on connect so the
viewer opens on the 3D map (camera image/depth views left out). "🗺 Show
annotated in Rerun" then
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

Auto-levelling: some recordings carry a "leveled" ancestor transform that
doesn't actually leave the ground plane level in the viewer (e.g. written
by a newer rerun than the viewer, or an over-correcting SLAM gravity
estimate). On open, the map cloud is sampled, its ground plane fitted, and
a corrective static Transform3D is streamed onto the ancestor carrying the
recorded transform, so the map renders flat (+Z up). The .rrd on disk is
never modified.
"""

from __future__ import annotations

import math
import os
import re
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

# Auto-levelling: the map clouds can hold 100M+ points, so the ground-plane
# fit runs on a bounded sample — up to this many points per chunk, and per
# recording overall. No corrective transform is logged when the fitted
# ground is already within this many degrees of level.
_MAP_SAMPLE_PER_CHUNK = 20_000
_MAP_SAMPLE_MAX = 2_000_000
_LEVEL_MIN_TILT_DEG = 1.0


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
        # (entity, quat_xyzw, translation) corrective static transform to
        # level the map's ground plane, derived on open; None when the map
        # is already level or no ground plane could be fitted.
        self._level_transform: Optional[Tuple] = None
        # Embedded mode: X11 window id of the native viewer process we
        # spawned (set when opened with embed=True; the GUI reparents that
        # window into its waypoint view; the process is terminated via
        # shutdown()).
        self.win_id: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._embed = False  # respawn mode once the viewer drops away
        # Windowed viewer process we spawned (embed=False). Left running on
        # shutdown()/GUI close on purpose, but terminated when the same
        # recording is re-opened embedded so both views don't coexist.
        self._ext_proc: Optional[subprocess.Popen] = None
        # The `rerun` on PATH is a console script whose real viewer process
        # DETACHES (re-parented to the session leader), so terminating the
        # Popen handle kills an already-dead wrapper and leaks the viewer
        # (it re-emerges as a standalone window when its Qt container is
        # destroyed). We therefore track the real viewer pid, read from the
        # window's _NET_WM_PID property, and kill that.
        self._viewer_pid: Optional[int] = None      # embedded viewer
        self._ext_viewer_pid: Optional[int] = None  # windowed viewer
        # Incremented on every open_recording: X11 recycles window ids, so
        # a re-open can legitimately yield the SAME win_id for a NEW viewer
        # process — the GUI must compare (win_id, open_seq), not win_id
        # alone, to decide whether to rebuild its container.
        self.open_seq = 0

    # ------------------------------------------------------------------ #

    def open_recording(self, rrd_path: str, embed: bool = False) -> bool:
        """Remember an .rrd as the marker target and open it in a viewer.

        ``embed=True`` spawns the native viewer and sets ``self.win_id``
        to its X11 window id so the GUI can reparent the window into its
        waypoint view (``win_id`` stays None when the window couldn't be
        found — the viewer then simply stays a standalone window).
        """
        if not self.enabled:
            return False
        app_id, rec_id = self._read_store_id(rrd_path)
        self._marker_entity, self._level_transform = \
            self._inspect_recording(rrd_path, rec_id)
        try:
            self._stream = rr.RecordingStream(app_id, recording_id=rec_id)
        except Exception as exc:
            print(f"WARNING: could not init Rerun stream for {rrd_path}: {exc}")
            self._stream = None
            return False
        self.rrd_path = rrd_path
        self._port = None
        self.win_id = None
        self._embed = embed
        self.open_seq += 1
        self.shutdown()  # an earlier embedded viewer process is useless now
        if embed:
            # Moving the recording into the embedded view — close the
            # windowed viewer we spawned for it so both don't coexist.
            self._shutdown_external()
        return self._spawn_viewer(embed=embed)

    def shutdown(self) -> None:
        """Terminate the viewer process spawned for the embedded view
        (a windowed external viewer is left alone on purpose)."""
        proc, self._proc = self._proc, None
        self.win_id = None
        pid, self._viewer_pid = self._viewer_pid, None
        if pid is not None:
            self._kill_viewer_pid(pid)
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _shutdown_external(self) -> None:
        """Terminate the windowed viewer process we spawned earlier."""
        proc, self._ext_proc = self._ext_proc, None
        pid, self._ext_viewer_pid = self._ext_viewer_pid, None
        if pid is not None:
            self._kill_viewer_pid(pid)
        if proc is not None and proc.poll() is None:
            proc.terminate()

    @staticmethod
    def _kill_viewer_pid(pid: int) -> None:
        """SIGTERM the real viewer process (see ``_viewer_pid``).

        Safety-checked against /proc so a recycled pid that no longer
        belongs to a rerun binary is left alone.
        """
        import signal
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"rerun" not in fh.read():
                    return
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

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

        With ``embed=True`` the viewer's top-level X11 window is looked up
        (``self.win_id``) so the GUI can reparent it into its waypoint
        view. The window may flash standalone for a moment before the GUI
        grabs it; when the lookup fails the recording still opens as a
        normal external window (``win_id`` stays None).

        In both modes the real viewer pid is resolved from the window's
        _NET_WM_PID (the Popen handle only tracks the short-lived wrapper
        script — see ``_viewer_pid``), so shutdown can actually kill it.
        """
        try:
            # A respawn (marker logging after the connection dropped)
            # replaces any previous viewer of the same mode instead of
            # stacking a second window for the same recording.
            if embed:
                self.shutdown()
            else:
                self._shutdown_external()
            # Grab a free port instead of using the default (9876): with the
            # default port occupied, the CLI "helpfully" streams into the
            # already-running process instead of opening a new window.
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            before = self._x11_rerun_windows()
            proc = subprocess.Popen(["rerun", self.rrd_path,
                                     "--port", str(port)])
            if embed:
                # The embedded window dies with the GUI.
                self._proc = proc
            else:
                self._ext_proc = proc
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
                print(f"WARNING: rerun viewer did not start listening "
                      f"on port {port}")
                return False
            self._stream.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
            self._port = port
            if embed:
                self.win_id = self._wait_for_window(before)
                if self.win_id is None:
                    print("WARNING: could not find the rerun viewer's X11 "
                          "window; it stays a standalone window")
                else:
                    self._viewer_pid = self._window_pid(self.win_id)
            else:
                # Best-effort pid for _shutdown_external; a short timeout
                # keeps a windowless environment from stalling the open.
                ext_wid = self._wait_for_window(before, timeout=5.0)
                if ext_wid is not None:
                    self._ext_viewer_pid = self._window_pid(ext_wid)
            self._send_map_blueprint()
            self._apply_level_transform()
            return True
        except Exception as exc:
            print(f"WARNING: could not spawn the rerun viewer: {exc}")
            return False

    @staticmethod
    def _window_pid(wid: int) -> Optional[int]:
        """The owning process id of an X11 window (_NET_WM_PID)."""
        try:
            out = subprocess.run(["xprop", "-id", str(wid), "_NET_WM_PID"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            return None
        m = re.search(r"_NET_WM_PID\(CARDINAL\)\s*=\s*(\d+)", out)
        return int(m.group(1)) if m else None

    @staticmethod
    def _x11_rerun_windows() -> set:
        """Window ids of top-level X11 windows titled 'Rerun'."""
        try:
            out = subprocess.run(["xwininfo", "-root", "-children"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            return set()
        return {int(m.group(1), 16) for m in
                re.finditer(r'^\s*(0x[0-9a-fA-F]+)\s+"Rerun"', out,
                            re.MULTILINE)}

    def _wait_for_window(self, before: set, timeout: float = 20.0) \
            -> Optional[int]:
        """Poll for the spawned viewer's top-level window (a 'Rerun'-titled
        window that didn't exist before the spawn)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            new = self._x11_rerun_windows() - before
            if new:
                return new.pop()
            time.sleep(0.25)
        return None

    def _send_map_blueprint(self) -> None:
        """Default the viewer layout to a single 3D view of the map.

        Without a blueprint the viewer auto-generates views for every
        entity root — including 2D camera image/depth views. We instead
        send one Spatial3DView over the map point cloud's directory (the
        markers' parent), so the viewer opens on the 3D map, with the
        camera-body subtree (camera_left/right, axes, cloud_body scans)
        excluded. Users can add the camera views back from the viewer's
        blueprint menu.
        """
        try:
            import rerun.blueprint as rrb
            parent = self._marker_entity.strip("/").rpartition("/")[0]
            self._stream.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(
                name="Map", origin=f"/{parent}",
                # The camera-body subtree (camera_left/right, axes and the
                # per-frame cloud_body scans) is hidden by default; the
                # streams panel can re-include it.
                contents=["+ $origin/**", "- $origin/body/**"])))
        except Exception as exc:
            print(f"WARNING: could not send the map blueprint: {exc}")

    def _apply_level_transform(self) -> None:
        """Stream the auto-levelling corrective transform (when derived on
        open) so the map's ground plane renders level.

        Logged statically into the same store, it shadows a mis-recorded
        static transform on that entity (later static data wins), while the
        markers — children of that entity — rotate along with the map and
        stay glued to it. The .rrd file itself is never modified.
        """
        if self._level_transform is None or self._stream is None:
            return
        try:
            entity, quat, translation = self._level_transform
            self._stream.log(f"/{entity}",
                             rr.Transform3D(
                                 rotation=rr.Quaternion(xyzw=quat),
                                 translation=translation),
                             static=True)
            self._stream.flush()
            print(f"Rerun auto-level: flattened the map's ground plane "
                  f"via a corrective transform at '/{entity}'")
        except Exception as exc:
            print(f"WARNING: could not log the levelling transform: {exc}")

    @staticmethod
    def _find_marker_entity(rrd_path: str, rec_id: Optional[str]) -> str:
        """Entity path for the markers (see _inspect_recording)."""
        return RerunLogger._inspect_recording(rrd_path, rec_id)[0]

    @staticmethod
    def _inspect_recording(rrd_path: str, rec_id: Optional[str]) -> Tuple:
        """Single pass over the recording: derive the marker entity and the
        auto-levelling transform.

        Returns ``(marker_entity, level)``. ``marker_entity`` is a sibling
        of the recording's map point cloud, so markers inherit the same
        Transform3D chain (e.g. a levelling rotation on an ancestor) as the
        map itself. ``level`` is None, or ``(entity, quat_xyzw,
        translation)`` — the static Transform3D to log at ``entity`` (the
        deepest map ancestor already carrying a static transform) so the
        map's fitted ground plane renders level. The pass also samples the
        map clouds (decimated) for the ground fit.
        """
        try:
            import numpy as np
            import rerun_bindings
            reader = rerun_bindings.RrdReaderInternal(rrd_path)
            entry = next(
                (e for e in reader.store_entries()
                 if getattr(e, "application_id", None)
                 and getattr(e, "recording_id", None) == rec_id), None)
            if entry is None:
                return MARKER_ENTITY, None
            clouds = []
            samples = []
            static_tf = {}    # entity path -> (quat_xyzw, translation)
            temporal_tf = set()
            for chunk in reader.stream(store=entry):
                path = chunk.entity_path.strip("/")
                rb = chunk.to_record_batch()
                cols = {rb.schema.field(i).name: i
                        for i in range(rb.num_columns)}
                if any(c.startswith("Points3D:") for c in cols):
                    clouds.append(path)
                    if "map" in path.lower() and \
                            "Points3D:positions" in cols:
                        arr = rb.column(cols["Points3D:positions"])
                        if hasattr(arr, "combine_chunks"):
                            arr = arr.combine_chunks()
                        pts = np.asarray(
                            arr.flatten().values
                            .to_numpy(zero_copy_only=False),
                            dtype=np.float32).reshape(-1, 3)
                        stride = max(1, len(pts) // _MAP_SAMPLE_PER_CHUNK)
                        # .copy(): the strided view would otherwise keep
                        # the whole chunk buffer alive until concatenate.
                        samples.append(pts[::stride].copy())
                quat_i = cols.get("Transform3D:quaternion")
                trans_i = cols.get("Transform3D:translation")
                if quat_i is not None or trans_i is not None:
                    if not chunk.is_static:
                        temporal_tf.add(path)
                    else:
                        quat = trans = None
                        if quat_i is not None:
                            quat = [v for row in
                                    rb.column(quat_i).to_pylist()
                                    for v in row][0]
                        if trans_i is not None:
                            trans = [v for row in
                                     rb.column(trans_i).to_pylist()
                                     for v in row][0]
                        static_tf[path] = (quat, trans)
            if not clouds:
                return MARKER_ENTITY, None
            clouds.sort(key=lambda p: ("map" not in p.lower(), p))
            parent, _, _ = clouds[0].rpartition("/")
            entity = f"{parent}/annotated_frames" if parent else \
                MARKER_ENTITY
            if entity != MARKER_ENTITY:
                print(f"Rerun markers will be logged at '{entity}' "
                      f"(next to the map '{clouds[0]}')")
            level = None
            if samples:
                points = np.concatenate(samples, axis=0)
                if len(points) > _MAP_SAMPLE_MAX:
                    pick = np.linspace(0, len(points) - 1,
                                       _MAP_SAMPLE_MAX).astype(np.int64)
                    points = points[pick]
                level = RerunLogger._derive_level_transform(
                    points, clouds[0], static_tf, temporal_tf)
            return entity, level
        except Exception as exc:
            print(f"WARNING: could not inspect {rrd_path} for the map "
                  f"point cloud: {exc}")
            return MARKER_ENTITY, None

    @staticmethod
    def _derive_level_transform(points, cloud: str, static_tf: dict,
                                temporal_tf: set) -> Optional[Tuple]:
        """Fit the map cloud's ground plane and compute the corrective
        static transform levelling it.

        ``points`` is the decimated Nx3 cloud (in the cloud's frame);
        ``cloud`` its entity path; ``static_tf``/``temporal_tf`` the
        recorded transforms per entity. Returns ``(entity, quat_xyzw,
        translation)`` for the deepest ancestor of the cloud carrying a
        static transform (or the cloud's parent when the chain is
        transform-free), or None when the map is already level, the fit is
        degenerate, or a temporal transform on the chain would shadow a
        static correction.
        """
        import numpy as np

        def quat_to_R(q):
            x, y, z, w = q
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

        def R_to_quat(Rm):
            tr = float(np.trace(Rm))
            if tr > 0:
                s = math.sqrt(tr + 1.0) * 2
                return [(Rm[2, 1] - Rm[1, 2]) / s,
                        (Rm[0, 2] - Rm[2, 0]) / s,
                        (Rm[1, 0] - Rm[0, 1]) / s, 0.25 * s]
            i = int(np.argmax(np.diag(Rm)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = math.sqrt(max(1e-12, 1.0 + Rm[i, i] - Rm[j, j]
                              - Rm[k, k])) * 2
            xyz = [0.0, 0.0, 0.0]
            xyz[i] = 0.25 * s
            xyz[j] = (Rm[j, i] + Rm[i, j]) / s
            xyz[k] = (Rm[k, i] + Rm[i, k]) / s
            return [xyz[0], xyz[1], xyz[2], (Rm[k, j] - Rm[j, k]) / s]

        def compose(a, b):  # (R, t) pair a applied after b
            return a[0] @ b[0], a[0] @ b[1] + a[1]

        def invert(a):
            Rm = a[0].T
            return Rm, -Rm @ a[1]

        ident = (np.eye(3), np.zeros(3))

        def tf_of(path):
            quat, trans = static_tf.get(path, (None, None))
            return (quat_to_R(quat) if quat is not None else np.eye(3),
                    np.array(trans, dtype=float) if trans is not None
                    else np.zeros(3))

        # Robust ground fit: lowest slice of the cloud, PCA plane with
        # inlier refits.
        z = points[:, 2]
        low = points[z <= np.quantile(z, 0.15)]
        normal = np.array([0.0, 0.0, 1.0])
        centroid = low.mean(axis=0)
        cur = low
        for _ in range(4):
            centroid = cur.mean(axis=0)
            _, _, vt = np.linalg.svd(cur - centroid, full_matrices=False)
            normal = vt[-1]
            if normal[2] < 0:
                normal = -normal
            d = np.abs((low - centroid) @ normal)
            cur = low[d < 0.25]
        if len(cur) < 50:
            return None

        # Ancestor chain of the cloud, root first, the cloud itself last.
        parts = cloud.split("/")
        chain = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
        target = next((p for p in reversed(chain) if p in static_tf), None)
        if target is None:
            if any(p in temporal_tf for p in chain):
                print("WARNING: not auto-levelling: a temporal transform "
                      "on the map's transform chain would shadow a static "
                      "correction")
                return None
            target = chain[-2] if len(chain) >= 2 else chain[-1]

        def compose_range(paths):
            acc = ident
            for p in paths:
                acc = compose(acc, tf_of(p))
            return acc

        idx = chain.index(target)
        above = compose_range(chain[:idx])
        below = compose_range(chain[idx + 1:])
        full = compose_range(chain)

        n_disp = full[0] @ normal
        n_disp = n_disp / np.linalg.norm(n_disp)
        tilt = math.degrees(math.acos(float(np.clip(n_disp[2], -1, 1))))
        if tilt < _LEVEL_MIN_TILT_DEG:
            return None

        # Minimal rotation taking the displayed normal to +Z (no yaw).
        axis = np.cross(n_disp, [0.0, 0.0, 1.0])
        s = float(np.linalg.norm(axis))
        if s > 1e-9:
            k = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            r_corr = np.eye(3) + k + k @ k * ((1 - n_disp[2]) / (s * s))
        else:
            r_corr = np.eye(3)

        # Levelled total transform, rotating about the displayed ground
        # centroid so the map stays in place.
        c_disp = full[0] @ centroid + full[1]
        total = (r_corr @ full[0],
                 c_disp - r_corr @ c_disp + r_corr @ full[1])
        new = compose(invert(above), compose(total, invert(below)))
        return (target, [float(v) for v in R_to_quat(new[0])],
                [float(v) for v in new[1]])

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
