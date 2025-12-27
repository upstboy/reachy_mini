"""Teleop Reachy Mini head with laptop face tracking (orientation only).

How it works (quick map for beginners):
1) A laptop webcam sees your face and MediaPipe finds face landmarks.
2) A small set of landmarks goes into solvePnP to estimate head pose (yaw/pitch/roll).
3) Press "s" to capture a neutral baseline so your natural head position becomes zero.
4) Each update, we apply gains, clamp angles, and smooth the motion to reduce jitter.
5) The robot head pose is sent directly, and a cube overlay shows the tracked pose.

Controls & tips:
- Keys: s=start calibration, p=pause, r=reset, d=toggle debug, q=quit.
- If motion feels reversed, try --mirror-pitch or --mirror-yaw.
- If the preview feels odd, try --no-flip-webcam.
- Increase --update-hz for snappier motion or lower it for stability.
"""

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import INIT_HEAD_POSE
from reachy_mini.utils import create_head_pose

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:  # Optional dependency
    mp = None
    python = None
    vision = None


@dataclass
class FaceResult:
    nx: float
    ny: float
    yaw: float
    pitch: float
    roll: float
    active: bool
    bbox: Tuple[int, int, int, int]
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    camera_matrix: Optional[np.ndarray]
    dist_coeffs: Optional[np.ndarray]


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent / ".models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"

# Key face points used to solve a simple 3D head pose.
LANDMARK_IDXS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye outer corner
        (43.3, 32.7, -26.0),      # Right eye outer corner
        (-28.9, -28.9, -24.1),    # Left mouth corner
        (28.9, -28.9, -24.1),     # Right mouth corner
    ],
    dtype=np.float32,
)


class LaptopCamera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(index)
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        if not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


def rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return yaw, pitch, roll


class FaceTracker:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.available = mp is not None and python is not None and vision is not None
        self.landmarker = None
        self.min_confidence = min_confidence
        if self.available:
            self.landmarker = self._create_landmarker()

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    def _create_landmarker(self) -> Optional[vision.FaceLandmarker]:
        if not self.available:
            return None
        model_path = self._ensure_model()
        if model_path is None:
            return None
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _ensure_model(self) -> Optional[str]:
        if MODEL_PATH.exists():
            return str(MODEL_PATH)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # The model is cached locally so it only downloads once.
        print("Downloading face landmarker model...")
        try:
            urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:
            print(f"Failed to download model: {exc}")
            return None
        return str(MODEL_PATH)

    def _landmark_to_point(self, lm, w: int, h: int) -> np.ndarray:
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def _solve_head_pose(
        self, landmarks: list, w: int, h: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        # Pick a few stable 2D points, then solve the 3D head pose with PnP.
        image_points = np.array(
            [
                self._landmark_to_point(landmarks[LANDMARK_IDXS["nose_tip"]], w, h),
                self._landmark_to_point(landmarks[LANDMARK_IDXS["chin"]], w, h),
                self._landmark_to_point(landmarks[LANDMARK_IDXS["left_eye_outer"]], w, h),
                self._landmark_to_point(landmarks[LANDMARK_IDXS["right_eye_outer"]], w, h),
                self._landmark_to_point(landmarks[LANDMARK_IDXS["mouth_left"]], w, h),
                self._landmark_to_point(landmarks[LANDMARK_IDXS["mouth_right"]], w, h),
            ],
            dtype=np.float32,
        )

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # rvec/tvec describe the 3D pose of the head relative to the camera.
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        return rvec, tvec, camera_matrix, dist_coeffs

    def estimate(self, frame: np.ndarray) -> Optional[FaceResult]:
        if self.landmarker is None:
            return None

        # Convert to MediaPipe's expected input format.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # Store the nose point in normalized coordinates for quick debug drawing.
        nx = float(np.clip(landmarks[LANDMARK_IDXS["nose_tip"]].x, 0.0, 1.0))
        ny = float(np.clip(landmarks[LANDMARK_IDXS["nose_tip"]].y, 0.0, 1.0))
        # Simple 2D face box for drawing a preview marker.
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = int(np.clip(min(xs) * w, 0, w - 1))
        y1 = int(np.clip(min(ys) * h, 0, h - 1))
        x2 = int(np.clip(max(xs) * w, 0, w - 1))
        y2 = int(np.clip(max(ys) * h, 0, h - 1))
        bbox = (x1, y1, x2, y2)

        # Estimate head pose from a few stable landmarks using solvePnP.
        pose = self._solve_head_pose(landmarks, w, h)
        if pose is None:
            return None
        rvec, tvec, camera_matrix, dist_coeffs = pose
        # Convert rotation vector to Euler angles (yaw/pitch/roll).
        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = rotation_matrix_to_euler(rmat)
        return FaceResult(
            nx=nx,
            ny=ny,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            active=True,
            bbox=bbox,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )


def draw_marker(frame: np.ndarray, x: int, y: int, active: bool) -> None:
    color = (0, 200, 0) if active else (0, 0, 200)
    cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)


def draw_head_cube(
    frame: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    if rvec is None or tvec is None or camera_matrix is None or dist_coeffs is None:
        return
    # Project a 3D cube into the 2D image using the head pose.
    size = 120.0
    half = size / 2.0
    cube_points = np.array(
        [
            (-half, -half, 0),
            (half, -half, 0),
            (half, half, 0),
            (-half, half, 0),
            (-half, -half, size),
            (half, -half, size),
            (half, half, size),
            (-half, half, size),
        ],
        dtype=np.float32,
    )

    img_pts, _ = cv2.projectPoints(
        cube_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    img_pts = img_pts.reshape(-1, 2).astype(int)
    color = (0, 200, 255)

    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[(i + 1) % 4]), color, 2)
    for i in range(4, 8):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[4 + (i + 1) % 4]), color, 2)
    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i + 4]), color, 2)

    eye_left = np.array([(-half / 2, -half / 4, 0)], dtype=np.float32)
    eye_right = np.array([(half / 2, -half / 4, 0)], dtype=np.float32)
    for eye in (eye_left, eye_right):
        eye_pts, _ = cv2.projectPoints(
            eye,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        eye_pt = tuple(eye_pts.reshape(-1, 2)[0].astype(int))
        cv2.circle(frame, eye_pt, 6, (0, 0, 255), -1)


def overlay_preview(base: np.ndarray, preview: np.ndarray, scale: float, margin: int) -> None:
    if scale <= 0.0:
        return
    target_w = int(base.shape[1] * scale)
    if target_w <= 0:
        return
    aspect = preview.shape[0] / preview.shape[1]
    target_h = int(target_w * aspect)
    if target_h <= 0:
        return
    resized = cv2.resize(preview, (target_w, target_h))
    y1 = margin
    y2 = min(base.shape[0], y1 + target_h)
    x2 = base.shape[1] - margin
    x1 = max(0, x2 - target_w)
    if y2 - y1 <= 0 or x2 - x1 <= 0:
        return
    base[y1:y2, x1:x2] = resized[: y2 - y1, : x2 - x1]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def ema_value(prev: float, current: float, alpha: float) -> float:
    # Simple exponential smoothing to reduce jitter.
    return (1.0 - alpha) * prev + alpha * current


def ema_vec(prev: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    # Vector version of EMA used for rvec/tvec smoothing.
    return (1.0 - alpha) * prev + alpha * current


def clamp_step(prev: float, current: float, max_step: float) -> float:
    # Limit how fast the target can change per update.
    return prev + clamp(current - prev, -max_step, max_step)


def main(args: argparse.Namespace) -> None:
    if mp is None:
        print("mediapipe is required. Please install mediapipe to use this script.")
        return
    if python is None or vision is None:
        print("mediapipe tasks are required. Please install mediapipe to use this script.")
        return

    cv2.namedWindow("Reachy Mini Face Teleop")
    debug = True
    control_state = {"enabled": False, "request_start": False}

    laptop = LaptopCamera(index=args.webcam, width=args.webcam_width, height=args.webcam_height)
    tracker = FaceTracker(min_confidence=args.face_confidence)

    last_command = 0.0
    last_active_time = 0.0
    last_filtered_rpy: Optional[Tuple[float, float, float]] = None
    last_pose_rvec: Optional[np.ndarray] = None
    last_pose_tvec: Optional[np.ndarray] = None
    baseline: Optional[FaceResult] = None
    baseline_samples: deque[FaceResult] = deque(maxlen=120)
    baseline_started: Optional[float] = None
    neutral_sent = False

    with ReachyMini(media_backend=args.backend) as reachy_mini:
        try:
            reachy_mini.goto_target(INIT_HEAD_POSE, antennas=[0.0, 0.0], duration=1.0)
            while True:
                # Grab frames from both cameras.
                reachy_frame = reachy_mini.media.get_frame()
                laptop_frame = laptop.read()

                if reachy_frame is None:
                    print("Failed to grab Reachy Mini frame.")
                    time.sleep(0.01)
                    continue
                if laptop_frame is None:
                    print("Failed to grab laptop webcam frame.")
                    time.sleep(0.01)
                    continue

                if args.flip_webcam:
                    laptop_frame = cv2.flip(laptop_frame, 1)

                # Detect head pose in the laptop webcam.
                face = tracker.estimate(laptop_frame)
                now = time.monotonic()
                face_active = face is not None
                if face_active:
                    last_active_time = now
                    neutral_sent = False

                # Start tracking on the next frame (after pressing "s").
                if control_state["request_start"]:
                    control_state["enabled"] = True
                    control_state["request_start"] = False
                    baseline = None
                    baseline_samples.clear()
                    baseline_started = None
                    last_filtered_rpy = None
                    last_pose_rvec = None
                    last_pose_tvec = None

                # Collect a short neutral baseline to define "zero" head pose.
                if control_state["enabled"] and baseline is None and face_active:
                    if baseline_started is None:
                        baseline_started = now
                    baseline_samples.append(face)
                    if (now - baseline_started) >= args.calib_seconds and len(baseline_samples) >= 5:
                        avg = np.mean(
                            np.array([[f.nx, f.ny, f.yaw, f.pitch, f.roll] for f in baseline_samples]),
                            axis=0,
                        )
                        baseline = FaceResult(
                            nx=float(avg[0]),
                            ny=float(avg[1]),
                            yaw=float(avg[2]),
                            pitch=float(avg[3]),
                            roll=float(avg[4]),
                            active=True,
                            bbox=(0, 0, 0, 0),
                            rvec=None,
                            tvec=None,
                            camera_matrix=None,
                            dist_coeffs=None,
                        )

                # Compute and send a new target at a fixed update rate.
                if (
                    control_state["enabled"]
                    and baseline is not None
                    and face_active
                    and (now - last_command) >= 1.0 / max(1.0, args.update_hz)
                ):
                    # Direct mode: map operator yaw/pitch/roll to robot head angles.
                    dyaw = face.yaw - baseline.yaw
                    dpitch = face.pitch - baseline.pitch
                    droll = face.roll - baseline.roll
                    if args.mirror_yaw:
                        dyaw = -dyaw
                    if args.mirror_pitch:
                        dpitch = -dpitch
                    if args.mirror_roll:
                        droll = -droll

                    tyaw = dyaw * args.orientation_gain_yaw
                    tpitch = dpitch * args.orientation_gain_pitch
                    troll = droll * args.orientation_gain_roll

                    max_yaw = math.radians(args.max_yaw_deg)
                    max_pitch = math.radians(args.max_pitch_deg)
                    max_roll = math.radians(args.max_roll_deg)
                    tyaw = clamp(tyaw, -max_yaw, max_yaw)
                    tpitch = clamp(tpitch, -max_pitch, max_pitch)
                    troll = clamp(troll, -max_roll, max_roll)

                    if last_filtered_rpy is not None and args.max_step_deg > 0.0:
                        max_step = math.radians(args.max_step_deg)
                        tyaw = clamp_step(last_filtered_rpy[0], tyaw, max_step)
                        tpitch = clamp_step(last_filtered_rpy[1], tpitch, max_step)
                        troll = clamp_step(last_filtered_rpy[2], troll, max_step)

                    # Smooth the motion to avoid jitter.
                    if last_filtered_rpy is None or args.smoothing <= 0.0:
                        fyaw, fpitch, froll = tyaw, tpitch, troll
                    else:
                        fyaw = ema_value(last_filtered_rpy[0], tyaw, args.smoothing)
                        fpitch = ema_value(last_filtered_rpy[1], tpitch, args.smoothing)
                        froll = ema_value(last_filtered_rpy[2], troll, args.smoothing)
                    last_filtered_rpy = (fyaw, fpitch, froll)

                    # Send the head pose directly.
                    head_pose = create_head_pose(
                        roll=froll,
                        pitch=fpitch,
                        yaw=fyaw,
                        degrees=False,
                        mm=False,
                    )
                    reachy_mini.set_target(head=head_pose)
                    last_command = now

                display_reachy = reachy_frame.copy()
                display_laptop = laptop_frame.copy()

                # Draw target markers and pose cube for the operator.
                if face is not None:
                    gx = int(face.nx * display_laptop.shape[1])
                    gy = int(face.ny * display_laptop.shape[0])
                    draw_marker(display_laptop, gx, gy, True)
                    if face.rvec is not None and face.tvec is not None:
                        if (
                            last_pose_rvec is None
                            or last_pose_tvec is None
                            or args.smoothing <= 0.0
                        ):
                            smoothed_rvec = face.rvec
                            smoothed_tvec = face.tvec
                        else:
                            smoothed_rvec = ema_vec(last_pose_rvec, face.rvec, args.smoothing)
                            smoothed_tvec = ema_vec(last_pose_tvec, face.tvec, args.smoothing)
                        last_pose_rvec = smoothed_rvec
                        last_pose_tvec = smoothed_tvec
                        draw_head_cube(
                            display_laptop,
                            smoothed_rvec,
                            smoothed_tvec,
                            face.camera_matrix,
                            face.dist_coeffs,
                        )

                if debug:
                    status = "tracking" if control_state["enabled"] else "paused"
                    face_status = "active" if face_active else "lost"
                    calib_status = "ready" if baseline is not None else "calibrating"
                    cv2.putText(
                        display_reachy,
                        f"Status: {status} | Face: {face_status} | Calib: {calib_status}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 0) if face_active else (0, 0, 200),
                        2,
                    )
                    if baseline is not None and face is not None:
                        if last_filtered_rpy is not None:
                            yaw_deg = math.degrees(last_filtered_rpy[0])
                            pitch_deg = math.degrees(last_filtered_rpy[1])
                            roll_deg = math.degrees(last_filtered_rpy[2])
                        cv2.putText(
                            display_reachy,
                            f"Head RPY (deg): {roll_deg:+.1f}  {pitch_deg:+.1f}  {yaw_deg:+.1f}",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                    cv2.putText(
                        display_reachy,
                        "Keys: s=start  p=pause  r=reset  d=debug  q=quit",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                overlay_preview(
                    display_reachy,
                    display_laptop,
                    scale=args.webcam_preview_scale,
                    margin=args.webcam_preview_margin,
                )
                cv2.imshow("Reachy Mini Face Teleop", display_reachy)

                # Handle keyboard input for quick control.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    debug = not debug
                if key == ord("s"):
                    control_state["request_start"] = True
                if key == ord("p"):
                    # Pause tracking and clear the baseline.
                    control_state["enabled"] = False
                    baseline = None
                    baseline_samples.clear()
                    baseline_started = None
                    last_filtered_rpy = None
                    last_pose_rvec = None
                    last_pose_tvec = None
                if key == ord("r"):
                    # Reset the robot to its neutral pose.
                    reachy_mini.goto_target(
                        INIT_HEAD_POSE,
                        antennas=[0.0, 0.0],
                        duration=0.6,
                    )
                    baseline = None
                    baseline_samples.clear()
                    baseline_started = None
                    last_filtered_rpy = None
                    last_pose_rvec = None
                    last_pose_tvec = None
                    neutral_sent = False

                # If the face is lost for too long, return to neutral.
                if not face_active and (now - last_active_time) > args.lost_timeout:
                    if control_state["enabled"] and not neutral_sent:
                        reachy_mini.goto_target(
                            INIT_HEAD_POSE,
                            antennas=[0.0, 0.0],
                            duration=0.6,
                        )
                        baseline = None
                        baseline_samples.clear()
                        baseline_started = None
                        last_filtered_rpy = None
                        last_pose_rvec = None
                        last_pose_tvec = None
                        neutral_sent = True
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            tracker.close()
            laptop.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Troubleshooting tips:
    # - Looking up/down feels reversed? Try --mirror-pitch (or --mirror-yaw).
    # - Motion is jittery? Increase --smoothing or lower --max-step-deg.
    # - No landmarks? Ensure good lighting and the model is downloaded in .models/.
    parser = argparse.ArgumentParser(
        description="Use laptop face tracking to drive Reachy Mini's head target."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer", "webrtc"],
        default="default",
        help="Media backend to use.",
    )
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--webcam-width", type=int, default=640, help="Webcam width.")
    parser.add_argument("--webcam-height", type=int, default=480, help="Webcam height.")
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.5,
        help="Minimum detection/tracking confidence for face landmarks.",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=15.0,
        help="Target update rate.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="EMA smoothing factor for pose and targets (0 disables).",
    )
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=6.0,
        help="Max per-update change in degrees for direct head control.",
    )
    parser.add_argument(
        "--orientation-gain-yaw",
        type=float,
        default=1.0,
        help="Scale operator yaw (radians) for direct head control.",
    )
    parser.add_argument(
        "--orientation-gain-pitch",
        type=float,
        default=1.0,
        help="Scale operator pitch (radians) for direct head control.",
    )
    parser.add_argument(
        "--orientation-gain-roll",
        type=float,
        default=1.0,
        help="Scale operator roll (radians) for direct head control.",
    )
    parser.add_argument(
        "--max-yaw-deg",
        type=float,
        default=60.0,
        help="Max absolute head yaw in degrees for direct control.",
    )
    parser.add_argument(
        "--max-pitch-deg",
        type=float,
        default=30.0,
        help="Max absolute head pitch in degrees for direct control.",
    )
    parser.add_argument(
        "--max-roll-deg",
        type=float,
        default=25.0,
        help="Max absolute head roll in degrees for direct control.",
    )
    parser.add_argument(
        "--mirror-yaw",
        action="store_true",
        help="Invert yaw direction (useful if control feels mirrored).",
    )
    parser.add_argument(
        "--mirror-pitch",
        action="store_true",
        help="Invert pitch direction (useful if control feels mirrored).",
    )
    parser.add_argument(
        "--mirror-roll",
        action="store_true",
        help="Invert roll direction (useful if control feels mirrored).",
    )
    parser.add_argument(
        "--calib-seconds",
        type=float,
        default=0.35,
        help="Seconds to average for neutral calibration.",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=1.5,
        help="Seconds until tracking loss triggers neutral pose.",
    )
    parser.add_argument(
        "--webcam-preview-scale",
        type=float,
        default=0.25,
        help="Preview size as a fraction of the Reachy frame width.",
    )
    parser.add_argument(
        "--webcam-preview-margin",
        type=int,
        default=10,
        help="Margin in pixels for the webcam preview.",
    )
    parser.add_argument(
        "--flip-webcam",
        action="store_true",
        default=True,
        help="Flip the webcam horizontally for mirror-like control.",
    )
    parser.add_argument(
        "--no-flip-webcam",
        action="store_false",
        dest="flip_webcam",
        help="Disable webcam flipping.",
    )
    parser.add_argument(
        "--aspect-compensate",
        action="store_true",
        help="Compensate for differing aspect ratios between cameras.",
    )

    main(parser.parse_args())
