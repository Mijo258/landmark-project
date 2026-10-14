"""Eye landmark extraction and driver drowsiness detection.

This module implements the classic Eye Aspect Ratio (EAR) drowsiness detector
described by Soukupova and Cech (2016), *Real-Time Eye Blink Detection using
Facial Landmarks*, built on dlib's 68-point iBUG face landmark model and
OpenCV.

The public surface is a single importable class, :class:`DrowsinessDetector`,
which is deliberately free of any I/O policy: it never opens a camera, a
window or a topic. Feeding it frames is the caller's job, so the exact same
object can be driven by a webcam test harness (see :func:`main`) or wrapped in
a ROS 2 node with a ``sensor_msgs/Image`` subscription.

Example:
    Basic integration from a ROS 2 node callback::

        from eye_tracker import DrowsinessDetector

        detector = DrowsinessDetector(ear_threshold=0.25, consecutive_frames=20)

        def on_image(msg) -> None:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated, analysis = detector.analyze(frame, draw=False)
            if analysis.drowsy:
                publish_alert(analysis.ear)

Landmark indices (iBUG 68):
    * Left eye  (image-left,  subject's right): 36-41
    * Right eye (image-right, subject's left):  42-47
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, Sequence, Tuple, Union

import cv2
import dlib
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import euclidean

__all__ = [
    "DrowsinessDetector",
    "FrameAnalysis",
    "LEFT_EYE_INDICES",
    "RIGHT_EYE_INDICES",
    "DEFAULT_PREDICTOR_PATH",
]

#: dlib shape-predictor file expected next to this module by default.
DEFAULT_PREDICTOR_PATH: Final[Path] = Path(__file__).with_name(
    "shape_predictor_68_face_landmarks.dat"
)

#: iBUG 68 landmark indices for the left eye (subject's right).
LEFT_EYE_INDICES: Final[Tuple[int, ...]] = (36, 37, 38, 39, 40, 41)

#: iBUG 68 landmark indices for the right eye (subject's left).
RIGHT_EYE_INDICES: Final[Tuple[int, ...]] = (42, 43, 44, 45, 46, 47)

#: Either a BGR colour frame (H, W, 3) or an already-grayscale frame (H, W).
Frame = NDArray[np.uint8]

#: (6, 2) array of ``(x, y)`` pixel coordinates for one eye.
EyePoints = NDArray[np.float64]


@dataclass(frozen=True)
class FrameAnalysis:
    """Immutable result of analysing a single frame.

    Attributes:
        ear: Mean Eye Aspect Ratio of both eyes, or ``None`` when no face was
            found (the ratio is undefined without landmarks).
        face_detected: Whether the detector localized a face in this frame.
        num_faces: Number of faces returned by the detector (the largest is
            the one that was measured).
        frames_below_threshold: Running count of consecutive frames whose EAR
            fell below the configured threshold.
        drowsy: ``True`` while the drowsiness criterion is satisfied.
    """

    ear: Optional[float]
    face_detected: bool
    num_faces: int
    frames_below_threshold: int
    drowsy: bool


class DrowsinessDetector:
    """Detects driver drowsiness from eye closure measured over time.

    Eye closure is quantified per frame with the Eye Aspect Ratio:

    .. math::

        EAR = \\frac{\\|p_2 - p_6\\| + \\|p_3 - p_5\\|}{2 \\, \\|p_1 - p_4\\|}

    where ``p_1..p_6`` are the six landmarks of one eye ordered clockwise from
    the outer corner. The ratio is scale invariant because both numerator and
    denominator shrink with apparent face size, so a single threshold works at
    any distance from the camera.

    A frame is treated as "eyes closed" when the mean EAR of both eyes drops
    below ``ear_threshold``. Drowsiness is raised only after
    ``consecutive_frames`` such frames in a row, which rejects blinks (a blink
    lasts roughly 100-400 ms, i.e. ~3-12 frames at 30 FPS).

    Attributes:
        ear_threshold: EAR below which an eye is considered closed.
        consecutive_frames: Closed-eye frames required to raise an alert.

    Threading:
        Instances are **not** thread-safe: they keep per-stream state. Use one
        instance per camera or video stream.
    """

    def __init__(
        self,
        predictor_path: Union[str, Path] = DEFAULT_PREDICTOR_PATH,
        ear_threshold: float = 0.25,
        consecutive_frames: int = 20,
        detector_upsample: int = 1,
    ) -> None:
        """Loads the dlib models and configures the drowsiness criterion.

        Args:
            predictor_path: Path to dlib's
                ``shape_predictor_68_face_landmarks.dat`` file.
            ear_threshold: EAR value below which an eye counts as closed.
                Typical range 0.18-0.28; 0.25 is a safe general default.
            consecutive_frames: Number of consecutive closed-eye frames that
                must accumulate before :meth:`check_drowsiness` reports
                drowsiness.
            detector_upsample: Number of times the grayscale image is
                upsampled before HOG face detection. ``1`` finds small/distant
                faces at roughly 2x the cost; ``0`` is fastest.

        Raises:
            FileNotFoundError: If ``predictor_path`` does not exist.
            ValueError: If ``ear_threshold`` is outside ``(0, 1)``,
                ``consecutive_frames`` is not positive, or
                ``detector_upsample`` is negative.
        """
        if not 0.0 < ear_threshold < 1.0:
            raise ValueError(
                f"ear_threshold must be in (0, 1), got {ear_threshold!r}."
            )
        if consecutive_frames <= 0:
            raise ValueError(
                f"consecutive_frames must be > 0, got {consecutive_frames!r}."
            )
        if detector_upsample < 0:
            raise ValueError(
                f"detector_upsample must be >= 0, got {detector_upsample!r}."
            )

        resolved_path = Path(predictor_path).expanduser()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"dlib shape predictor not found: {resolved_path}\n"
                "Download it with:\n"
                "  curl -L -o shape_predictor_68_face_landmarks.dat.bz2 "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "  bunzip2 -f shape_predictor_68_face_landmarks.dat.bz2\n"
                "then place the .dat file next to eye_tracker.py, or pass its "
                "path via the `predictor_path` argument."
            )

        self.ear_threshold: float = float(ear_threshold)
        self.consecutive_frames: int = int(consecutive_frames)
        self.detector_upsample: int = int(detector_upsample)

        # HOG + linear SVM frontal face detector: CPU-only, ~30-60 FPS at VGA,
        # and far cheaper than dlib's CNN detector, which needs a GPU to keep
        # up with a live camera.
        self._detector: dlib.fhog_object_detector = dlib.get_frontal_face_detector()
        self._predictor: dlib.shape_predictor = dlib.shape_predictor(
            str(resolved_path)
        )

        # Rolling state for the temporal criterion.
        self._frames_below_threshold: int = 0
        self._last_ear: Optional[float] = None
        self._num_faces: int = 0

    # ------------------------------------------------------------------ #
    # Public state (read-only views for logging / ROS 2 diagnostics)
    # ------------------------------------------------------------------ #

    @property
    def ear(self) -> Optional[float]:
        """Latest measured Eye Aspect Ratio, or ``None`` if no face was found."""
        return self._last_ear

    @property
    def frames_below_threshold(self) -> int:
        """Number of consecutive frames currently below the EAR threshold."""
        return self._frames_below_threshold

    @property
    def num_faces(self) -> int:
        """Number of faces detected in the most recently processed frame."""
        return self._num_faces

    @property
    def is_drowsy(self) -> bool:
        """Whether the consecutive-frame criterion is currently satisfied."""
        return self._frames_below_threshold >= self.consecutive_frames

    def reset(self) -> None:
        """Clears all temporal state.

        Call this after a stream discontinuity (camera reconnect, topic gap,
        driver change) so stale closed-eye frames cannot trigger a spurious
        alert.
        """
        self._frames_below_threshold = 0
        self._last_ear = None
        self._num_faces = 0

    # ------------------------------------------------------------------ #
    # Core math
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_ear(eye_landmarks: Union[EyePoints, Sequence[Sequence[float]]]) -> float:
        """Computes the Eye Aspect Ratio for a single eye.

        Implements the Soukupova and Cech (2016) formula::

            EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

        with the six landmarks ordered clockwise starting at the outer corner
        (dlib returns them in exactly this order for indices 36-41 / 42-47).

        Args:
            eye_landmarks: Six ``(x, y)`` points, either a ``(6, 2)`` array or
                any sequence of six 2-element sequences. Coordinates may be
                floats or ints.

        Returns:
            The Eye Aspect Ratio as a float. Roughly 0.30 for a fully open eye
            and below ~0.20 for a closed one.

        Raises:
            ValueError: If the input cannot be interpreted as six 2D points.

        Note:
            The Euclidean distances use :func:`scipy.spatial.distance.euclidean`,
            which is a thin wrapper over ``numpy.linalg.norm``; the calls are
            kept here (rather than vectorised by hand) so the formula stays
            readable and directly comparable with the reference paper.
        """
        points = np.asarray(eye_landmarks, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        if points.shape != (6, 2):
            raise ValueError(
                f"expected 6 landmarks of shape (6, 2), got shape {points.shape!r}."
            )

        # Vertical distances: (p2, p6) and (p3, p5).
        vertical_a = euclidean(points[1], points[5])
        vertical_b = euclidean(points[2], points[4])
        # Horizontal distance: (p1, p4), the eye's full width.
        horizontal = euclidean(points[0], points[3])

        if horizontal <= 0.0:
            # Degenerate landmarks (all six points collapsed): both eyes shut
            # is the safe failure mode for a safety system.
            return 0.0

        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    # ------------------------------------------------------------------ #
    # Per-frame pipeline
    # ------------------------------------------------------------------ #

    def _eye_points(
        self, shape: dlib.full_object_detection, indices: Sequence[int]
    ) -> EyePoints:
        """Extracts the pixel coordinates of the requested landmark indices.

        Args:
            shape: The 68-point landmark set for one face.
            indices: Landmark indices to pull out, in the order they should
                appear in the returned array.

        Returns:
            A ``(len(indices), 2)`` float64 array of ``(x, y)`` coordinates.
        """
        return np.array(
            [(shape.part(i).x, shape.part(i).y) for i in indices], dtype=np.float64
        )

    @staticmethod
    def _largest_face(faces: dlib.rectangles) -> Optional[dlib.rectangle]:
        """Selects the largest face box from a detection set.

        With more than one face in frame (passenger, rear-seat, a face on a
        billboard), the driver is assumed to be the closest to the camera, i.e.
        the one with the greatest box area.

        Args:
            faces: Detection boxes returned by the dlib face detector.

        Returns:
            The box with the largest area, or ``None`` if ``faces`` is empty.
        """
        if len(faces) == 0:
            return None
        return max(faces, key=lambda rect: rect.area())

    def process_frame(
        self, frame: Frame, draw: bool = True
    ) -> Tuple[Frame, Optional[float]]:
        """Measures the mean EAR of the driver's eyes in one frame.

        Converts to grayscale, detects faces, keeps the largest one, regresses
        the 68 landmarks, and averages the EAR of both eyes.

        Args:
            frame: BGR image (``H x W x 3``, ``uint8``) straight from OpenCV, or
                an already-grayscale ``H x W`` image. Passed by reference; the
                array is annotated in place when ``draw`` is true, which is the
                intended zero-copy behaviour for a video pipeline.
            draw: When ``True``, overlays the eye contours (green while open,
                red while closed) and the face box on ``frame``.

        Returns:
            A ``(frame, ear)`` tuple, where ``ear`` is the mean EAR of both
            eyes, or ``None`` when no face was detected.

        Raises:
            ValueError: If ``frame`` is ``None``, empty, or is neither a 2D
                grayscale nor a 3D colour image.
        """
        if frame is None or frame.size == 0:
            raise ValueError("`frame` is empty; nothing to process.")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f"unsupported frame shape {frame.shape!r}; expected (H, W) or (H, W, 3)."
            )

        faces = self._detector(gray, self.detector_upsample)
        self._num_faces = len(faces)
        face = self._largest_face(faces)

        if face is None:
            # No face -> no landmarks -> EAR undefined. Report None rather than
            # a fake 0.0 so the caller can distinguish "eyes shut" from
            # "driver not visible", which need different responses.
            self._last_ear = None
            return frame, None

        shape = self._predictor(gray, face)
        left_eye = self._eye_points(shape, LEFT_EYE_INDICES)
        right_eye = self._eye_points(shape, RIGHT_EYE_INDICES)

        # Averaging both eyes cancels out asymmetric noise (head yaw, one eye
        # occluded by a hand) and halves the variance of the closure signal.
        ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
        self._last_ear = ear

        if draw:
            color = (0, 0, 255) if ear < self.ear_threshold else (0, 255, 0)
            self._draw_eye(frame, left_eye, color)
            self._draw_eye(frame, right_eye, color)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)

        return frame, ear

    @staticmethod
    def _draw_eye(frame: Frame, eye: EyePoints, color: Tuple[int, int, int]) -> None:
        """Draws a closed contour through one eye's six landmarks.

        Args:
            frame: Image to annotate in place.
            eye: ``(6, 2)`` array of the eye's landmark coordinates.
            color: BGR line colour.
        """
        contour = eye.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=1)

    # ------------------------------------------------------------------ #
    # Temporal criterion
    # ------------------------------------------------------------------ #

    def check_drowsiness(self, ear: Optional[float]) -> bool:
        """Updates the closed-eye frame counter and reports drowsiness.

        The counter increments on every frame with ``ear`` below
        ``ear_threshold`` and resets to zero on every frame at or above it. A
        missing measurement (``None``, e.g. the face left the frame) also
        resets the counter: an occlusion is not evidence of eye closure, and
        counting it would fire alerts whenever the driver looks away.

        Args:
            ear: Mean EAR from :meth:`process_frame`, or ``None`` when no face
                was detected.

        Returns:
            ``True`` once the counter reaches ``consecutive_frames``, i.e.
            while the drowsiness criterion is satisfied; ``False`` otherwise.
        """
        if ear is None or ear >= self.ear_threshold:
            self._frames_below_threshold = 0
            return False

        self._frames_below_threshold += 1
        return self._frames_below_threshold >= self.consecutive_frames

    def analyze(
        self, frame: Frame, draw: bool = True
    ) -> Tuple[Frame, FrameAnalysis]:
        """Runs the full per-frame pipeline and returns a typed result.

        Convenience wrapper combining :meth:`process_frame` and
        :meth:`check_drowsiness`; this is the method a ROS 2 node should call.

        Args:
            frame: BGR (or grayscale) image.
            draw: Whether to annotate ``frame`` with the eye contours.

        Returns:
            A ``(frame, analysis)`` tuple with the annotated frame and a
            :class:`FrameAnalysis` snapshot of the detector state.
        """
        annotated, ear = self.process_frame(frame, draw=draw)
        drowsy = self.check_drowsiness(ear)
        return annotated, FrameAnalysis(
            ear=ear,
            face_detected=ear is not None,
            num_faces=self._num_faces,
            frames_below_threshold=self._frames_below_threshold,
            drowsy=drowsy,
        )


# ---------------------------------------------------------------------- #
# Rendering helpers (test harness only)
# ---------------------------------------------------------------------- #

_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX


def _draw_centered_text(
    frame: Frame,
    text: str,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    y_ratio: float = 0.5,
) -> None:
    """Draws horizontally centred text with a black outline for legibility.

    Args:
        frame: Image to annotate in place.
        text: The string to render.
        font_scale: OpenCV font scale factor.
        color: BGR fill colour.
        thickness: Stroke thickness of the coloured text.
        y_ratio: Vertical position as a fraction of frame height.
    """
    text_size, _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x = max(0, (frame.shape[1] - text_size[0]) // 2)
    y = int(frame.shape[0] * y_ratio)
    origin = (x, y)
    # Outline first, then fill: keeps the text readable over any background.
    cv2.putText(frame, text, origin, _FONT, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for the standalone webcam harness.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Webcam demo for the DrowsinessDetector (press 'q' or Esc to quit)."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument(
        "--predictor",
        type=Path,
        default=DEFAULT_PREDICTOR_PATH,
        help=f"Path to shape_predictor_68_face_landmarks.dat (default: {DEFAULT_PREDICTOR_PATH.name}).",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.25,
        help="EAR below which the eyes count as closed (default: 0.25).",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=20,
        help="Closed-eye frames needed to raise the alert (default: 20).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the standalone webcam drowsiness monitor.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on a clean quit, ``1`` on setup failure.
    """
    args = _build_arg_parser().parse_args(argv)

    try:
        detector = DrowsinessDetector(
            predictor_path=args.predictor,
            ear_threshold=args.ear_threshold,
            consecutive_frames=args.consecutive_frames,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"[ERROR] Could not open camera index {args.camera}. "
            "It may be in use by another application, or the index is wrong.",
            file=sys.stderr,
        )
        capture.release()
        return 1

    print(
        f"[INFO] Monitoring started (EAR threshold={detector.ear_threshold}, "
        f"consecutive frames={detector.consecutive_frames}). Press 'q' to quit."
    )

    session_start = time.perf_counter()
    prev_time = session_start
    fps = 0.0
    frame_count = 0
    alert_count = 0
    read_failures = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                # Transient USB/decoder hiccups happen; only give up if the
                # stream stays dead, otherwise a single dropped frame would
                # kill the whole session.
                read_failures += 1
                if read_failures >= 30:
                    print("[ERROR] Camera stopped delivering frames; exiting.", file=sys.stderr)
                    break
                continue
            read_failures = 0
            frame_count += 1

            annotated, analysis = detector.analyze(frame, draw=True)

            now = time.perf_counter()
            delta = now - prev_time
            prev_time = now
            if delta > 0.0:
                instant_fps = 1.0 / delta
                # Exponential moving average: the raw per-frame value is too
                # jittery to read at 30 Hz.
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            # --- HUD: EAR and FPS ------------------------------------------
            ear_text = "EAR: n/a" if analysis.ear is None else f"EAR: {analysis.ear:.3f}"
            cv2.putText(annotated, ear_text, (10, 30), _FONT, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(
                annotated,
                ear_text,
                (10, 30),
                _FONT,
                0.7,
                (0, 0, 255) if analysis.drowsy else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(annotated, f"FPS: {fps:5.1f}", (10, 60), _FONT, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated, f"FPS: {fps:5.1f}", (10, 60), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"Closed frames: {analysis.frames_below_threshold}/{detector.consecutive_frames}",
                (10, 90),
                _FONT,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Closed frames: {analysis.frames_below_threshold}/{detector.consecutive_frames}",
                (10, 90),
                _FONT,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # --- Edge case: no face in frame -------------------------------
            if not analysis.face_detected:
                _draw_centered_text(
                    annotated, "NO FACE DETECTED", 0.9, (0, 255, 255), 2, y_ratio=0.5
                )

            # --- Drowsiness alert ------------------------------------------
            if analysis.drowsy:
                alert_count += 1
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (annotated.shape[1] - 1, annotated.shape[0] - 1),
                    (0, 0, 255),
                    8,
                )
                _draw_centered_text(
                    annotated,
                    "WARNING: DROWSINESS DETECTED",
                    font_scale=max(0.8, annotated.shape[1] / 900.0),
                    color=(0, 0, 255),
                    thickness=3,
                    y_ratio=0.85,
                )

            cv2.imshow("DMS - Eye Landmark & Drowsiness Tracker", annotated)

            # 0xFF mask: waitKey returns a full int on some platforms, and
            # ord('q') would never match without it.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # Always release the device and destroy windows, including on the
        # error paths above, so the camera is not left locked.
        capture.release()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - session_start, 1e-9)
    avg_fps = frame_count / elapsed
    print(
        f"[INFO] Session finished: {frame_count} frames in {elapsed:.1f}s "
        f"(~{avg_fps:.1f} FPS average), {alert_count} frames in the drowsy state."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
