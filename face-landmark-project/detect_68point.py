#!/usr/bin/env python3
"""
68-point facial landmark detection using dlib + OpenCV.

Implements the iBUG 300-W 68-point markup topology on top of dlib's HOG face
detector and `shape_predictor_68_face_landmarks.dat`:

    indices  0-16   jaw line      (open polyline)
    indices 17-21   right eyebrow
    indices 22-26   left eyebrow
    indices 27-35   nose
    indices 36-41   right eye     (closed polygon)
    indices 42-47   left eye      (closed polygon)
    indices 48-59   outer lip     (closed polygon)
    indices 60-67   inner lip     (closed polygon)

Each region is drawable independently and colour-coded, so the output is
readable at a glance rather than a uniform cloud of dots.

REQUIRED MODEL ASSET (NOT a pip package)
-----------------------------------------
This script needs `shape_predictor_68_face_landmarks.dat`, which is a separate
~100 MB download from the 5-point model. See `--help` or run the script with a
missing model to get copy-paste download instructions.

Usage
-----
    python detect_68point.py
    python detect_68point.py --camera 1 --upsample 1
    python detect_68point.py --save output.mp4 --indices
    python detect_68point.py --landmarks-json landmarks.jsonl

Press `q` (or ESC) in the video window to quit.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import dlib
import numpy as np

LOGGER = logging.getLogger("face68")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
MODEL_DOWNLOAD_URL = (
    "https://github.com/davisking/dlib-models/raw/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)
EXPECTED_NUM_PARTS = 68

Color = Tuple[int, int, int]  # OpenCV uses BGR, not RGB

# (name, start, end_exclusive, closed, BGR color)
#
# Ranges partition 0..67 exactly, so a landmark index maps to exactly one
# region for point colouring, while the polylines reuse the same slicing.
REGION_SPECS: Tuple[Tuple[str, int, int, bool, Color], ...] = (
    ("jaw",           0, 17, False, (0, 200, 255)),    # amber
    ("right_eyebrow", 17, 22, False, (255, 144, 30)),  # blue
    ("left_eyebrow",  22, 27, False, (255, 144, 30)),  # blue
    ("nose",          27, 36, False, (0, 255, 255)),   # yellow
    ("right_eye",     36, 42, True,  (60, 76, 231)),   # red
    ("left_eye",      42, 48, True,  (60, 76, 231)),   # red
    ("outer_lip",     48, 60, True,  (80, 220, 80)),   # green
    ("inner_lip",     60, 68, True,  (200, 80, 200)),  # violet
)

FACE_BOX_COLOR: Color = (0, 255, 0)
TEXT_COLOR: Color = (255, 255, 255)
TEXT_SHADOW: Color = (0, 0, 0)


# --------------------------------------------------------------------------- #
# Model / colour helpers
# --------------------------------------------------------------------------- #

def build_point_colors() -> List[Color]:
    """Return a length-68 list mapping each landmark index to its region color."""
    colors: List[Color] = [(255, 255, 255)] * EXPECTED_NUM_PARTS
    for _name, start, end, _closed, color in REGION_SPECS:
        for idx in range(start, end):
            colors[idx] = color
    return colors


def default_model_path() -> Path:
    """Resolve the model next to this script, independent of the CWD."""
    return Path(__file__).resolve().parent / MODEL_FILENAME


def model_missing_message(path: Path) -> str:
    """Build actionable download instructions for a missing model file."""
    return (
        f"Model file not found: {path}\n"
        f"This is a ~100 MB asset and is separate from the 5-point model.\n\n"
        f"Download and decompress it with:\n"
        f"  curl -L -o {MODEL_FILENAME}.bz2 {MODEL_DOWNLOAD_URL}\n"
        f"  python -c \"import bz2,shutil; shutil.copyfileobj("
        f"bz2.open('{MODEL_FILENAME}.bz2','rb'), open('{MODEL_FILENAME}','wb'))\"\n\n"
        f"Then place it beside this script, or point at it with --model PATH."
    )


def predictor_num_parts(predictor: "dlib.shape_predictor") -> Optional[int]:
    """Read `num_parts` whether the binding exposes it as a property or method."""
    value = getattr(predictor, "num_parts", None)
    if value is None:
        return None
    try:
        return int(value() if callable(value) else value)
    except Exception:  # pragma: no cover - defensive against binding changes
        return None


def load_predictor(model_path: Path) -> "dlib.shape_predictor":
    """Load and validate the 68-point shape predictor.

    Raises:
        FileNotFoundError: the model file does not exist.
        RuntimeError: dlib could not parse the file, or it is not a 68-point
            model (e.g. the 5-point predictor was passed by mistake).
    """
    if not model_path.is_file():
        raise FileNotFoundError(model_missing_message(model_path))

    try:
        predictor = dlib.shape_predictor(str(model_path))
    except Exception as exc:  # dlib raises RuntimeError on malformed files
        raise RuntimeError(
            f"dlib could not load '{model_path}': {exc}\n"
            "The file may be truncated or still bzip2-compressed. "
            "Confirm it is decompressed (it should be ~99.7 MB)."
        ) from exc

    parts = predictor_num_parts(predictor)
    if parts is not None and parts != EXPECTED_NUM_PARTS:
        raise RuntimeError(
            f"'{model_path.name}' predicts {parts} points, but this script "
            f"expects {EXPECTED_NUM_PARTS}.\n"
            "You probably passed the 5-point model "
            "(shape_predictor_5_face_landmarks.dat). "
            "Use detect_5point.py for that file instead."
        )

    return predictor


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def draw_regions(
    frame: np.ndarray,
    points: np.ndarray,
    point_colors: Sequence[Color],
    *,
    draw_points: bool = True,
    draw_indices: bool = False,
    line_thickness: int = 2,
    point_radius: int = 2,
) -> None:
    """Draw the 68-point markup onto `frame` in place.

    Args:
        frame: BGR image to annotate.
        points: (68, 2) integer array of landmark coordinates.
        point_colors: per-index BGR colors, length 68.
        draw_points: draw a dot at each landmark.
        draw_indices: annotate each landmark with its index (cluttered; opt-in).
    """
    for _name, start, end, closed, color in REGION_SPECS:
        segment = points[start:end]
        if len(segment) >= 2:
            cv2.polylines(
                frame,
                [segment.reshape(-1, 1, 2)],
                isClosed=closed,
                color=color,
                thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )

    if draw_points:
        for idx, (x, y) in enumerate(points):
            cv2.circle(
                frame,
                (int(x), int(y)),
                point_radius,
                point_colors[idx],
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    if draw_indices:
        for idx, (x, y) in enumerate(points):
            origin = (int(x) + 4, int(y) - 4)
            # Shadow first so labels stay legible over bright skin tones.
            cv2.putText(frame, str(idx), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, TEXT_SHADOW, 2, cv2.LINE_AA)
            cv2.putText(frame, str(idx), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render an FPS readout in the top-left corner."""
    label = f"FPS: {fps:5.1f}"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, TEXT_SHADOW, 4, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, TEXT_COLOR, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def detect_landmarks(
    frame: np.ndarray,
    detector: "dlib.fhog_object_detector",
    predictor: "dlib.shape_predictor",
    *,
    upsample: int = 1,
    process_width: Optional[int] = None,
) -> List[np.ndarray]:
    """Detect faces and return their 68 landmark points in `frame` coordinates.

    Detection runs on a grayscale (optionally downscaled) copy for speed, then
    the landmarks are scaled back into the original frame's coordinate space.

    Args:
        frame: BGR frame.
        detector: dlib frontal face detector.
        predictor: 68-point shape predictor.
        upsample: dlib upsample passes; 1 helps with small/distant faces.
        process_width: if set and the frame is wider, detection runs on a
            downscaled copy. Landmarks are still returned at full resolution.

    Returns:
        A list of (68, 2) int32 arrays.
    """
    scale = 1.0
    detect_image = frame
    if process_width is not None and frame.shape[1] > process_width:
        scale = process_width / float(frame.shape[1])
        detect_image = cv2.resize(
            frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)

    results: List[np.ndarray] = []
    for rect in detector(gray, upsample):
        shape = predictor(gray, rect)
        points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)
        if scale != 1.0:
            points /= scale
        results.append(points.astype(np.int32))

    return results


def fourcc(code: str) -> int:
    """Build a FOURCC int across OpenCV 4 and 5 bindings.

    OpenCV 5 moved the helper to `cv2.VideoWriter.fourcc`; the module-level
    `cv2.VideoWriter_fourcc` of OpenCV 4 is deprecated or absent there.
    """
    factory = getattr(cv2, "VideoWriter_fourcc", None)
    if factory is None:
        factory = cv2.VideoWriter.fourcc
    return int(factory(*code))


def make_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create a VideoWriter, preferring H.264 then falling back to mp4v."""
    for code in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), fourcc(code), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open a video writer for '{path}'. "
        "Try a different container/extension, or omit --save."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time 68-point facial landmark detection (dlib + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index. Try 1 or 2 for external webcams.")
    parser.add_argument("--width", type=int, default=0,
                        help="Requested capture width (0 = driver default)")
    parser.add_argument("--height", type=int, default=0,
                        help="Requested capture height (0 = driver default)")
    parser.add_argument("--upsample", type=int, default=1,
                        help="dlib upsample passes; raise for small faces")
    parser.add_argument("--process-width", type=int, default=None,
                        help="Downscale to this width for detection (faster)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable the mirrored (selfie) view")
    parser.add_argument("--hide-points", action="store_true",
                        help="Draw region outlines only, no landmark dots")
    parser.add_argument("--indices", action="store_true",
                        help="Annotate each landmark with its index")
    parser.add_argument("--hide-fps", action="store_true",
                        help="Hide the FPS readout")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write the annotated stream to this video file")
    parser.add_argument("--landmarks-json", type=Path, default=None,
                        help="Append detected landmarks per frame as JSON Lines")
    parser.add_argument("--max-read-failures", type=int, default=30,
                        help="Abort after this many consecutive dropped frames")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera and surface a diagnosis if it fails."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend) if backend != cv2.CAP_ANY \
        else cv2.VideoCapture(index)

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera index {index}.\n"
            "Check that: the webcam is not in use by another app; "
            "Windows camera privacy settings allow desktop apps; "
            "and try a different --camera index (0, 1, 2...)."
        )
    return cap


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    model_path = args.model if args.model is not None else default_model_path()

    LOGGER.info("Loading 68-point shape predictor from %s", model_path)
    try:
        predictor = load_predictor(model_path)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    detector = dlib.get_frontal_face_detector()
    point_colors = build_point_colors()

    LOGGER.info("Opening camera %d ...", args.camera)
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    writer: Optional[cv2.VideoWriter] = None
    json_file = None
    window = "68-Point Facial Landmark Detection"
    frame_index = 0
    read_failures = 0
    frame_times: collections.deque = collections.deque(maxlen=30)
    previous = time.perf_counter()

    try:
        ok, frame = cap.read()
        if not ok:
            LOGGER.error("Camera opened but returned no frames. Try another --camera index.")
            return 1

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        LOGGER.info("Camera stream started at %dx%d. Press 'q' to quit.", actual_w, actual_h)

        if args.save is not None:
            writer = make_writer(args.save, 30.0, (frame.shape[1], frame.shape[0]))
            LOGGER.info("Recording annotated stream to %s", args.save)

        if args.landmarks_json is not None:
            json_file = open(args.landmarks_json, "a", encoding="utf-8")
            LOGGER.info("Appending landmarks to %s", args.landmarks_json)

        while True:
            if not ok:
                read_failures += 1
                if read_failures >= args.max_read_failures:
                    LOGGER.error("Dropped %d consecutive frames; stopping.", read_failures)
                    break
                LOGGER.debug("Dropped frame (%d/%d)", read_failures, args.max_read_failures)
            else:
                read_failures = 0

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                faces = detect_landmarks(
                    frame, detector, predictor,
                    upsample=args.upsample,
                    process_width=args.process_width,
                )

                for points in faces:
                    # Bounding box derived from the landmark extents.
                    x0, y0 = points[:, 0].min(), points[:, 1].min()
                    x1, y1 = points[:, 0].max(), points[:, 1].max()
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                                  FACE_BOX_COLOR, 1, cv2.LINE_AA)

                    draw_regions(
                        frame, points, point_colors,
                        draw_points=not args.hide_points,
                        draw_indices=args.indices,
                    )

                if json_file is not None:
                    json_file.write(json.dumps({
                        "frame": frame_index,
                        "timestamp": time.time(),
                        "faces": [pts.tolist() for pts in faces],
                    }) + "\n")

            now = time.perf_counter()
            frame_times.append(now - previous)
            previous = now
            if not args.hide_fps:
                avg = sum(frame_times) / len(frame_times)
                draw_fps(frame, 1.0 / avg if avg > 0 else 0.0)

            cv2.imshow(window, frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break

            frame_index += 1
            ok, frame = cap.read()

        LOGGER.info("Processed %d frames. Shutting down.", frame_index)
        return 0

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if json_file is not None:
            json_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
