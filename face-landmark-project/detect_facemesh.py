#!/usr/bin/env python3
"""
468 / 478-point face mesh detection using MediaPipe Face Landmarker.

This is the third markup in the project and it uses a DIFFERENT runtime from
the dlib scripts:

    detect_5point.py    dlib  ->  5 landmarks   (shape_predictor_5_face_landmarks.dat)
    detect_68point.py   dlib  -> 68 landmarks   (shape_predictor_68_face_landmarks.dat)
    detect_facemesh.py  MediaPipe -> 468 / 478 landmarks (face_landmarker.task)

Landmark layout (MediaPipe Face Mesh topology)
----------------------------------------------
    indices   0-467    face mesh: face oval, eyes, brows, nose, lips, cheeks
    indices 468-472    right iris ring (468 = centre, 469-472 = ring)
    indices 473-477    left  iris ring (473 = centre, 474-477 = ring)

So the model natively predicts 478 points. Pass `--mesh-only` to ignore the
final 10 iris points and work in the classic 468-point space.

REQUIRED MODEL ASSET (NOT a pip package)
----------------------------------------
`face_landmarker.task` (~3.7 MB), downloaded separately from the pip package.
Run with a missing model to print copy-paste download instructions.

Usage
-----
    python detect_facemesh.py                       # 478 points, webcam 0
    python detect_facemesh.py --mesh-only           # 468 points
    python detect_facemesh.py --image photo.jpg     # single still image
    python detect_facemesh.py --tesselation         # full mesh wireframe
    python detect_facemesh.py --save out.mp4 --landmarks-json lm.jsonl

Press `q` (or ESC) in the video window to quit.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe is an optional-at-import dependency so that `--help` still works
# and so the failure mode is an actionable message rather than a raw traceback.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on the environment
    mp = None
    mp_python = None
    vision = None
    MEDIAPIPE_IMPORT_ERROR = exc

LOGGER = logging.getLogger("facemesh")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_FILENAME = "face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

MESH_COUNT = 468
IRIS_COUNT = 478
IRIS_START = MESH_COUNT

Color = Tuple[int, int, int]  # OpenCV uses BGR

# --------------------------------------------------------------------------- #
# Face Mesh region topology
#
# These index sequences follow the published MediaPipe Face Mesh topology. They
# are curated closed/open loops rather than the full edge list, which keeps the
# output readable. Use --tesselation for the ground-truth full wireframe, which
# is read from MediaPipe's own connection tables when available.
# --------------------------------------------------------------------------- #

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
NOSE = [168, 6, 197, 195, 5, 4, 1, 2, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64, 98]
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]

# (name, indices, closed, BGR color, requires_iris)
REGION_SPECS: Tuple[Tuple[str, List[int], bool, Color, bool], ...] = (
    ("face_oval",     FACE_OVAL,     True,  (0, 200, 255),   False),  # amber
    ("right_eyebrow", RIGHT_EYEBROW, False, (255, 144, 30),  False),  # blue
    ("left_eyebrow",  LEFT_EYEBROW,  False, (255, 144, 30),  False),  # blue
    ("nose",          NOSE,          False, (0, 255, 255),   False),  # yellow
    ("right_eye",     RIGHT_EYE,     True,  (60, 76, 231),   False),  # red
    ("left_eye",      LEFT_EYE,      True,  (60, 76, 231),   False),  # red
    ("lips_outer",    LIPS_OUTER,    True,  (80, 220, 80),   False),  # green
    ("lips_inner",    LIPS_INNER,    True,  (200, 80, 200),  False),  # violet
    ("right_iris",    RIGHT_IRIS,    True,  (255, 255, 255), True),   # white
    ("left_iris",     LEFT_IRIS,     True,  (255, 255, 255), True),   # white
)

FACE_BOX_COLOR: Color = (0, 255, 0)
TEXT_COLOR: Color = (255, 255, 255)
TEXT_SHADOW: Color = (0, 0, 0)


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def default_model_path() -> Path:
    """Resolve the model next to this script, independent of the CWD."""
    return Path(__file__).resolve().parent / MODEL_FILENAME


def model_missing_message(path: Path) -> str:
    """Build actionable download instructions for a missing model file."""
    return (
        f"Model file not found: {path}\n"
        f"This is a ~3.7 MB asset, separate from the pip package.\n\n"
        f"Download it with (PowerShell):\n"
        f"  curl.exe -L -o {MODEL_FILENAME} {MODEL_DOWNLOAD_URL}\n\n"
        f"Or with Python:\n"
        f"  python -c \"import urllib.request; urllib.request.urlretrieve('"
        f"{MODEL_DOWNLOAD_URL}', '{MODEL_FILENAME}')\"\n\n"
        f"Then place it beside this script, or point at it with --model PATH."
    )


def mediapipe_missing_message() -> str:
    """Explain how to install MediaPipe when the import failed."""
    detail = f"\nUnderlying import error: {MEDIAPIPE_IMPORT_ERROR!r}" \
        if MEDIAPIPE_IMPORT_ERROR is not None else ""
    return (
        "MediaPipe is not importable in this interpreter.\n"
        "Install it with:\n"
        "  pip install -r requirements-facemesh.txt\n"
        "or:\n"
        "  pip install mediapipe\n"
        f"{detail}"
    )


def create_landmarker(
    model_path: Path, num_faces: int, running_mode: "vision.RunningMode"
) -> "vision.FaceLandmarker":
    """Create a MediaPipe FaceLandmarker.

    The running mode must match the call used later: IMAGE for one-shot stills
    via `detect()`, VIDEO for `detect_for_video()`. MediaPipe raises if they
    disagree, so the mode is chosen at creation time by the caller.

    Raises:
        FileNotFoundError: the .task bundle is absent.
        RuntimeError: MediaPipe could not load the bundle.
    """
    if not model_path.is_file():
        raise FileNotFoundError(model_missing_message(model_path))

    try:
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        raise RuntimeError(
            f"MediaPipe could not load '{model_path}': {exc}\n"
            "Confirm the file is a real .task bundle (~3.7 MB), not an HTML "
            "error page saved by a failed download."
        ) from exc


def load_tesselation_edges() -> Optional[frozenset]:
    """Return MediaPipe's own FACEMESH_TESSELATION edge set, or None.

    The connection tables live in different module paths across MediaPipe
    releases, and are absent in the Tasks-only builds, so this is best-effort.
    """
    for module_path in (
        "mediapipe.python.solutions.face_mesh_connections",
        "mediapipe.solutions.face_mesh_connections",
    ):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        edges = getattr(module, "FACEMESH_TESSELATION", None)
        if edges:
            return edges
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def draw_regions(
    frame: np.ndarray,
    points: np.ndarray,
    *,
    iris: bool,
    draw_points: bool = False,
    line_thickness: int = 1,
    point_radius: int = 1,
) -> None:
    """Draw the curated face-mesh regions onto `frame` in place.

    Args:
        frame: BGR image to annotate.
        points: (N, 2) integer array of landmark coordinates, N >= 468.
        iris: whether the iris points (468-477) are present in `points`.
        draw_points: draw a dot at every landmark (dense; opt-in).
    """
    count = len(points)
    for _name, indices, closed, color, needs_iris in REGION_SPECS:
        if needs_iris and not iris:
            continue
        if indices and max(indices) >= count:
            continue  # defensive: model returned fewer points than expected
        segment = points[indices].reshape(-1, 1, 2)
        cv2.polylines(frame, [segment], isClosed=closed, color=color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)

    if draw_points:
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), point_radius, (255, 255, 255),
                       thickness=-1, lineType=cv2.LINE_AA)


def draw_tesselation(
    frame: np.ndarray, points: np.ndarray, edges: frozenset, num_points: int
) -> None:
    """Draw the full mesh wireframe from MediaPipe's own edge table."""
    count = len(points)
    for a, b in edges:
        if a >= count or b >= count or a >= num_points or b >= num_points:
            continue
        cv2.line(frame, tuple(points[a]), tuple(points[b]),
                 (90, 90, 90), 1, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render an FPS readout in the top-left corner."""
    label = f"FPS: {fps:5.1f}"
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, TEXT_SHADOW, 4, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str) -> None:
    """Render a small informational label in the bottom-left corner."""
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, TEXT_SHADOW, 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, TEXT_COLOR, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def to_mp_image(frame_bgr: np.ndarray) -> "mp.Image":
    """Convert a BGR OpenCV frame into a contiguous MediaPipe SRGB image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb))


def landmarks_to_arrays(
    result: "vision.FaceLandmarkerResult", width: int, height: int,
    mesh_only: bool,
) -> List[np.ndarray]:
    """Convert normalised MediaPipe landmarks into pixel-space (N, 2) arrays.

    MediaPipe returns normalised [0, 1] coordinates; they are scaled to the
    frame's pixel grid. When `mesh_only` is set the iris points are dropped,
    yielding the classic 468-point layout.
    """
    faces: List[np.ndarray] = []
    for face in result.face_landmarks:
        points = np.array(
            [[lm.x * width, lm.y * height] for lm in face], dtype=np.float64
        )
        if mesh_only:
            points = points[:MESH_COUNT]
        faces.append(points.astype(np.int32))
    return faces


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="468/478-point face mesh detection (MediaPipe Face Landmarker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=None,
                        help=f"Path to {MODEL_FILENAME}")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index. Try 1 or 2 for external webcams.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Process a single still image instead of the webcam")
    parser.add_argument("--output-image", type=Path, default=None,
                        help="Where to write the annotated still image")
    parser.add_argument("--width", type=int, default=0,
                        help="Requested capture width (0 = driver default)")
    parser.add_argument("--height", type=int, default=0,
                        help="Requested capture height (0 = driver default)")
    parser.add_argument("--num-faces", type=int, default=1,
                        help="Maximum simultaneous faces to track")
    parser.add_argument("--mesh-only", action="store_true",
                        help="Drop the 10 iris points and work in 468-point space")
    parser.add_argument("--tesselation", action="store_true",
                        help="Draw the full mesh wireframe (slower, dense)")
    parser.add_argument("--hide-points", action="store_true",
                        help="Suppress the per-landmark dots")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable the mirrored (selfie) view")
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


def fourcc(code: str) -> int:
    """Build a FOURCC int across OpenCV 4 and 5 bindings."""
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
# Entry point
# --------------------------------------------------------------------------- #

def annotate(
    frame: np.ndarray,
    faces: Sequence[np.ndarray],
    *,
    iris: bool,
    tesselation_edges: Optional[frozenset],
    draw_tesselation: bool,
    draw_points: bool,
    mesh_limit: int,
) -> None:
    """Draw every detected face onto `frame` in place."""
    for points in faces:
        x0, y0 = points[:, 0].min(), points[:, 1].min()
        x1, y1 = points[:, 0].max(), points[:, 1].max()
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                      FACE_BOX_COLOR, 1, cv2.LINE_AA)

        if draw_tesselation and tesselation_edges:
            draw_tesselation(frame, points, tesselation_edges, mesh_limit)

        draw_regions(frame, points, iris=iris, draw_points=draw_points)


def run_image(
    args: argparse.Namespace, landmarker: "vision.FaceLandmarker",
    tesselation_edges: Optional[frozenset], mesh_limit: int,
) -> int:
    """Process a single still image and return an exit code."""
    frame = cv2.imread(str(args.image))
    if frame is None:
        LOGGER.error("Could not read image: %s", args.image)
        return 1

    height, width = frame.shape[:2]
    result = landmarker.detect(to_mp_image(frame))
    faces = landmarks_to_arrays(result, width, height, args.mesh_only)
    LOGGER.info("Detected %d face(s), %d points each.",
                len(faces), len(faces[0]) if faces else 0)

    annotate(frame, faces, iris=not args.mesh_only,
             tesselation_edges=tesselation_edges,
             draw_tesselation=args.tesselation,
             draw_points=not args.hide_points, mesh_limit=mesh_limit)

    out = args.output_image or args.image.with_name(
        f"{args.image.stem}_mesh{args.image.suffix}"
    )
    if not cv2.imwrite(str(out), frame):
        LOGGER.error("Could not write output image: %s", out)
        return 1
    LOGGER.info("Wrote annotated image to %s", out)
    return 0


def run_stream(
    args: argparse.Namespace, landmarker: "vision.FaceLandmarker",
    tesselation_edges: Optional[frozenset], mesh_limit: int,
) -> int:
    """Process the webcam stream and return an exit code."""
    LOGGER.info("Opening camera %d ...", args.camera)
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    writer: Optional[cv2.VideoWriter] = None
    json_file = None
    window = "468/478-Point Face Mesh Detection"
    frame_index = 0
    read_failures = 0
    frame_times: collections.deque = collections.deque(maxlen=30)
    previous = time.perf_counter()
    last_timestamp_ms = -1

    try:
        ok, frame = cap.read()
        if not ok:
            LOGGER.error("Camera opened but returned no frames. Try another --camera index.")
            return 1

        LOGGER.info("Camera stream started at %dx%d. Press 'q' to quit.",
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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

                height, width = frame.shape[:2]

                # VIDEO mode requires strictly increasing millisecond stamps.
                timestamp_ms = int(time.perf_counter() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    to_mp_image(frame), timestamp_ms
                )
                faces = landmarks_to_arrays(result, width, height, args.mesh_only)

                annotate(frame, faces, iris=not args.mesh_only,
                         tesselation_edges=tesselation_edges,
                         draw_tesselation=args.tesselation,
                         draw_points=not args.hide_points, mesh_limit=mesh_limit)

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

            draw_label(frame, "478-point" if not args.mesh_only else "468-point")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if mp is None:
        LOGGER.error("%s", mediapipe_missing_message())
        return 1

    model_path = args.model if args.model is not None else default_model_path()

    # Still images use IMAGE mode; the webcam stream uses VIDEO mode. The mode
    # is fixed when the landmarker is built, so pick it from the chosen input.
    running_mode = (
        vision.RunningMode.IMAGE if args.image is not None
        else vision.RunningMode.VIDEO
    )

    LOGGER.info("Loading MediaPipe Face Landmarker from %s", model_path)
    try:
        landmarker = create_landmarker(model_path, args.num_faces, running_mode)
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    tesselation_edges = load_tesselation_edges() if args.tesselation else None
    if args.tesselation and tesselation_edges is None:
        LOGGER.warning(
            "MediaPipe connection tables unavailable; --tesselation disabled. "
            "Region outlines will still be drawn."
        )
    mesh_limit = MESH_COUNT if args.mesh_only else IRIS_COUNT

    try:
        if args.image is not None:
            return run_image(args, landmarker, tesselation_edges, mesh_limit)
        return run_stream(args, landmarker, tesselation_edges, mesh_limit)
    finally:
        landmarker.close()


if __name__ == "__main__":
    sys.exit(main())
