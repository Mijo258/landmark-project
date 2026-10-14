"""Production 21-point hand landmark tracking for an autonomous-vehicle perception stack.

This module wraps the MediaPipe Tasks hand-landmarker graph in a small,
dependency-injectable object so it can be dropped into a larger ROS 2 node
without dragging the capture/visualisation loop along with it::

    from hand_tracker import HandDetector, HandLandmark

    detector = HandDetector(max_num_hands=2)
    while rclpy.ok():
        ok, frame = cap.read()
        hand = detector.get_landmark_positions(frame)   # None when no hand
        if hand is not None:
            angle = detector.calculate_angle(hand.get_normalized(HandLandmark.WRIST),
                                             hand.get_normalized(HandLandmark.INDEX_FINGER_MCP),
                                             hand.get_normalized(HandLandmark.INDEX_FINGER_TIP))

REQUIRES A MODEL FILE
    This uses the MediaPipe Tasks API (``mediapipe.tasks.python.vision``), which
    does NOT bundle its models inside the wheel the way the old
    ``mp.solutions.hands`` namespace did. You must download
    ``hand_landmarker.task`` once and keep it next to this module (or point
    ``model_path=`` at it)::

        curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

    Ship that file with the node: an autonomous platform should never fetch a
    model over the network at runtime.

Design notes for edge devices (Jetson Nano / Xavier NX):

* The BGR->RGB conversion is done once with ``cv2.cvtColor`` (SIMD-optimised)
  and wrapped in a ``mp.Image``, which is what the Tasks API consumes.
* Landmark extraction happens once per frame into a contiguous ``(21, 3)``
  float32 array; all downstream maths is vectorised.
* The skeleton is drawn with plain ``cv2`` primitives, avoiding a second pass
  over the landmark lists and the per-call drawing-spec machinery.
* ``delegate="gpu"`` selects the GPU delegate where the platform supports it.

Run ``python hand_tracker.py --help`` for the built-in webcam smoke test.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError(
        "mediapipe is required for hand_tracker: pip install mediapipe"
    ) from exc

try:
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError as exc:  # pragma: no cover - version guard
    raise ImportError(
        "hand_tracker requires the MediaPipe Tasks API (mediapipe.tasks.python), "
        f"which was not found in mediapipe {getattr(mp, '__version__', 'unknown')}. "
        "Install mediapipe>=0.10.9."
    ) from exc


LOGGER = logging.getLogger(__name__)

__all__ = [
    "HandDetector",
    "HandLandmark",
    "HandLandmarks",
    "FPSCounter",
    "HAND_CONNECTIONS",
    "NUM_LANDMARKS",
]

#: Number of landmarks MediaPipe Hands emits per hand.
NUM_LANDMARKS: int = 21

#: Official MediaPipe hand-landmarker model bundle (float16), used by the Tasks
#: API. Unlike the legacy ``solutions`` namespace, the Tasks API does NOT bundle
#: its models inside the wheel -- this file must be downloaded once and then
#: shipped alongside the node.
MODEL_DOWNLOAD_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

#: Default model location: next to this module, so a deployed node finds it
#: without any path configuration.
DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent / "hand_landmarker.task"

#: Bone connections of the 21-point topology, as index pairs. Hard-coded rather
#: than read from ``mp.solutions.hands.HAND_CONNECTIONS`` so this module keeps
#: working if the legacy ``solutions`` namespace is ever removed.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
)

#: Input types accepted by :meth:`HandDetector.calculate_angle`.
PointLike = Union[np.ndarray, Sequence[float], Sequence[int]]


class HandLandmark(IntEnum):
    """Canonical MediaPipe Hands landmark indices.

    Using the enum instead of magic numbers keeps gesture rules readable::

        if hand.get_normalized(HandLandmark.INDEX_FINGER_TIP)[1] < \\
           hand.get_normalized(HandLandmark.INDEX_FINGER_PIP)[1]:
            ...  # index finger raised (image y grows downwards)
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(eq=False)
class HandLandmarks:
    """Immutable snapshot of one detected hand for a single frame.

    Attributes:
        landmarks: ``(21, 3)`` float32 array of normalised ``(x, y, z)``
            coordinates. ``x`` and ``y`` are in ``[0, 1]`` relative to the image
            width/height; ``z`` is depth relative to the wrist, on roughly the
            same scale as ``x`` (smaller = closer to the camera).
        pixel_landmarks: ``(21, 2)`` int32 array of ``(x, y)`` pixels, already
            scaled to the frame the hand was detected in. Use this for drawing;
            use :attr:`landmarks` for anything that must be resolution
            independent (angles, gesture ratios, ROS 2 messages).
        world_landmarks: ``(21, 3)`` float32 array of metric ``(x, y, z)``
            coordinates in metres, origin at the hand's approximate geometric
            centre, or ``None`` if MediaPipe did not return a world solution.
        handedness: ``"Left"`` or ``"Right"`` as reported by the classifier.
            Note this is the *anatomical* label assuming a non-mirrored input
            image; it flips if you feed a horizontally flipped frame.
        handedness_score: Classifier confidence for :attr:`handedness` in
            ``[0, 1]``.
        image_shape: ``(height, width)`` of the frame the hand was found in.
    """

    landmarks: np.ndarray
    pixel_landmarks: np.ndarray
    world_landmarks: Optional[np.ndarray]
    handedness: str
    handedness_score: float
    image_shape: Tuple[int, int]

    def __len__(self) -> int:
        """Return the number of landmarks (always 21)."""
        return NUM_LANDMARKS

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the normalised ``(x, y, z)`` row at ``index``."""
        return self.get_normalized(index)

    def get_normalized(self, index: Union[int, HandLandmark]) -> Tuple[float, float, float]:
        """Return the normalised ``(x, y, z)`` tuple for one landmark.

        Args:
            index: Landmark index, ideally a :class:`HandLandmark` member.

        Returns:
            ``(x, y, z)`` as plain floats, safe to serialise into a ROS 2
            message or ``json.dumps``.

        Raises:
            IndexError: If ``index`` is outside ``[0, 20]``.
        """
        row = self.landmarks[self._checked_index(index)]
        return (float(row[0]), float(row[1]), float(row[2]))

    def get_pixel(self, index: Union[int, HandLandmark]) -> Tuple[int, int]:
        """Return the integer ``(x, y)`` pixel tuple for one landmark.

        Args:
            index: Landmark index, ideally a :class:`HandLandmark` member.

        Returns:
            ``(x, y)`` in pixels of the frame the hand was detected in.

        Raises:
            IndexError: If ``index`` is outside ``[0, 20]``.
        """
        row = self.pixel_landmarks[self._checked_index(index)]
        return (int(row[0]), int(row[1]))

    def as_dict(
        self,
        *,
        normalized: bool = True,
        include_depth: bool = True,
    ) -> Dict[str, Tuple[float, ...]]:
        """Return all 21 landmarks keyed by name, for logging or messaging.

        Args:
            normalized: ``True`` for the resolution-independent coordinates,
                ``False`` for pixel coordinates.
            include_depth: Keep the ``z`` component (ignored for pixel output,
                which is always 2D).

        Returns:
            Mapping of landmark name (e.g. ``"INDEX_FINGER_TIP"``) to its
            coordinate tuple, ordered from wrist to pinky tip.
        """
        out: Dict[str, Tuple[float, ...]] = {}
        for landmark in HandLandmark:
            if normalized:
                x, y, z = self.get_normalized(landmark)
                out[landmark.name] = (x, y, z) if include_depth else (x, y)
            else:
                px, py = self.get_pixel(landmark)
                out[landmark.name] = (float(px), float(py))
        return out

    def as_list(self, *, normalized: bool = True) -> List[Tuple[float, ...]]:
        """Return all 21 landmarks as a name-free list of coordinate tuples.

        Args:
            normalized: ``True`` for normalized ``(x, y, z)`` tuples, ``False``
                for pixel ``(x, y)`` tuples.

        Returns:
            List of length 21 indexed by landmark id.
        """
        if normalized:
            return [self.get_normalized(i) for i in range(NUM_LANDMARKS)]
        return [self.get_pixel(i) for i in range(NUM_LANDMARKS)]

    @staticmethod
    def _checked_index(index: Union[int, HandLandmark]) -> int:
        """Validate a landmark index.

        Args:
            index: Landmark index or :class:`HandLandmark` member.

        Returns:
            The index as a plain ``int``.

        Raises:
            IndexError: If ``index`` is outside ``[0, 20]``.
        """
        idx = int(index)
        if not 0 <= idx < NUM_LANDMARKS:
            raise IndexError(
                f"landmark index {idx} out of range [0, {NUM_LANDMARKS - 1}]"
            )
        return idx


class FPSCounter:
    """Rolling-window frames-per-second estimator.

    A windowed mean is used instead of an exponential moving average so a
    single stalled frame (a common thing on a Jetson when the thermal governor
    kicks in) does not visibly skew the reported number.

    Args:
        window: Number of frame timestamps to average over.

    Raises:
        ValueError: If ``window`` is smaller than 2.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._timestamps: Deque[float] = deque(maxlen=window)
        self.fps: float = 0.0

    def update(self) -> float:
        """Register a frame arrival and return the current FPS estimate.

        Returns:
            Frames per second over the rolling window, or ``0.0`` until at
            least two frames have been registered.
        """
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        self.fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0.0 else 0.0
        return self.fps


class _LandmarkListAdapter:
    """Wraps a Tasks API landmark list so it looks like a legacy LandmarkList.

    The legacy ``solutions`` API returned objects exposing ``.landmark``; the
    Tasks API returns plain Python lists. Normalising in one place keeps every
    downstream consumer -- :meth:`HandDetector._extract`, the skeleton renderer,
    and the test doubles in ``selftest.py`` -- working against a single shape.
    """

    __slots__ = ("landmark",)

    def __init__(self, landmark: Sequence[Any]) -> None:
        self.landmark = landmark


class _CategoryAdapter:
    """One classification result, in the legacy ``.label`` / ``.score`` shape."""

    __slots__ = ("label", "score")

    def __init__(self, label: str, score: float) -> None:
        self.label = label
        self.score = score


class _HandednessAdapter:
    """Legacy-shaped handedness entry exposing a ``.classification`` list."""

    __slots__ = ("classification",)

    def __init__(self, label: str, score: float) -> None:
        self.classification = [_CategoryAdapter(label, score)]


class _ResultsAdapter:
    """Presents Tasks API output in the legacy ``mp.solutions.hands`` shape.

    Exposes exactly the three attributes the rest of this module reads:
    ``multi_hand_landmarks``, ``multi_hand_world_landmarks`` and
    ``multi_handedness``.
    """

    __slots__ = (
        "multi_hand_landmarks",
        "multi_hand_world_landmarks",
        "multi_handedness",
    )

    def __init__(self, result: Any) -> None:
        self.multi_hand_landmarks = [
            _LandmarkListAdapter(hand) for hand in (result.hand_landmarks or [])
        ]

        world = getattr(result, "hand_world_landmarks", None)
        self.multi_hand_world_landmarks = (
            [_LandmarkListAdapter(hand) for hand in world] if world else None
        )

        handedness: List[_HandednessAdapter] = []
        for group in getattr(result, "handedness", None) or []:
            if group:
                top = group[0]
                label = getattr(top, "category_name", "") or getattr(
                    top, "display_name", ""
                )
                handedness.append(
                    _HandednessAdapter(str(label or "Unknown"), float(top.score))
                )
            else:
                handedness.append(_HandednessAdapter("Unknown", 0.0))
        self.multi_handedness = handedness or None


class HandDetector:
    """Thin, reusable wrapper around the MediaPipe Hands solution.

    The instance owns the MediaPipe graph and the most recent inference result.
    A typical consumer processes each frame once and then queries that result:

    ::

        detector.find_hands(frame, draw=True)          # infer + render
        hand = detector.get_landmark_positions()       # reuses last result

    Args:
        static_image_mode: ``True`` runs palm detection on every frame (use for
            still images or a batch pipeline). ``False`` runs palm detection
            only on the first frame and tracks afterwards, which is
            significantly cheaper and should be used for live video. When
            ``True``, ``min_tracking_confidence`` has no effect.
        max_num_hands: Maximum number of hands to detect.
        model_complexity: ``0`` for the lite landmark model (recommended for
            Jetson-class hardware), ``1`` for the full-accuracy model.
        min_detection_confidence: Palm-detection confidence threshold in
            ``[0, 1]``.
        min_tracking_confidence: Landmark-tracking confidence threshold in
            ``[0, 1]``; below this the graph falls back to palm detection.
        smooth_landmarks: Apply MediaPipe's temporal one-euro filter to the
            landmarks. Keep enabled for video; disable for static images.
        input_is_bgr: Set ``False`` if frames handed to :meth:`find_hands` are
            already RGB (e.g. from a ROS 2 ``sensor_msgs/Image`` after
            ``cv_bridge`` conversion), avoiding a redundant colour conversion.
        auto_flush: Release the MediaPipe graph via :meth:`close` when used as a
            context manager; see :meth:`__exit__`.

    Raises:
        ValueError: If any argument is out of its valid range.
        RuntimeError: If the MediaPipe ``solutions.hands`` API is unavailable in
            the installed mediapipe build.
    """

    #: Rendering style (BGR). Tuned once at class level so the per-frame draw
    #: path allocates nothing.
    _POINT_COLOR: Tuple[int, int, int] = (0, 255, 0)
    _CONNECTION_COLOR: Tuple[int, int, int] = (255, 255, 255)
    _TIP_COLOR: Tuple[int, int, int] = (0, 0, 255)
    _POINT_RADIUS: int = 3
    _TIP_RADIUS: int = 6
    _LINE_THICKNESS: int = 2
    #: ``cv2.LINE_8`` is materially cheaper than ``cv2.LINE_AA``; swap it if you
    #: have GPU/thermal headroom and want smoother bones.
    _LINE_TYPE: int = cv2.LINE_8

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        input_is_bgr: bool = True,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        delegate: str = "cpu",
    ) -> None:
        if max_num_hands < 1:
            raise ValueError(f"max_num_hands must be >= 1, got {max_num_hands}")
        if model_complexity not in (0, 1):
            raise ValueError(
                f"model_complexity must be 0 (lite) or 1 (full), got {model_complexity}"
            )
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if delegate not in ("cpu", "gpu"):
            raise ValueError(f"delegate must be 'cpu' or 'gpu', got {delegate!r}")

        self.static_image_mode: bool = bool(static_image_mode)
        self.max_num_hands: int = int(max_num_hands)
        self.model_complexity: int = int(model_complexity)
        self.min_detection_confidence: float = float(min_detection_confidence)
        self.min_tracking_confidence: float = float(min_tracking_confidence)
        self.smooth_landmarks: bool = bool(smooth_landmarks)
        self.input_is_bgr: bool = bool(input_is_bgr)
        self.model_path: Path = Path(model_path)
        self.delegate: str = delegate

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"hand landmarker model not found: {self.model_path}\n"
                "The MediaPipe Tasks API needs an explicit model bundle (the legacy "
                "solutions API bundled it, this one does not). Download it once with:\n"
                "  curl -L -o hand_landmarker.task "
                f"{MODEL_DOWNLOAD_URL}\n"
                "or pass model_path=... to point at an existing copy."
            )

        self._results: Optional[Any] = None
        self._closed: bool = False
        #: ``(height, width)`` of the most recent inference frame; keeps pixel
        #: scaling correct when a caller queries a cached result later.
        self._frame_shape: Tuple[int, int] = (1, 1)
        #: Tasks API VIDEO mode requires strictly increasing timestamps.
        self._last_timestamp_ms: int = -1
        #: Set by :meth:`_build_graph`; selects detect() vs detect_for_video().
        self._running_mode: Any = None
        self._hands = self._build_graph()

        LOGGER.debug(
            "HandDetector ready (max_hands=%d, static=%s, bgr_input=%s, delegate=%s, "
            "model=%s)",
            self.max_num_hands,
            self.static_image_mode,
            self.input_is_bgr,
            self.delegate,
            self.model_path.name,
        )
        LOGGER.debug(
            "model_complexity and smooth_landmarks are accepted for API "
            "compatibility but have no effect under the Tasks API; the model "
            "bundle embeds the network and VIDEO mode smooths/tracks internally."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        """Instantiate the MediaPipe Tasks hand-landmarker graph.

        Returns:
            The configured ``vision.HandLandmarker`` instance.

        Raises:
            RuntimeError: If the graph cannot be created (e.g. an unavailable
                GPU delegate, or a corrupt/incompatible model bundle).
        """
        base_options = BaseOptions(model_asset_path=str(self.model_path))
        if self.delegate == "gpu":
            # Worth trying on Jetson/Xavier. There is no silent fallback: an
            # unsupported delegate raises from create_from_options below.
            base_options.delegate = BaseOptions.Delegate.GPU

        # VIDEO mode tracks between frames (cheap, and what live capture wants);
        # IMAGE mode re-detects every frame and is required for stills/batches.
        running_mode = (
            vision.RunningMode.IMAGE
            if self.static_image_mode
            else vision.RunningMode.VIDEO
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._running_mode = running_mode
        return vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release the MediaPipe graph. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._results = None
        try:
            self._hands.close()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive teardown
            LOGGER.debug("MediaPipe graph close raised; ignoring", exc_info=True)

    def __enter__(self) -> "HandDetector":
        """Return self for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Release the graph on context exit."""
        self.close()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def results(self) -> Optional[Any]:
        """Raw MediaPipe result of the most recent :meth:`find_hands` call."""
        return self._results

    @property
    def num_hands_detected(self) -> int:
        """Number of hands found in the most recent frame."""
        landmarks = getattr(self._results, "multi_hand_landmarks", None)
        return len(landmarks) if landmarks else 0

    def find_hands(
        self,
        frame: Optional[np.ndarray],
        draw: bool = True,
    ) -> Optional[np.ndarray]:
        """Run hand inference on a frame and optionally render the skeleton.

        Args:
            frame: ``(H, W, 3)`` uint8 image. BGR by default; pass
                ``input_is_bgr=False`` to the constructor for RGB input.
            draw: Draw the 21-point skeleton and bone connections in place.

        Returns:
            The same ``frame`` object, annotated when ``draw`` is ``True`` and a
            hand was found, or ``None`` if an empty/``None`` frame was passed
            (which is logged and skipped rather than raised, so a flaky camera
            driver cannot take the node down). It is returned for convenient
            chaining; note the annotation mutates the input buffer, so copy
            first if the original frame feeds another consumer (e.g. a
            recording or a second model).

        Raises:
            ValueError: If ``frame`` is not a 3-channel uint8 image.
            RuntimeError: If the detector has already been closed.
        """
        if self._closed:
            raise RuntimeError("HandDetector is closed; construct a new instance")

        if frame is None or frame.size == 0:
            LOGGER.warning("find_hands received an empty frame; skipping inference")
            self._results = None
            return frame

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) frame, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 frame, got dtype {frame.dtype}")

        self._frame_shape = (int(frame.shape[0]), int(frame.shape[1]))

        # Colour conversion is done once here (cv2.cvtColor is SIMD-optimised),
        # then wrapped in a mediapipe Image, which is what the Tasks API takes.
        if self.input_is_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == vision.RunningMode.IMAGE:
            raw = self._hands.detect(mp_image)  # type: ignore[attr-defined]
        else:
            raw = self._hands.detect_for_video(  # type: ignore[attr-defined]
                mp_image, self._next_timestamp_ms()
            )

        self._results = _ResultsAdapter(raw)

        if draw and self._results.multi_hand_landmarks:
            self._draw_skeleton(frame)

        return frame

    def _next_timestamp_ms(self) -> int:
        """Return a strictly increasing millisecond timestamp for VIDEO mode.

        The Tasks API rejects a timestamp that does not advance, which happens
        whenever two frames land inside the same millisecond -- common at high
        frame rates, and guaranteed when replaying a recorded bag faster than
        real time.

        Returns:
            A monotonic timestamp in milliseconds, always greater than the one
            returned by the previous call.
        """
        now_ms = int(time.perf_counter() * 1000.0)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------
    def get_landmark_positions(
        self,
        frame: Optional[np.ndarray] = None,
        hand_index: int = 0,
        *,
        draw: bool = False,
    ) -> Optional[HandLandmarks]:
        """Return the 21 landmark positions for one detected hand.

        Args:
            frame: Frame to run inference on. Pass ``None`` (the default) to
                reuse the result of the most recent :meth:`find_hands` call,
                which avoids a second inference when one node both draws and
                consumes the landmarks.
            hand_index: Which detected hand to describe, ordered by detector
                confidence (``0`` is the most confident).
            draw: When ``frame`` is supplied, also render the skeleton.

        Returns:
            A :class:`HandLandmarks` snapshot, or ``None`` when no hand was
            detected -- callers must handle ``None``; nothing is raised.
            The returned object is a copy, so it stays valid after the next
            inference call.

        Raises:
            IndexError: If ``hand_index`` is outside the number of hands found
                in this frame.
        """
        if frame is not None:
            self.find_hands(frame, draw=draw)

        results = self._results
        hand_landmarks = getattr(results, "multi_hand_landmarks", None)
        if not hand_landmarks:
            return None

        count = len(hand_landmarks)
        if not 0 <= hand_index < count:
            raise IndexError(
                f"hand_index {hand_index} out of range: {count} hand(s) detected"
            )
        return self._extract(hand_index)

    def get_all_hand_landmarks(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        draw: bool = False,
    ) -> List[HandLandmarks]:
        """Return a snapshot for every detected hand.

        Args:
            frame: Frame to run inference on, or ``None`` to reuse the last
                :meth:`find_hands` result.
            draw: When ``frame`` is supplied, also render the skeletons.

        Returns:
            List of :class:`HandLandmarks`, ordered by detector confidence.
            Empty (never ``None``) when no hand was detected.
        """
        if frame is not None:
            self.find_hands(frame, draw=draw)

        hand_landmarks = getattr(self._results, "multi_hand_landmarks", None)
        if not hand_landmarks:
            return []
        return [self._extract(i) for i in range(len(hand_landmarks))]

    def _extract(self, hand_index: int) -> HandLandmarks:
        """Copy one hand out of the MediaPipe protobufs into numpy arrays.

        This is the only place the protobuf landmark objects are touched; doing
        it once per frame keeps the hot path out of Python attribute lookups.

        Args:
            hand_index: Index into the current result's hand list.

        Returns:
            The extracted :class:`HandLandmarks` snapshot.
        """
        results = self._results
        proto = results.multi_hand_landmarks[hand_index]  # type: ignore[attr-defined]

        # (21, 3) float32 in one comprehension -> one allocation, no per-point
        # Python objects kept alive.
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in proto.landmark], dtype=np.float32
        )

        height, width = self._frame_shape
        # Round rather than truncate so the drawn point sits on the true pixel.
        pixel_landmarks = np.rint(
            landmarks[:, :2] * np.array((width, height), dtype=np.float32)
        ).astype(np.int32)
        np.clip(pixel_landmarks[:, 0], 0, width - 1, out=pixel_landmarks[:, 0])
        np.clip(pixel_landmarks[:, 1], 0, height - 1, out=pixel_landmarks[:, 1])

        world_landmarks: Optional[np.ndarray] = None
        world_proto = getattr(results, "multi_hand_world_landmarks", None)
        if world_proto is not None and hand_index < len(world_proto):
            world_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in world_proto[hand_index].landmark],
                dtype=np.float32,
            )

        handedness = "Unknown"
        handedness_score = 0.0
        handedness_proto = getattr(results, "multi_handedness", None)
        if handedness_proto is not None and hand_index < len(handedness_proto):
            classification = handedness_proto[hand_index].classification
            if classification:
                handedness = str(classification[0].label)
                handedness_score = float(classification[0].score)

        return HandLandmarks(
            landmarks=landmarks,
            pixel_landmarks=pixel_landmarks,
            world_landmarks=world_landmarks,
            handedness=handedness,
            handedness_score=handedness_score,
            image_shape=(height, width),
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_angle(
        a: PointLike,
        b: PointLike,
        c: PointLike,
        *,
        use_z: bool = False,
        degrees: bool = True,
    ) -> float:
        """Return the interior angle at ``b`` formed by the points ``a-b-c``.

        Vectors ``b->a`` and ``b->c`` are formed and the angle recovered from
        the dot product, ``theta = acos( (u . v) / (|u| |v|) )``. The result is
        invariant to translation, rotation and scale, so it works equally well
        on normalized and pixel coordinates.

        Example -- index finger extension at the MCP joint::

            HandDetector.calculate_angle(
                hand.get_normalized(HandLandmark.WRIST),
                hand.get_normalized(HandLandmark.INDEX_FINGER_MCP),
                hand.get_normalized(HandLandmark.INDEX_FINGER_TIP),
            )   # ~180 deg when straight, ~30-60 deg when curled

        Args:
            a: First point, ``(x, y)`` or ``(x, y, z)``.
            b: Vertex point where the angle is measured.
            c: Third point.
            use_z: Include the ``z`` axis for a true 3D angle. MediaPipe's ``z``
                is a relative, noisy depth estimate, so 2D is usually more
                stable for gesture thresholds -- enable this only if you
                genuinely need out-of-plane angles.
            degrees: Return degrees when ``True``, radians otherwise.

        Returns:
            The interior angle in ``[0, 180]`` degrees (or ``[0, pi]`` radians),
            or ``nan`` if either vector has (near) zero length, i.e. the angle
            is geometrically undefined. ``nan`` is returned rather than a
            sentinel like ``0.0`` so a bad frame cannot silently be read as a
            confidently closed finger -- guard with ``math.isnan`` if your rules
            do not already compare against thresholds (any comparison with
            ``nan`` is ``False``).
        """
        dim = 3 if use_z else 2
        try:
            pa = np.asarray(a, dtype=np.float64).reshape(-1)[:dim]
            pb = np.asarray(b, dtype=np.float64).reshape(-1)[:dim]
            pc = np.asarray(c, dtype=np.float64).reshape(-1)[:dim]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"angle points must be numeric sequences, got {exc}") from exc

        if pa.size < dim or pb.size < dim or pc.size < dim:
            raise ValueError(
                f"each point needs at least {dim} components for use_z={use_z}"
            )

        ba = pa - pb
        bc = pc - pb
        norm_ba = float(np.linalg.norm(ba))
        norm_bc = float(np.linalg.norm(bc))
        if norm_ba < 1e-9 or norm_bc < 1e-9:
            return float("nan")

        cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
        # Clip before acos: float error can push a collinear triple to 1.0000001,
        # which would otherwise raise a domain warning and return nan.
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        return math.degrees(angle) if degrees else angle

    @classmethod
    def calculate_landmark_angle(
        cls,
        hand: HandLandmarks,
        a: Union[int, HandLandmark],
        b: Union[int, HandLandmark],
        c: Union[int, HandLandmark],
        *,
        use_z: bool = False,
        degrees: bool = True,
    ) -> float:
        """Convenience wrapper around :meth:`calculate_angle` for a hand snapshot.

        Args:
            hand: Landmarks returned by :meth:`get_landmark_positions`.
            a: Index of the first landmark.
            b: Index of the vertex landmark.
            c: Index of the third landmark.
            use_z: Include the ``z`` axis; see :meth:`calculate_angle`.
            degrees: Return degrees when ``True``.

        Returns:
            The interior angle at ``b``, or ``nan`` if it is undefined.

        Raises:
            IndexError: If any index is outside ``[0, 20]``.
        """
        return cls.calculate_angle(
            hand.get_normalized(a),
            hand.get_normalized(b),
            hand.get_normalized(c),
            use_z=use_z,
            degrees=degrees,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _draw_skeleton(self, frame: np.ndarray) -> None:
        """Draw all detected skeletons onto ``frame`` in place.

        Args:
            frame: BGR frame to annotate; mutated.
        """
        height, width = frame.shape[:2]
        for proto in self._results.multi_hand_landmarks:  # type: ignore[attr-defined]
            # Single vectorised conversion to pixel space, then integer indexing
            # inside the draw loops.
            points = np.rint(
                np.array([(lm.x, lm.y) for lm in proto.landmark], dtype=np.float32)
                * np.array((width, height), dtype=np.float32)
            ).astype(np.int32)

            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    (int(points[start, 0]), int(points[start, 1])),
                    (int(points[end, 0]), int(points[end, 1])),
                    self._CONNECTION_COLOR,
                    self._LINE_THICKNESS,
                    self._LINE_TYPE,
                )
            for index in range(points.shape[0]):
                cv2.circle(
                    frame,
                    (int(points[index, 0]), int(points[index, 1])),
                    self._POINT_RADIUS,
                    self._POINT_COLOR,
                    -1,
                    self._LINE_TYPE,
                )
            tip_x, tip_y = (int(points[HandLandmark.INDEX_FINGER_TIP, 0]),
                            int(points[HandLandmark.INDEX_FINGER_TIP, 1]))
            cv2.circle(frame, (tip_x, tip_y), self._TIP_RADIUS, self._TIP_COLOR, 2, self._LINE_TYPE)

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """``(height, width)`` of the frame passed to the last inference.

        Returns:
            Frame shape, falling back to ``(1, 1)`` before the first frame so
            pixel maths on a cached result cannot divide by zero.
        """
        return self._frame_shape


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the built-in webcam smoke test.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="21-point hand landmark tracker (webcam smoke test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    parser.add_argument("--width", type=int, default=640, help="requested capture width")
    parser.add_argument("--height", type=int, default=480, help="requested capture height")
    parser.add_argument("--max-hands", type=int, default=2, help="maximum hands to track")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="path to the hand_landmarker.task model bundle",
    )
    parser.add_argument(
        "--delegate",
        choices=("cpu", "gpu"),
        default="cpu",
        help="inference delegate; 'gpu' where the platform supports it",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="run palm detection every frame (still images / no tracking)",
    )
    parser.add_argument("--no-draw", action="store_true", help="skip skeleton rendering")
    parser.add_argument(
        "--print-hz",
        type=float,
        default=5.0,
        help="console print rate for the index tip; 0 prints every frame",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the webcam smoke test: capture, infer, annotate, print FPS.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on a clean quit, ``1`` if the camera could not
        be opened or stopped delivering frames.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cv2.setUseOptimized(True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        LOGGER.error(
            "could not open camera index %d. Check the device is connected and not "
            "held by another process, or pass --camera <index>.",
            args.camera,
        )
        cap.release()
        return 1

    # Requested, not guaranteed: drivers may hand back a different size, and
    # that is fine because MediaPipe works in normalised coordinates anyway.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    # A 1-frame buffer keeps latency low by preventing a backlog of stale frames
    # building up if inference is slower than capture.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    LOGGER.info(
        "camera %d open at %dx%d (max_hands=%d, delegate=%s, static=%s)",
        args.camera,
        actual_w,
        actual_h,
        args.max_hands,
        args.delegate,
        args.static,
    )

    fps_counter = FPSCounter(window=30)
    print_interval = 1.0 / args.print_hz if args.print_hz > 0 else 0.0
    last_print = 0.0
    consecutive_read_failures = 0
    max_read_failures = 30  # ~1 s of dropped frames before declaring a disconnect
    exit_code = 0
    quit_requested = False

    try:
        with HandDetector(
            static_image_mode=args.static,
            max_num_hands=args.max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_path=args.model_path,
            delegate=args.delegate,
        ) as detector:
            while not quit_requested:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    # Transient USB hiccups are common; only give up after a
                    # sustained run of failures so a single bad frame is not
                    # treated as a disconnect.
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= max_read_failures:
                        LOGGER.error(
                            "camera stopped delivering frames (%d consecutive read "
                            "failures); shutting down.",
                            consecutive_read_failures,
                        )
                        exit_code = 1
                        break
                    LOGGER.debug(
                        "empty frame (%d/%d); retrying",
                        consecutive_read_failures,
                        max_read_failures,
                    )
                    continue
                consecutive_read_failures = 0

                detector.find_hands(frame, draw=not args.no_draw)
                fps = fps_counter.update()
                hand = detector.get_landmark_positions()  # reuse this frame's result

                status_lines = [f"FPS: {fps:5.1f}"]
                if hand is None:
                    status_lines.append("Hand: none")
                else:
                    tip_x, tip_y = hand.get_pixel(HandLandmark.INDEX_FINGER_TIP)
                    status_lines.append(f"Hand: {hand.handedness} ({hand.handedness_score:.2f})")
                    status_lines.append(f"Index tip (8): x={tip_x} y={tip_y}")
                    now = time.perf_counter()
                    if now - last_print >= print_interval:
                        last_print = now
                        norm_x, norm_y, norm_z = hand.get_normalized(HandLandmark.INDEX_FINGER_TIP)
                        print(
                            f"[{hand.handedness}] landmark 8 (index tip) "
                            f"pixel=({tip_x}, {tip_y}) "
                            f"normalized=({norm_x:.4f}, {norm_y:.4f}, {norm_z:.4f})"
                        )

                for row, text in enumerate(status_lines):
                    cv2.putText(
                        frame,
                        text,
                        (10, 25 + row * 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Hand Tracker - 21 landmarks (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or ESC
                    quit_requested = True
    except FileNotFoundError as exc:
        # The Tasks API needs an explicit model bundle; the message already
        # carries the download command, so just surface it and exit.
        LOGGER.error("%s", exc)
        exit_code = 1
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user; shutting down.")
    finally:
        # Always release, even on an exception mid-loop, so the camera is not
        # left locked for the next process.
        cap.release()
        cv2.destroyAllWindows()

    LOGGER.info("clean shutdown.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
