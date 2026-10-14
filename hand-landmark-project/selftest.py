"""Offline correctness self-test for hand_tracker.py.

Run this *after* installing requirements.txt and *before* trusting the live
webcam demo. It verifies the parts of the module that a webcam cannot verify
objectively -- the angle maths and the landmark-to-pixel conversion -- against
known-good values, and it exercises the no-hand path without needing a hand in
front of the camera.

Usage:
    python selftest.py

Exits 0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hand_tracker import (
    HAND_CONNECTIONS,
    MODEL_DOWNLOAD_URL,
    NUM_LANDMARKS,
    FPSCounter,
    HandDetector,
    HandLandmark,
)

# --------------------------------------------------------------------------
# Minimal test harness -- no pytest dependency, so this runs on a bare venv.
# --------------------------------------------------------------------------

_FAILURES: List[str] = []
_CHECKS_RUN = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record and report one boolean check."""
    global _CHECKS_RUN
    _CHECKS_RUN += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(name)
        suffix = f"  ({detail})" if detail else ""
        print(f"  FAIL  {name}{suffix}")


def check_close(name: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    """Record and report one floating-point comparison."""
    ok = math.isnan(actual) and math.isnan(expected)
    if not ok:
        ok = abs(actual - expected) <= tol
    check(name, ok, f"expected {expected}, got {actual}")


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{title}")


# --------------------------------------------------------------------------
# Duck-typed stand-ins for the MediaPipe protobuf result objects, so the
# landmark-extraction path can be tested without running the neural network.
# --------------------------------------------------------------------------


class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class _FakeLandmarkList:
    def __init__(self, points: Sequence[Tuple[float, float, float]]) -> None:
        self.landmark = [_FakeLandmark(*p) for p in points]


class _FakeClassification:
    def __init__(self, label: str, score: float) -> None:
        self.label, self.score = label, score


class _FakeHandedness:
    def __init__(self, label: str, score: float) -> None:
        self.classification = [_FakeClassification(label, score)]


class _FakeResults:
    """Mimics the attributes hand_tracker reads off an mp Hands result."""

    def __init__(
        self,
        hands: Sequence[Sequence[Tuple[float, float, float]]],
        handedness: Optional[Sequence[Tuple[str, float]]] = None,
        world: Optional[Sequence[Sequence[Tuple[float, float, float]]]] = None,
    ) -> None:
        self.multi_hand_landmarks = [_FakeLandmarkList(h) for h in hands]
        self.multi_handedness = (
            [_FakeHandedness(label, score) for label, score in handedness]
            if handedness is not None
            else None
        )
        self.multi_hand_world_landmarks = (
            [_FakeLandmarkList(w) for w in world] if world is not None else None
        )


def _ramp_hand() -> List[Tuple[float, float, float]]:
    """Build 21 landmarks on a known ramp: point i = (i/20, i/40, 0)."""
    return [(i / 20.0, i / 40.0, 0.0) for i in range(NUM_LANDMARKS)]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_angle_known_geometry() -> None:
    """calculate_angle must reproduce hand-checkable angles exactly."""
    section("calculate_angle -- known geometry")
    angle = HandDetector.calculate_angle

    check_close("right angle (1,0)-(0,0)-(0,1) = 90", angle((1, 0), (0, 0), (0, 1)), 90.0)
    check_close("straight line = 180", angle((1, 0), (0, 0), (-1, 0)), 180.0)
    check_close("fully folded back = 0", angle((1, 0), (0, 0), (1, 0)), 0.0)
    check_close("45 degrees", angle((1, 0), (0, 0), (1, 1)), 45.0)
    check_close("120 degrees", angle((1, 0), (0, 0), (-0.5, math.sqrt(3) / 2)), 120.0)
    check_close(
        "radians mode returns pi/2",
        angle((1, 0), (0, 0), (0, 1), degrees=False),
        math.pi / 2,
    )
    check_close(
        "3D mode: 90 deg in the x-z plane",
        angle((1, 0, 0), (0, 0, 0), (0, 0, 1), use_z=True),
        90.0,
    )
    check_close(
        "2D mode ignores a large z on the same points (still 0)",
        angle((1, 0, 999), (0, 0, 0), (1, 0, -999), use_z=False),
        0.0,
    )
    # A collinear-but-opposite triple is the float-error worst case for acos:
    # the cosine lands on exactly -1.0 and rounding can push it past the domain,
    # which is what the clamp-before-acos guard exists for.
    check_close(
        "collinear points give exactly 180 (acos clamp path)",
        angle((1000, 1000), (0, 0), (-1000, -1000)),
        180.0,
    )
    check_close(
        "collinear points, same direction, give exactly 0",
        angle((1000, 1000), (0, 0), (1000, 1000)),
        0.0,
    )


def test_angle_degenerate_and_invariance() -> None:
    """Degenerate input returns nan; the angle ignores translation and scale."""
    section("calculate_angle -- degenerate input and invariance")

    check(
        "zero-length vector returns nan (not 0.0)",
        math.isnan(HandDetector.calculate_angle((5, 5), (5, 5), (9, 9))),
        "a coincident vertex must not read as a confidently closed finger",
    )
    check(
        "nan survives a threshold comparison as False",
        not (HandDetector.calculate_angle((5, 5), (5, 5), (9, 9)) > 150.0),
        "finger-straight checks must fail closed on a bad frame",
    )

    base = HandDetector.calculate_angle((1, 0), (0, 0), (0.3, 0.9))
    shifted = HandDetector.calculate_angle((101, 55), (100, 55), (100.3, 55.9))
    scaled = HandDetector.calculate_angle((1000, 0), (0, 0), (300, 900))
    check_close("translation invariant", shifted, base, tol=1e-9)
    check_close("scale invariant (normalized vs pixel coords agree)", scaled, base, tol=1e-9)

    # 'a' and 'c' swapped is the same interior angle.
    check_close(
        "argument order of a and c does not matter",
        HandDetector.calculate_angle((0, 1), (0, 0), (1, 0)),
        HandDetector.calculate_angle((1, 0), (0, 0), (0, 1)),
    )


def test_landmark_extraction_and_pixel_scaling(detector: HandDetector) -> None:
    """_extract must scale normalized coords to pixels with (h, w) in the right order."""
    section("landmark extraction -- pixel scaling")

    detector._results = _FakeResults([_ramp_hand()], handedness=[("Right", 0.98)])
    detector._frame_shape = (480, 640)  # (height, width)

    hand = detector.get_landmark_positions()
    check("a hand is returned", hand is not None)
    if hand is None:
        return

    check("len(hand) == 21", len(hand) == NUM_LANDMARKS)
    check("landmarks shape is (21, 3)", hand.landmarks.shape == (21, 3), str(hand.landmarks.shape))
    check("pixel_landmarks shape is (21, 2)", hand.pixel_landmarks.shape == (21, 2))
    check("image_shape reports (height, width)", hand.image_shape == (480, 640))

    # Point i = (i/20, i/40, 0) in a 640x480 frame -> (i/20*640, i/40*480).
    # x must scale by WIDTH and y by HEIGHT; swapping them is the classic bug.
    for idx in (0, 8):
        expected = (round(idx / 20.0 * 640), round(idx / 40.0 * 480))
        check(
            f"landmark {idx} pixel == {expected}",
            hand.get_pixel(idx) == expected,
            f"got {hand.get_pixel(idx)}",
        )

    # The ramp's final point sits exactly on x=1.0 / y=0.5, i.e. pixel 640 is one
    # past the last addressable column, so the clamp must pull it back to 639.
    check(
        "landmark 20 clamps x to width-1 (639) and keeps y=240",
        hand.get_pixel(20) == (639, 240),
        f"got {hand.get_pixel(20)}",
    )

    check_close("normalized x round-trips", hand.get_normalized(8)[0], 0.4, tol=1e-6)
    check_close("normalized y round-trips", hand.get_normalized(8)[1], 0.2, tol=1e-6)
    check("handedness is passed through", hand.handedness == "Right", hand.handedness)
    check_close("handedness score is passed through", hand.handedness_score, 0.98, tol=1e-6)

    # Index Finger Tip must be landmark 8, and the enum must agree with the id.
    check("INDEX_FINGER_TIP is index 8", int(HandLandmark.INDEX_FINGER_TIP) == 8)
    check(
        "get_pixel(HandLandmark.INDEX_FINGER_TIP) == get_pixel(8)",
        hand.get_pixel(HandLandmark.INDEX_FINGER_TIP) == hand.get_pixel(8),
    )


def test_pixel_clipping(detector: HandDetector) -> None:
    """Out-of-frame predictions must be clamped inside the image bounds."""
    section("landmark extraction -- out-of-frame clamping")

    wild = [(1.5, 1.5, 0.0)] * NUM_LANDMARKS
    detector._results = _FakeResults([wild])
    detector._frame_shape = (480, 640)
    hand = detector.get_landmark_positions()
    check("clamping: hand returned", hand is not None)
    if hand is None:
        return
    check("x clamped to width-1", hand.get_pixel(0)[0] == 639, str(hand.get_pixel(0)))
    check("y clamped to height-1", hand.get_pixel(0)[1] == 479, str(hand.get_pixel(0)))

    negative = [(-0.5, -0.5, 0.0)] * NUM_LANDMARKS
    detector._results = _FakeResults([negative])
    hand = detector.get_landmark_positions()
    check("negative coords clamp to 0", hand is not None and hand.get_pixel(0) == (0, 0))


def test_no_hand_edge_case(detector: HandDetector) -> None:
    """The no-hand path must return None / [] and never raise."""
    section("no-hand edge case")

    detector._results = _FakeResults([])
    detector._frame_shape = (480, 640)

    check("get_landmark_positions -> None", detector.get_landmark_positions() is None)
    check("get_all_hand_landmarks -> []", detector.get_all_hand_landmarks() == [])
    check("num_hands_detected -> 0", detector.num_hands_detected == 0)

    # Before any inference at all, the cached result is unset.
    fresh = HandDetector(max_num_hands=1, model_complexity=0)
    try:
        check("no inference yet -> None", fresh.get_landmark_positions() is None)
        check("no inference yet -> []", fresh.get_all_hand_landmarks() == [])
    finally:
        fresh.close()


def test_multiple_hands_and_errors(detector: HandDetector) -> None:
    """hand_index must select, and out-of-range access must raise IndexError."""
    section("multiple hands, index bounds, and error handling")

    detector._results = _FakeResults(
        [_ramp_hand(), [(0.1, 0.1, 0.0)] * NUM_LANDMARKS],
        handedness=[("Right", 0.9), ("Left", 0.8)],
        world=[[(0.0, 0.0, 0.0)] * NUM_LANDMARKS, [(1.0, 1.0, 1.0)] * NUM_LANDMARKS],
    )
    detector._frame_shape = (480, 640)

    check("two hands detected", detector.num_hands_detected == 2)

    first = detector.get_landmark_positions(0)
    second = detector.get_landmark_positions(1)
    if first is None or second is None:
        check("both hands extracted", False, "a fake hand came back as None")
        return

    check("hand_index=0 is the first hand", first.handedness == "Right", first.handedness)
    check("hand_index=1 is the second hand", second.handedness == "Left", second.handedness)
    check("get_all_hand_landmarks returns 2", len(detector.get_all_hand_landmarks()) == 2)

    world = second.world_landmarks
    check("world landmarks extracted", world is not None and world.shape == (21, 3))
    if world is not None:
        check_close("world landmark value", float(world[0][0]), 1.0)

    try:
        detector.get_landmark_positions(5)
        check("hand_index out of range raises IndexError", False, "no exception raised")
    except IndexError:
        check("hand_index out of range raises IndexError", True)

    for bad in (21, -1, 999):
        try:
            first.get_normalized(bad)
            check(f"get_normalized({bad}) raises IndexError", False, "no exception raised")
        except IndexError:
            check(f"get_normalized({bad}) raises IndexError", True)

    # Missing world/handedness must degrade gracefully, not crash.
    detector._results = _FakeResults([_ramp_hand()])
    sparse = detector.get_landmark_positions()
    if sparse is None:
        check("hand available for graceful-degradation tests", False)
        return
    check("missing world_landmarks -> None", sparse.world_landmarks is None)
    check("missing handedness -> 'Unknown'", sparse.handedness == "Unknown", sparse.handedness)
    check_close("missing handedness score -> 0.0", sparse.handedness_score, 0.0)


def test_structured_output(detector: HandDetector) -> None:
    """as_dict / as_list must expose all 21 landmarks consistently."""
    section("structured output")

    detector._results = _FakeResults([_ramp_hand()], handedness=[("Right", 0.9)])
    detector._frame_shape = (480, 640)
    hand = detector.get_landmark_positions()
    if hand is None:
        check("hand available for structured-output tests", False)
        return

    as_dict = hand.as_dict()
    check("as_dict has 21 entries", len(as_dict) == NUM_LANDMARKS, str(len(as_dict)))
    check("as_dict is keyed by landmark name", "INDEX_FINGER_TIP" in as_dict)
    check("as_dict values are 3-tuples", all(len(v) == 3 for v in as_dict.values()))
    check_close("as_dict tip x matches get_normalized", as_dict["INDEX_FINGER_TIP"][0], 0.4, tol=1e-6)
    check("as_dict(include_depth=False) gives 2-tuples",
          all(len(v) == 2 for v in hand.as_dict(include_depth=False).values()))

    as_list = hand.as_list()
    check("as_list has 21 entries", len(as_list) == NUM_LANDMARKS)
    check("as_list[8] matches get_normalized(8)", as_list[8] == hand.get_normalized(8))

    pixel_list = hand.as_list(normalized=False)
    check("pixel as_list[8] == (256, 96)", pixel_list[8] == (256.0, 96.0), str(pixel_list[8]))

    # The landmark-indexed convenience wrapper must agree with the raw helper.
    direct = HandDetector.calculate_angle(
        hand.get_normalized(HandLandmark.WRIST),
        hand.get_normalized(HandLandmark.INDEX_FINGER_MCP),
        hand.get_normalized(HandLandmark.INDEX_FINGER_TIP),
    )
    wrapped = HandDetector.calculate_landmark_angle(
        hand,
        HandLandmark.WRIST,
        HandLandmark.INDEX_FINGER_MCP,
        HandLandmark.INDEX_FINGER_TIP,
    )
    check_close("calculate_landmark_angle matches calculate_angle", wrapped, direct, tol=1e-12)


def test_fps_counter() -> None:
    """FPSCounter must report 0 before it has two samples, then a sane rate."""
    section("FPSCounter")

    try:
        FPSCounter(window=1)
        check("window < 2 raises ValueError", False, "no exception raised")
    except ValueError:
        check("window < 2 raises ValueError", True)

    counter = FPSCounter(window=30)
    check_close("first sample reports 0.0", counter.update(), 0.0)
    check("fps stays 0.0 after one sample", counter.fps == 0.0)

    # 20 back-to-back samples cannot produce a negative or absurd rate.
    for _ in range(20):
        fps = counter.update()
    check("fps is finite and positive after many fast samples", math.isfinite(fps) and fps > 0.0, str(fps))
    check("fps attribute mirrors the last update", counter.fps == fps)


def test_topology() -> None:
    """The hard-coded skeleton must be a valid 21-node, 21-edge tree plus one loop."""
    section("skeleton topology")

    check("21 connections defined", len(HAND_CONNECTIONS) == 21, str(len(HAND_CONNECTIONS)))
    check(
        "every connection references a valid landmark id",
        all(0 <= a < NUM_LANDMARKS and 0 <= b < NUM_LANDMARKS for a, b in HAND_CONNECTIONS),
    )
    check("no self-loops", all(a != b for a, b in HAND_CONNECTIONS))
    check(
        "no duplicate connections",
        len(set(HAND_CONNECTIONS)) == len(HAND_CONNECTIONS),
    )
    covered = {i for pair in HAND_CONNECTIONS for i in pair}
    check("all 21 landmarks are wired into the skeleton", covered == set(range(NUM_LANDMARKS)),
          f"uncovered: {sorted(set(range(NUM_LANDMARKS)) - covered)}")
    check("each finger tip is a leaf (appears once)",
          all(sum(1 for pair in HAND_CONNECTIONS if tip in pair) == 1
              for tip in (HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP,
                          HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP,
                          HandLandmark.PINKY_TIP)))


def test_input_validation(detector: HandDetector) -> None:
    """Bad frames must raise a clear ValueError, not a deep MediaPipe crash."""
    section("input validation and lifecycle")

    try:
        HandDetector(max_num_hands=0)
        check("max_num_hands=0 raises ValueError", False, "no exception raised")
    except ValueError:
        check("max_num_hands=0 raises ValueError", True)

    try:
        HandDetector(model_complexity=7)
        check("model_complexity=7 raises ValueError", False, "no exception raised")
    except ValueError:
        check("model_complexity=7 raises ValueError", True)

    try:
        HandDetector(min_detection_confidence=1.5)
        check("confidence > 1 raises ValueError", False, "no exception raised")
    except ValueError:
        check("confidence > 1 raises ValueError", True)

    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        detector.find_hands(np.zeros((480, 640), dtype=np.uint8))
        check("2D grayscale frame raises ValueError", False, "no exception raised")
    except ValueError:
        check("2D grayscale frame raises ValueError", True)

    try:
        detector.find_hands(np.zeros((480, 640, 3), dtype=np.float32))
        check("float32 frame raises ValueError", False, "no exception raised")
    except ValueError:
        check("float32 frame raises ValueError", True)

    result = detector.find_hands(blank, draw=True)
    check("empty frame is handled without raising", result is None or result.shape == (480, 640, 3))
    check("blank frame yields no hand", detector.get_landmark_positions() is None)


def test_real_graph_blank_frame(detector: HandDetector) -> None:
    """End-to-end run of the actual MediaPipe graph on a synthetic frame."""
    section("integration -- real MediaPipe graph on a blank frame")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    before = blank.copy()

    returned = detector.find_hands(blank, draw=True)
    check("find_hands returns the same buffer object", returned is blank)
    check("frame is still uint8 BGR (H, W, 3)", blank.dtype == np.uint8 and blank.shape == (480, 640, 3))
    check("no hand hallucinated on a black frame", detector.get_landmark_positions() is None)
    check("drawing was a no-op when nothing was detected", np.array_equal(blank, before))

    check("frame_shape recorded as (height, width)", detector.frame_shape == (480, 640),
          str(detector.frame_shape))

    # A real hand-shaped blob still will not be detected, but must not raise.
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    detector.find_hands(noise, draw=True)
    check("random noise frame does not raise", True)

    detector.close()
    try:
        detector.close()
        check("close() is idempotent", True)
    except Exception as exc:  # noqa: BLE001 - a second close must not raise
        check("close() is idempotent", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    """Run every self-test and report a summary.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    print("hand_tracker self-test")
    print(f"numpy {np.__version__}")

    # The constructor builds the real MediaPipe graph, so a failure here means
    # the environment itself is broken -- report it clearly and stop.
    try:
        detector = HandDetector(max_num_hands=2)
    except FileNotFoundError as exc:
        print(f"\nFATAL: {exc}")
        print("\nThe Tasks API does not bundle its model inside the wheel.")
        print("Download it once, next to hand_tracker.py:")
        print(f"  curl -L -o hand_landmarker.task {MODEL_DOWNLOAD_URL}")
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any environment failure
        print(f"\nFATAL: could not construct HandDetector: {type(exc).__name__}: {exc}")
        print("Fix the environment first (see requirements.txt).")
        return 1

    try:
        test_angle_known_geometry()
        test_angle_degenerate_and_invariance()
        test_landmark_extraction_and_pixel_scaling(detector)
        test_pixel_clipping(detector)
        test_no_hand_edge_case(detector)
        test_multiple_hands_and_errors(detector)
        test_structured_output(detector)
        test_fps_counter()
        test_topology()
        test_input_validation(detector)
        test_real_graph_blank_frame(detector)
    finally:
        try:
            detector.close()
        except Exception:  # noqa: BLE001 - teardown must not mask test results
            pass

    print("\n" + "-" * 62)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} of {_CHECKS_RUN} checks failed")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print(f"ALL {_CHECKS_RUN} CHECKS PASSED")
    print("-" * 62)
    print("Maths and landmark extraction verified. Now run the live demo:")
    print("    python hand_tracker.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
