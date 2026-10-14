# Hand Landmark Tracking — 21 Points (MediaPipe Tasks API)

This folder tracks a hand in real time and gives you **21 three-dimensional
landmarks** per hand: the wrist, every finger joint, and every fingertip.

It contains two scripts:

| Script | What it is |
|---|---|
| `hand_tracker.py` | The library (`HandDetector`, `HandLandmarks`, geometry helpers) plus a built-in webcam smoke test |
| `selftest.py` | An offline correctness test for the maths and the landmark extraction — **run this before trusting the live demo** |

The design target stated in the module docstring is an autonomous-vehicle
perception stack: a small, dependency-injectable object you can drop into a ROS 2
node without dragging the capture and visualisation loop along with it. The code
is written with edge devices (Jetson Nano / Xavier NX) in mind, which shows up in
several concrete choices we will look at.

Press **`q`** or **`Esc`** to quit the smoke test.

> **Read the root [`../README.md`](../README.md) first** if you have not set up
> Git, Python or the virtual environment yet.

---

## 1. Setup

```powershell
cd hand-landmark-project
python -m venv hand_venv
.\hand_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### The model file — the step everyone forgets

```powershell
curl.exe -L -o hand_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Expected size: **~7,819,105 bytes (~7.5 MB)**. If you cloned this repository,
the file is already committed — check it is there before downloading it again.

**Why this is a separate download.** Older MediaPipe versions exposed
`mp.solutions.hands`, which *bundled* its model inside the pip wheel — `pip
install mediapipe` and you were done. The newer **Tasks API**
(`mediapipe.tasks.python.vision`) does not. The model is an explicit `.task`
bundle you fetch once and keep next to the script. The docstring makes the
reason for shipping it with your code rather than fetching at runtime explicit:

> Ship that file with the node: an autonomous platform should never fetch a
> model over the network at runtime.

That is a sound principle for anything safety-related — an unavailable network
must not be able to stop your perception stack from starting.

### Test it

```powershell
python selftest.py        # offline checks — no camera, no hand needed
python hand_tracker.py    # live webcam smoke test
```

Run them in that order. `selftest.py` verifies the geometry maths and the
coordinate conversion against known-good values, which a webcam cannot do
objectively — if the angle formula were subtly wrong, a live demo would still
*look* plausible.

### Dependency note: NumPy is capped below 2.0

```python
numpy>=1.24,<2.0
```

This is deliberate and the `requirements.txt` explains it: the MediaPipe wheels
ship native extensions compiled against the **NumPy 1.x C ABI**. Under NumPy 2.x
those imports fail with a binary incompatibility error rather than a clean Python
exception. The verified combination on this project is **mediapipe 0.10.35 +
numpy 1.26.4**.

Also note that `opencv-python` is **deliberately not listed** — `mediapipe`
already depends on `opencv-contrib-python`, a strict superset. Installing both
puts two distributions in charge of the same `cv2/` folder and whichever pip
writes last wins, which is a well-known source of environment-specific `cv2`
breakage. `import cv2` works either way.

This is the same conflict described in the face project — and it is why this
folder also needs its own environment.

---

## 2. The 21 landmarks

The topology is fixed and worth memorising, because everything you build
(gestures, angles, pinch detection) is expressed in these indices. The
`HandLandmark` enum in the script exists precisely so you never have to write a
magic number:

| Index | Enum name | Index | Enum name |
|---|---|---|---|
| 0 | `WRIST` | 11 | `MIDDLE_FINGER_DIP` |
| 1 | `THUMB_CMC` | 12 | `MIDDLE_FINGER_TIP` |
| 2 | `THUMB_MCP` | 13 | `RING_FINGER_MCP` |
| 3 | `THUMB_IP` | 14 | `RING_FINGER_PIP` |
| 4 | `THUMB_TIP` | 15 | `RING_FINGER_DIP` |
| 5 | `INDEX_FINGER_MCP` | 16 | `RING_FINGER_TIP` |
| 6 | `INDEX_FINGER_PIP` | 17 | `PINKY_MCP` |
| 7 | `INDEX_FINGER_DIP` | 18 | `PINKY_PIP` |
| 8 | `INDEX_FINGER_TIP` | 19 | `PINKY_DIP` |
| 9 | `MIDDLE_FINGER_MCP` | 20 | `PINKY_TIP` |
| 10 | `MIDDLE_FINGER_PIP` | | |

The joint suffixes are standard anatomy:

- **MCP** — *metacarpophalangeal*, the knuckle where a finger meets the palm.
- **PIP** — *proximal interphalangeal*, the middle joint.
- **DIP** — *distal interphalangeal*, the joint nearest the tip.

The thumb is different, and it is worth knowing why: it has only three joints
because it has no middle phalanx. Its joints are **CMC** (*carpometacarpal*, at
the base of the thumb, tucked against the wrist), **MCP**, and **IP**
(*interphalangeal*). That is why the thumb's base point is index 1, not 5 — the
pattern is structurally different from the four fingers, not an inconsistency.

**The enumeration is used as the index.** Because `HandLandmark` is an
`IntEnum`, `HandLandmark.INDEX_FINGER_TIP` *is* the integer `8`, so it can be
used anywhere a plain index is expected:

```python
hand.get_normalized(HandLandmark.INDEX_FINGER_TIP)   # same as get_normalized(8)
```

The docstring shows the payoff — gesture rules become readable:

```python
if hand.get_normalized(HandLandmark.INDEX_FINGER_TIP)[1] < \
   hand.get_normalized(HandLandmark.INDEX_FINGER_PIP)[1]:
    ...  # index finger raised
```

Remember that in image coordinates **y grows downwards**, so a smaller `y` means
*higher on the screen*. That is why "tip above the PIP joint" is written as `<`.

---

## 3. The skeleton

```python
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
)
```

Twenty-one edges, as index pairs. Read the structure and you will see it is a
**chain per finger**, with the chains linked through the palm:

```
                 0  WRIST
              /  |    |    \   \
             1   5    9    13   17     <- knuckles (MCPs)
             |   |    |    |    |
             2   6    10   14   18
             |   |    |    |    |
             3   7    11   15   19
             |   |    |    |    |
             4   8    12   16   20     <- fingertips
```

(The thumb chain hangs off 0 directly; the palm edge `(0, 17)` closes the base of
the hand.) The `selftest.py` checks this structure explicitly: 21 edges, no
self-loops, no duplicates, every one of the 21 landmarks wired in, and each of
the five fingertips appearing in exactly one edge — that last check is what
proves the tips are **leaves**, i.e. endpoints rather than junctions.

**`HAND_CONNECTIONS` is hard-coded on purpose**, and the docstring says why:
rather than reading `mp.solutions.hands.HAND_CONNECTIONS` at runtime, the list is
written out so the module keeps working if the legacy `solutions` namespace is
removed. Given that recent MediaPipe versions *did* remove it, this decision has
already paid off. In general, inlining a small, stable constant is often more
robust than depending on an API that may not exist.

---

## 4. Reading `hand_tracker.py`

### The MediaPipe version guard

```python
try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required for hand_tracker: pip install mediapipe") from exc

try:
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError as exc:
    raise ImportError(
        "hand_tracker requires the MediaPipe Tasks API (mediapipe.tasks.python), "
        f"which was not found in mediapipe {getattr(mp, '__version__', 'unknown')}. "
        "Install mediapipe>=0.10.9."
    ) from exc
```

Two separate checks: is MediaPipe installed at all, and does *this* MediaPipe
have the Tasks API? The second message names the installed version, which is
exactly the information you need when a version mismatch is the problem. Both
re-raise `from exc`, so the original traceback is preserved rather than hidden.

### `HandLandmarks` — the data object

```python
@dataclass(eq=False)
class HandLandmarks:
    landmarks: np.ndarray            # (21, 3) float32, normalised
    pixel_landmarks: np.ndarray      # (21, 2) int32, pixels
    world_landmarks: Optional[np.ndarray]  # (21, 3) float32, metres
    handedness: str                  # "Left" / "Right"
    handedness_score: float
    image_shape: Tuple[int, int]
```

**Why three coordinate arrays.** This is a genuinely good design decision and
worth understanding rather than skimming:

- `landmarks` — **normalised `(x, y, z)`**, `x` and `y` in `[0, 1]` relative to
  the image width and height. Scale-independent, so this is the right array for
  anything geometric: angles, ratios, gesture rules, ROS 2 messages, logging.
  A gesture rule written against normalised coordinates keeps working when you
  change the camera resolution.
- `pixel_landmarks` — **integer pixel `(x, y)`**, already scaled to the frame
  the hand was found in. This is what you draw with.
- `world_landmarks` — **metric `(x, y, z)` in metres**, with the origin at the
  hand's approximate geometric centre. This is the only one with real-world
  units, and it is `Optional` because MediaPipe may not return a world solution.

Keeping all three avoids the classic either/or: you are not forced to choose
between "resolution-independent maths" and "drawable pixels".

**The `z` coordinate is relative, not metric** (except in `world_landmarks`).
In `landmarks`, `z` is depth relative to the wrist, on roughly the same scale as
`x`, with smaller meaning closer to the camera. It is a useful *relative*
signal — is the index fingertip in front of the palm? — but it is not a distance
measurement, and it is noticeably noisier than `x` and `y`. This is why
`calculate_angle()` defaults to `use_z=False`.

**`handedness` is anatomical, assuming a non-mirrored image.** The docstring
flags the gotcha directly: it **flips if you feed a horizontally flipped frame**.
The smoke test in this module does *not* mirror frames, so the label is correct
there — but if you add selfie mirroring (as the face scripts in this repository
do), the reported handedness will be inverted. Worth knowing before you build a
gesture rule on it.

**`eq=False`** on the dataclass means equality falls back to identity comparison
rather than comparing NumPy arrays element-wise. Comparing dataclasses containing
arrays with the default `__eq__` raises an ambiguous-truth-value error, so this
sidesteps a real trap.

**The convenience API:**

```python
hand.get_normalized(HandLandmark.INDEX_FINGER_TIP)  # -> (x, y, z) plain floats
hand.get_pixel(HandLandmark.INDEX_FINGER_TIP)       # -> (x, y) plain ints
hand.as_dict()                                      # -> {"INDEX_FINGER_TIP": (x, y, z), ...}
hand.as_list()                                      # -> [(x, y, z), ...] indexed by id
```

Both accessors return **plain Python floats and ints**, not NumPy scalars, and
that is deliberate: the docstring notes they are "safe to serialise into a ROS 2
message or `json.dumps`". A `np.float32` is not JSON-serialisable, so this
conversion at the boundary is what saves you from a confusing
`TypeError: Object of type float32 is not JSON serializable` far away from its
cause.

Every accessor routes through `_checked_index()`, which raises `IndexError` for
anything outside `[0, 20]`. `selftest.py` verifies this for 21, -1 and 999.

### The adapter layer — bridging two MediaPipe APIs

Four small classes sit between MediaPipe and the rest of the module:

```python
class _LandmarkListAdapter:  __slots__ = ("landmark",)      # .landmark
class _CategoryAdapter:      __slots__ = ("label", "score")
class _HandednessAdapter:    __slots__ = ("classification",)
class _ResultsAdapter:       __slots__ = ("multi_hand_landmarks",
                                          "multi_hand_world_landmarks",
                                          "multi_handedness")
```

**The problem they solve.** The legacy `mp.solutions.hands` API returned protobuf
objects with a `.landmark` attribute. The Tasks API returns **plain Python
lists**. Every downstream consumer — `_extract()`, `_draw_skeleton()`, and the
fake objects in `selftest.py` — was written against the older shape. Rather than
scatter `if hasattr(...)` checks through the hot path, `_ResultsAdapter`
normalises the Tasks output into the legacy shape **once per frame**, in one
place.

The docstring states the benefit precisely: normalising in one place keeps every
downstream consumer "working against a single shape". This is the Adapter
pattern, and it is the right call when you are gluing a new library version onto
code written for the old one — the alternative is version-checking branches
sprinkled through your rendering and extraction code.

`__slots__` on all four is a small memory/attribute-lookup optimisation that
matters when you allocate these objects every frame on a Jetson.

`_ResultsAdapter` also handles the messy reality of the handedness field:

```python
label = getattr(top, "category_name", "") or getattr(top, "display_name", "")
handedness.append(_HandednessAdapter(str(label or "Unknown"), float(top.score)))
```

It probes two different attribute names (MediaPipe has used both across versions)
and falls back to `"Unknown"` with score `0.0`. An empty group produces an
`"Unknown"` entry rather than being skipped, so the handedness list stays aligned
with the landmark list by index — which `_extract()` depends on.

### `HandDetector.__init__` — validation and the model check

```python
if max_num_hands < 1:              raise ValueError(...)
if model_complexity not in (0, 1): raise ValueError(...)
for name, value in (("min_detection_confidence", min_detection_confidence),
                    ("min_tracking_confidence", min_tracking_confidence)):
    if not 0.0 <= value <= 1.0:    raise ValueError(...)
if delegate not in ("cpu", "gpu"): raise ValueError(...)
```

Then the model check, which is the one that will actually fire in practice:

```python
if not self.model_path.is_file():
    raise FileNotFoundError(
        f"hand landmarker model not found: {self.model_path}\n"
        "The MediaPipe Tasks API needs an explicit model bundle (the legacy "
        "solutions API bundled it, this one does not). Download it once with:\n"
        f"  curl -L -o hand_landmarker.task {MODEL_DOWNLOAD_URL}\n"
        ...)
```

The error message explains *why* MediaPipe needs a separate download, which is
the part that is genuinely confusing if you have used older MediaPipe.

**An honest compatibility note.** The constructor accepts `model_complexity` and
`smooth_landmarks`, but logs:

```
model_complexity and smooth_landmarks are accepted for API compatibility but
have no effect under the Tasks API; the model bundle embeds the network and
VIDEO mode smooths/tracks internally.
```

These parameters existed in the legacy `solutions` API. The Tasks API takes the
network *from the model bundle*, so there is no complexity knob to turn, and
VIDEO-mode tracking does the temporal smoothing internally. Accepting the
arguments and documenting them as no-ops is far better than silently ignoring
them or breaking every caller — but you should know that passing
`model_complexity=1` will not make anything more accurate here.

### `_build_graph()` and the lifecycle

```python
base_options = BaseOptions(model_asset_path=str(self.model_path))
if self.delegate == "gpu":
    base_options.delegate = BaseOptions.Delegate.GPU

running_mode = (vision.RunningMode.IMAGE if self.static_image_mode
                else vision.RunningMode.VIDEO)
options = vision.HandLandmarkerOptions(
    base_options=base_options, running_mode=running_mode,
    num_hands=self.max_num_hands,
    min_hand_detection_confidence=self.min_detection_confidence,
    min_hand_presence_confidence=self.min_detection_confidence,
    min_tracking_confidence=self.min_tracking_confidence,
)
self._running_mode = running_mode
return vision.HandLandmarker.create_from_options(options)
```

**IMAGE vs VIDEO mode** is the same distinction explained in the face project's
README. Here it is wired to `static_image_mode`:

- `static_image_mode=False` (default) → **VIDEO** mode. Palm detection runs on
  the first frame and the graph **tracks** afterwards. Significantly cheaper and
  smoother. This is what live capture wants.
- `static_image_mode=True` → **IMAGE** mode. Palm detection on **every** frame.
  Use for stills or a batch pipeline. `min_tracking_confidence` has no effect in
  this mode, because there is no tracking.

**The GPU delegate has no silent fallback.** The comment is explicit: "There is
no silent fallback: an unsupported delegate raises from `create_from_options`
below." That is a deliberate choice — if you ask for GPU and the platform cannot
provide it, you find out immediately with an exception rather than silently
running 5× slower and wondering why. On a Jetson with the right runtime, `gpu` is
worth trying.

**Teardown.** `close()` is **idempotent** (guarded by `self._closed`), catches
exceptions from the underlying `close()`, and is driven automatically by the
context manager:

```python
with HandDetector(max_num_hands=2) as detector:
    ...
# graph released here, even if the body raised
```

Freeing the MediaPipe graph matters on an embedded device where handles are
finite. The smoke test uses this form.

After `close()`, any call to `find_hands()` raises a clear
`RuntimeError("HandDetector is closed; construct a new instance")` rather than
failing deep inside MediaPipe.

### `find_hands()` — inference and the hot path

```python
if frame is None or frame.size == 0:
    LOGGER.warning("find_hands received an empty frame; skipping inference")
    self._results = None
    return frame

if frame.ndim != 3 or frame.shape[2] != 3:
    raise ValueError(f"expected an (H, W, 3) frame, got shape {frame.shape}")
if frame.dtype != np.uint8:
    raise ValueError(f"expected a uint8 frame, got dtype {frame.dtype}")
```

Note the split in error handling policy, which is a deliberate design choice:

- **An empty frame is tolerated** — logged and skipped, returning rather than
  raising. The docstring explains: "a flaky camera driver cannot take the node
  down." In a long-running perception node, one bad frame should not be fatal.
- **A malformed frame raises** — the wrong shape or dtype is a *programming*
  error, and failing loudly at the boundary is much better than letting a
  `float32` array reach MediaPipe and produce an obscure error three layers down.

The validated checks come **before** any work, so you never pay for a conversion
that is about to be thrown away.

```python
self._frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
```

`(height, width)` is recorded — in that order, matching NumPy's `shape`. This is
cached because `_extract()` needs it later, and because a caller can query a
cached result after the frame is gone.

```python
if self.input_is_bgr:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
else:
    rgb = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
```

OpenCV gives **BGR**; MediaPipe wants **RGB**. The `input_is_bgr=False` path
exists for a ROS 2 `sensor_msgs/Image` that has already been converted by
`cv_bridge`, so you do not convert twice. The `c_contiguous` check avoids a
pointless copy when the array is already in the right layout, but guarantees
MediaPipe gets the contiguous buffer it requires.

`find_hands(..., draw=True)` **annotates the input frame in place and returns the
same object.** The docstring warns about this explicitly:

> The annotation mutates the input buffer, so copy first if the original frame
> feeds another consumer (e.g. a recording or a second model).

This is zero-copy by design for speed on an edge device — but if you are also
writing the frame to a video file, you will record the annotated version. Copy
first if that matters.

### `_next_timestamp_ms()` — the monotonic guard

```python
now_ms = int(time.perf_counter() * 1000.0)
if now_ms <= self._last_timestamp_ms:
    now_ms = self._last_timestamp_ms + 1
self._last_timestamp_ms = now_ms
return now_ms
```

The same rule as the face mesh script, with a clearer justification in the
docstring:

> The Tasks API rejects a timestamp that does not advance, which happens whenever
> two frames land inside the same millisecond — common at high frame rates, and
> guaranteed when replaying a recorded bag faster than real time.

That second case is the important one for this project: replaying a ROS 2 bag
faster than real time would otherwise crash on the very first repeated
millisecond.

### `get_landmark_positions()` — the reusable-result trick

```python
def get_landmark_positions(self, frame=None, hand_index=0, *, draw=False):
    if frame is not None:
        self.find_hands(frame, draw=draw)
    ...
    return self._extract(hand_index)
```

Passing `frame=None` (the default) **reuses the result of the most recent
`find_hands()` call** instead of running inference again. The docstring gives the
motivation: "avoids a second inference when one node both draws and consumes the
landmarks." Running the network twice per frame on a Jetson, where inference is
the bottleneck, would halve your frame rate for no reason. The smoke test uses
this pattern:

```python
detector.find_hands(frame, draw=not args.no_draw)   # infer + render
hand = detector.get_landmark_positions()            # reuse — no second inference
```

`get_all_hand_landmarks()` is the same idea for every hand, and returns an
**empty list** rather than `None` when nothing is found, so a `for` loop over it
is always safe.

### `_extract()` — protobuf to NumPy, once per frame

This is the only place the protobuf landmark objects are touched:

```python
landmarks = np.array([(lm.x, lm.y, lm.z) for lm in proto.landmark], dtype=np.float32)

height, width = self._frame_shape
pixel_landmarks = np.rint(
    landmarks[:, :2] * np.array((width, height), dtype=np.float32)
).astype(np.int32)
np.clip(pixel_landmarks[:, 0], 0, width - 1,  out=pixel_landmarks[:, 0])
np.clip(pixel_landmarks[:, 1], 0, height - 1, out=pixel_landmarks[:, 1])
```

Four things happening, all of them deliberate:

1. **One list comprehension → one array allocation.** The docstring notes this
   keeps "no per-point Python objects alive" — the hot path avoids creating 21
   objects per hand per frame.

2. **`x` scales by `width`, `y` scales by `height`.** This is the single most
   common bug in MediaPipe code. Swap them and the points still look
   *approximately* right — just subtly stretched — so it can survive a long time
   before anyone notices. `selftest.py` tests this specific case with a ramp
   whose expected pixel positions can be computed by hand.

3. **`np.rint` rounds rather than truncates**, so the drawn point sits on the
   true pixel. Truncation would bias every point half a pixel up and left.

4. **The clamp is not paranoia.** MediaPipe can predict a landmark slightly
   outside the image — a fingertip at the frame edge, or briefly during a fast
   motion. Drawing at `x = 640` on a 640-wide image is out of bounds. Clamping to
   `width - 1` / `height - 1` keeps `cv2` calls valid. The `out=` argument makes
   `np.clip` write into the existing array rather than allocating a new one.

The world-landmarks and handedness blocks that follow both use
`getattr(..., None)` plus a length check, so a MediaPipe build that omits either
field degrades to `world_landmarks=None` and `handedness="Unknown"` instead of
raising. `selftest.py` verifies that graceful degradation.

### `calculate_angle()` — the geometry helper

```python
ba = pa - pb
bc = pc - pb
norm_ba = float(np.linalg.norm(ba))
norm_bc = float(np.linalg.norm(bc))
if norm_ba < 1e-9 or norm_bc < 1e-9:
    return float("nan")

cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
angle = math.acos(max(-1.0, min(1.0, cosine)))
return math.degrees(angle) if degrees else angle
```

This computes the interior angle at vertex `b` formed by `a–b–c`, from the dot
product:

```
θ = acos( (u · v) / (|u| |v|) )
```

The docstring's example is the one you will actually use — index finger extension
at the MCP joint:

```python
HandDetector.calculate_angle(
    hand.get_normalized(HandLandmark.WRIST),
    hand.get_normalized(HandLandmark.INDEX_FINGER_MCP),
    hand.get_normalized(HandLandmark.INDEX_FINGER_TIP),
)   # ~180 deg when straight, ~30-60 deg when curled
```

Points to take away:

- **The result is invariant to translation, rotation and scale**, because it
  depends only on the two vectors `b→a` and `b→c` and their lengths. Concretely:
  you get the same answer from normalised coordinates and from pixel
  coordinates, which `selftest.py` asserts directly. That means one set of
  gesture thresholds works at any resolution and any distance from the camera.

- **The clamp before `acos` is load-bearing.** A collinear triple should give
  `cosine = 1.0` exactly, but floating-point error can produce `1.0000001`,
  which is outside `acos`'s domain `[-1, 1]`. NumPy would emit a domain warning
  and return `nan`. Clamping first makes the straight-finger case return exactly
  `180.0`. `selftest.py` has a dedicated test for this
  (`angle((1000,1000),(0,0),(-1000,-1000)) == 180.0`).

- **Degenerate input returns `nan`, not `0.0`.** If two of the points coincide,
  a vector has zero length and the angle is geometrically undefined. The
  docstring explains the reasoning, and it is a subtle and important point:

  > `nan` is returned rather than a sentinel like `0.0` so a bad frame cannot
  > silently be read as a confidently closed finger — guard with `math.isnan` if
  > your rules do not already compare against thresholds (any comparison with
  > `nan` is `False`).

  This is fail-safe by construction: a "is the finger straight?" test written as
  `angle > 150.0` returns `False` on a garbage frame rather than accidentally
  `True`. `selftest.py` asserts exactly this
  (`not (nan > 150.0)`). Compare this with the eye project's EAR, which returns
  `0.0` on degenerate input — opposite choices, both correct, because in that
  safety context a missing measurement must fail *toward* an alert.

- **`use_z=False` by default.** MediaPipe's `z` is a relative, noisy depth
  estimate, so for gesture thresholds the 2D angle is more stable. The docstring
  is direct: "enable this only if you genuinely need out-of-plane angles."
  `selftest.py` verifies that 2D mode ignores `z` entirely.

`calculate_landmark_angle()` is a convenience wrapper that takes the hand
snapshot and three indices directly, so you do not have to unpack the
coordinates yourself. `selftest.py` asserts it agrees with the raw helper to
within `1e-12`.

### `_draw_skeleton()` — rendering tuned for an edge device

```python
_POINT_COLOR = (0, 255, 0);  _CONNECTION_COLOR = (255, 255, 255)
_TIP_COLOR = (0, 0, 255);    _POINT_RADIUS = 3;  _TIP_RADIUS = 6
_LINE_THICKNESS = 2
_LINE_TYPE = cv2.LINE_8
```

The style constants live at **class level**, "tuned once … so the per-frame draw
path allocates nothing."

`_LINE_TYPE = cv2.LINE_8` rather than `cv2.LINE_AA` is a considered performance
choice, and the comment says so: "`cv2.LINE_8` is materially cheaper than
`cv2.LINE_AA`; swap it if you have GPU/thermal headroom and want smoother
bones." Anti-aliased line drawing is noticeably more expensive, and on a
thermal-limited Jetson every millisecond of CPU time competes with inference.
Note that the face scripts in this repository make the *opposite* choice
(`LINE_AA` everywhere) — they are desktop demos where smoothness matters more
than throughput.

The drawing itself converts to pixels with a single vectorised operation, then
loops with integer indexing:

```python
points = np.rint(
    np.array([(lm.x, lm.y) for lm in proto.landmark], dtype=np.float32)
    * np.array((width, height), dtype=np.float32)
).astype(np.int32)
```

The docstring notes the deliberate avoidance of MediaPipe's own drawing spec
machinery ("avoiding a second pass over the landmark lists and the per-call
drawing-spec machinery"). One vectorised conversion, then plain `cv2` primitives.
The index fingertip is drawn last, larger and in red, as a visual anchor —
useful when debugging a gesture rule.

### `FPSCounter`

```python
class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        if window < 2: raise ValueError("window must be >= 2")
        self._timestamps: Deque[float] = deque(maxlen=window)

    def update(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        self.fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0.0 else 0.0
        return self.fps
```

A rolling-window mean over frame timestamps, and the docstring gives a
hardware-specific reason for preferring it to an exponential moving average:

> A windowed mean is used instead of an exponential moving average so a single
> stalled frame (a common thing on a Jetson when the thermal governor kicks in)
> does not visibly skew the reported number.

Note the arithmetic: `len - 1` intervals across `last - first` elapsed time. With
`n` timestamps there are `n-1` gaps. Dividing `n` by that elapsed time — an easy
off-by-one — would over-report the rate slightly at small window sizes. Two
guards are present: fewer than 2 samples returns `0.0` (a single timestamp has no
interval), and a zero denominator is handled in case two calls land on the same
clock tick.

### `main()` — the smoke test

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(args.width))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

The first two are **requests, not commands** — the comment acknowledges that
drivers may return a different size, and that this is fine because MediaPipe
works in normalised coordinates anyway.

`CAP_PROP_BUFFERSIZE = 1` is the interesting one: it asks the driver to keep only
one frame queued, "to prevent a backlog of stale frames building up if inference
is slower than capture." Without it, a pipeline where inference is the bottleneck
accumulates a queue of old frames and you end up looking at the past — the
classic "my webcam has a one-second delay" problem. Honest caveat: buffer-size
control is **backend-dependent**; it works on V4L2 (Linux) and is often ignored
by the Windows backends, so on Windows this line may be a no-op.

The loop mirrors the pattern used in the other scripts: read-failure tolerance
(`max_read_failures = 30`, about a second at 30 FPS) so a USB hiccup is not fatal,
status lines drawn top-left, the index fingertip printed to the console at a
throttled rate (`--print-hz`, default 5 per second, so the console stays readable
at 30 FPS), and release in a `finally` block so the camera is never left locked.

---

## 5. Reading `selftest.py`

`selftest.py` deserves its own section, because the habit it represents is worth
copying more than any individual line in it.

### Why it exists

The module docstring:

> Run this *after* installing requirements.txt and *before* trusting the live
> webcam demo. It verifies the parts of the module that a webcam cannot verify
> objectively — the angle maths and the landmark-to-pixel conversion — against
> known-good values, and it exercises the no-hand path without needing a hand in
> front of the camera.

That is the correct instinct. A live webcam demo **cannot** tell you that your
angle formula is off by a sign, or that `x` is being scaled by height. Everything
still *looks* plausible on screen. Only a test against known-good values catches
that class of bug.

### No pytest — a deliberate choice

```python
def check(name, condition, detail=""): ...
def check_close(name, actual, expected, tol=1e-6): ...
def section(title): ...
```

A tiny hand-rolled harness so the file "runs on a bare venv" with no test
framework to install. Three helpers, a module-level failure list and a counter,
and a summary at the end that exits `0` or `1`. For a project whose whole point
is minimal dependencies on an edge device, adding `pytest` as a dev dependency
would be a real cost.

`check_close` is worth a look:

```python
ok = math.isnan(actual) and math.isnan(expected)
if not ok:
    ok = abs(actual - expected) <= tol
```

The `nan` case is handled first, because `nan == nan` is `False` — a naive
`abs(actual - expected) <= tol` would fail on a correctly-returned `nan`.

### Faking MediaPipe

```python
class _FakeLandmark:      # .x .y .z
class _FakeLandmarkList:  # .landmark
class _FakeClassification:# .label .score
class _FakeHandedness:    # .classification
class _FakeResults:       # multi_hand_landmarks / multi_handedness / multi_hand_world_landmarks
```

These duck-type the protobuf result objects, so the landmark-extraction path can
be tested **without running the neural network**. Tests then inject them directly:

```python
detector._results = _FakeResults([_ramp_hand()], handedness=[("Right", 0.98)])
detector._frame_shape = (480, 640)
hand = detector.get_landmark_positions()
```

Fast, deterministic, and no hand required in front of a camera. This is only
possible because `_extract()` reads a **narrow, well-defined set of attributes** —
which is a good argument for keeping that interface small. Note the fakes mirror
exactly the shape the `_ResultsAdapter` produces, which is what lets the same
code path serve both.

### `_ramp_hand()` — designing test data you can compute by hand

```python
def _ramp_hand():
    """Build 21 landmarks on a known ramp: point i = (i/20, i/40, 0)."""
    return [(i / 20.0, i / 40.0, 0.0) for i in range(NUM_LANDMARKS)]
```

This is the neatest thing in the file. Because point `i` is at `(i/20, i/40)`, the
expected pixel position in a 640×480 frame is arithmetic you can do in your head:

```
x = i/20 × 640 = i × 32
y = i/40 × 480 = i × 12
```

The test then asserts exactly that, including the fact that `x` uses **width** and
`y` uses **height** — "swapping them is the classic bug", as the comment says.
It also exercises the clamp: point 20 sits at normalised `x = 1.0`, which is
pixel 640, one past the last addressable column of a 640-wide image, so the
clamp must pull it back to **639**. Synthetic data with hand-computable expected
values is far more valuable than data copied from a previous run.

### What is tested

| Test | Covers |
|---|---|
| `test_angle_known_geometry` | 90°, 180°, 0°, 45°, 120°, radians mode, 3D mode, `z` ignored in 2D, acos clamp on collinear points |
| `test_angle_degenerate_and_invariance` | `nan` on coincident points; `nan > 150` is `False`; translation invariance; scale invariance; argument order of `a`/`c` |
| `test_landmark_extraction_and_pixel_scaling` | `(21,3)` / `(21,2)` shapes, `image_shape`, x-by-width / y-by-height, the 639 clamp, round-trip of normalised values, handedness passthrough, enum agrees with the integer id |
| `test_pixel_clipping` | Out-of-frame predictions (`1.5`, `-0.5`) clamp into bounds |
| `test_no_hand_edge_case` | `None` / `[]` / `0` on no hand, before any inference at all |
| `test_multiple_hands_and_errors` | `hand_index` selection, `IndexError` out of range, world landmarks, graceful degradation when world/handedness are missing |
| `test_structured_output` | `as_dict` / `as_list`, key names, tuple lengths, agreement with `get_normalized`, `calculate_landmark_angle` vs `calculate_angle` |
| `test_fps_counter` | `window=1` rejected, `0.0` before two samples, finite and positive afterwards |
| `test_topology` | 21 connections, valid ids, no self-loops, no duplicates, all landmarks wired, tips are leaves |
| `test_input_validation` | Bad constructor args, 2-D grayscale frame, `float32` frame, blank frame handled |
| `test_real_graph_blank_frame` | The real MediaPipe graph on a blank and a noise frame: returns the same buffer, no hallucinated hand, no drawing when nothing is found, `close()` idempotent |

**One important practical detail:** `main()` constructs a **real `HandDetector`**
(the real MediaPipe graph), and some tests construct additional instances. So
`selftest.py` **requires `hand_landmarker.task` to be present** — it is not a
pure unit test. If the model is missing it catches the `FileNotFoundError` and
prints the download command rather than a traceback.

The last test is the one that earns the "integration" label:
`test_real_graph_blank_frame` runs the actual network on a black frame and on
random noise, asserting that no hand is hallucinated and that drawing is a no-op
when nothing is detected (`np.array_equal(blank, before)`). That is a real check
that the pipeline does not invent landmarks — a failure mode you would otherwise
only notice as mysterious phantom hands in production.

---

## 6. Using it as a library

```python
from hand_tracker import HandDetector, HandLandmark

with HandDetector(max_num_hands=2, model_path="hand_landmarker.task") as detector:
    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            continue

        detector.find_hands(frame, draw=True)
        hand = detector.get_landmark_positions()      # reuses this frame's result

        if hand is None:
            continue                                   # no hand — handle it

        # Angles from NORMALISED coordinates: resolution-independent.
        index_angle = detector.calculate_angle(
            hand.get_normalized(HandLandmark.WRIST),
            hand.get_normalized(HandLandmark.INDEX_FINGER_MCP),
            hand.get_normalized(HandLandmark.INDEX_FINGER_TIP),
        )

        if not math.isnan(index_angle) and index_angle > 150.0:
            ...   # index finger is straight
```

Habits worth carrying over from this module's design:

- **Use normalised coordinates for logic, pixels for drawing.** Angles, ratios
  and gesture thresholds against `hand.get_normalized(...)` survive a resolution
  change; drawing against `hand.get_pixel(...)` is simpler and exact.
- **Always handle `None`.** No hand detected is the normal case, not an error.
- **Guard `nan`** before comparing an angle to a threshold — or write the
  comparison so that `nan` fails safe, as the docstring suggests.
- **Call `get_landmark_positions()` without a frame** when you have already run
  `find_hands()` in the same iteration. It saves a full inference.
- **Use one detector per stream**, and use the context manager so the graph is
  released.

### Gesture building blocks

With angles from three landmarks, most gesture rules are a couple of comparisons.
A worked example — which fingers are extended — using normalised `y` (remember,
`y` grows downwards, so a tip above its PIP joint means extended):

```python
def fingers_up(hand) -> list[bool]:
    up = []
    for tip, pip in ((HandLandmark.INDEX_FINGER_TIP,  HandLandmark.INDEX_FINGER_PIP),
                     (HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_PIP),
                     (HandLandmark.RING_FINGER_TIP,   HandLandmark.RING_FINGER_PIP),
                     (HandLandmark.PINKY_TIP,         HandLandmark.PINKY_PIP)):
        up.append(hand.get_normalized(tip)[1] < hand.get_normalized(pip)[1])
    return up
```

The thumb needs its own rule — it moves sideways rather than up, so compare its
tip against the index MCP on the **x** axis, or use
`calculate_angle(THUMB_MCP, THUMB_IP, THUMB_TIP)` and threshold it. This
asymmetry is exactly why the thumb has a different joint naming scheme, and it is
the usual reason a naive "count extended fingers" implementation gets four
instead of five.

---

## 7. CLI reference

### `hand_tracker.py`

```powershell
python hand_tracker.py [options]
```

| Flag | Default | Effect |
|---|---|---|
| `--camera` | `0` | `cv2.VideoCapture` index |
| `--width` / `--height` | `640` / `480` | Requested capture size (a request, not a guarantee) |
| `--max-hands` | `2` | Maximum hands to track |
| `--model-path` | `hand_landmarker.task` beside the script | Path to the model bundle |
| `--delegate` | `cpu` | `cpu` or `gpu` (no silent fallback — see above) |
| `--static` | off | IMAGE mode: re-detect every frame, no tracking |
| `--no-draw` | off | Skip skeleton rendering |
| `--print-hz` | `5.0` | Console print rate for the index tip; `0` prints every frame |

Exit codes: `0` on a clean quit, `1` if the camera could not be opened or stopped
delivering frames, or if the model is missing.

### `selftest.py`

```powershell
python selftest.py
```

No arguments. Exits `0` if every check passed, `1` otherwise, and prints a
summary listing any failures by name.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `hand landmarker model not found` | Download `hand_landmarker.task` (section 1) into this folder, or pass `--model-path`. |
| `mediapipe is required` / `requires the MediaPipe Tasks API` | Wrong venv, or a MediaPipe older than 0.10.9. Use `hand_venv` and check `pip show mediapipe`. |
| `numpy.core.multiarray failed to import` or a binary incompatibility error | NumPy 2.x with a MediaPipe wheel built for the 1.x ABI. `pip install "numpy<2.0"`. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | You are reading older MediaPipe tutorials. This module uses the Tasks API on purpose; ignore `mp.solutions` examples. |
| `ValueError: expected an (H, W, 3) frame` | The frame is grayscale or BGRA. Convert to 3-channel BGR first. |
| `ValueError: expected a uint8 frame` | You have a `float32` image (often after normalising to 0–1). Scale back to 0–255 and cast. |
| `RuntimeError: HandDetector is closed` | You called `find_hands()` after `close()`. Construct a new instance. |
| Handedness labels look swapped | You are feeding a horizontally mirrored frame. Either stop mirroring or invert the label. |
| The webcam view lags behind reality | Inference is slower than capture and a frame queue is building up. `CAP_PROP_BUFFERSIZE` (set to 1 here) is backend-dependent and may be ignored on Windows. Reduce `--width`/`--height`. |
| Skeleton looks jittery | VIDEO mode tracks and smooths internally; make sure `--static` is **off**. |
| `--delegate gpu` raises | The platform has no supported GPU delegate. Use `cpu`. There is no silent fallback by design. |
| Slow on a Jetson | Lower the capture resolution, keep `--static` off (tracking is cheaper than detection), and consider `--delegate gpu`. |
| `selftest.py` fails on `FATAL: could not construct HandDetector` | An environment problem, not a code problem — it builds the real graph. Check the venv and the model file. |

---

## 9. Where to go next

- **Check the topology visually.** Run `hand_tracker.py` and compare what you see
  against the table in section 2 — then modify `_TIP_COLOR` to highlight a
  different landmark and watch which one lights up.
- **Build a gesture.** `calculate_angle()` plus the `HandLandmark` enum is enough
  for finger counting, pinch detection, thumbs-up, and simple sign recognition.
  Start with the `fingers_up` example in section 6.
- **Compare with the face project.** `../face-landmark-project/` solves the same
  problem for faces with dlib (68 points, pixel coordinates) and MediaPipe (478
  points, normalised coordinates). The coordinate-convention difference between
  those two is the single most transferable lesson in this repository.
- **Track over time.** Landmark positions are noisy frame to frame. For anything
  smooth (drawing, cursor control, gesture trajectories) add a filter — a
  one-euro filter or a simple Kalman filter — before you act on the coordinates.
