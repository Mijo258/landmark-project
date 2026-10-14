# Eye Landmarks & Driver Drowsiness Detection (EAR)

This folder contains a **Driver Monitoring System (DMS)** built on the classic
**Eye Aspect Ratio (EAR)** — a small, fast, interpretable measure of how open a
person's eyes are, computed from facial landmarks.

It is one script, `eye_tracker.py`, and it does two things:

1. **Library mode** — it exposes a reusable `DrowsinessDetector` class with no
   camera, no window and no I/O policy of its own, so it can be dropped into a
   larger system (the docstrings repeatedly show a ROS 2 node as the target).
2. **Demo mode** — run it directly and it opens your webcam and shows a live
   drowsiness monitor.

Press **`q`** or **`Esc`** to quit the demo.

> **Read the root [`../README.md`](../README.md) first** if you have not set up
> Git, Python or the virtual environment yet.

---

## 1. The idea

A blink and a microsleep look similar frame by frame. The difference is
**duration**: a blink lasts roughly 100–400 ms, while a drowsy eye closure lasts
longer. So the algorithm needs two ingredients:

- a per-frame measurement of "how open are the eyes" — that is the **EAR**;
- a rule about **how long** the eyes have been shut — that is a **frame counter**.

That is the entire method. Its appeal is that it needs no training data, no
neural network beyond the face landmark model you already have, and it runs in
real time on a CPU. Its weakness is that the threshold is a fixed number, so
lighting, glasses, camera angle and individual eye shape all affect it — see
section 6 on tuning.

The method comes from Soukupova and Cech (2016), *Real-Time Eye Blink Detection
using Facial Landmarks*, and this module is a faithful implementation of it.

---

## 2. Setup

```powershell
cd eye-landmark-project
python -m venv eye_venv
.\eye_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This environment needs four things: `numpy`, `scipy`, `opencv-python`, and
`dlib`.

**On Windows, `dlib` is the hard part.** dlib ships no prebuilt wheels for
CPython on PyPI, so pip compiles it from source. You need, before running
`pip install`:

1. **CMake** — <https://cmake.org/download/>, and tick "Add CMake to the system
   PATH" during install.
2. **Visual Studio Build Tools** with the **"Desktop development with C++"**
   workload (this provides MSVC and the Windows SDK).

Without them the install fails with `CMake is not installed` or
`Cannot find Visual Studio`. The build then takes **several minutes**. That is
normal and expected, not a hang.

> **Note on versions.** This project deliberately targets **OpenCV 4** and
> **dlib 19.x** (`opencv-python>=4.8,<5.0`, `dlib>=19.24,<20.0`), whereas
> `../face-landmark-project/` pins **OpenCV 5** and **dlib 20**. They are
> genuinely different dependency sets, so **each project needs its own virtual
> environment**. Do not try to share one.

### The model file

```powershell
curl.exe -L -o shape_predictor_68_face_landmarks.dat.bz2 `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
python -c "import bz2,shutil; shutil.copyfileobj(bz2.open('shape_predictor_68_face_landmarks.dat.bz2','rb'), open('shape_predictor_68_face_landmarks.dat','wb'))"
```

The result must be **~99,693,937 bytes**. This is the *same* 68-point model used
by `detect_68point.py` in the face project — if you already have it there, just
copy it across.

Use `curl.exe`, not `curl` (in PowerShell, `curl` is an alias for
`Invoke-WebRequest` and will not behave as expected).

### Run the demo

```powershell
python eye_tracker.py
python eye_tracker.py --ear-threshold 0.22 --consecutive-frames 15
```

---

## 3. The Eye Aspect Ratio

### Which six points

The EAR is computed from six landmarks per eye. Using the iBUG 68 indices these
are:

| Point in the formula | Right eye (image-left) | Left eye (image-right) | Where it is |
|---|---|---|---|
| `p1` | 36 | 42 | outer corner |
| `p2` | 37 | 43 | upper lid, outer |
| `p3` | 38 | 44 | upper lid, inner |
| `p4` | 39 | 45 | inner corner |
| `p5` | 40 | 46 | lower lid, inner |
| `p6` | 41 | 47 | lower lid, outer |

The order is **clockwise starting at the outer corner**, which is exactly the
order dlib returns them in. That ordering is not decorative — the formula's
correctness depends on it.

### The formula

```
        ‖p2 − p6‖ + ‖p3 − p5‖
EAR  =  ───────────────────────
             2 · ‖p1 − p4‖
```

Reading it in plain language:

- **Numerator** — two vertical distances across the eye. `p2–p6` and `p3–p5` each
  span from an upper lid point to the lower lid point, so together they measure
  the **height** of the eye opening. There are two of them because one would be
  noisy; averaging both sides makes the measure robust to the eyelid not being
  symmetric.
- **Denominator** — one horizontal distance, `p1–p4`, the **full width** of the
  eye from corner to corner.

So the EAR is a **ratio of eye height to eye width**. For a fully open eye it is
roughly **0.30**; as the eye closes, the height shrinks toward zero while the
width barely changes, so the ratio falls below roughly **0.20**.

### Why a fixed threshold works at any distance

This is the elegant part. If you move away from the camera, every distance in
the formula shrinks — but they all shrink by *the same factor*. A ratio of two
quantities that scale together is unchanged. That is what "scale invariant"
means here, and it is why a single threshold like `0.25` works whether the face
is 30 cm or 2 m from the lens. The same argument applies to translation and
rotation.

The practical consequence: **you do not need to re-tune the threshold when the
driver moves**, only when something changes the *shape* of the eye signal —
glasses, heavy lighting change, a different person.

### Why both eyes are averaged

```python
ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0
```

Two reasons:

- **Winking or one eye occluded.** If a hand or the steering wheel hides one
  eye, that eye's EAR collapses to something meaningless. Averaging halves its
  influence, so the measurement degrades gracefully instead of firing a false
  alert.
- **Noise.** Averaging two independent measurements reduces the variance of the
  closure signal, which means you can use a shorter `consecutive_frames` window
  without getting more false positives.

### The degenerate case

```python
if horizontal <= 0.0:
    return 0.0
```

If all six points collapse to the same pixel, the denominator is zero and the
ratio is mathematically undefined (division by zero → `inf` or `nan`). The code
returns `0.0` instead. That is a deliberate choice, and the comment says why:
for a **safety system, "eyes shut" is the safe failure mode.** A `nan` would
propagate through every comparison in the temporal check and silently evaluate
to `False`, which means *no alert* — the exact opposite of what you want when
the landmark model has produced garbage.

> Worth noting the contrast with `hand_tracker.py` in this repository, which
> returns `nan` for its degenerate angle case and documents that callers must
> guard with `math.isnan`. The two modules make **opposite** choices, each
> correct for its context: here a missing measurement must fail *toward* an
> alert.

---

## 4. Reading the code

### Design: no I/O inside the class

The module docstring is explicit about this:

> The public surface is a single importable class, `DrowsinessDetector`, which
> is deliberately free of any I/O policy: it never opens a camera, a window or a
> topic.

Everything about *where frames come from* and *what happens on an alert* lives
outside the class. The class does exactly one thing: given a frame, tell you the
EAR and whether the drowsiness criterion is currently met. That is what makes
the same object usable from a webcam demo (`main()` in this same file) and from
a ROS 2 subscription callback, with no changes. It is a good pattern to copy.

### `FrameAnalysis` — the result object

```python
@dataclass(frozen=True)
class FrameAnalysis:
    ear: Optional[float]
    face_detected: bool
    num_faces: int
    frames_below_threshold: int
    drowsy: bool
```

`frozen=True` makes instances immutable — the result of analysing a frame is a
historical fact and should not be mutated afterwards.

Note `ear` is `Optional[float]`: **`None` means "no face found", not "eyes
closed".** This distinction is deliberate and important, because the two
situations need different responses. If the driver has turned away, an alert
saying "DROWSY" is wrong — the right response is "driver not visible". The code
keeps them apart all the way through:

```python
if ear is None:
    self._last_ear = None
    return frame, None
```

collapsing a face detection failure to a fake `0.0` would be a design bug, since
`0.0` is a perfectly valid EAR meaning "eyes definitely shut".

### The constructor and its validation

```python
if not 0.0 < ear_threshold < 1.0:      raise ValueError(...)
if consecutive_frames <= 0:             raise ValueError(...)
if detector_upsample < 0:               raise ValueError(...)
```

Fail fast and loudly on nonsense parameters. `dlib.shape_predictor` is only
constructed **after** the path has been checked with `.is_file()`, so you get
the actionable `FileNotFoundError` with download instructions rather than a
cryptic dlib failure.

Two models are built once, in the constructor, and reused for every frame:

- `dlib.get_frontal_face_detector()` — the HOG + linear SVM face detector.
  The comment explains the choice: it is **CPU-only, ~30–60 FPS at VGA**, and
  far cheaper than dlib's CNN detector, which needs a GPU to keep up with a live
  camera. On a vehicle's embedded computer with no discrete GPU, HOG is the
  practical option.
- `dlib.shape_predictor(path)` — the 68-point regressor.

Both are loaded once because model loading is expensive (the 95 MB predictor
takes a moment) and doing it per frame would destroy the frame rate.

### `process_frame()` — the per-frame pipeline

```python
if frame.ndim == 2:      gray = frame
elif frame.ndim == 3:    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else: raise ValueError(...)
```

The function accepts **either** a colour BGR frame **or** an already-grayscale
frame. That matters in a ROS 2 pipeline, where the incoming `sensor_msgs/Image`
may already be `mono8` — converting again would be wasted work. The `else` branch
catches genuinely malformed input (a 4-channel BGRA image, say) with a clear
message instead of letting it reach dlib.

```python
faces = self._detector(gray, self.detector_upsample)
self._num_faces = len(faces)
face = self._largest_face(faces)
```

`_largest_face()` picks the detection with the greatest box area:

> With more than one face in frame (passenger, rear-seat, a face on a
> billboard), the driver is assumed to be the closest to the camera.

That is a **heuristic**, and worth being honest about: it is right in the common
case and wrong when a passenger leans forward. A production DMS for a real
vehicle would need seat-position priors, or a face-recognition step to identify
the driver. For a demo or a research prototype, largest-face is the standard
approach.

The `upsample` parameter is dlib's resolution boost — `1` doubles each dimension
(so **four times** the pixels to scan) to find smaller faces at a substantial
speed cost; `0` is fastest. Since the driver sits close to the camera in a car,
`0` is often the right choice, but the default of `1` is more forgiving when
testing at a desk.

```python
shape = self._predictor(gray, face)
left_eye = self._eye_points(shape, LEFT_EYE_INDICES)
right_eye = self._eye_points(shape, RIGHT_EYE_INDICES)
```

`_eye_points()` is a thin helper that pulls the six requested indices out of the
`full_object_detection` into a `(6, 2)` float64 array — the exact shape
`calculate_ear()` expects. `float64` rather than `int32` because the
coordinates come out of the regressor as sub-pixel values and the distance maths
benefits from the precision.

Note that **both eyes are always measured**, even if the analysis only needs one,
because the average is the signal.

### `calculate_ear()` — the maths

```python
points = np.asarray(eye_landmarks, dtype=np.float64)
if points.ndim == 1:
    points = points.reshape(-1, 2)
if points.shape != (6, 2):
    raise ValueError(f"expected 6 landmarks of shape (6, 2), got shape {points.shape!r}.")

vertical_a = euclidean(points[1], points[5])   # ‖p2 − p6‖
vertical_b = euclidean(points[2], points[4])   # ‖p3 − p5‖
horizontal = euclidean(points[0], points[3])   # ‖p1 − p4‖

return float((vertical_a + vertical_b) / (2.0 * horizontal))
```

Details worth noticing:

- The `ndim == 1` reshape accepts a flat `[x1, y1, x2, y2, ...]` sequence as a
  convenience, so callers with flattened data do not have to reshape first.
- The index arithmetic is **0-based** while the paper's notation is 1-based.
  `points[1]` is `p2`, `points[5]` is `p6`, and so on. When comparing this code
  against the formula, subtract one from every subscript. This trips people up;
  the inline comments label each line with the paper's notation.
- It is a `@staticmethod`, so you can call
  `DrowsinessDetector.calculate_ear(points)` without constructing a detector —
  handy for testing or for reusing the metric on recorded landmark data.
- `scipy.spatial.distance.euclidean` is used rather than a hand-rolled
  `np.linalg.norm` call. The docstring states the reason honestly: the calls are
  kept readable and directly comparable to the reference paper rather than
  micro-optimised. This runs six times per frame, which is nothing next to the
  face detection; clarity wins.

### `check_drowsiness()` — the temporal rule

This is the part that separates a blink from a microsleep:

```python
if ear is None or ear >= self.ear_threshold:
    self._frames_below_threshold = 0
    return False

self._frames_below_threshold += 1
return self._frames_below_threshold >= self.consecutive_frames
```

The counter **increments** on a closed-eye frame and **resets to zero**
otherwise. Drowsiness is reported once the counter reaches
`consecutive_frames`, and stays reported while the eyes stay shut.

The important subtlety is in the first line: **a missing measurement resets the
counter too.** The docstring explains why:

> an occlusion is not evidence of eye closure, and counting it would fire alerts
> whenever the driver looks away.

If `None` were treated as "closed" instead, then every time the driver checked a
mirror or turned their head, the counter would keep climbing and you would get a
false drowsiness alert within a second. Resetting is the correct behaviour.

**Doing the arithmetic on the default settings:** at 30 FPS, `consecutive_frames
= 20` corresponds to 20 / 30 ≈ **0.67 seconds** of continuous closure. A blink is
100–400 ms (roughly 3–12 frames), so it never reaches the threshold. That gap
between ~0.4 s and ~0.67 s is the safety margin the default is chosen for. In
microsleep research the standard criterion is closer to **1–3 seconds** of
closure, so for a real system you would raise `consecutive_frames` to roughly
30–90 at 30 FPS depending on how early you want to warn.

### State and threading

```python
def reset(self) -> None:
    """Clears all temporal state."""
```

The detector keeps per-stream state (`_frames_below_threshold`, `_last_ear`,
`_num_faces`), so:

- It is **not thread-safe**. Use one instance per camera or video stream.
- Call `reset()` after any discontinuity — a camera reconnect, a gap in a ROS 2
  topic, or a driver change — so stale closed-eye frames from before the gap
  cannot combine with new ones to trigger a spurious alert.

Read-only properties (`ear`, `frames_below_threshold`, `num_faces`,
`is_drowsy`) expose the state for logging and diagnostics without letting a
caller corrupt the counters.

### `analyze()` — the method to integrate against

```python
annotated, ear = self.process_frame(frame, draw=draw)
drowsy = self.check_drowsiness(ear)
return annotated, FrameAnalysis(ear=ear, face_detected=ear is not None,
                                num_faces=self._num_faces,
                                frames_below_threshold=self._frames_below_threshold,
                                drowsy=drowsy)
```

Combines the measurement and the temporal rule, and returns a typed snapshot.
This is what a ROS 2 callback should call, as the module docstring demonstrates.

Note `process_frame()` annotates the input array **in place** when `draw=True` —
it is the same object returned. That is intentional zero-copy behaviour for a
video pipeline, but it means you must copy the frame first if it feeds another
consumer (a recording, a second model). Pass `draw=False` when you only want the
number.

### `main()` — the demo harness

The demo is written with more care than a throwaway test loop, and the details
are worth borrowing:

**FPS smoothing.**
```python
fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
```
An **exponential moving average**. The raw per-frame rate at 30 Hz is too jittery
to read on screen; the EMA keeps 90% of the previous estimate and blends in 10%
of the new one, so the number settles down while still responding to real changes
within a fraction of a second.

**Read-failure tolerance.**
```python
if not ok or frame is None:
    read_failures += 1
    if read_failures >= 30:
        break
    continue
read_failures = 0
```
Transient USB and decoder hiccups happen constantly with webcams. Giving up on
the first dropped frame would make the demo unusable; the counter allows about a
second of failures (`continue` skips the frame without treating it as a
shutdown), and a successful read resets it.

**The HUD.** Three readouts — EAR (turning red when drowsy), FPS, and
`Closed frames: 12/20`. That last one is the most useful for tuning: you can
watch the counter climb as you close your eyes and immediately see how close you
are to the threshold.

**The `NO FACE DETECTED` overlay.** The visible expression of the `None` vs
`0.0` distinction from section 4 — the driver not being visible is reported as
its own state, in yellow, not as a drowsiness alert.

**The alert.** A red border around the whole frame plus
`WARNING: DROWSINESS DETECTED` at 85% height. Deliberately hard to miss. In a
real vehicle this is also where you would publish a ROS 2 alert message or
trigger a haptic/audio warning.

**Cleanup on every path.**
```python
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
finally:
    capture.release()
    cv2.destroyAllWindows()
```
The `finally` block runs on the error paths, on `Ctrl+C`, and on a normal quit,
so the camera is never left locked. `KeyboardInterrupt` is caught so that `Ctrl+C`
prints a tidy message rather than a traceback.

**The session summary.** On exit it prints frames processed, elapsed time,
average FPS, and how many frames were spent in the drowsy state — a quick sanity
check that the run behaved.

---

## 5. Using it as a library

```python
from eye_tracker import DrowsinessDetector

# Build once, outside the loop.
detector = DrowsinessDetector(
    predictor_path="shape_predictor_68_face_landmarks.dat",
    ear_threshold=0.25,
    consecutive_frames=20,
)

# Then, per frame:
annotated, analysis = detector.analyze(frame, draw=True)

if analysis.ear is None:
    ...                     # driver not visible — say so, do not alert
elif analysis.drowsy:
    ...                     # sustained eye closure — raise the alert
else:
    ...                     # normal; analysis.ear has the current ratio
```

The ROS 2 integration the module was written for:

```python
detector = DrowsinessDetector(ear_threshold=0.25, consecutive_frames=20)

def on_image(msg) -> None:
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    annotated, analysis = detector.analyze(frame, draw=False)
    if analysis.drowsy:
        publish_alert(analysis.ear)
```

Two things to carry over from the class design if you build on it: call
`detector.reset()` whenever the topic stops publishing for a while, and create
**one detector per stream** rather than sharing one across callbacks.

---

## 6. Tuning

The two parameters interact, and both depend on your frame rate.

| Parameter | Meaning | Default |
|---|---|---|
| `ear_threshold` | EAR below which an eye counts as closed | `0.25` |
| `consecutive_frames` | Closed frames in a row needed to alert | `20` |

**Calibrating the EAR threshold.** The published range is 0.18–0.28, but it is
person- and setup-dependent: glasses, contact lenses, eye shape, camera height
and lighting all shift it. The reliable way to set it is empirically:

1. Run the demo and watch the `EAR:` readout.
2. Note the value with your eyes comfortably **open** (look straight ahead), and
   with your eyes **closed**.
3. Set the threshold **between** the two, closer to the open value — typically
   60–75% of the way down.

If open-eye EAR is 0.31 and closed is 0.12, a threshold of ~0.22 sits in the
gap with margin on both sides. The default `0.25` is a safe general starting
point, not an optimum.

**Setting `consecutive_frames`.** Convert your intended closure duration into
frames: `frames = seconds × fps`. At 30 FPS:

| Intended criterion | Frames |
|---|---|
| Longer than a blink (~0.4 s) | 12 |
| Default (~0.67 s) | 20 |
| Microsleep onset (~1 s) | 30 |
| Conservative warning (~2 s) | 60 |

Use `--consecutive-frames` to experiment without editing code.

**Failure modes to be aware of:**

- **Yawning** raises the eyebrows and squints the eyes; heavy yawning can push
  the EAR down and cause false positives. Real DMS systems measure mouth opening
  as well (landmarks 48–67) to disambiguate.
- **Sunglasses** block the eye region entirely, so the landmark model produces
  poor eye points. The EAR becomes unreliable. Production systems typically need
  infrared illumination and a dedicated eye model for this case.
- **Head pose.** A large yaw makes one eye nearly edge-on, flattening its EAR.
  The averaging over two eyes mitigates but does not eliminate this.
- **Low light** hurts the HOG detector first — faces stop being found at all,
  which surfaces as `EAR: n/a` rather than a wrong number.

A production system would add per-driver calibration, head-pose compensation,
and often a PERCLOS-style statistic (the *percentage* of time the eyes are closed
over a rolling window) instead of a single consecutive-frame count.

---

## 7. CLI reference

```powershell
python eye_tracker.py [--camera N] [--predictor PATH]
                     [--ear-threshold F] [--consecutive-frames N]
```

| Flag | Default | Effect |
|---|---|---|
| `--camera` | `0` | OpenCV camera index |
| `--predictor` | `shape_predictor_68_face_landmarks.dat` beside the script | Path to the model |
| `--ear-threshold` | `0.25` | EAR below which the eyes count as closed |
| `--consecutive-frames` | `20` | Closed frames needed to raise the alert |

Exit codes: `0` on a clean quit, `1` on a setup failure (missing model, invalid
threshold, camera unavailable).

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dlib shape predictor not found` | Download the model (section 2), or pass `--predictor PATH`. |
| `pip install dlib` fails with `CMake is not installed` | Install CMake and add it to PATH. |
| `pip install dlib` fails with `Cannot find Visual Studio` | Install VS Build Tools with the "Desktop development with C++" workload. |
| The dlib build takes many minutes | Expected. It compiles from source; there are no Windows wheels on PyPI. |
| `EAR: n/a` all the time | No face is being detected. Improve the lighting, move closer, or raise `detector_upsample`. |
| `Could not open camera index 0` | Another app holds the webcam, or Windows camera privacy blocks desktop apps. |
| Constantly alerting | Threshold too high for your eyes/brightness — calibrate per section 6. |
| Never alerting | Threshold too low, or `consecutive_frames` too high for your frame rate. |
| `ModuleNotFoundError: No module named 'scipy'` | `pip install -r requirements.txt` was not run in this venv. |
| Importing this into a project that already uses OpenCV 5 | Use a separate environment — this project targets OpenCV 4. |

---

## 9. Where to go next

- **Understand the landmarks.** Run `python ../face-landmark-project/detect_68point.py --indices`
  with the face project's environment to see indices 36–47 on your own eyes.
  That is the data this whole module is built on.
- **Record a session.** `detect_68point.py --landmarks-json lm.jsonl` writes the
  raw eye coordinates per frame, so you can develop and test the EAR logic
  offline, against real data, without a camera.
- **Add mouth analysis.** Landmarks 48–67 give you the mouth opening, which
  disambiguates yawning from drowsiness.
- **Compute PERCLOS.** Replace the consecutive-frame counter with the fraction of
  closed-eye frames in a rolling window; it is the metric used in the drowsiness
  literature and is more robust to blink bursts.
