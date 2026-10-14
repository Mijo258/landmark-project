# Face Landmark Detection — 5, 68 and 478 Points

This folder answers one question three different ways: **where are the important
points on a human face in this image?**

It contains three independent programs. Each one uses a different model and a
different notion of what a "landmark" is, so you can see the same problem solved
at three levels of detail:

| Script | Runtime | Points | Model file | Best for |
|---|---|---|---|---|
| `detect_5point.py` | dlib | 5 | `shape_predictor_5_face_landmarks.dat` (9 MB) | Face alignment, fast head-pose, face recognition pipelines |
| `detect_68point.py` | dlib | 68 | `shape_predictor_68_face_landmarks.dat` (95 MB) | Classic facial analysis — eyes, brows, nose, lips as separate shapes |
| `detect_facemesh.py` | MediaPipe | 468 or 478 | `face_landmarker.task` (3.6 MB) | Dense mesh, AR filters, iris/gaze, blendshapes |

All three read your webcam with OpenCV and draw the result live in a window.
Press **`q`** or **`Esc`** to quit any of them.

> **Read the root [`../README.md`](../README.md) first** if you have not set up
> Git, Python or the virtual environments yet. This file assumes those are done.

---

## 1. Two environments, not one

This is the single most important thing to understand about this folder:
**`detect_facemesh.py` cannot share an environment with the other two scripts.**

| Environment | Folder | Serves | Why |
|---|---|---|---|
| Main | `venv/` | `detect_5point.py`, `detect_68point.py` | Needs `dlib` + `opencv-python` |
| Face Mesh | `venv_facemesh/` | `detect_facemesh.py` | Needs `mediapipe` |

The reason is a genuine packaging conflict, and it is worth knowing because you
will meet it again:

`mediapipe` depends on **`opencv-contrib-python`**. The dlib scripts are built
on **`opencv-python`**. These are two *different distributions on PyPI that
both install the same `cv2` package* into the same folder. If you install both
into one environment, `cv2` becomes a mixture of files from whichever wheel was
unpacked last, and neither project can be trusted afterwards. `mediapipe` also
has opinions about the `numpy` version, and the dlib/OpenCV 5 stack in
`requirements.txt` is pinned to `numpy==2.4.6` — letting pip re-resolve `numpy`
underneath `dlib` risks the NumPy 1.x/2.x ABI mismatch that compiled extensions
are sensitive to.

Keeping them apart means nothing installed for the mesh script can break the
5-point or 68-point scripts.

### Set up the main environment (dlib scripts)

```powershell
cd face-landmark-project
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Set up the Face Mesh environment

```powershell
cd face-landmark-project
python -m venv venv_facemesh
.\venv_facemesh\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-facemesh.txt
```

### The model files

Two of the three model files are committed to this repository (they are small).
The large dlib one is not — download it once:

```powershell
# ~95 MB decompressed. Required by detect_68point.py.
curl.exe -L -o shape_predictor_68_face_landmarks.dat.bz2 `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
python -c "import bz2,shutil; shutil.copyfileobj(bz2.open('shape_predictor_68_face_landmarks.dat.bz2','rb'), open('shape_predictor_68_face_landmarks.dat','wb'))"

# 9 MB. Required by detect_5point.py.
curl.exe -L -o shape_predictor_5_face_landmarks.dat `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
```

Afterwards, check the sizes — a truncated download is the most common cause of
a confusing "could not load model" error:

| File | Expected size |
|---|---|
| `shape_predictor_68_face_landmarks.dat` | ~99,693,937 bytes (~95 MB) |
| `shape_predictor_5_face_landmarks.dat` | ~9,150,489 bytes (~9 MB) |
| `face_landmarker.task` | ~3,758,596 bytes (~3.6 MB) |

Use `curl.exe`, **not** `curl`. In PowerShell, plain `curl` is an alias for
`Invoke-WebRequest`, which takes different arguments and will not do what the
command above says.

---

## 2. `detect_5point.py` — the minimal markup

The shortest of the three scripts, and the clearest place to learn the dlib
pipeline. It has no command-line options at all: `python detect_5point.py` and
it runs.

### What the 5 points are

dlib's 5-point model regresses the four **eye corners** and the **nose tip**.
The conventional index order (right/left here mean image-left / image-right) is:

| Index | Point |
|---|---|
| 0 | right eye, outer corner |
| 1 | right eye, inner corner |
| 2 | left eye, inner corner |
| 3 | left eye, outer corner |
| 4 | nose tip |

Five points is enough to compute an affine transform that scales, rotates and
translates a face into a canonical pose — which is exactly what a face
recognition system needs before it feeds a crop to its embedding network. That
is why this model is small (9 MB) and fast: it does one job well.

### Reading the code

**Locating the model (lines 10–11).** The path is built from
`os.path.dirname(os.path.abspath(__file__))` rather than a relative string like
`"shape_predictor_5_face_landmarks.dat"`. `__file__` is the path of the script
itself, so the model is always found *next to the script* no matter which
directory you launched Python from. Using a bare relative path instead would
make the program work from this folder and fail from everywhere else — a very
common beginner bug.

**Loading the two models (lines 16–17).**

```python
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
```

These are two separate models doing two separate jobs, and the distinction
matters for the whole rest of this folder:

- `get_frontal_face_detector()` is a **detector**. It answers "is there a face
  here, and roughly where?" — it returns bounding boxes, not precise points. It
  uses HOG (Histogram of Oriented Gradients) features and a linear SVM over a
  sliding window, runs entirely on the CPU, and is fast because it is coarse.
- `shape_predictor(...)` is a **regressor**. It answers "given this box, where
  exactly are the features?" It is an ensemble of regression trees that takes
  the pixels inside the box and predicts point coordinates directly. It cannot
  find a face on its own — it needs the detector's box to start from.

The `try/except` around them converts a crash into a readable message, which
matters most for the missing-file case.

**Opening the camera (line 23).** `cv2.VideoCapture(0)` grabs the default
built-in webcam. The `0` is a device index; if you have an external USB webcam
and want it instead, change this to `1` or `2`. `cap.isOpened()` returning
`False` means the camera was busy, blocked by Windows privacy settings, or the
index does not exist — the script checks this rather than failing later with a
confusing error.

**The frame loop (lines 31–70).**

```python
ret, frame = cap.read()
```

Every iteration pulls one frame. `ret` is a success flag and `frame` is a
NumPy array of shape `(height, width, 3)` in **BGR** channel order — not RGB.
OpenCV has used BGR as its native order since its beginnings, and forgetting
this is the classic cause of "why is my face blue?" bugs. Nothing here converts
to RGB, so it does not come up; but `detect_facemesh.py` later *does* have to
convert, and that is where it bites.

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 0)
```

Every dlib model here works on a single-channel grayscale image. Colour carries
no information about the *shape* of a face, so throwing it away cuts the data
volume to a third and speeds everything up. The second argument to `detector` is
the **upsample count**: how many times dlib doubles the image before scanning.
`0` means no upsampling — fastest, and fine for a face that fills a good part of
the frame. Raising it finds smaller, more distant faces but costs a lot: one
upsample doubles each dimension, which is **four times the pixels** to scan.

**Drawing the box (lines 45–48).** A dlib detection is a `rectangle` object,
queried with `.left()`, `.top()`, `.width()`, `.height()`. `cv2.rectangle` wants
two corner points — top-left and bottom-right — so the bottom-right is computed
as `(x + w, y + h)`. The `2` is line thickness in pixels and `(0, 255, 0)` is
BGR green.

**Predicting and plotting the points (lines 51–63).**

```python
shape = predictor(gray, face)
for i in range(5):
    pt = shape.part(i)
    pt_coords = (pt.x, pt.y)
    cv2.circle(frame, pt_coords, 3, (0, 0, 255), -1)
```

`predictor(gray, face)` returns a `full_object_detection` — a container of
points. `.part(i)` returns one point with integer `.x` and `.y` in the
original image's pixel coordinates. Note that the predictor is given the *same*
grayscale image the detector saw, so the coordinates are already in the right
space; no rescaling is needed.

`cv2.circle(frame, center, radius, color, thickness)`: radius `3`, BGR red
`(0, 0, 255)`, and **thickness `-1`**, which is a special value meaning "fill
the shape" rather than "draw an outline of this width".

The label is offset by `(pt.x + 4, pt.y - 4)` so the number sits beside the dot
instead of on top of it.

**Exit and cleanup (lines 69–74).** `cv2.waitKey(1)` waits one millisecond for a
keypress and *also* pumps the GUI event queue — without it, the window would
freeze and never repaint. The `& 0xFF` masks the return value to the low byte
because on some platforms `waitKey` returns extra bits in the higher bytes;
`ord('q')` is only 113, so a raw comparison would silently never match on those
platforms. This idiom appears in all five scripts in this repository.

`cap.release()` hands the camera back to the operating system. Skipping it
leaves the device locked and the *next* run of the script will fail to open the
camera until you close the Python process.

---

## 3. `detect_68point.py` — the classic iBUG markup

The 68-point markup is the standard of classical facial analysis. It was
defined for the **iBUG 300-W** challenge and it is what almost every face
paper published between 2014 and 2020 assumes. dlib's
`shape_predictor_68_face_landmarks.dat` is the reference implementation of it.

### The topology

The 68 points are not 68 independent dots — they are eight **ordered shapes**
concatenated into one list. Knowing the index ranges is the whole game, because
everything you do downstream (measure the eyes, draw the lips) is slicing these
ranges:

| Indices | Region | Count | Geometry |
|---|---|---|---|
| `0–16` | jaw line | 17 | open polyline, ear to ear |
| `17–21` | right eyebrow | 5 | open polyline |
| `22–26` | left eyebrow | 5 | open polyline |
| `27–35` | nose | 9 | bridge (27–30) + nostrils (31–35) |
| `36–41` | right eye | 6 | **closed** polygon |
| `42–47` | left eye | 6 | **closed** polygon |
| `48–59` | outer lip | 12 | **closed** polygon |
| `60–67` | inner lip | 8 | **closed** polygon |

"Right" and "left" are from the **subject's** point of view, so index 36–41 is
the eye on the *left* of your screen. The eyes and lips are closed loops — the
last point connects back to the first — which is why they can be drawn as
outlined shapes and why a contour or a convex hull of them is meaningful.

This script encodes exactly that table in `REGION_SPECS`, as
`(name, start, end_exclusive, closed, colour)` tuples. Two design choices are
worth noticing:

1. The ranges **partition `0..67` exactly** — no gaps, no overlaps. That means
   any index maps to exactly one region, which is what lets
   `build_point_colors()` build a length-68 colour table by simple assignment.
2. `end` is **exclusive**, matching Python slicing, so the drawing code can use
   `points[start:end]` directly with no off-by-one arithmetic.

### Reading the code

#### Model loading, done defensively

`load_predictor()` (lines 129–159) does something the 5-point script does not:
it **validates what it loaded**. After constructing the predictor it calls
`predictor_num_parts()` and checks the answer is 68. Feed it
`shape_predictor_5_face_landmarks.dat` by mistake and you get:

> `'shape_predictor_5_face_landmarks.dat' predicts 5 points, but this script
> expects 68. You probably passed the 5-point model... Use detect_5point.py`

instead of an `IndexError` from deep inside the drawing loop. This is the
difference between a script and a tool.

`predictor_num_parts()` exists because the dlib Python bindings have exposed
`num_parts` both as an attribute and as a method across versions. The helper
reads it either way and returns `None` if it cannot, in which case the check is
skipped rather than crashing — a small piece of defensive engineering against
a dependency changing shape underneath you.

`model_missing_message()` builds a multi-line error containing the exact
`curl` and decompression commands to fix the problem. The failure mode teaches
you how to fix it.

#### Detection at reduced resolution

This is the most interesting function in the file —
`detect_landmarks()`, lines 231–273:

```python
scale = 1.0
detect_image = frame
if process_width is not None and frame.shape[1] > process_width:
    scale = process_width / float(frame.shape[1])
    detect_image = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)

results = []
for rect in detector(gray, upsample):
    shape = predictor(gray, rect)
    points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)
    if scale != 1.0:
        points /= scale
    results.append(points.astype(np.int32))
```

Three things are happening:

**Downscaling for speed.** A 1080p frame is ~2 million pixels; at
`--process-width 640` it becomes ~0.3 million. HOG detection is roughly linear
in pixel count, so this is a large constant-factor win. Both the detector *and*
the predictor run on the small image, so the regression is also cheaper.

**Mapping coordinates back.** The landmarks come out in the *downscaled*
image's coordinate system and must be expressed in the original frame's
coordinates or they will be drawn in the wrong place. Since
`scale = process_width / original_width`, dividing by `scale` multiplies by
`original_width / process_width` — the exact inverse. Both `x` and `y` use the
same factor, which is correct because the resize preserved the aspect ratio
(`fx == fy`). Note the landmarks are returned as `int32` pixels, rounded at the
end.

**`cv2.INTER_AREA` is the right interpolation for shrinking.** It averages
blocks of source pixels, which is a proper low-pass filter: it does not invent
detail that is not there, so the downscaled image is a faithful, anti-aliased
version. `INTER_LINEAR` or `INTER_NEAREST` used for downscaling would alias and
hurt detection accuracy.

One subtlety: the frame has **already been mirrored** by the time this runs
(see below), so the returned points are in mirrored coordinates and are drawn
correctly without any extra flip.

#### The read-ahead frame loop

`main()` uses a loop shape that is slightly unusual and worth understanding:

```python
ok, frame = cap.read()          # one read BEFORE the loop
...
while True:
    if not ok:
        read_failures += 1
        if read_failures >= args.max_read_failures:
            break
    else:
        read_failures = 0
        ...process frame...
    ...display frame...
    frame_index += 1
    ok, frame = cap.read()       # read again at the END
```

Reading at the bottom means the loop body always has a frame in hand without a
`continue`-and-retry dance, and a failed read does not abort immediately.
Cameras drop frames — a USB hiccup, a driver stall — and treating one dropped
frame as fatal would make the program useless. Only
`--max-read-failures` consecutive failures (default 30, about a second at 30 FPS)
end the session.

#### Frame mirroring

```python
if not args.no_mirror:
    frame = cv2.flip(frame, 1)
```

`flip(frame, 1)` mirrors horizontally (`1` = around the vertical axis). A
selfie view feels natural because you expect a mirror. Disable it with
`--no-mirror` when you need the image as the camera actually sees it — which
matters if you are recording data or comparing against another sensor.

#### FPS measurement

```python
frame_times = collections.deque(maxlen=30)
...
now = time.perf_counter()
frame_times.append(now - previous)
previous = now
avg = sum(frame_times) / len(frame_times)
draw_fps(frame, 1.0 / avg if avg > 0 else 0.0)
```

A `deque` with `maxlen=30` is a rolling window: appending the 31st item silently
drops the oldest. Averaging 30 frame intervals gives a number you can actually
read, where a single-frame reading would flicker too fast to see. Note it
averages *intervals* and then inverts, rather than averaging per-frame rates —
those are not the same thing, and the interval average is the correct one.
`time.perf_counter()` is used rather than `time.time()` because it is monotonic
and high-resolution; `time.time()` can jump if the system clock is adjusted.

#### Video recording compatibility

```python
def fourcc(code):
    factory = getattr(cv2, "VideoWriter_fourcc", None)
    if factory is None:
        factory = cv2.VideoWriter.fourcc
    return int(factory(*code))
```

`requirements.txt` pins **OpenCV 5**, where this helper moved onto the
`VideoWriter` class; older OpenCV 4 code calls it at module level. The
`getattr` probe lets the same script run on both. `make_writer()` then tries
H.264 (`"avc1"`) first and falls back to `"mp4v"`:

- `avc1` gives smaller files and wider compatibility, but depends on an H.264
  encoder being present on the system. When it is missing, `VideoWriter` does
  not raise — it silently fails and `isOpened()` returns `False`.
- `mp4v` (MPEG-4 Part 2) is available in essentially every OpenCV build.

Hence the loop: try a codec, check `isOpened()`, release and try the next. Both
codecs carry no audio; there is no audio track in this pipeline.

#### JSON Lines output

`--landmarks-json FILE` appends one JSON object per frame, one per line:

```json
{"frame": 0, "timestamp": 1750000000.12, "faces": [[[123, 45], [130, 47], ...]]}
```

This format — **JSON Lines** (`.jsonl`) — is chosen deliberately over a single
big JSON array. Each line is independently parseable, so a crash or a power cut
during a long recording still leaves you a valid file with all the frames
written up to that point, and you can stream it line by line without loading
the whole thing into memory. The `"faces"` field is a list (one entry per
detected face) of 68 `[x, y]` pairs. Coordinates are in the **mirrored** frame
unless you passed `--no-mirror`.

### Running it

```powershell
.\venv\Scripts\Activate.ps1
python detect_68point.py
```

| Flag | Effect |
|---|---|
| `--camera N` | Camera index (default 0). Try 1 or 2 for external webcams |
| `--width N` / `--height N` | Requested capture resolution. A request, not a command — drivers may ignore it |
| `--upsample N` | dlib upsample passes (default 1). Raise it for small/distant faces |
| `--process-width N` | Downscale to this width for detection. Faster on big frames |
| `--model PATH` | Use a predictor file somewhere else |
| `--indices` | Label every landmark with its index number (busy, but the fastest way to learn the topology) |
| `--hide-points` | Outlines only, no dots |
| `--no-mirror` | Turn off the selfie view |
| `--hide-fps` | Hide the FPS readout |
| `--save out.mp4` | Record the annotated stream |
| `--landmarks-json lm.jsonl` | Append landmarks per frame |
| `--max-read-failures N` | Consecutive dropped frames before giving up (default 30) |
| `-v` | Debug logging |

```powershell
# Label the points so you can see which index is which
python detect_68point.py --indices

# Smooth performance on a slow laptop
python detect_68point.py --process-width 640

# Record a clip and dump the landmarks at the same time
python detect_68point.py --save demo.mp4 --landmarks-json landmarks.jsonl
```

**Tip for learning the topology:** run with `--indices`, look at the screen, and
match what you see against the table in section 3. Watching index 30 land on the
tip of your nose makes the whole scheme click in a way a table cannot.

---

## 4. `detect_facemesh.py` — the dense MediaPipe mesh

A different runtime, a different model format, and a different coordinate
convention. This script produces the dense mesh you have seen in AR filters and
face-tracking demos.

### The 478-point layout

MediaPipe's Face Mesh predicts **478** points:

| Indices | Region |
|---|---|
| `0–467` | the face mesh — face oval, eyes, eyebrows, nose, lips, cheeks, forehead |
| `468–472` | **right** iris: 468 is the centre, 469–472 are the ring |
| `473–477` | **left** iris: 473 is the centre, 474–477 are the ring |

The first 468 are the original Face Mesh topology. The final 10 were added later
to support iris tracking, which is what makes gaze estimation and eye-contact
detection possible. `--mesh-only` drops them, giving you the classic 468-point
space — useful because a lot of published code and index tables assume 468, and
because a stale index table plus 10 extra points is a quiet source of bugs.

The mesh is genuinely dense: 468 points across a face means the average spacing
is a handful of pixels. That is why this script draws **curated outlines**
rather than all the points by default — 468 white dots is a white blob.
`--tesselation` draws the full wireframe instead, using MediaPipe's own edge
list, which has thousands of edges.

### Reading the code

#### MediaPipe is imported defensively

```python
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_IMPORT_ERROR = None
except Exception as exc:
    mp = mp_python = vision = None
    MEDIAPIPE_IMPORT_ERROR = exc
```

The import is wrapped so that `--help` still works in an environment without
MediaPipe, and so that running the script in the wrong venv produces
`mediapipe_missing_message()` — a message telling you to
`pip install -r requirements-facemesh.txt` — instead of a raw traceback. The
original exception is preserved and appended, so you still get the underlying
reason.

#### Two running modes, and they must match the call

```python
running_mode = (vision.RunningMode.IMAGE if args.image is not None
                else vision.RunningMode.VIDEO)
```

The MediaPipe Task API has separate entry points that are **not**
interchangeable:

- `RunningMode.IMAGE` — call `landmarker.detect(image)`. Detects from scratch on
  every call. Correct for stills.
- `RunningMode.VIDEO` — call `landmarker.detect_for_video(image, timestamp_ms)`.
  Tracks across frames, which is cheaper *and* smoother, because the model uses
  the previous frame to inform the current one. Correct for a live camera.

The mode is baked into the landmarker when it is built and MediaPipe raises if
you then call the wrong method, so the mode is chosen from the input type at
construction time.

**The timestamp rule.** `VIDEO` mode requires **strictly increasing**
millisecond timestamps. Two frames landing in the same millisecond is not a
theoretical concern at 30–60 FPS on a fast machine, and MediaPipe rejects a
repeated timestamp. Hence:

```python
timestamp_ms = int(time.perf_counter() * 1000)
if timestamp_ms <= last_timestamp_ms:
    timestamp_ms = last_timestamp_ms + 1
last_timestamp_ms = timestamp_ms
```

The clock is read, then forced to advance by at least 1 ms if it did not. The
*ordering* is what the model cares about; the exact wall-clock value does not
matter for a live stream.

#### The coordinate convention — the key difference from dlib

```python
points = np.array([[lm.x * width, lm.y * height] for lm in face], dtype=np.float64)
```

MediaPipe returns landmarks **normalised to `[0, 1]`** relative to the image,
not in pixels. To draw them you multiply `x` by the image **width** and `y` by
the image **height**. Getting that the wrong way round is one of the most common
bugs in MediaPipe code — the points look *almost* right, just subtly stretched,
which is easy to miss and hard to diagnose.

`lm.z` is also returned but deliberately ignored here: it is a relative depth
estimate, roughly on the same scale as `x`, with its origin near the head
centre. It is not metric and it is not a reliable distance measurement.

Because the coordinates are normalised, this model is resolution-independent,
which is why `--width` / `--height` can be treated as hints rather than
requirements.

#### The colour conversion

```python
rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
```

OpenCV hands you **BGR**; MediaPipe expects **RGB**. This `cvtColor` call is
where the earlier warning about channel order becomes real — skip it and the
mesh still tracks perfectly (because *shape* is what the model uses) while the
displayed colours are wrong, which is a confusing combination.

`np.ascontiguousarray` guarantees the buffer is C-contiguous, which MediaPipe's
zero-copy ingestion requires. `cv2.cvtColor` already returns a contiguous array
in practice, so this is cheap insurance; it would matter if the frame were a
non-contiguous slice or view.

#### Curated regions and the `requires_iris` flag

`REGION_SPECS` here is `(name, indices, closed, colour, requires_iris)`. The
indices are *not* contiguous ranges like the 68-point model — the mesh indexes
points by their position in the topology, so, for example, the left eye is
`[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]`,
scattered all over the index space. That is normal for a mesh.

The `requires_iris` boolean is checked before drawing:

```python
if needs_iris and not iris:
    continue
if indices and max(indices) >= count:
    continue   # defensive: model returned fewer points than expected
```

The second check guards against requesting point 477 from an array that only has
468 entries. It should never fire, but if a future model version or a
`--mesh-only` mismatch ever hands over a shorter array, this degrades to a
missing outline instead of an `IndexError` mid-frame.

#### Tesselation, and an honest limitation

`--tesselation` draws the full wireframe using MediaPipe's own connection table.
`load_tesselation_edges()` tries to import it:

```python
for module_path in ("mediapipe.python.solutions.face_mesh_connections",
                    "mediapipe.solutions.face_mesh_connections"):
```

MediaPipe **removed the legacy `mp.solutions` namespace in recent versions**, so
on a current install this import usually fails and the function returns `None`.
The script handles that honestly — it logs a warning and continues with the
region outlines:

> `MediaPipe connection tables unavailable; --tesselation disabled. Region
> outlines will still be drawn.`

So `--tesselation` is best-effort by design. The curated `REGION_SPECS`
outlines, by contrast, need nothing but the landmark coordinates and always work.

#### Single-image mode

```powershell
python detect_facemesh.py --image photo.jpg
```

Reads the file, runs one `detect()` call, draws, and writes
`photo_mesh.jpg` beside the input (or wherever `--output-image` points). This is
the mode to use for testing on a fixed image rather than waving at a webcam, and
it is the only path that uses `RunningMode.IMAGE`.

### Running it

```powershell
.\venv_facemesh\Scripts\Activate.ps1
python detect_facemesh.py
```

| Flag | Effect |
|---|---|
| `--image FILE` | Process one still instead of the webcam |
| `--output-image FILE` | Where to write the annotated still (default `<name>_mesh.<ext>`) |
| `--mesh-only` | 468-point space, iris dropped |
| `--tesselation` | Full wireframe (best-effort, see above) |
| `--num-faces N` | Maximum faces to track (default 1) |
| `--camera N`, `--width`, `--height` | Capture settings |
| `--hide-points`, `--no-mirror`, `--hide-fps` | Display options |
| `--save FILE`, `--landmarks-json FILE` | Recording and landmark dumps |
| `--model PATH` | Use a different `.task` bundle |
| `--max-read-failures N`, `-v` | Robustness and logging |

```powershell
python detect_facemesh.py --mesh-only           # classic 468-point space
python detect_facemesh.py --image selfie.jpg    # still image
python detect_facemesh.py --num-faces 3         # track a group
```

---

## 5. How the three compare

**Coordinate output.** dlib returns **absolute pixel** coordinates. MediaPipe
returns **normalised `[0, 1]`** coordinates. If you are writing code that
consumes landmarks from both, normalise the dlib output or you will spend an
afternoon confused.

**Model format.** dlib models are `.dat` files loaded by path. MediaPipe Tasks
models are `.task` bundles (a zipped TFLite graph plus metadata). Neither is a
pip package — `pip install dlib` gives you the *code*, not the *model*. This
surprises almost everyone the first time.

**Accuracy vs. density.** The 68-point model puts a handful of points on
semantically meaningful locations and is very accurate at each. The mesh puts
roughly seven times as many points almost uniformly across the face. For
measuring the eye opening or the mouth width, 68 points are easier to reason
about. For warping a texture onto a face or driving a 3D avatar, you need the
mesh.

**Speed.** For a rough ordering on a typical laptop CPU at VGA: the 5-point
pipeline is fastest, the 68-point pipeline is close behind at `--upsample 0` but
loses ground at higher upsample counts, and the 478-point mesh is the heaviest
of the three — though MediaPipe's VIDEO-mode tracking recovers much of that by
not re-detecting every frame.

**Iris.** Only the MediaPipe model gives you the iris. If you need gaze
direction or "is the driver looking at the road", that is the one to use — the
68-point model has no points inside the eye at all, only the eyelid contour.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Could not open camera index 0` | Another app holds the webcam (Teams, Zoom, Camera). Close it. Also check **Settings → Privacy & security → Camera** allows desktop apps. Try `--camera 1`. |
| `Model file not found` | The download did not run, or ran into the wrong folder. The file must sit **next to the script**. Check the size against the table in section 1. |
| `dlib could not load ... The file may be truncated or still bzip2-compressed` | You pointed at the `.bz2` file, or the download was cut short. Decompress it. A `.dat` that starts with `BZh` is still compressed. |
| `predicts 5 points, but this script expects 68` | You passed the 5-point model to `detect_68point.py`. Use `detect_5point.py`, or point `--model` at the 68-point file. |
| `ModuleNotFoundError: No module named 'mediapipe'` | You are in the wrong venv. `detect_facemesh.py` needs `venv_facemesh`, not `venv`. |
| `MediaPipe could not load ... not an HTML error page` | The `.task` download saved a 404 page. Re-download and check the size is ~3.6 MB. |
| `.\venv\Scripts\Activate.ps1 ... running scripts is disabled` | PowerShell execution policy. See the root README's troubleshooting section. |
| Video window freezes or never appears | The `waitKey` call is what pumps the GUI event queue. If you modify the script, make sure you keep it inside the loop. |
| `--save` produces a 0-byte file | No codec opened. The script already falls back from `avc1` to `mp4v`; if both fail, try a `.avi` extension. |
| Detection is slow | Add `--process-width 640`. Lower `--upsample` to 0. Close other apps using the GPU. |

---

## 7. Where to go next

- **Learn the topology hands-on.** `python detect_68point.py --indices` and
  watch which number lands where.
- **Dump and analyse.** `--landmarks-json` gives you a clean dataset to plot,
  measure or feed to a classifier.
- **Blink and drowsiness.** The `../eye-landmark-project/` folder uses this same
  68-point model to measure eye opening over time. It is the natural next step
  from here.
- **Hands.** The `../hand-landmark-project/` folder applies the same ideas to a
  21-point hand skeleton.
