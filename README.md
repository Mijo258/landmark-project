# Landmark Project — Face, Eye and Hand Landmark Detection

A collection of computer-vision projects that find **landmarks** — specific,
named points on the human body — in images and live webcam video. All of them
use Python, OpenCV and a pre-trained machine-learning model, and all of them run
in real time on an ordinary laptop CPU.

| Project | What it does | Landmarks | Runtime |
|---|---|---|---|
| [`face-landmark-project/`](face-landmark-project/) | Face detection and landmark detection, three ways | 5, 68, 468/478 | dlib, MediaPipe |
| [`eye-landmark-project/`](eye-landmark-project/) | Driver drowsiness detection from eye opening | 6 per eye (from 68) | dlib |
| [`hand-landmark-project/`](hand-landmark-project/) | Hand tracking and finger geometry | 21 | MediaPipe |

Each project has its own README with a full code walkthrough. This file is the
**setup and download guide** — read it first, in order, top to bottom.

**Never used Git, GitHub, a terminal or a virtual environment before?** That is
exactly who this document is written for. Follow it literally and you will end
up with all three projects running. Nothing is assumed.

---

## Table of contents

1. [What you are about to install](#1-what-you-are-about-to-install)
2. [Part A — Git and GitHub from zero](#part-a--git-and-github-from-zero)
3. [Part B — Put this project on GitHub](#part-b--put-this-project-on-github)
4. [Part C — Clone it on another computer](#part-c--clone-it-on-another-computer)
5. [Part D — Install and run the projects](#part-d--install-and-run-the-projects)
6. [Part E — Everyday Git commands](#part-e--everyday-git-commands)
7. [Troubleshooting](#troubleshooting)
8. [Quick reference](#quick-reference)

---

## 1. What you are about to install

You need three things on your computer:

| Software | Why | Where |
|---|---|---|
| **Python 3.11** | Runs all the code | <https://www.python.org/downloads/> |
| **Git** | Downloads the code and tracks your changes | <https://git-scm.com/download/win> |
| **A code editor** (optional but recommended) | To read and edit the code | <https://code.visualstudio.com/> |

Everything else — OpenCV, dlib, MediaPipe, NumPy — is installed automatically in
Part D. **Do not install those by hand**; that is what the requirements files
are for.

### The folder layout

Once set up, this is what you will have:

```
landmark-project/
├── README.md                       <- you are here
├── .gitignore                      <- tells Git what NOT to save
│
├── face-landmark-project/
│   ├── README.md                   <- full code walkthrough for all 3 scripts
│   ├── detect_5point.py            5 landmarks, dlib
│   ├── detect_68point.py           68 landmarks, dlib
│   ├── detect_facemesh.py          468/478 landmarks, MediaPipe
│   ├── requirements.txt            environment for the two dlib scripts
│   ├── requirements-facemesh.txt   environment for the MediaPipe script
│   ├── shape_predictor_5_face_landmarks.dat     (model, ~9 MB, downloaded)
│   ├── shape_predictor_68_face_landmarks.dat    (model, ~95 MB, downloaded)
│   └── face_landmarker.task                     (model, ~3.6 MB)
│
├── eye-landmark-project/
│   ├── README.md                   <- full code walkthrough
│   ├── eye_tracker.py              EAR drowsiness detection
│   ├── requirements.txt
│   └── shape_predictor_68_face_landmarks.dat    (model, ~95 MB, downloaded)
│
└── hand-landmark-project/
    ├── README.md                   <- full code walkthrough for both scripts
    ├── hand_tracker.py             21-point hand tracking
    ├── selftest.py                 offline correctness tests
    ├── requirements.txt
    └── hand_landmarker.task        (model, ~7.5 MB)
```

### The one thing to understand before you start

There is no single "install everything" command in this repository. That is
deliberate, and it is the most important thing on this page:

**There are four separate Python environments, and each project needs its own.**

| Environment | Serves | Key packages |
|---|---|---|
| `face-landmark-project/venv` | `detect_5point.py`, `detect_68point.py` | dlib, **opencv-python 5** |
| `face-landmark-project/venv_facemesh` | `detect_facemesh.py` | mediapipe |
| `eye-landmark-project/eye_venv` | `eye_tracker.py` | dlib, **opencv-python 4**, scipy |
| `hand-landmark-project/hand_venv` | `hand_tracker.py`, `selftest.py` | mediapipe, **numpy < 2** |

A **virtual environment** is a private box of installed packages, one per
project, so that installing something for one project cannot break another. If
that sounds like unnecessary ceremony, here is the concrete reason it is not:

- `mediapipe` depends on **`opencv-contrib-python`**, while the dlib scripts use
  **`opencv-python`**. These are two different packages that install the *same*
  `cv2` folder. Install both into one environment and `cv2` becomes a mixture of
  files from whichever was installed last — quietly broken, in ways that are
  painful to diagnose.
- `mediapipe`'s compiled extensions need **NumPy 1.x**; the face project pins
  **NumPy 2.4.6**. They cannot coexist.
- The face project pins **OpenCV 5** and **dlib 20**; the eye project pins
  **OpenCV 4** and **dlib 19**.

So: one environment per project. Part D sets all four up. It takes a while —
mostly waiting for dlib to compile — but you only do it once.

---

## Part A — Git and GitHub from zero

### A.1 What Git and GitHub actually are

**Git** is a program that runs on your computer. It records snapshots of your
files over time, so you can see what changed, undo mistakes, and go back to any
earlier version. It works entirely offline.

**GitHub** is a website that hosts copies of Git repositories. It is not Git —
it is a place to put your Git history so it is backed up, shareable, and
downloadable from any computer.

The workflow you will use:

```
   your computer                    GitHub
   ┌──────────────┐                ┌──────────────┐
   │  your files  │   git push ──► │   a copy     │
   │  + .git      │ ◄── git clone  │   online     │
   └──────────────┘                └──────────────┘
```

`push` sends your work up. `clone` brings a copy down. That is 90% of Git for a
single person working alone.

Three words you will see constantly:

- **Repository** (or "repo") — a project folder that Git is tracking.
- **Commit** — one saved snapshot, with a message describing what changed.
- **Remote** — the online copy your local repo is linked to. By convention it is
  named `origin`.

### A.2 Install Python

1. Go to <https://www.python.org/downloads/> and download **Python 3.11** for
   Windows. (3.11 specifically — it is what these projects were developed and
   tested against. Newer versions may work; 3.11 is known good.)

   > Avoid the Microsoft Store version of Python. It sandboxes the install in
   > ways that confuse virtual environments and camera access.

2. Run the installer. **On the first screen, tick "Add python.exe to PATH"** at
   the bottom before clicking Install. This is the single most commonly missed
   step, and skipping it means `python` will not be recognised in the terminal
   and you will have to reinstall.

3. Verify. Open **PowerShell** (press `Win`, type `powershell`, press Enter) and
   type:

   ```powershell
   python --version
   ```

   You should see `Python 3.11.x`. If you see an error, the PATH box was not
   ticked — re-run the installer, choose "Modify", and tick it.

### A.3 Install Git

1. Download from <https://git-scm.com/download/win>. The download starts
   automatically.
2. Run the installer. **Accept all the defaults** — click "Next" through every
   screen. The defaults are sensible, and one of them (Git Credential Manager)
   is what makes GitHub sign-in painless later.
3. Verify:

   ```powershell
   git --version
   ```

   You should see something like `git version 2.51.0.windows.1`.

### A.4 Tell Git who you are

Every commit is stamped with a name and email. Do this once:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Use the **same email you will use for GitHub** — it is how commits get linked to
your account. These are just labels on commits; nothing is sent anywhere.

Two more settings that save real pain on Windows:

```powershell
git config --global core.autocrlf true      # handle Windows/Mac line endings
git config --global core.longpaths true     # allow long folder paths
git config --global init.defaultBranch main # match GitHub's default branch name
```

Check it worked:

```powershell
git config --global --list
```

### A.5 Create a GitHub account

1. Go to <https://github.com> and click **Sign up**.
2. Pick a username. It becomes part of your repository URLs, so keep it simple —
   for example `jane-doe` gives you `github.com/jane-doe/landmark-project`.
3. Verify your email address. GitHub will not let you push code until you do.

### A.6 How signing in works (you can skip reading this if you like)

When you first push, Git needs to prove to GitHub that you are you. Git for
Windows installs **Git Credential Manager**, which pops up a browser window, you
click "Authorize", and it stores the token securely. You do this **once** and
never think about it again.

If a browser window does not appear and it asks for a username and password in
the terminal instead: **your GitHub password will not work.** GitHub removed
password authentication for Git in 2021. Options:

- **Recommended:** cancel out and check that Git Credential Manager is
  installed — `git config --global credential.helper` should print `manager`.
- **Alternative:** create a **Personal Access Token** at
  <https://github.com/settings/tokens> → *Tokens (classic)* → *Generate new
  token (classic)*, tick the **`repo`** scope, and use that token as the
  *password* when prompted. **Copy the token immediately** — GitHub shows it only
  once. Treat it like a password: never commit it, never paste it into a file
  that gets pushed.

---

## Part B — Put this project on GitHub

Do this on the computer that **already has the project files**. It publishes
them to GitHub so you (or anyone you invite) can download them anywhere.

> **If you received this project by cloning it from GitHub already, skip to
> Part C** — your repo is set up.

### B.1 Open PowerShell in the project folder

In File Explorer, navigate to `C:\Users\Lenovo\landmark-project`. Then:

- Hold `Shift` and **right-click** on some empty space inside the folder.
- Choose **"Open PowerShell window here"** (or "Open in Terminal").

A terminal opens already pointed at the project folder. Confirm with:

```powershell
pwd
```

It should print `C:\Users\Lenovo\landmark-project`.

> **Why this specific folder?** `git init` in B.2 turns *this* folder and
> everything under it into a repository. Run it one level up and your repo
> includes unrelated files; run it inside a single project subfolder and you get
> three separate repositories instead of one. We want one repo containing all
> three projects.

### B.2 Create the repository

```powershell
git init
```

Expected output: `Initialized empty Git repository in C:/Users/Lenovo/landmark-project/.git/`

Git just created a hidden `.git` folder. That folder *is* the repository — it
holds the entire history. Delete it and you delete the history (your files
survive, but Git's memory of them does not). Never edit anything inside `.git`
by hand.

### B.3 Check what Git can see — BEFORE you commit anything

This step is not optional. Read it carefully.

```powershell
git status --short
```

You will see a list of every file Git has found, each prefixed by `??`
(meaning "untracked"). Now **scan that list**.

**You should NOT see any of these:**

- `venv/`, `venv_facemesh/`, `eye_venv/`, `hand_venv/` — hundreds of megabytes of
  packages that pip can reinstall in a minute
- `__pycache__/` — Python's compiled bytecode cache
- `shape_predictor_68_face_landmarks.dat` — a ~95 MB model file
- Anything ending in `.bz2`

If you *do* see them, the `.gitignore` file is missing or is not being read.
Stop and fix that before continuing. The reason is concrete: **GitHub rejects
any file larger than 100 MB**, and warns above 50 MB. Pushing the 95 MB dlib
model will make the push fail, and — worse — once a large file is in a commit,
removing it properly requires rewriting history, which is genuinely painful.

The `.gitignore` in this repository is already written to exclude all of the
above. Confirm it is there:

```powershell
Get-Content .gitignore | Select-Object -First 5
```

If the file is missing, create it — the contents are specified in section 1.
Then re-run `git status --short` and check again.

### B.4 Stage and commit

```powershell
git add .
git commit -m "Initial commit: face, eye and hand landmark detection projects"
```

What these do:

- `git add .` — the dot means "everything in this folder". This *stages* the
  files: it tells Git which changes to include in the next snapshot.
  (`git add` does not save anything yet; it fills a holding area.)
- `git commit -m "..."` — takes the staged snapshot and records it permanently
  in the history with your message. The `-m` is the message.

The message matters more than beginners expect. Six months from now, "Initial
commit" tells you what happened; "stuff" does not. Write what changed and why.

If this is your first commit on this machine, Git may open a text editor asking
for a message. Save and close it (`:wq` in Vim, or `Ctrl+S` then close in Nano)
to finish the commit.

Verify:

```powershell
git log --oneline
```

You should see one line — your commit, with a short hash and your message.

Rename the branch to `main`, matching GitHub's default:

```powershell
git branch -M main
```

### B.5 Create the empty repository on GitHub

1. Go to <https://github.com/new>.
2. **Repository name:** `landmark-project` (or anything you like).
3. **Description:** optional. "Face, eye and hand landmark detection with dlib
   and MediaPipe" is a good one.
4. **Public or Private:** your choice. Public is visible to everyone; Private
   only to you and people you invite. Both are free. Start with Private if you
   are unsure.
5. **IMPORTANT — leave all three initialisation boxes UNTICKED:**
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license

   Ticking any of these makes GitHub create its own commit, which puts your
   local repo and the remote out of sync and causes a confusing "rejected"
   error on your first push. You already have a README and a `.gitignore`;
   you do not need GitHub's.
6. Click **Create repository**.
7. GitHub shows a page with setup instructions. **Ignore all of them** except
   the URL — it looks like:

   ```
   https://github.com/YOUR-USERNAME/landmark-project.git
   ```

   Copy it, substituting your actual username.

### B.6 Connect your folder to GitHub

```powershell
git remote add origin https://github.com/YOUR-USERNAME/landmark-project.git
```

Substitute your real username. This registers the GitHub URL under the nickname
`origin`, so from now on `origin` means "my repo on GitHub".

Check it:

```powershell
git remote -v
```

You should see the URL listed twice, once for `fetch` and once for `push`.

### B.7 Push

```powershell
git push -u origin main
```

- `push` — upload your commits.
- `-u origin main` — link your local `main` branch to `origin/main`, so that
  from now on you can type just `git push` with no arguments. You only need the
  full form this first time.

A browser window should open asking you to authorise Git Credential Manager.
Click through and approve. Back in the terminal you will see a progress bar and
something close to this (the object count and size depend on how many files you
have; roughly 12 MB is expected, mostly the two `.task` models):

```
Writing objects: 100% (25/25), 12.42 MiB | 3.10 MiB/s, done.
To https://github.com/YOUR-USERNAME/landmark-project.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

Refresh the GitHub page in your browser — **your files are there.** Click into
any `.py` file to see it rendered with syntax highlighting.

> **Do not worry about the model files.** The two small `.task` files (3.6 MB and
> 7.5 MB) are committed and will appear on GitHub. The large dlib `.dat` files
> (95 MB) are excluded by `.gitignore` — that is intentional. Part D.4 gives the
> download commands, and every script prints them if a model is missing.

### B.8 Alternative: keep everything in the repo with Git LFS

If you would rather have the 95 MB models stored in Git too — so that a clone is
completely self-contained — use **Git LFS** (Large File Storage):

```powershell
git lfs install
git lfs track "*.dat"
git add .gitattributes
git add -f face-landmark-project/shape_predictor_68_face_landmarks.dat
git add -f eye-landmark-project/shape_predictor_68_face_landmarks.dat
git commit -m "Track dlib model files with Git LFS"
git push
```

You must also remove `*.dat` from `.gitignore` for this to work smoothly. Be
aware of the trade-offs: Git LFS needs installing separately on every machine
that clones, and GitHub's free tier includes **1 GB of storage and 1 GB/month of
bandwidth** — the 95 MB model duplicated across two folders uses 190 MB of that.
For most people, downloading the model with one `curl` command (Part D.4) is
simpler and cheaper. Both approaches are legitimate; pick one and stay
consistent.

---

## Part C — Clone it on another computer

"Cloning" is downloading a repository from GitHub, with its full history
attached, so it is a working Git repo and not just files.

**On the new computer**, first complete Part A (install Python, install Git,
configure your name and email).

Then open PowerShell and navigate to where you want the project — for example
your user folder:

```powershell
cd C:\Users\YourName
git clone https://github.com/YOUR-USERNAME/landmark-project.git
```

Substitute your username. You will see:

```
Cloning into 'landmark-project'...
remote: Enumerating objects: 25, done.
...
Receiving objects: 100% (25/25), 12.42 MiB | 5.20 MiB/s, done.
```

Then:

```powershell
cd landmark-project
git log --oneline
```

You will see your commit history — proof the history came across, not just the
files.

That is it. **The clone is complete and identical to the original**, minus the
files `.gitignore` excludes. Now go to Part D to make it run.

> **Getting a repo that is private?** You need to be signed in to GitHub as an
> account that has access, and Git will prompt you to authenticate the first
> time — same Credential Manager flow as in B.7.

---

## Part D — Install and run the projects

Do these in order. Each project is independent, so you can stop after any of
them and come back later.

### D.0 The PowerShell execution policy — read this first

Windows blocks PowerShell scripts by default. Activating a virtual environment
runs a script (`Activate.ps1`), so without this you will hit:

```
.\venv\Scripts\Activate.ps1 : File ... cannot be loaded because running scripts
is disabled on this system.
```

Fix it once, for your user account only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` and press Enter when asked. `RemoteSigned` allows scripts you wrote
locally, and requires a signature on scripts downloaded from the internet — a
reasonable balance. `-Scope CurrentUser` means you do not need administrator
rights, and it does not change anything for other users of the machine.

If you would rather not change it permanently, this works for one session only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Either way, you can always skip activation entirely by calling the Python
executable by its full path:

```powershell
.\venv\Scripts\python.exe detect_68point.py
```

That is more typing but it always works.

### D.1 Each environment follows the same four steps

The pattern never changes:

```powershell
cd <project-folder>                        # 1. go to the project
python -m venv <env-name>                  # 2. create the environment
.\<env-name>\Scripts\Activate.ps1          # 3. activate it
python -m pip install --upgrade pip        # 4. upgrade pip, then install
pip install -r requirements.txt
```

**How to tell an environment is active:** your prompt is prefixed with its name:

```
(venv) PS C:\Users\Lenovo\landmark-project\face-landmark-project>
```

If you do not see `(venv)`, you are installing into your global Python, and you
will hit "module not found" errors later. Always check the prefix.

**To leave an environment:** type `deactivate`.
**To switch projects:** `deactivate`, `cd` to the other folder, activate its env.

### D.2 Download the models

Three model files are too large to live in Git and must be downloaded. Each goes
in a **specific folder** — next to the script that uses it. All commands below
are one line each; note the backtick `` ` `` at the end of a wrapped line, which
is PowerShell's line-continuation character.

```powershell
# 1) dlib 68-point model (~95 MB decompressed) -- needed by BOTH
#    face-landmark-project/detect_68point.py AND eye-landmark-project/eye_tracker.py
#    Run this once inside each of those two folders.

curl.exe -L -o shape_predictor_68_face_landmarks.dat.bz2 `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2

python -c "import bz2,shutil; shutil.copyfileobj(bz2.open('shape_predictor_68_face_landmarks.dat.bz2','rb'), open('shape_predictor_68_face_landmarks.dat','wb'))"

# 2) dlib 5-point model (~9 MB) -- needed by face-landmark-project/detect_5point.py
curl.exe -L -o shape_predictor_5_face_landmarks.dat `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
```

**Use `curl.exe`, not `curl`.** In PowerShell, plain `curl` is an alias for
`Invoke-WebRequest`, which takes completely different arguments and will not
behave as the command above intends.

**Always verify the size afterwards.** A truncated download is by far the most
common cause of a confusing "could not load model" error:

```powershell
Get-ChildItem *.dat, *.task | Select-Object Name, Length
```

| File | Expected size (bytes) |
|---|---|
| `shape_predictor_68_face_landmarks.dat` | 99,693,937 |
| `shape_predictor_5_face_landmarks.dat` | 9,150,489 |
| `face_landmarker.task` | 3,758,596 |
| `hand_landmarker.task` | 7,819,105 |

The two `.task` files come with the repository, so only the `.dat` files need
downloading.

> **Sanity check on the `.dat` file.** If `Get-ChildItem` shows the file but it
> starts with `BZh`, it is still compressed and dlib cannot read it. Run the
> `python -c` decompression command from step 1.

### D.3 Project 1 — Face landmarks (two environments)

```powershell
cd C:\Users\Lenovo\landmark-project\face-landmark-project

# --- Environment 1: the dlib scripts ---
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download the 68-point and 5-point models here (commands in D.2).

# Test the 5-point script
python detect_5point.py            # press 'q' to quit

# Test the 68-point script
python detect_68point.py --indices # press 'q' to quit
```

**dlib compiles from source on Windows and takes several minutes.** If you see
`CMake is not installed` or `Cannot find Visual Studio`, you need two tools
first:

1. **CMake** — <https://cmake.org/download/>. Tick "Add CMake to the system
   PATH" during install.
2. **Visual Studio Build Tools** — <https://visualstudio.microsoft.com/downloads/>
   → "Tools for Visual Studio" → **Build Tools for Visual Studio**. In the
   installer, tick the **"Desktop development with C++"** workload. This
   provides MSVC and the Windows SDK.

Then close and reopen PowerShell (so the new PATH is picked up) and retry.

Now the third face script, which needs its **own** environment:

```powershell
deactivate
python -m venv venv_facemesh
.\venv_facemesh\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-facemesh.txt

# The face_landmarker.task model is already in this folder.
python detect_facemesh.py                       # 478 points, webcam
python detect_facemesh.py --mesh-only           # classic 468-point space
python detect_facemesh.py --image photo.jpg     # single still image
```

Why the separate environment is genuinely required, not just tidy: `mediapipe`
depends on `opencv-contrib-python`, and the dlib scripts use `opencv-python` —
two different packages that install the *same* `cv2` folder. Installing both
into `venv` leaves you with a `cv2` assembled from whichever wheel was unpacked
last, which neither project can rely on. `requirements.txt` in this folder
explains it in more detail.

### D.4 Project 2 — Eye landmarks and drowsiness

```powershell
cd C:\Users\Lenovo\landmark-project\eye-landmark-project

python -m venv eye_venv
.\eye_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download the 68-point model here too (commands in D.2).

python eye_tracker.py
```

The starter criterion is **20 consecutive frames with the eyes closed**, which at
30 FPS is about 0.67 seconds — comfortably longer than a blink (100–400 ms), so
blinking does not trigger it. Tune it live:

```powershell
python eye_tracker.py --ear-threshold 0.22 --consecutive-frames 15
```

The eye project's README explains how to calibrate the threshold for your own
eyes and lighting — the default `0.25` is a safe starting point, not an optimum.

### D.5 Project 3 — Hand landmarks

```powershell
cd C:\Users\Lenovo\landmark-project\hand-landmark-project

python -m venv hand_venv
.\hand_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the offline self-test FIRST -- it checks the maths without needing a camera.
python selftest.py

# Then the live webcam demo.
python hand_tracker.py
```

`selftest.py` should print `ALL <n> CHECKS PASSED` and exit cleanly. It verifies
the geometry maths and the coordinate conversion against hand-computable
values — the things a webcam cannot check objectively, because a subtly wrong
angle formula still *looks* fine on screen.

> **NumPy is pinned below 2.0** in this project's requirements. MediaPipe's
> compiled extensions are built against the NumPy 1.x binary interface, and
> under NumPy 2.x they fail to import with a binary incompatibility error rather
> than a clean message. The pin is deliberate — do not "upgrade" it.

### D.6 A note about the webcam

Every demo opens your default camera with `cv2.VideoCapture(0)`. If a window
never appears, or you get "Could not open camera index 0":

1. **Close other apps using the camera** — Teams, Zoom, Skype, the Windows
   Camera app, and browser tabs with video calls.
2. **Check Windows permissions:** Settings → Privacy & security → Camera → make
   sure "Camera access" and "Let desktop apps access your camera" are both on.
3. **Try a different index** — `--camera 1` or `--camera 2` for a USB webcam.
4. **Check the terminal.** The scripts print a diagnostic on failure rather than
   failing silently — read the message; it usually names the cause.

Press **`q`** or **`Esc`** to quit any demo window. If the window appears frozen,
click on it first so it has keyboard focus, then press `q`.

---

## Part E — Everyday Git commands

Ninety percent of daily Git use is these seven commands.

| Command | What it does |
|---|---|
| `git status` | What has changed, what is staged, what Git is ignoring |
| `git add <file>` | Stage one file (or `git add .` for everything) |
| `git commit -m "msg"` | Save a snapshot of the staged changes |
| `git push` | Upload your commits to GitHub |
| `git pull` | Download commits made elsewhere |
| `git log --oneline` | The commit history, one line each |
| `git diff` | Exactly what changed, line by line, since the last commit |

### The everyday loop

```powershell
# 1. See where you are
git status

# 2. Look at what you actually changed
git diff

# 3. Stage and save
git add .
git commit -m "Add brighter lighting handling to the EAR threshold"

# 4. Send it to GitHub
git push
```

### Working with someone else (or your other computer)

Always `pull` before you start, so you are building on the latest version:

```powershell
git pull
# ... make changes, add, commit ...
git push
```

If `git push` is rejected with "Updates were rejected because the remote
contains work that you do not have locally", it means GitHub has commits you do
not. Run `git pull`, resolve any conflicts, then `git push` again.

### Undoing things

| You want to | Command |
|---|---|
| Discard changes to one file | `git restore <file>` |
| Unstage a file (keep the changes) | `git restore --staged <file>` |
| Fix the message of the last commit | `git commit --amend -m "new message"` |
| Undo the last commit, keep the changes staged | `git reset --soft HEAD~1` |
| See one commit in full | `git show <hash>` |

`git restore` **permanently discards** uncommitted changes to that file — there
is no undo. Check with `git diff` first.

### Useful extras

```powershell
git branch                  # list branches
git switch -c my-feature    # create and switch to a new branch
git switch main             # switch back
git remote -v               # show the GitHub URL you are linked to
git log --oneline --graph   # history with a simple branch diagram
```

**A branch** is a parallel line of development. Try changes on a branch, and
merge them into `main` only when they work — `main` then always stays in a
working state. For solo work it is optional, but it is a good habit and costs
nothing.

---

## Troubleshooting

### Git and GitHub

| Problem | Fix |
|---|---|
| `git: command not found` / `'git' is not recognized` | Git is not installed, or PATH was not updated. Reinstall Git, then **close and reopen PowerShell**. |
| `python: command not found` | Python's "Add to PATH" was not ticked. Re-run the Python installer → Modify → tick it. Reopen PowerShell. |
| `fatal: not a git repository` | You are in the wrong folder, or `git init` was never run. `pwd` to check where you are. |
| `src refspec main does not match any` | You have no commits yet, or your branch is not called `main`. Run `git log --oneline` to confirm a commit exists, then `git branch -M main`. |
| `remote origin already exists` | You already added a remote. Change it: `git remote set-url origin <new-url>`. |
| Push rejected — remote contains work you do not have | GitHub has commits you lack. `git pull` (resolve conflicts if any), then `git push`. |
| `Support for password authentication was removed` | Use Git Credential Manager (browser sign-in) or a Personal Access Token as the password. See A.6. |
| `file is 99.69 MB; this exceeds GitHub's file size limit of 100.00 MB` | A model file got committed. See the row below. |
| I accidentally committed a large file | Remove it from tracking and commit again:<br>`git rm --cached path/to/bigfile`<br>`git commit -m "Remove large file from tracking"`<br>Check `.gitignore` covers it, then push. If the push still fails, the file is in an *earlier* commit and the history must be rewritten (`git filter-repo`), or you can start the remote repo fresh. |
| `Please tell me who you are` | Set your identity: `git config --global user.name` / `user.email`. See A.4. |
| Cloned repo but the models are missing | Expected — they are excluded by `.gitignore`. Download them per D.2. |
| The `.gitignore` does not seem to work | `git status --ignored` shows what is being ignored. If a file was already committed, `.gitignore` will not untrack it — use `git rm --cached <file>`. |

### Python, environments and installs

| Problem | Fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (D.0). |
| `pip: command not found` | Use `python -m pip` instead of bare `pip` — it always works when Python works. |
| Packages install but `import cv2` fails | The environment is not activated, or you have the `opencv-python` / `opencv-contrib-python` conflict. Check for the `(venv)` prompt prefix; use the project's own environment. |
| `pip install dlib` fails: `CMake is not installed` | Install CMake and add it to PATH (D.3). |
| `pip install dlib` fails: `Cannot find Visual Studio` | Install VS Build Tools with the "Desktop development with C++" workload (D.3). |
| The dlib install takes many minutes and appears stuck | Expected — it is compiling from source. There are no prebuilt Windows wheels on PyPI. |
| `ModuleNotFoundError: No module named 'mediapipe'` | Wrong environment. `detect_facemesh.py` and `hand_tracker.py` each need their own MediaPipe environment. |
| `numpy.core.multiarray failed to import` | NumPy 2.x with a MediaPipe wheel built for the 1.x ABI. `pip install "numpy<2.0"`. |
| `ERROR: Could not find a version that satisfies the requirement` | No internet, or a typo in the package name. |
| Two environments, same package, different versions | That is by design here — see section 1. Never install the face project's `requirements.txt` into the eye project's environment. |

### Running the scripts

| Problem | Fix |
|---|---|
| `Could not access the laptop camera` | Another app holds it; or Windows privacy settings block desktop apps; or try `--camera 1`. See D.6. |
| `Model file not found` | The model is not next to the script. Check the size against the table in D.2. Every script prints the exact download command for its model. |
| `dlib could not load ... truncated or still bzip2-compressed` | You pointed at the `.bz2` file, or the download was cut short. Run the decompression command from D.2. |
| `predicts 5 points, but this script expects 68` | You passed the 5-point model to the 68-point script. Use `detect_5point.py`, or point `--model` at the right file. |
| The window opens but shows a frozen frame | Click the window to give it keyboard focus, then press `q`. Check you have not removed the `cv2.waitKey` call if you edited the script. |
| Video window never appears at all, no error | A headless OpenCV build, or the script crashed before `imshow`. Read the terminal output. |
| Everything is very slow | Lower the capture resolution (`--width 640 --height 480`), and for the 68-point script add `--process-width 640`. Close other apps. |
| `EAR: n/a` constantly | No face is detected — improve the lighting or move closer to the camera. |

---

## Quick reference

### One-time setup

```powershell
# Install Python 3.11 (tick "Add python.exe to PATH") and Git (accept defaults)

git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global core.autocrlf true
git config --global core.longpaths true
git config --global init.defaultBranch main
git config --global credential.helper manager

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Publish this folder to GitHub

```powershell
cd C:\Users\Lenovo\landmark-project
git init
git status --short          # <-- CHECK: no venv/, no *.dat, no __pycache__
git add .
git commit -m "Initial commit: face, eye and hand landmark detection projects"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/landmark-project.git
git push -u origin main
```

### Clone on another machine

```powershell
cd C:\Users\YourName
git clone https://github.com/YOUR-USERNAME/landmark-project.git
cd landmark-project
git log --oneline
```

### Set up and run each project

```powershell
# --- Face: dlib scripts (5-point and 68-point) ---
cd face-landmark-project
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python detect_5point.py
python detect_68point.py --indices
deactivate

# --- Face: MediaPipe mesh (separate environment!) ---
python -m venv venv_facemesh
.\venv_facemesh\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-facemesh.txt
python detect_facemesh.py
deactivate

# --- Eye: drowsiness detection ---
cd ..\eye-landmark-project
python -m venv eye_venv
.\eye_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python eye_tracker.py
deactivate

# --- Hand: 21-point tracking ---
cd ..\hand-landmark-project
python -m venv hand_venv
.\hand_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python selftest.py
python hand_tracker.py
deactivate
```

### Download the models

```powershell
# 68-point (~95 MB) -- run inside face-landmark-project/ AND eye-landmark-project/
curl.exe -L -o shape_predictor_68_face_landmarks.dat.bz2 `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
python -c "import bz2,shutil; shutil.copyfileobj(bz2.open('shape_predictor_68_face_landmarks.dat.bz2','rb'), open('shape_predictor_68_face_landmarks.dat','wb'))"

# 5-point (~9 MB) -- run inside face-landmark-project/
curl.exe -L -o shape_predictor_5_face_landmarks.dat `
  https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
```

### Everyday Git

```powershell
git status                 # what changed?
git diff                   # show me the changes
git add .                  # stage everything
git commit -m "message"    # save a snapshot
git push                   # upload to GitHub
git pull                   # download from GitHub
git log --oneline          # history
```

---

## Where to go next

Each project README is a full walkthrough of the code, written to be readable
without prior computer-vision experience:

- **[`face-landmark-project/README.md`](face-landmark-project/README.md)** — the
  iBUG 68-point topology, the 478-point mesh layout, why the same problem has
  three answers, and how dlib and MediaPipe differ in coordinates and models.
- **[`eye-landmark-project/README.md`](eye-landmark-project/README.md)** — the
  Eye Aspect Ratio, the maths and the intuition, how a blink is distinguished
  from a microsleep, and how to calibrate the thresholds for your own face.
- **[`hand-landmark-project/README.md`](hand-landmark-project/README.md)** — the
  21-point hand skeleton, the three-coordinate-system design, finger-angle
  geometry, and what the self-test proves.

**The single most transferable lesson in this repository:** dlib returns
landmark coordinates in **absolute pixels**, while MediaPipe returns them
**normalised to `[0, 1]`**. Mixing the two up produces results that look almost
right — subtly stretched, or a mesh drawn at a tenth of its proper size — which
is much harder to debug than something that fails outright. When landmarks look
wrong, check the coordinate convention first.

If you are new to this area, a good order to work through is:

1. `detect_5point.py` — the smallest possible complete pipeline.
2. `detect_68point.py` — the same idea with a richer output.
3. `eye_tracker.py` — turning 68 static points into a measurement over time.
4. `hand_tracker.py` and `selftest.py` — the same ideas on a different body
   part, with the geometry helpers and tests that a production system needs.
