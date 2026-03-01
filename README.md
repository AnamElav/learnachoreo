# Learn a Choreo

A dance learning platform for ingestion, pose extraction, and segmentation of choreography from video.

## Requirements

- **Python 3.9–3.12** (MediaPipe does not support Python 3.13 yet.)
- [ffmpeg](https://ffmpeg.org/) on your system (e.g. `brew install ffmpeg` on macOS).
- **CMake** (needed to build `llvmlite` from source when no wheel is available; e.g. `brew install cmake` on macOS).
- **Intel Mac only:** there is no pre-built `llvmlite` wheel for macOS x86_64. You need **LLVM 20** installed so llvmlite can build from source (see below), or use the minimal install without librosa.

## Setup

If your default `python3` is 3.13, create the virtual environment with a supported version:

```bash
# macOS: install Python 3.12 if needed
brew install python@3.12

# remove existing venv and recreate with 3.12
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
```

### Option A: Full install (includes librosa)

Install dependencies in three steps so `llvmlite` (used by `librosa`) builds correctly. If step 2 fails with “llvmlite needs CMake”, install CMake first (`brew install cmake`).

**On Intel Mac (x86_64):** there is no llvmlite wheel; you must install LLVM 20 and point the build at it before step 2:

```bash
brew install llvm@20 cmake
export CMAKE_PREFIX_PATH="/usr/local/opt/llvm@20"   # Intel Homebrew; use /opt/homebrew/opt/llvm@20 on Apple Silicon if building from source
```

Then:

```bash
# 1) Install build tools with compatible setuptools
pip install "setuptools>=58,<70" wheel

# 2) Build llvmlite (uses venv setuptools + LLVM from CMAKE_PREFIX_PATH on Intel Mac)
pip install --no-build-isolation llvmlite

# 3) Install everything else (numba will use the llvmlite from step 2)
pip install -r requirements.txt
```

### Option B: Minimal install (no librosa)

To avoid building llvmlite (e.g. on Intel Mac without installing LLVM), install everything except librosa:

```bash
pip install -r requirements-minimal.txt
```

You get mediapipe, opencv, yt-dlp, ffmpeg-python, and numpy. Add `soundfile` and `scipy` later if you need basic audio I/O.

---

## Install all requirements (pipeline + API)

Use this when you need the full pipeline **and** the choreo-ai-api (e.g. for `./choreo-ai-api/start_local.sh`).

**1. Create and activate venv** (Python 3.9–3.12):

```bash
cd /path/to/learnachoreo
python3 -m venv .venv
source .venv/bin/activate
```

**2. Intel Mac only (no prebuilt llvmlite wheel):** install LLVM 20 and CMake, then set `CMAKE_PREFIX_PATH` so llvmlite can build from source:

```bash
brew install llvm@20 cmake
export CMAKE_PREFIX_PATH="/usr/local/opt/llvm@20"
```

(On Apple Silicon, if you ever build llvmlite from source: `export CMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm@20"`.)

**3. Full pip install (three steps so llvmlite builds correctly):**

```bash
pip install "setuptools>=58,<70" wheel
pip install --no-build-isolation llvmlite
pip install -r requirements.txt
pip install -r choreo-ai-api/requirements.txt
```

**4. System deps:** ffmpeg and Redis (for local API):

```bash
brew install ffmpeg redis
```

After this, `./choreo-ai-api/start_local.sh` and the pipeline (including beat detection via librosa) will work.
