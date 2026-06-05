# Plumbline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only Python CLI that reads a scroll segment's ink-prediction image and emits a self-contained HTML "trace-quality dashboard" flagging likely segmentation errors (sheet jumps, drift, garbled regions).

**Architecture:** Pure image-processing pipeline — `load → tile → per-tile coherence features (orientation, line-pitch, density) → consensus + flagging → global score → HTML report`. No model inference, no GPU, no server. Validated by synthetic perturbations of a known-good striped field.

**Tech Stack:** Python 3.13, numpy, scipy, scikit-image (structure tensor / FFT), Pillow + tifffile (image IO), matplotlib (Agg, heatmap rendering), jinja2 (report), argparse (CLI). `vesuvius` library used only on the optional fetch-by-id path. pytest for tests.

---

## File Structure

```
plumbline/
  __init__.py
  util.py          # to01(), wrap_angle() — tiny pure helpers
  model.py         # dataclasses: Tile, TileFeatures, FlagMap, ScoreReport
  tiles.py         # tile_grid(shape, tile, overlap) -> (list[Tile], n_rows, n_cols)
  synthetic.py     # striped_field() + rotate_band/splice_shift/garble_patch perturbations
  coherence.py     # orientation_and_anisotropy(), dominant_pitch(), ink_density(), analyze_tiles()
  score.py         # orientation/pitch/structure flags, flag_tiles(), trace_health()
  io.py            # load_image01(), load_mask(), fetch_segment()
  render.py        # overlay_png/heatmap_png/orientation_png/flags_png() + flagged_regions()
  report.py        # render_report(), write_report(), write_json()
  cli.py           # argparse entry point: `plumbline run ...`
templates/
  report.html.j2
tests/
  test_smoke.py test_tiles.py test_synthetic.py test_coherence.py
  test_score.py test_io.py test_render.py test_report.py test_cli.py
examples/          # committed example report
pyproject.toml
README.md
```

Each module has one responsibility. `model.py` defines the shared data types every later module references — build it early.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `plumbline/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "plumbline"
version = "0.1.0"
description = "Trace-quality reports for Vesuvius Challenge scroll segments"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-image>=0.22",
    "Pillow>=10.0",
    "tifffile>=2024.1",
    "matplotlib>=3.8",
    "jinja2>=3.1",
]

[project.optional-dependencies]
fetch = ["vesuvius"]
dev = ["pytest>=8.0"]

[project.scripts]
plumbline = "plumbline.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["plumbline"]
```

- [ ] **Step 2: Create `plumbline/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the smoke test** in `tests/test_smoke.py`

```python
import plumbline


def test_version_present():
    assert plumbline.__version__ == "0.1.0"
```

- [ ] **Step 4: Create venv, install editable, run test**

Run:
```bash
cd ~/Documents/plumbline
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: PASS (1 passed). If a dependency wheel fails to build on macOS, note it and stop.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml plumbline/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold plumbline package"
```

---

## Task 2: Shared data model

**Files:**
- Create: `plumbline/model.py`
- Test: `tests/test_smoke.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_smoke.py`)

```python
import numpy as np
from plumbline.model import Tile, TileFeatures, FlagMap, ScoreReport


def test_model_dataclasses_construct():
    t = Tile(row=0, col=1, y0=0, y1=256, x0=256, x1=512)
    assert (t.row, t.col, t.x0) == (0, 1, 256)

    z = np.zeros((2, 2))
    feats = TileFeatures(
        n_rows=2, n_cols=2, theta=z.copy(), anisotropy=z.copy(),
        pitch=z.copy(), strength=z.copy(), density=z.copy(),
        confidence=z.astype(bool), tiles=[t],
    )
    assert feats.n_rows == 2 and feats.tiles[0] is t

    flags = FlagMap(
        orient_break=z.astype(bool), pitch_break=z.astype(bool),
        structure_loss=z.astype(bool),
    )
    assert flags.any_flag.shape == (2, 2)
    assert not flags.any_flag.any()

    rep = ScoreReport(score=100, n_orient=0, n_pitch=0, n_structure=0, low_conf_frac=0.0)
    assert rep.score == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py::test_model_dataclasses_construct -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.model'`

- [ ] **Step 3: Write `plumbline/model.py`**

```python
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class TileFeatures:
    n_rows: int
    n_cols: int
    theta: np.ndarray        # (n_rows, n_cols) radians, text-line orientation, mod pi
    anisotropy: np.ndarray   # (n_rows, n_cols) 0..1, how linear the texture is
    pitch: np.ndarray        # (n_rows, n_cols) pixels between lines, NaN if none
    strength: np.ndarray     # (n_rows, n_cols) 0..1 spectral peak strength
    density: np.ndarray      # (n_rows, n_cols) 0..1 ink fraction
    confidence: np.ndarray   # (n_rows, n_cols) bool, enough ink+coverage to judge
    tiles: List[Tile]        # maps grid cell -> pixel box


@dataclass
class FlagMap:
    orient_break: np.ndarray     # bool (n_rows, n_cols)
    pitch_break: np.ndarray      # bool (n_rows, n_cols)
    structure_loss: np.ndarray   # bool (n_rows, n_cols)

    @property
    def any_flag(self) -> np.ndarray:
        return self.orient_break | self.pitch_break | self.structure_loss


@dataclass
class ScoreReport:
    score: int            # 0..100 trace health
    n_orient: int
    n_pitch: int
    n_structure: int
    low_conf_frac: float  # fraction of grid that was low-confidence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/model.py tests/test_smoke.py
git commit -m "feat: add shared data model dataclasses"
```

---

## Task 3: Tiny pure helpers (util)

**Files:**
- Create: `plumbline/util.py`
- Test: `tests/test_coherence.py`

- [ ] **Step 1: Write the failing test** in `tests/test_coherence.py`

```python
import numpy as np
from plumbline.util import to01, wrap_angle


def test_to01_normalizes_uint8():
    a = np.array([[0, 255], [128, 255]], dtype=np.uint8)
    out = to01(a)
    assert out.dtype == np.float64
    assert out.min() == 0.0 and out.max() == 1.0


def test_to01_collapses_rgb():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to01(rgb).shape == (4, 4)


def test_wrap_angle_mod_pi():
    # pi and 0 are the same orientation
    assert abs(wrap_angle(np.pi)) < 1e-9
    assert abs(wrap_angle(np.pi / 2) - np.pi / 2) < 1e-9 or \
           abs(wrap_angle(np.pi / 2) + np.pi / 2) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coherence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.util'`

- [ ] **Step 3: Write `plumbline/util.py`**

```python
import numpy as np


def to01(img) -> np.ndarray:
    """Grayscale float64 image in [0, 1]."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.size and a.max() > 1.0:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def wrap_angle(a):
    """Wrap an orientation angle (mod pi) into [-pi/2, pi/2)."""
    return (a + np.pi / 2) % np.pi - np.pi / 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coherence.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/util.py tests/test_coherence.py
git commit -m "feat: add to01 and wrap_angle helpers"
```

---

## Task 4: Tile grid

**Files:**
- Create: `plumbline/tiles.py`
- Test: `tests/test_tiles.py`

- [ ] **Step 1: Write the failing test** in `tests/test_tiles.py`

```python
from plumbline.tiles import tile_grid
from plumbline.model import Tile


def test_tile_grid_counts_and_coords():
    tiles, n_rows, n_cols = tile_grid((512, 512), tile=256, overlap=0.5)
    # step = 128 -> origins 0,128,256 in each axis
    assert (n_rows, n_cols) == (3, 3)
    assert len(tiles) == 9
    assert all(isinstance(t, Tile) for t in tiles)
    first = tiles[0]
    assert (first.row, first.col, first.y0, first.x0) == (0, 0, 0, 0)
    # boxes are clamped to the image bounds
    assert all(t.y1 <= 512 and t.x1 <= 512 for t in tiles)


def test_tile_grid_small_image_yields_single_tile():
    tiles, n_rows, n_cols = tile_grid((100, 80), tile=256, overlap=0.5)
    assert (n_rows, n_cols) == (1, 1)
    assert len(tiles) == 1
    assert (tiles[0].y1, tiles[0].x1) == (100, 80)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.tiles'`

- [ ] **Step 3: Write `plumbline/tiles.py`**

```python
from typing import List, Tuple
from plumbline.model import Tile


def tile_grid(shape, tile: int = 256, overlap: float = 0.5) -> Tuple[List[Tile], int, int]:
    """Build a full rectangular grid of (possibly overlapping) tile boxes.

    Returns (tiles, n_rows, n_cols). Boxes are clamped to image bounds. The
    grid always covers the whole image; mask/ink emptiness is handled later
    via per-tile confidence, not by dropping tiles here.
    """
    h, w = shape[0], shape[1]
    step = max(1, int(round(tile * (1.0 - overlap))))
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    tiles: List[Tile] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tiles.append(Tile(r, c, y0, min(y0 + tile, h), x0, min(x0 + tile, w)))
    return tiles, len(ys), len(xs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tiles.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/tiles.py tests/test_tiles.py
git commit -m "feat: add tile_grid"
```

---

## Task 5: Synthetic striped field + perturbations

**Files:**
- Create: `plumbline/synthetic.py`
- Test: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test** in `tests/test_synthetic.py`

```python
import numpy as np
from plumbline.synthetic import striped_field, rotate_band, splice_shift, garble_patch


def test_striped_field_shape_and_range():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    assert f.shape == (256, 256)
    assert 0.0 <= f.min() and f.max() <= 1.0


def test_striped_field_is_periodic_along_rows_for_zero_angle():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    profile = f.mean(axis=1)            # average across columns -> vary along rows
    profile = profile - profile.mean()
    spec = np.abs(np.fft.rfft(profile))
    freqs = np.fft.rfftfreq(len(profile))
    peak_pitch = 1.0 / freqs[1:][np.argmax(spec[1:])]
    assert abs(peak_pitch - 20) < 3


def test_perturbations_preserve_shape_and_change_content():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    assert rotate_band(f, 100, 160, ddeg=30).shape == f.shape
    assert splice_shift(f, x_split=128, dy=15).shape == f.shape
    g = garble_patch(f, 80, 140, 80, 140)
    assert g.shape == f.shape
    assert not np.allclose(g[80:140, 80:140], f[80:140, 80:140])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.synthetic'`

- [ ] **Step 3: Write `plumbline/synthetic.py`**

```python
import numpy as np
from scipy.ndimage import rotate as _ndrotate


def striped_field(shape=(512, 512), pitch=20, angle=0.0, sharpness=0.85, noise=0.03, seed=0):
    """A clean 'ink-like' field of parallel text lines running along `angle`
    (radians), with the given line `pitch` (pixels). Square-ish stripes."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    # coordinate perpendicular to lines running along `angle`
    s = yy * np.cos(angle) - xx * np.sin(angle)
    stripes = (np.sin(2 * np.pi * s / pitch) > 0).astype(np.float64)
    rng = np.random.default_rng(seed)
    img = stripes * sharpness + rng.normal(0.0, noise, shape)
    return np.clip(img, 0.0, 1.0)


def rotate_band(field, y0, y1, ddeg=25):
    """Rotate a horizontal band in place -> local orientation discontinuity."""
    out = field.copy()
    out[y0:y1] = _ndrotate(field[y0:y1], ddeg, reshape=False, order=1, mode="reflect")
    return out


def splice_shift(field, x_split, dy=15):
    """Vertically shift everything right of x_split -> seam / line break."""
    out = field.copy()
    out[:, x_split:] = np.roll(field[:, x_split:], dy, axis=0)
    return out


def garble_patch(field, y0, y1, x0, x1, seed=1):
    """Replace a patch with noise -> high ink density, no line structure."""
    out = field.copy()
    rng = np.random.default_rng(seed)
    out[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/synthetic.py tests/test_synthetic.py
git commit -m "feat: add synthetic striped field and perturbations"
```

---

## Task 6: Orientation & anisotropy

**Files:**
- Create: `plumbline/coherence.py`
- Test: `tests/test_coherence.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_coherence.py`)

```python
from plumbline.synthetic import striped_field
from plumbline.coherence import orientation_and_anisotropy


def test_orientation_horizontal_lines_near_zero():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    theta, aniso = orientation_and_anisotropy(f)
    # horizontal lines -> orientation ~0 (mod pi), strongly anisotropic
    assert min(abs(theta), abs(abs(theta) - np.pi)) < np.radians(12)
    assert aniso > 0.4


def test_orientation_anisotropy_low_for_noise():
    rng = np.random.default_rng(3)
    noise = rng.random((256, 256))
    _, aniso = orientation_and_anisotropy(noise)
    assert aniso < 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coherence.py -v`
Expected: FAIL with `ImportError: cannot import name 'orientation_and_anisotropy'`

- [ ] **Step 3: Write `orientation_and_anisotropy` in `plumbline/coherence.py`**

```python
import numpy as np
from skimage.feature import structure_tensor
from plumbline.util import to01, wrap_angle


def orientation_and_anisotropy(img, sigma=2.0):
    """Dominant text-line orientation (radians, mod pi) and anisotropy (0..1)
    of a tile, from the averaged structure tensor."""
    a = to01(img)
    arr, arc, acc = structure_tensor(a, sigma=sigma, order="rc")
    jrr, jrc, jcc = float(arr.mean()), float(arc.mean()), float(acc.mean())
    # eigenvalues of [[jrr, jrc], [jrc, jcc]]
    tmp = np.sqrt((jrr - jcc) ** 2 + 4 * jrc ** 2)
    l1 = 0.5 * ((jrr + jcc) + tmp)
    l2 = 0.5 * ((jrr + jcc) - tmp)
    denom = l1 + l2
    anisotropy = 0.0 if denom <= 0 else float((l1 - l2) / denom)
    # gradient orientation; text lines run perpendicular to the dominant gradient
    grad_theta = 0.5 * np.arctan2(2 * jrc, jrr - jcc)
    line_theta = wrap_angle(grad_theta + np.pi / 2)
    return float(line_theta), anisotropy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coherence.py -v`
Expected: PASS. If the orientation convention is off by pi/2 for horizontal lines, the structure_tensor axis convention differs — drop the `+ np.pi/2` (or add it) so horizontal lines give ~0, then rerun.

- [ ] **Step 5: Commit**

```bash
git add plumbline/coherence.py tests/test_coherence.py
git commit -m "feat: add orientation_and_anisotropy"
```

---

## Task 7: Dominant line pitch

**Files:**
- Modify: `plumbline/coherence.py`
- Test: `tests/test_coherence.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_coherence.py`)

```python
from plumbline.coherence import dominant_pitch


def test_dominant_pitch_detects_known_spacing():
    f = striped_field((256, 256), pitch=24, angle=0.0)
    pitch, strength = dominant_pitch(f, theta=0.0)
    assert abs(pitch - 24) < 4
    assert strength > 0.0


def test_dominant_pitch_nan_on_flat_image():
    flat = np.zeros((256, 256))
    pitch, strength = dominant_pitch(flat, theta=0.0)
    assert np.isnan(pitch)
    assert strength == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coherence.py -v`
Expected: FAIL with `ImportError: cannot import name 'dominant_pitch'`

- [ ] **Step 3: Add `dominant_pitch` to `plumbline/coherence.py`**

```python
from scipy.ndimage import rotate as _ndrotate


def dominant_pitch(img, theta, min_pitch=4.0, max_pitch=200.0):
    """Spacing (pixels) between text lines and its spectral strength (0..1).
    Rotates the tile so lines are horizontal, averages along them, FFTs the
    cross-line profile. Returns (nan, 0.0) when there is no periodic signal."""
    a = to01(img)
    deg = np.degrees(theta)
    rot = _ndrotate(a, deg, reshape=True, order=1, mode="constant", cval=0.0)
    profile = rot.mean(axis=1)
    profile = profile - profile.mean()
    n = profile.size
    if n < 8 or not np.any(profile):
        return float("nan"), 0.0
    spec = np.abs(np.fft.rfft(profile * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        pitch = np.where(freqs > 0, 1.0 / freqs, np.inf)
    valid = (pitch >= min_pitch) & (pitch <= max_pitch)
    if not valid.any() or spec[valid].max() <= 0:
        return float("nan"), 0.0
    masked = np.where(valid, spec, -1.0)
    idx = int(np.argmax(masked))
    strength = float(spec[idx] / (spec.sum() + 1e-9))
    return float(pitch[idx]), strength
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coherence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/coherence.py tests/test_coherence.py
git commit -m "feat: add dominant_pitch via directional FFT"
```

---

## Task 8: Ink density + analyze_tiles

**Files:**
- Modify: `plumbline/coherence.py`
- Test: `tests/test_coherence.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_coherence.py`)

```python
from plumbline.coherence import ink_density, analyze_tiles
from plumbline.model import TileFeatures


def test_ink_density_blank_vs_dense():
    assert ink_density(np.zeros((64, 64))) < 0.05
    assert ink_density(np.ones((64, 64))) > 0.9


def test_analyze_tiles_on_clean_field():
    f = striped_field((512, 512), pitch=24, angle=0.0)
    mask = np.ones((512, 512), dtype=bool)
    feats = analyze_tiles(f, mask, tile=256, overlap=0.5)
    assert isinstance(feats, TileFeatures)
    assert feats.theta.shape == (feats.n_rows, feats.n_cols)
    # clean field -> confident everywhere, low orientation spread, pitch ~24
    assert feats.confidence.all()
    confident_pitch = feats.pitch[np.isfinite(feats.pitch)]
    assert abs(np.median(confident_pitch) - 24) < 5
    assert feats.theta.std() < np.radians(10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coherence.py -v`
Expected: FAIL with `ImportError: cannot import name 'ink_density'`

- [ ] **Step 3: Add `ink_density` and `analyze_tiles` to `plumbline/coherence.py`**

```python
from plumbline.model import TileFeatures
from plumbline.tiles import tile_grid


def ink_density(img, thresh=0.25):
    """Fraction of pixels above an ink threshold."""
    a = to01(img)
    return float((a > thresh).mean())


def analyze_tiles(ink, mask, tile=256, overlap=0.5, sigma=2.0,
                  min_density=0.02, min_coverage=0.5):
    """Compute per-tile features over the whole grid. Tiles with too little
    mask coverage or ink are marked low-confidence (orientation still recorded,
    pitch left NaN)."""
    a = to01(ink)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    tiles, nr, nc = tile_grid(a.shape, tile, overlap)
    theta = np.zeros((nr, nc)); aniso = np.zeros((nr, nc))
    pitch = np.full((nr, nc), np.nan); strength = np.zeros((nr, nc))
    density = np.zeros((nr, nc)); conf = np.zeros((nr, nc), dtype=bool)
    for t in tiles:
        sub = a[t.y0:t.y1, t.x0:t.x1]
        cov = float(mask[t.y0:t.y1, t.x0:t.x1].mean())
        d = ink_density(sub)
        th, an = orientation_and_anisotropy(sub, sigma=sigma)
        density[t.row, t.col] = d
        theta[t.row, t.col] = th
        aniso[t.row, t.col] = an
        if cov < min_coverage or d < min_density:
            continue
        p, s = dominant_pitch(sub, th)
        pitch[t.row, t.col] = p
        strength[t.row, t.col] = s
        conf[t.row, t.col] = True
    return TileFeatures(nr, nc, theta, aniso, pitch, strength, density, conf, tiles)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coherence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/coherence.py tests/test_coherence.py
git commit -m "feat: add ink_density and analyze_tiles"
```

---

## Task 9: Orientation-break flag + consensus

**Files:**
- Create: `plumbline/score.py`
- Test: `tests/test_score.py`

- [ ] **Step 1: Write the failing test** in `tests/test_score.py`

```python
import numpy as np
from plumbline.synthetic import striped_field, rotate_band
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_orientation


def _feats(field):
    return analyze_tiles(field, np.ones(field.shape, bool), tile=128, overlap=0.5)


def test_orientation_flag_quiet_on_clean_control():
    feats = _feats(striped_field((512, 512), pitch=24, angle=0.0))
    flags = flag_orientation(feats)
    assert flags.sum() <= 1  # essentially no false alarms


def test_orientation_flag_fires_on_rotated_band():
    f = rotate_band(striped_field((512, 512), pitch=24, angle=0.0), 200, 320, ddeg=35)
    feats = _feats(f)
    flags = flag_orientation(feats)
    assert flags.sum() >= 1
    # flagged cells should fall in the band's row range (rows ~ y/step, step=64)
    rows = np.where(flags.any(axis=1))[0]
    assert any(2 <= r <= 5 for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.score'`

- [ ] **Step 3: Write `plumbline/score.py`**

```python
import numpy as np
from scipy.ndimage import uniform_filter
from plumbline.model import FlagMap, ScoreReport


def orientation_deviation(theta, confidence, radius=2):
    """Angular deviation (radians, mod pi) of each tile's orientation from a
    confidence-weighted local consensus, using doubled-angle vectors."""
    u = np.cos(2 * theta)
    v = np.sin(2 * theta)
    w = confidence.astype(float)
    size = 2 * radius + 1
    su = uniform_filter(u * w, size=size, mode="nearest")
    sv = uniform_filter(v * w, size=size, mode="nearest")
    sw = uniform_filter(w, size=size, mode="nearest") + 1e-9
    consensus = 0.5 * np.arctan2(sv / sw, su / sw)
    dev = np.abs(((theta - consensus) + np.pi / 2) % np.pi - np.pi / 2)
    return dev


def flag_orientation(features, deg_thresh=15.0, radius=2):
    dev = orientation_deviation(features.theta, features.confidence, radius)
    return (dev > np.radians(deg_thresh)) & features.confidence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score.py -v`
Expected: PASS. If the control trips a flag, raise `deg_thresh` slightly (clean fields have near-zero spread, so a real margin exists); if the band is missed, lower it. Keep both tests green.

- [ ] **Step 5: Commit**

```bash
git add plumbline/score.py tests/test_score.py
git commit -m "feat: add orientation-break flagging with local consensus"
```

---

## Task 10: Pitch-break flag

**Files:**
- Modify: `plumbline/score.py`
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_score.py`)

```python
from plumbline.score import flag_pitch


def test_pitch_flag_quiet_on_clean_control():
    feats = _feats(striped_field((512, 512), pitch=24, angle=0.0))
    assert flag_pitch(feats).sum() <= 1


def test_pitch_flag_fires_when_spacing_changes():
    # left half pitch 16, right half pitch 40 -> spacing discontinuity
    left = striped_field((512, 256), pitch=16, angle=0.0)
    right = striped_field((512, 256), pitch=40, angle=0.0)
    f = np.hstack([left, right])
    feats = _feats(f)
    assert flag_pitch(feats).sum() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ImportError: cannot import name 'flag_pitch'`

- [ ] **Step 3: Add `flag_pitch` to `plumbline/score.py`**

```python
from scipy.ndimage import median_filter


def flag_pitch(features, rel_thresh=0.35, radius=2):
    """Flag tiles whose line pitch departs from the local median pitch."""
    p = features.pitch
    valid = features.confidence & np.isfinite(p)
    if not valid.any():
        return np.zeros_like(valid)
    fill = np.nanmedian(p[valid])
    pf = np.where(valid, p, fill)
    med = median_filter(pf, size=2 * radius + 1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(pf - med) / np.where(med > 0, med, np.nan)
    return (np.nan_to_num(rel) > rel_thresh) & valid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/score.py tests/test_score.py
git commit -m "feat: add pitch-break flagging"
```

---

## Task 11: Structure-loss flag

**Files:**
- Modify: `plumbline/score.py`
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_score.py`)

```python
from plumbline.synthetic import garble_patch
from plumbline.score import flag_structure_loss


def test_structure_loss_quiet_on_clean_control():
    feats = _feats(striped_field((512, 512), pitch=24, angle=0.0))
    assert flag_structure_loss(feats).sum() == 0


def test_structure_loss_fires_on_garbled_patch():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0),
                     192, 320, 192, 320)
    feats = _feats(f)
    assert flag_structure_loss(feats).sum() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ImportError: cannot import name 'flag_structure_loss'`

- [ ] **Step 3: Add `flag_structure_loss` (and `flag_tiles`) to `plumbline/score.py`**

```python
def flag_structure_loss(features, text_density=0.15, aniso_thresh=0.3):
    """Flag tiles that have ink (so text is expected) but no linear structure."""
    return ((features.density >= text_density)
            & (features.anisotropy < aniso_thresh)
            & features.confidence)


def flag_tiles(features) -> FlagMap:
    return FlagMap(
        orient_break=flag_orientation(features),
        pitch_break=flag_pitch(features),
        structure_loss=flag_structure_loss(features),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/score.py tests/test_score.py
git commit -m "feat: add structure-loss flag and flag_tiles aggregator"
```

---

## Task 12: Global trace-health score

**Files:**
- Modify: `plumbline/score.py`
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_score.py`)

```python
from plumbline.score import flag_tiles, trace_health
from plumbline.model import ScoreReport


def test_trace_health_high_on_clean_low_on_garbled():
    clean = _feats(striped_field((512, 512), pitch=24, angle=0.0))
    clean_rep = trace_health(clean, flag_tiles(clean))
    assert isinstance(clean_rep, ScoreReport)
    assert clean_rep.score >= 90

    bad = _feats(garble_patch(striped_field((512, 512), pitch=24, angle=0.0),
                              128, 384, 128, 384))
    bad_rep = trace_health(bad, flag_tiles(bad))
    assert bad_rep.score < clean_rep.score
    assert 0 <= bad_rep.score <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score.py -v`
Expected: FAIL with `ImportError: cannot import name 'trace_health'`

- [ ] **Step 3: Add `trace_health` to `plumbline/score.py`**

```python
def trace_health(features, flags) -> ScoreReport:
    """0..100 health = 100 * (1 - flagged_confident_fraction)."""
    conf = features.confidence
    n_conf = int(conf.sum())
    flagged = int((flags.any_flag & conf).sum())
    frac_bad = (flagged / n_conf) if n_conf else 0.0
    score = int(round(100 * (1.0 - frac_bad)))
    score = max(0, min(100, score))
    total = features.confidence.size
    low_conf = 1.0 - (n_conf / total) if total else 1.0
    return ScoreReport(
        score=score,
        n_orient=int(flags.orient_break.sum()),
        n_pitch=int(flags.pitch_break.sum()),
        n_structure=int(flags.structure_loss.sum()),
        low_conf_frac=float(low_conf),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/score.py tests/test_score.py
git commit -m "feat: add trace_health global score"
```

---

## Task 13: Image IO

**Files:**
- Create: `plumbline/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test** in `tests/test_io.py`

```python
import numpy as np
from PIL import Image
from plumbline.io import load_image01, load_mask


def test_load_image01_roundtrip(tmp_path):
    arr = (np.linspace(0, 255, 64 * 64).reshape(64, 64)).astype(np.uint8)
    p = tmp_path / "ink.png"
    Image.fromarray(arr).save(p)
    out = load_image01(str(p))
    assert out.shape == (64, 64)
    assert out.dtype == np.float64
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_load_mask_is_bool(tmp_path):
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 8:24] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(arr).save(p)
    m = load_mask(str(p))
    assert m.dtype == bool
    assert m.sum() == 16 * 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.io'`

- [ ] **Step 3: Write `plumbline/io.py`**

```python
import numpy as np
from plumbline.util import to01


def load_image01(path) -> np.ndarray:
    """Load a PNG/TIF as a grayscale float64 image in [0, 1]."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        arr = np.asarray(Image.open(path))
    return to01(arr)


def load_mask(path) -> np.ndarray:
    """Load a mask image as a boolean array (any nonzero pixel is True)."""
    img = load_image01(path)
    return img > 0.0


def fetch_segment(segment_id, scroll="Scroll1"):
    """Optional: fetch (ink_prediction, flat_mask) for a segment id via the
    `vesuvius` library. Requires `pip install plumbline[fetch]` and network
    access. Returns (ink01, mask_bool)."""
    try:
        import vesuvius  # noqa: F401
    except ImportError as e:  # pragma: no cover - network/optional path
        raise RuntimeError(
            "Install the optional fetch extra: pip install 'plumbline[fetch]'"
        ) from e
    raise NotImplementedError(
        "fetch_segment is a stub: wire to the vesuvius API during Task 16 "
        "once the exact ink/flat_mask accessor for the chosen segment is known."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_io.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/io.py tests/test_io.py
git commit -m "feat: add image/mask IO with optional segment fetch stub"
```

---

## Task 14: Render views + flagged regions

**Files:**
- Create: `plumbline/render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing test** in `tests/test_render.py`

```python
import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles
from plumbline.render import overlay_png, heatmap_png, orientation_png, flags_png, flagged_regions


def _setup():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    return f, feats, flag_tiles(feats)


def test_png_renderers_return_png_bytes():
    f, feats, flags = _setup()
    for png in (overlay_png(f, feats, flags), heatmap_png(feats),
                orientation_png(feats), flags_png(f, feats, flags)):
        assert isinstance(png, (bytes, bytearray))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_flagged_regions_lists_pixel_locations():
    f, feats, flags = _setup()
    regions = flagged_regions(feats, flags)
    assert isinstance(regions, list)
    assert regions, "garbled patch should produce at least one region"
    r = regions[0]
    assert {"x", "y", "mode"} <= set(r.keys())
    assert r["mode"] in {"orientation", "pitch", "structure"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.render'`

- [ ] **Step 3: Write `plumbline/render.py`**

```python
import io as _io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_png(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _cell_to_pixel(features, r, c):
    t = next(t for t in features.tiles if t.row == r and t.col == c)
    return (t.x0 + t.x1) // 2, (t.y0 + t.y1) // 2


def _flag_extent(features, flags):
    # draw flag rectangles in pixel space
    rects = []
    fm = flags.any_flag
    for t in features.tiles:
        if fm[t.row, t.col]:
            rects.append((t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0))
    return rects


def overlay_png(ink01, features, flags) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ink01, cmap="gray", origin="upper")
    for (x, y, w, h) in _flag_extent(features, flags):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    ax.set_axis_off()
    return _fig_to_png(fig)


def heatmap_png(features) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.where(features.confidence, features.anisotropy, np.nan)
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="line coherence")
    ax.set_axis_off()
    return _fig_to_png(fig)


def orientation_png(features) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    nr, nc = features.n_rows, features.n_cols
    yy, xx = np.mgrid[0:nr, 0:nc]
    u = np.cos(features.theta) * features.confidence
    v = -np.sin(features.theta) * features.confidence  # image y points down
    ax.quiver(xx, yy, u, v, pivot="mid", scale=nc * 1.5)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_axis_off()
    return _fig_to_png(fig)


def flags_png(ink01, features, flags) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ink01, cmap="gray", origin="upper", alpha=0.6)
    colors = {"orientation": "#ff5c5c", "pitch": "#ffce5c", "structure": "#7fb0e0"}
    layers = {"orientation": flags.orient_break, "pitch": flags.pitch_break,
              "structure": flags.structure_loss}
    for mode, grid in layers.items():
        for t in features.tiles:
            if grid[t.row, t.col]:
                ax.add_patch(plt.Rectangle((t.x0, t.y0), t.x1 - t.x0, t.y1 - t.y0,
                                           fill=True, alpha=0.35, color=colors[mode]))
    ax.set_axis_off()
    return _fig_to_png(fig)


def flagged_regions(features, flags):
    """Flat list of flagged cells as {x, y, mode} in pixel coordinates."""
    out = []
    layers = [("orientation", flags.orient_break), ("pitch", flags.pitch_break),
              ("structure", flags.structure_loss)]
    for mode, grid in layers:
        for t in features.tiles:
            if grid[t.row, t.col]:
                x, y = _cell_to_pixel(features, t.row, t.col)
                out.append({"x": int(x), "y": int(y), "mode": mode})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/render.py tests/test_render.py
git commit -m "feat: add matplotlib view renderers and flagged_regions"
```

---

## Task 15: HTML report + JSON sidecar

**Files:**
- Create: `templates/report.html.j2`
- Create: `plumbline/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test** in `tests/test_report.py`

```python
import json
import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles, trace_health
from plumbline.report import render_report, write_report, write_json


def _bundle():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    flags = flag_tiles(feats)
    return f, feats, flags, trace_health(feats, flags)


def test_render_report_is_self_contained_html():
    f, feats, flags, rep = _bundle()
    html = render_report({"segment_id": "seg-test", "scroll": "Scroll1"},
                         f, feats, flags, rep)
    assert "<html" in html.lower()
    assert str(rep.score) in html
    assert "data:image/png;base64," in html        # images embedded, no external files
    assert "http://" not in html and "https://" not in html


def test_write_report_and_json(tmp_path):
    f, feats, flags, rep = _bundle()
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    write_report(str(html_path), {"segment_id": "seg-test", "scroll": "Scroll1"},
                 f, feats, flags, rep)
    write_json(str(json_path), {"segment_id": "seg-test"}, feats, flags, rep)
    assert html_path.exists() and html_path.stat().st_size > 1000
    data = json.loads(json_path.read_text())
    assert data["score"] == rep.score
    assert "regions" in data and isinstance(data["regions"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.report'`

- [ ] **Step 3: Write `templates/report.html.j2`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plumbline — {{ meta.segment_id }}</title>
<style>
  body { margin:0; background:#0a1521; color:#cfe0ee; font:14px/1.5 ui-monospace,Menlo,monospace; }
  header { display:flex; justify-content:space-between; align-items:center;
           padding:12px 18px; border-bottom:1px solid #22354a; }
  .score { font:700 20px Georgia,serif; padding:4px 12px; border-radius:4px; color:#0a1521; }
  main { display:flex; gap:16px; padding:18px; }
  .viewer { flex:1; }
  .tabs button { background:#13243a; color:#7fb0e0; border:1px solid #2a3f55;
                 padding:4px 10px; cursor:pointer; font:inherit; }
  .tabs button.active { background:#5aa0ff; color:#0a1521; }
  .viewer img { width:100%; border:1px solid #22354a; margin-top:8px; display:none; }
  .viewer img.active { display:block; }
  .rail { width:230px; }
  .rail h3 { color:#5a7a98; font-size:11px; letter-spacing:1px; text-transform:uppercase; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td,th { text-align:left; border-top:1px solid #22354a; padding:4px 6px; }
</style>
</head>
<body>
<header>
  <span>{{ meta.scroll }} · segment {{ meta.segment_id }}</span>
  <span class="score" style="background:{{ score_color }}">{{ report.score }} / 100</span>
</header>
<main>
  <div class="viewer">
    <div class="tabs">
      <button class="active" onclick="show('ink')">Ink + flags</button>
      <button onclick="show('heat')">Coherence</button>
      <button onclick="show('orient')">Orientation</button>
      <button onclick="show('flags')">Flags</button>
    </div>
    <img id="ink" class="active" src="data:image/png;base64,{{ img_overlay }}">
    <img id="heat" src="data:image/png;base64,{{ img_heat }}">
    <img id="orient" src="data:image/png;base64,{{ img_orient }}">
    <img id="flags" src="data:image/png;base64,{{ img_flags }}">
  </div>
  <div class="rail">
    <h3>Metrics</h3>
    <table>
      <tr><td>orientation breaks</td><td>{{ report.n_orient }}</td></tr>
      <tr><td>pitch jumps</td><td>{{ report.n_pitch }}</td></tr>
      <tr><td>structure loss</td><td>{{ report.n_structure }}</td></tr>
      <tr><td>low-confidence</td><td>{{ '%.0f' % (report.low_conf_frac * 100) }}%</td></tr>
    </table>
    <h3>Flagged regions</h3>
    <table>
      <tr><th>x</th><th>y</th><th>mode</th></tr>
      {% for r in regions %}<tr><td>{{ r.x }}</td><td>{{ r.y }}</td><td>{{ r.mode }}</td></tr>{% endfor %}
    </table>
  </div>
</main>
<script>
function show(id){
  document.querySelectorAll('.viewer img').forEach(i=>i.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
}
</script>
</body>
</html>
```

- [ ] **Step 4: Write `plumbline/report.py`**

```python
import base64
import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plumbline.render import (overlay_png, heatmap_png, orientation_png,
                              flags_png, flagged_regions)

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _score_color(score: int) -> str:
    if score >= 85:
        return "#2bd47a"
    if score >= 60:
        return "#ffce5c"
    return "#ff5c5c"


def render_report(meta, ink01, features, flags, report) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR),
                      autoescape=select_autoescape(["html"]))
    tmpl = env.get_template("report.html.j2")
    return tmpl.render(
        meta=meta, report=report,
        score_color=_score_color(report.score),
        regions=flagged_regions(features, flags),
        img_overlay=_b64(overlay_png(ink01, features, flags)),
        img_heat=_b64(heatmap_png(features)),
        img_orient=_b64(orientation_png(features)),
        img_flags=_b64(flags_png(ink01, features, flags)),
    )


def write_report(path, meta, ink01, features, flags, report):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(meta, ink01, features, flags, report))


def write_json(path, meta, features, flags, report):
    payload = {
        "segment_id": meta.get("segment_id"),
        "score": report.score,
        "n_orient": report.n_orient,
        "n_pitch": report.n_pitch,
        "n_structure": report.n_structure,
        "low_conf_frac": report.low_conf_frac,
        "regions": flagged_regions(features, flags),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add templates/report.html.j2 plumbline/report.py tests/test_report.py
git commit -m "feat: add self-contained HTML report and JSON sidecar"
```

---

## Task 16: CLI

**Files:**
- Create: `plumbline/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** in `tests/test_cli.py`

```python
import json
import numpy as np
from PIL import Image
from plumbline.synthetic import striped_field, garble_patch
from plumbline.cli import main


def test_cli_run_produces_html_and_json(tmp_path):
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    ink_path = tmp_path / "ink.png"
    Image.fromarray((f * 255).astype("uint8")).save(ink_path)
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"

    rc = main(["run", str(ink_path), "-o", str(out_html),
               "--json", str(out_json), "--tile", "128"])
    assert rc == 0
    assert out_html.exists() and out_html.stat().st_size > 1000
    data = json.loads(out_json.read_text())
    assert 0 <= data["score"] <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.cli'`

- [ ] **Step 3: Write `plumbline/cli.py`**

```python
import argparse
import os
import numpy as np
from plumbline import io as pio
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles, trace_health
from plumbline.report import write_report, write_json


def _load_inputs(args):
    if args.segment_id:
        ink, mask = pio.fetch_segment(args.segment_id, scroll=args.scroll)
        meta = {"segment_id": args.segment_id, "scroll": args.scroll}
        return ink, mask, meta
    ink = pio.load_image01(args.ink)
    mask = pio.load_mask(args.mask) if args.mask else np.ones(ink.shape, dtype=bool)
    meta = {"segment_id": os.path.splitext(os.path.basename(args.ink))[0],
            "scroll": args.scroll}
    return ink, mask, meta


def _cmd_run(args) -> int:
    ink, mask, meta = _load_inputs(args)
    feats = analyze_tiles(ink, mask, tile=args.tile, overlap=args.overlap)
    flags = flag_tiles(feats)
    report = trace_health(feats, flags)
    write_report(args.output, meta, ink, feats, flags, report)
    if args.json:
        write_json(args.json, meta, feats, flags, report)
    print(f"{meta['segment_id']}: trace health {report.score}/100 "
          f"(orient {report.n_orient}, pitch {report.n_pitch}, "
          f"structure {report.n_structure}) -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plumbline",
                                description="Trace-quality reports for scroll segments")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="analyze a segment ink prediction")
    run.add_argument("ink", nargs="?", help="path to ink-prediction PNG/TIF")
    run.add_argument("--segment-id", help="fetch by segment id (needs [fetch] extra)")
    run.add_argument("--scroll", default="Scroll1")
    run.add_argument("--mask", help="path to flat_mask image")
    run.add_argument("-o", "--output", default="report.html")
    run.add_argument("--json", help="also write a JSON sidecar to this path")
    run.add_argument("--tile", type=int, default=256)
    run.add_argument("--overlap", type=float, default=0.5)
    run.set_defaults(func=_cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run" and not args.ink and not args.segment_id:
        raise SystemExit("provide an ink path or --segment-id")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test + verify console entry point**

Run:
```bash
pytest tests/test_cli.py -v
plumbline run --help
```
Expected: test PASS; `--help` prints usage.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add plumbline/cli.py tests/test_cli.py
git commit -m "feat: add plumbline run CLI"
```

---

## Task 17: End-to-end on a real segment, README, example

**Files:**
- Create: `README.md`
- Create: `examples/` (committed example report + JSON)
- Modify: `plumbline/io.py` (`fetch_segment`) only if wiring the real fetch

- [ ] **Step 1: Pick a real published segment and obtain its ink prediction + flat_mask**

Browse `https://dl.ash2txt.org/` (or `s3://vesuvius-challenge-open-data/`) to a segment under a scroll's `paths/<id>/` that has a `<id>_prediction*.png` and `<id>_flat_mask.png`. Download both locally into `data/` (gitignored). Record the exact path in the README.

- [ ] **Step 2: Run Plumbline on the real segment**

Run:
```bash
plumbline run data/<id>_prediction.png --mask data/<id>_flat_mask.png \
  -o examples/<id>.html --json examples/<id>.json --tile 256
```
Expected: prints a trace-health line and writes the report. Open `examples/<id>.html` and confirm the four tabs render and flags look sane (sparse on a clean segment).

- [ ] **Step 3 (optional): Wire `fetch_segment`**

If the `vesuvius` library exposes ink + flat_mask by id, replace the `NotImplementedError` in `plumbline/io.py:fetch_segment` with the real calls returning `(ink01, mask_bool)`, and add a network-gated test marked `@pytest.mark.skipif` on absence of the extra. Otherwise leave the documented stub.

- [ ] **Step 4: Write `README.md`**

Include: one-paragraph purpose (trace-quality QA for Vesuvius segmentation), install (`pip install -e .`), usage (`plumbline run <ink.png> --mask <flat_mask.png> -o report.html`), what each of the four views means, how the score is computed, the synthetic-validation story (with the `striped_field` + perturbation snippet), an embedded screenshot of `examples/<id>.html`, and explicit non-goals (no inference, no 3D). State that it targets the Vesuvius Challenge segmentation wishlist item.

- [ ] **Step 5: Verify the suite is green and commit**

Run: `pytest -v`
Expected: all PASS.

```bash
git add README.md examples/
git commit -m "docs: add README, real-segment example, and walkthrough"
```

---

## Self-Review notes

- **Spec coverage:** ink-only coherence diagnostic (Tasks 6–8), consensus + three flag modes (Tasks 9–11), global score (Task 12), CLI + static HTML dashboard with view-toggle + metrics + flagged-regions table (Tasks 15–16), JSON sidecar (Task 15), synthetic-perturbation validation for all three failure modes + clean control (Tasks 9–12), local-path inputs + optional segment-id fetch stub (Task 13), real-segment end-to-end + README + example (Task 17). Deferred items (through-stack signal, batch index, 3D, inference) are explicitly out of scope.
- **Type consistency:** `TileFeatures`/`FlagMap`/`ScoreReport` fields defined in Task 2 are used unchanged throughout; `analyze_tiles`, `flag_tiles`, `trace_health`, `render_report`/`write_report`/`write_json`, and `flagged_regions` keep identical signatures wherever referenced.
- **Known calibration risk (flagged in steps, not placeholders):** the orientation-convention `+pi/2` (Task 6 Step 4) and the flag thresholds (`deg_thresh`, `rel_thresh`, `aniso_thresh`) may need a one-line nudge so synthetic perturbation tests stay green; each affected step says exactly which direction to adjust.
