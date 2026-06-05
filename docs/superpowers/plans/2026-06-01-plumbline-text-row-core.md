# Text-Row Analysis Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plumbline's stripe/anisotropy analysis core — which scored real legible text 0/100 — with projection-profile **band detection** on an **auto-adapted tile scale**, so genuine text scores reasonably while garble/drift are still flagged.

**Architecture:** A new one-pass `estimate_scale_and_skew` picks a tile size that spans several text rows and a global writing angle. Per tile, the core measures `band_strength` (detrended projection-profile contrast — "rows vs. gaps"), an orientation by contrast-maximization, and a secondary autocorrelation `pitch`. Flags become `garble` (ink but no bands), `orient_break` (drift), and a tightly-gated `spacing_break`. The tiling, scoring shell, report, dashboard, and JSON sidecar are reused unchanged except for renamed fields.

**Tech Stack:** Python 3.12, NumPy, SciPy (`ndimage`), scikit-image (being removed from the core), Pillow, Jinja2, Matplotlib (Agg), pytest. Run via the venv at `~/.venvs/plumbline` (NOT an in-repo `.venv` — iCloud corrupts editable installs).

**Spec:** `docs/superpowers/specs/2026-06-01-plumbline-text-row-core-design.md`

**Conventions for every task:**
- Tests: `~/.venvs/plumbline/bin/pytest -q` (full) or `~/.venvs/plumbline/bin/pytest <path>::<test> -v` (single).
- Work on branch `redesign-text-row-core` (already created; spec already committed there).
- Commit after each task with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | This plan |
|------|----------------|-----------|
| `plumbline/synthetic.py` | test-input generators | **add** `glyph_rows()` (Task 1) |
| `plumbline/model.py` | dataclasses | **rename** fields (Task 2) |
| `plumbline/coherence.py` | per-tile analysis core | **new primitives** (Task 3); **rewrite** `analyze_tiles`, drop stripe funcs (Task 4) |
| `plumbline/score.py` | flags + score + warning | **rewrite** flag bodies (Task 4) |
| `plumbline/render.py` | report figures | field reads (Tasks 2, 4) |
| `plumbline/report.py` | HTML + JSON writers | JSON keys (Task 2) |
| `plumbline/dashboard.py` | batch index | field reads (Task 2) |
| `plumbline/cli.py` | CLI wiring | rename + `--tile` auto (Tasks 2, 5) |
| `plumbline/templates/*.j2` | HTML | labels (Tasks 2, 5) |
| `tests/*` | suite | updated/added throughout |

---

## Task 1: `glyph_rows()` synthetic generator

The validation rule (hard-learned): validate on **text-like** inputs, never continuous stripes. This generator is the permanent dense-text fixture.

**Files:**
- Modify: `plumbline/synthetic.py`
- Test: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthetic.py`:

```python
import numpy as np
from plumbline.synthetic import glyph_rows


def test_glyph_rows_shape_and_range():
    f = glyph_rows((512, 512), row_pitch=40)
    assert f.shape == (512, 512)
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_glyph_rows_has_row_banding_unlike_noise():
    f = glyph_rows((512, 512), row_pitch=40)
    rows_profile = f.mean(axis=1)            # mean ink per image row
    noise = np.random.default_rng(0).random((512, 512)).mean(axis=1)
    # discrete rows + gaps -> projection profile oscillates far more than noise
    assert rows_profile.std() > 5 * noise.std()
```

- [ ] **Step 2: Run, verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_synthetic.py -q`
Expected: FAIL — `ImportError: cannot import name 'glyph_rows'`.

- [ ] **Step 3: Implement `glyph_rows`**

Add to `plumbline/synthetic.py` (after `striped_field`):

```python
def glyph_rows(shape=(512, 512), row_pitch=40, glyph=18, gap=8, angle=0.0,
               fill=0.85, sharpness=0.9, noise=0.03, seed=0):
    """Rows of discrete glyph blocks separated by inter-row gaps and
    intra-row letter/word spaces -- a stand-in for real text (NOT stripes).
    `angle` rotates the writing direction (radians)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float64)
    gh = max(1, int(glyph * 0.8))
    y = row_pitch // 2
    while y < h:
        x = gap
        while x < w:
            gw = max(1, glyph + int(rng.integers(-3, 4)))
            word_gap = gap * (3 if rng.random() < 0.2 else 1)
            if rng.random() < fill:                       # leave letter-ish gaps
                img[y:min(y + gh, h), x:min(x + gw, w)] = sharpness
            x += gw + word_gap
        y += row_pitch
    img = img + rng.normal(0.0, noise, shape)
    if angle:
        img = _ndrotate(img, np.degrees(angle), reshape=False, order=1,
                        mode="constant", cval=0.0)
    return np.clip(img, 0.0, 1.0)
```

`_ndrotate` is already imported at the top of `synthetic.py`.

- [ ] **Step 4: Run, verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_synthetic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plumbline/synthetic.py tests/test_synthetic.py
git commit -m "feat: add glyph_rows synthetic text generator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Rename data-model fields (mechanical, behavior unchanged)

Pure rename across the codebase so later tasks work in the new vocabulary. The stripe algorithm is untouched here; `band_strength` transiently holds the old anisotropy value until Task 4 replaces the computation. The suite must stay green.

Renames: `anisotropy→band_strength`, `strength→pitch_strength`, `structure_loss→garble`, `pitch_break→spacing_break`, `n_structure→n_garble`, `n_pitch→n_spacing`, `flag_structure_loss→flag_garble`, `flag_pitch→flag_spacing`.

**Files:** `plumbline/model.py`, `plumbline/score.py`, `plumbline/render.py`, `plumbline/report.py`, `plumbline/cli.py`, `plumbline/dashboard.py`, `plumbline/templates/report.html.j2`, `plumbline/templates/index.html.j2`, `tests/test_smoke.py`, `tests/test_score.py`.

- [ ] **Step 1: `model.py`** — edit the three dataclasses:

```python
# TileFeatures:
    band_strength: np.ndarray   # (n_rows, n_cols) 0..~1 band contrast (rows vs gaps)
# (was: anisotropy)
    pitch_strength: np.ndarray  # (n_rows, n_cols) 0..1 autocorrelation peak height
# (was: strength)

# FlagMap:
    orient_break: np.ndarray     # bool (n_rows, n_cols)
    spacing_break: np.ndarray    # bool (n_rows, n_cols)   (was: pitch_break)
    garble: np.ndarray           # bool (n_rows, n_cols)   (was: structure_loss)

    @property
    def any_flag(self) -> np.ndarray:
        return self.orient_break | self.spacing_break | self.garble

# ScoreReport:
    n_orient: int
    n_spacing: int     # (was: n_pitch)
    n_garble: int      # (was: n_structure)

# IndexRow:
    n_orient: int
    n_spacing: int     # (was: n_pitch)
    n_garble: int      # (was: n_structure)
```

Keep all other fields/positions identical (TileFeatures and IndexRow are built positionally, so order must not change: TileFeatures order stays `theta, band_strength, pitch, pitch_strength, density, confidence, tiles`; IndexRow order stays `seg_id, score, n_orient, n_spacing, n_garble, low_conf_frac, report_filename, thumb_b64, error`).

- [ ] **Step 2: `score.py`** — rename functions and attribute reads, logic unchanged:
  - `def flag_structure_loss(...)` → `def flag_garble(...)`; inside, `features.anisotropy` → `features.band_strength`.
  - `def flag_pitch(...)` → `def flag_spacing(...)` (body unchanged; it reads `features.pitch`).
  - In `flag_tiles`: `pitch_break=flag_pitch(features)` → `spacing_break=flag_spacing(features)`; `structure_loss=flag_structure_loss(features)` → `garble=flag_garble(features)`.
  - In `trace_health`: `n_pitch=int(flags.pitch_break.sum())` → `n_spacing=int(flags.spacing_break.sum())`; `n_structure=int(flags.structure_loss.sum())` → `n_garble=int(flags.garble.sum())`.
  - In `input_warning`: `flags.structure_loss` → `flags.garble`; rename local `struct_frac` → `garble_frac`.

- [ ] **Step 3: `render.py`**:
  - `heatmap_png`: `features.anisotropy` → `features.band_strength`.
  - `flags_png`: `colors = {"orientation": "#ff5c5c", "spacing": "#ffce5c", "garble": "#7fb0e0"}` and `layers = {"orientation": flags.orient_break, "spacing": flags.spacing_break, "garble": flags.garble}`.
  - `flagged_regions`: `layers = [("orientation", flags.orient_break), ("spacing", flags.spacing_break), ("garble", flags.garble)]`.

- [ ] **Step 4: `report.py`** — JSON keys:

```python
        "n_orient": report.n_orient,
        "n_spacing": report.n_spacing,
        "n_garble": report.n_garble,
```

- [ ] **Step 5: `cli.py`**:
  - Print: `f"(orient {report.n_orient}, spacing {report.n_spacing}, garble {report.n_garble}) -> {args.output}")`.
  - Batch `IndexRow(...)`: pass `rep.n_orient, rep.n_spacing, rep.n_garble` (positions unchanged).

- [ ] **Step 6: `dashboard.py`** — `(r.n_orient + r.n_spacing + r.n_garble) > 0`.

- [ ] **Step 7: `templates/report.html.j2`** (lines 47-48):

```html
      <tr><td>spacing breaks</td><td>{{ report.n_spacing }}</td></tr>
      <tr><td>garble</td><td>{{ report.n_garble }}</td></tr>
```

- [ ] **Step 8: `templates/index.html.j2`** — headers (lines 40-41) and cells (lines 51-52):

```html
  <th onclick="sortBy(4,'n')">spacing</th>
  <th onclick="sortBy(5,'n')">garble</th>
```
```html
  <td data-v="{{ r.n_spacing }}" class="{{ '' if r.n_spacing else 'muted' }}">{{ r.n_spacing }}</td>
  <td data-v="{{ r.n_garble }}" class="{{ '' if r.n_garble else 'muted' }}">{{ r.n_garble }}</td>
```

- [ ] **Step 9: `tests/test_smoke.py`** — update keyword args to construct the renamed dataclasses:
  - `anisotropy=z.copy()` → `band_strength=z.copy()`; `strength=z.copy()` → `pitch_strength=z.copy()`.
  - `pitch_break=z.astype(bool)` → `spacing_break=z.astype(bool)`; `structure_loss=z.astype(bool)` → `garble=z.astype(bool)`.
  - `ScoreReport(score=100, n_orient=0, n_pitch=0, n_structure=0, low_conf_frac=0.0)` → `... n_spacing=0, n_garble=0 ...`.
  - `IndexRow(seg_id="abc", score=73, n_orient=0, n_pitch=2, n_structure=1, ...)` → `... n_spacing=2, n_garble=1, ...`.

- [ ] **Step 10: `tests/test_score.py`** — rename imports/usages so it still imports (bodies rewritten in Task 4):
  - Import line → `from plumbline.score import flag_orientation, flag_spacing, flag_garble, flag_tiles, trace_health`.
  - `def test_pitch_flag_quiet_on_clean_control` body `flag_pitch(feats)` → `flag_spacing(feats)`; `def test_pitch_flag_fires_when_spacing_changes` body `flag_pitch(feats)` → `flag_spacing(feats)`.
  - `def test_structure_loss_quiet_on_clean_control` → `flag_structure_loss(feats)` → `flag_garble(feats)`; `def test_structure_loss_fires_on_garbled_patch` → `flag_structure_loss(feats)` → `flag_garble(feats)`.

- [ ] **Step 11: Run the full suite, verify green**

Run: `~/.venvs/plumbline/bin/pytest -q`
Expected: PASS (same count as before — pure rename).

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor: rename fields to band/spacing/garble vocabulary

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: New analysis primitives (additive, unit-tested)

Add the band-detection primitives as standalone functions. The old stripe functions remain until Task 4, so the suite stays green.

**Files:**
- Modify: `plumbline/coherence.py` (add functions + imports; don't touch `analyze_tiles` yet)
- Test: `tests/test_coherence.py` (append)

- [ ] **Step 1: Write failing tests** — append to `tests/test_coherence.py`:

```python
from plumbline.synthetic import glyph_rows
from plumbline.coherence import (projection_profile, band_contrast,
                                 orientation_by_contrast, row_pitch,
                                 estimate_scale_and_skew)


def test_band_contrast_text_beats_noise():
    text = glyph_rows((512, 512), row_pitch=40)
    noise = np.random.default_rng(1).random((512, 512))
    assert band_contrast(text, 0.0) > 2 * band_contrast(noise, 0.0)


def test_orientation_by_contrast_recovers_angle():
    for deg in (0, 12, -20):
        f = glyph_rows((512, 512), row_pitch=40, angle=np.radians(deg))
        est = np.degrees(orientation_by_contrast(f, seed=0.0))
        assert abs(est - deg) < 6, f"deg={deg} -> est={est:.1f}"


def test_row_pitch_detects_known_spacing():
    f = glyph_rows((512, 512), row_pitch=40)
    pitch, strength = row_pitch(f, 0.0, min_lag=10, max_lag=256)
    assert abs(pitch - 40) < 8
    assert strength > 0.0


def test_estimate_scale_and_skew_bounds_and_angle():
    f = glyph_rows((1024, 1024), row_pitch=48, angle=np.radians(8))
    tile, theta = estimate_scale_and_skew(f)
    assert 256 <= tile <= 2048
    assert abs(np.degrees(theta) - 8) < 8
```

- [ ] **Step 2: Run, verify failure**

Run: `~/.venvs/plumbline/bin/pytest tests/test_coherence.py -q`
Expected: FAIL — `ImportError` for the new names.

- [ ] **Step 3: Implement primitives** — add to `plumbline/coherence.py`. First extend the imports at the top:

```python
import numpy as np
from scipy.ndimage import rotate as _ndrotate, zoom as _zoom, uniform_filter1d
from plumbline.util import to01, wrap_angle
from plumbline.model import TileFeatures
from plumbline.tiles import tile_grid

TILE_MIN, TILE_MAX = 256, 2048
```

Then add the functions:

```python
def projection_profile(img, theta=0.0):
    """Mean ink per text-row: rotate so rows are horizontal, average along them."""
    a = to01(img)
    if abs(theta) > 1e-6:
        a = _ndrotate(a, -np.degrees(theta), reshape=False, order=1,
                      mode="constant", cval=0.0)
    return a.mean(axis=1)


def _detrend(profile):
    """Remove the broad density envelope, keep row-scale oscillation."""
    win = max(3, (profile.size // 2) | 1)
    return profile - uniform_filter1d(profile, size=win, mode="nearest")


def band_contrast(img, theta=0.0):
    """0..~1 'rowness': detrended projection-profile contrast. High when ink
    forms rows separated by gaps; ~0 for structureless mottle."""
    p = projection_profile(img, theta)
    det = _detrend(p)
    return float(det.std() / (float(p.mean()) + 1e-6))


def orientation_by_contrast(img, seed=0.0, span=np.radians(25), n=13):
    """Writing-direction angle (radians, mod pi) maximizing band contrast."""
    best_t, best_v = float(seed), -1.0
    for t in np.asarray(seed) + np.linspace(-span, span, n):
        v = band_contrast(img, float(t))
        if v > best_v:
            best_v, best_t = v, float(t)
    return float(wrap_angle(best_t))


def row_pitch(img, theta=0.0, min_lag=8, max_lag=None):
    """Row spacing (px) from the profile autocorrelation + its 0..1 peak height.
    Returns (nan, 0.0) when there is no clear periodic peak."""
    p = projection_profile(img, theta)
    p = p - p.mean()
    if not np.any(p):
        return float("nan"), 0.0
    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = p.size // 2 if max_lag is None else int(max_lag)
    lo = max(2, int(min_lag))
    seg = ac[lo:hi]
    if seg.size < 3:
        return float("nan"), 0.0
    i = np.arange(1, seg.size - 1)
    ismax = (seg[i] > seg[i - 1]) & (seg[i] > seg[i + 1])
    cand = i[ismax]
    if cand.size == 0:
        return float("nan"), 0.0
    b = int(cand[int(np.argmax(seg[cand]))])
    return float(b + lo), float(seg[b])


def estimate_scale_and_skew(ink, mask=None, k=2.5, target=1000.0):
    """Pick a tile size spanning several text rows + the global skew angle.
    Returns (tile_size:int, theta:float). Scale comes from the autocorrelation
    decay length of the detrended global profile (robust without a clean peak)."""
    a = to01(ink)
    if mask is not None and mask.shape == a.shape and mask.any():
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(1.0, target / max(a.shape))
    small = _zoom(a, scale, order=1) if scale < 1.0 else a
    theta = orientation_by_contrast(small, seed=0.0, span=np.radians(25), n=21)
    det = _detrend(projection_profile(small, theta))
    det = det - det.mean()
    if not np.any(det):
        return TILE_MIN, theta
    ac = np.correlate(det, det, mode="full")[det.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    below = np.where(ac[1:] < 0.2)[0]
    decay_small = int(below[0] + 1) if below.size else max(1, det.size // 4)
    decay_full = decay_small / max(scale, 1e-9)
    tile = int(np.clip(round(k * decay_full), TILE_MIN, TILE_MAX))
    return tile, theta
```

- [ ] **Step 4: Run, verify pass**

Run: `~/.venvs/plumbline/bin/pytest tests/test_coherence.py -q`
Expected: PASS (new + old tests both pass; old stripe funcs still present).

- [ ] **Step 5: Commit**

```bash
git add plumbline/coherence.py tests/test_coherence.py
git commit -m "feat: add projection-profile band primitives

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Swap the analysis core and flags to band-based

The regime change: `analyze_tiles` now produces band-based features, and the flags reinterpret them. Producer, consumers, and their tests change together so each commit is coherent.

**Files:**
- Modify: `plumbline/coherence.py` (rewrite `analyze_tiles`; delete `orientation_and_anisotropy`, `dominant_pitch`; keep `ink_density`)
- Modify: `plumbline/score.py` (rewrite `flag_garble`, `flag_spacing` bodies; keep `flag_orientation`, `trace_health`, `input_warning` shells)
- Modify: `tests/test_coherence.py` (drop obsolete stripe-primitive tests; rewrite the `analyze_tiles` test)
- Modify: `tests/test_score.py` (rewrite around `glyph_rows`)

- [ ] **Step 1: Rewrite `analyze_tiles` test** — in `tests/test_coherence.py`, delete `test_orientation_horizontal_lines_near_zero`, `test_orientation_anisotropy_low_for_noise`, `test_dominant_pitch_detects_known_spacing`, `test_dominant_pitch_nan_on_flat_image`, `test_dominant_pitch_detects_known_spacing_at_angles`, and their imports `from plumbline.coherence import orientation_and_anisotropy` / `dominant_pitch`. Keep `to01`/`wrap_angle`/`ink_density` tests. Replace `test_analyze_tiles_on_clean_field` with:

```python
def test_analyze_tiles_text_is_banded_noise_is_not():
    text = glyph_rows((512, 512), row_pitch=40)
    tf = analyze_tiles(text, tile=256, overlap=0.5)
    assert isinstance(tf, TileFeatures)
    assert tf.theta.shape == (tf.n_rows, tf.n_cols)
    text_band = np.median(tf.band_strength[tf.confidence])

    noise = np.random.default_rng(7).random((512, 512))
    nf = analyze_tiles(noise, tile=256, overlap=0.5)
    noise_band = np.median(nf.band_strength[nf.confidence])
    assert text_band > 1.5 * noise_band


def test_analyze_tiles_auto_picks_tile_in_bounds():
    text = glyph_rows((1024, 1024), row_pitch=48)
    tf = analyze_tiles(text, tile=None)      # auto-adapt
    assert tf.confidence.any()
```

- [ ] **Step 2: Run, verify failure**

Run: `~/.venvs/plumbline/bin/pytest tests/test_coherence.py -q`
Expected: FAIL — old `analyze_tiles` lacks band-based behavior / `band_strength` populated from anisotropy gives wrong magnitudes (and `tile=None` unsupported).

- [ ] **Step 3: Rewrite `analyze_tiles`; delete stripe funcs** — in `plumbline/coherence.py`, delete `orientation_and_anisotropy` and `dominant_pitch` entirely, remove the now-unused `from skimage.feature import structure_tensor` import, keep `ink_density`, and replace `analyze_tiles` with:

```python
def analyze_tiles(ink, mask=None, tile=None, overlap=0.5,
                  min_density=0.02, min_coverage=0.5):
    """Per-tile band features over the grid. Tile size auto-adapts to text
    scale when `tile` is None. Low-coverage / low-ink tiles stay low-confidence
    (orientation + band_strength still recorded; pitch left NaN)."""
    a = to01(ink)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    auto_tile, gtheta = estimate_scale_and_skew(a, mask)
    if tile is None:
        tile = auto_tile
    tiles, nr, nc = tile_grid(a.shape, tile, overlap)
    theta = np.zeros((nr, nc)); band = np.zeros((nr, nc))
    pitch = np.full((nr, nc), np.nan); pstr = np.zeros((nr, nc))
    density = np.zeros((nr, nc)); conf = np.zeros((nr, nc), dtype=bool)
    for t in tiles:
        sub = a[t.y0:t.y1, t.x0:t.x1]
        if min(sub.shape) < 8:
            continue
        cov = float(mask[t.y0:t.y1, t.x0:t.x1].mean())
        d = ink_density(sub)
        th = orientation_by_contrast(sub, seed=gtheta)
        theta[t.row, t.col] = th
        band[t.row, t.col] = band_contrast(sub, th)
        density[t.row, t.col] = d
        if cov < min_coverage or d < min_density:
            continue
        p, s = row_pitch(sub, th, min_lag=max(8, tile // 16),
                         max_lag=sub.shape[0] // 2)
        pitch[t.row, t.col] = p
        pstr[t.row, t.col] = s
        conf[t.row, t.col] = True
    return TileFeatures(nr, nc, theta, band, pitch, pstr, density, conf, tiles)
```

- [ ] **Step 4: Run coherence tests, verify pass**

Run: `~/.venvs/plumbline/bin/pytest tests/test_coherence.py -q`
Expected: PASS. If `test_analyze_tiles_text_is_banded_noise_is_not` is marginal, that is real signal to revisit `_detrend`/tile defaults — do not loosen the assertion below 1.5× without noting it.

- [ ] **Step 5: Rewrite the flag bodies in `score.py`** — replace `flag_garble` and `flag_spacing`:

```python
def flag_garble(features, band_thresh=0.12):
    """Confident ink but no row-band contrast -> structureless mottle (garble
    or non-text). The corrected, right-way-round structure rule."""
    return features.confidence & (features.band_strength < band_thresh)


def flag_spacing(features, rel_thresh=0.35, radius=2, strength_gate=0.30):
    """Row pitch departs from the local median -- ONLY on tiles with a
    confident periodic peak, so unreliable pitch can't manufacture flags."""
    p = features.pitch
    valid = features.confidence & np.isfinite(p) & (features.pitch_strength >= strength_gate)
    if not valid.any():
        return np.zeros_like(valid)
    fill = np.nanmedian(p[valid])
    pf = np.where(valid, p, fill)
    med = median_filter(pf, size=2 * radius + 1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(pf - med) / np.where(med > 0, med, np.nan)
    return (np.nan_to_num(rel) > rel_thresh) & valid
```

`flag_orientation`, `flag_tiles`, `trace_health`, and `input_warning` keep their Task-2 form (they already reference `band_strength`/`garble`/`spacing_break`).

- [ ] **Step 6: Rewrite `tests/test_score.py`** around `glyph_rows`:

```python
import numpy as np
from plumbline.synthetic import glyph_rows, rotate_band, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import (flag_orientation, flag_spacing, flag_garble,
                             flag_tiles, trace_health, input_warning)
from plumbline.model import ScoreReport


def _feats(field):
    return analyze_tiles(field, np.ones(field.shape, bool), tile=128, overlap=0.5)


def test_garble_quiet_on_clean_text():
    feats = _feats(glyph_rows((512, 512), row_pitch=40))
    assert flag_garble(feats).sum() == 0


def test_garble_fires_on_garbled_patch():
    f = garble_patch(glyph_rows((512, 512), row_pitch=40), 192, 384, 192, 384)
    feats = _feats(f)
    assert flag_garble(feats).sum() >= 1


def test_orientation_flag_quiet_on_clean_text():
    feats = _feats(glyph_rows((512, 512), row_pitch=40))
    assert flag_orientation(feats).sum() <= 1


def test_orientation_flag_fires_on_rotated_band():
    f = rotate_band(glyph_rows((512, 512), row_pitch=40), 200, 320, ddeg=35)
    feats = _feats(f)
    assert flag_orientation(feats).sum() >= 1


def test_trace_health_high_on_text_low_on_garbled():
    clean = _feats(glyph_rows((512, 512), row_pitch=40))
    clean_rep = trace_health(clean, flag_tiles(clean))
    assert isinstance(clean_rep, ScoreReport)
    assert clean_rep.score >= 80
    bad = _feats(garble_patch(glyph_rows((512, 512), row_pitch=40), 128, 384, 128, 384))
    bad_rep = trace_health(bad, flag_tiles(bad))
    assert bad_rep.score < clean_rep.score
    assert 0 <= bad_rep.score <= 100


def test_trace_health_all_low_confidence_is_not_healthy():
    feats = _feats(np.zeros((512, 512)))
    assert not feats.confidence.any()
    rep = trace_health(feats, flag_tiles(feats))
    assert rep.low_conf_frac == 1.0
    assert rep.score == 0


def test_input_warning_flags_noise_not_text():
    noise = np.random.default_rng(0).random((512, 512))
    nf = analyze_tiles(noise, np.ones(noise.shape, bool), tile=128, overlap=0.5)
    assert input_warning(nf, flag_tiles(nf)) is not None
    cf = _feats(glyph_rows((512, 512), row_pitch=40))
    assert input_warning(cf, flag_tiles(cf)) is None
```

- [ ] **Step 7: Run the full suite**

Run: `~/.venvs/plumbline/bin/pytest -q`
Expected: PASS. If `test_garble_quiet_on_clean_text` or `test_trace_health_high_on_text` fail, the `band_thresh=0.12` default needs tuning against `glyph_rows` — adjust it (and note the value) here; final tuning against real data is Task 6.

- [ ] **Step 8: Commit**

```bash
git add plumbline/coherence.py plumbline/score.py tests/test_coherence.py tests/test_score.py
git commit -m "feat: band-based analysis core and flags

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: CLI auto-tile + report/dashboard labels

Make `--tile` optional (auto by default) and finish the user-facing labels.

**Files:**
- Modify: `plumbline/cli.py`
- Modify: `plumbline/render.py` (heatmap label + scale)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test** — append to `tests/test_cli.py`:

```python
def test_run_auto_tile(tmp_path):
    import numpy as np
    from PIL import Image
    from plumbline.cli import main
    from plumbline.synthetic import glyph_rows
    ink = (glyph_rows((640, 640), row_pitch=40) * 255).astype("uint8")
    p = tmp_path / "ink.png"; Image.fromarray(ink).save(p)
    out = tmp_path / "r.html"
    rc = main(["run", str(p), "-o", str(out)])     # no --tile -> auto
    assert rc == 0 and out.exists()
```

- [ ] **Step 2: Run, verify failure**

Run: `~/.venvs/plumbline/bin/pytest tests/test_cli.py::test_run_auto_tile -q`
Expected: FAIL — current `--tile` default is `256` (int), but more importantly confirm the auto path; if it already passes by accident, still make the default change in Step 3.

- [ ] **Step 3: `cli.py`** — default `--tile` to `None` for both subcommands:

```python
    run.add_argument("--tile", type=int, default=None,
                     help="tile size in px (default: auto-adapt to text scale)")
```
```python
    batch.add_argument("--tile", type=int, default=None,
                       help="tile size in px (default: auto-adapt to text scale)")
```

`analyze_tiles(... tile=args.tile ...)` already forwards `None` correctly (Task 4 handles auto).

- [ ] **Step 4: `render.py`** — `heatmap_png` label + scale (band contrast is ~0..0.4, not 0..1):

```python
def heatmap_png(features) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.where(features.confidence, features.band_strength, np.nan)
    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=max(vmax, 1e-3))
    fig.colorbar(im, ax=ax, fraction=0.046, label="row coherence")
    ax.set_axis_off()
    return _fig_to_png(fig)
```

- [ ] **Step 5: Run, verify pass**

Run: `~/.venvs/plumbline/bin/pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add plumbline/cli.py plumbline/render.py tests/test_cli.py
git commit -m "feat: auto-adapt tile size by default; row-coherence heatmap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Real-IR regression, threshold tuning, docs

The keystone: enforce that genuine legible text (the input that scored 0/100) now scores reasonably and is not blanketed in garble flags. Tune `band_thresh` / `k` against both `glyph_rows` and the real IR until green.

**Files:**
- Create: `tests/test_real_ir.py`
- Modify (tuning only, if needed): `plumbline/score.py` (`band_thresh`), `plumbline/coherence.py` (`k`)
- Modify: `README.md` (known gap), `docs/superpowers/specs/2026-06-01-plumbline-text-row-core-design.md` (mark implemented)

- [ ] **Step 1: Write the regression test** — `tests/test_real_ir.py`:

```python
import os
import numpy as np
import pytest

IR = "data/frag1/ir.png"
MASK = "data/frag1/mask.png"


@pytest.mark.skipif(not os.path.exists(IR), reason="gitignored real IR image absent")
def test_real_ir_text_scores_and_is_not_blanketed():
    from PIL import Image
    from plumbline.coherence import analyze_tiles
    from plumbline.score import flag_tiles, trace_health
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(IR).convert("L"), dtype=np.float64) / 255.0
    ink = 1.0 - img                              # IR photo: invert so ink=high
    mask = None
    if os.path.exists(MASK):
        m = np.asarray(Image.open(MASK).convert("L")) > 127
        if m.shape == ink.shape:
            mask = m
    feats = analyze_tiles(ink, mask=mask, tile=None)   # auto-adapt scale
    flags = flag_tiles(feats)
    rep = trace_health(feats, flags)
    n_conf = int(feats.confidence.sum())
    assert n_conf > 0, "no analyzable tiles on real text"
    garble_frac = int((flags.garble & feats.confidence).sum()) / n_conf
    # The redesign's whole point: real legible text must NOT score ~0 nor be
    # blanketed in garble (the stripe core scored this exact image 0/100).
    assert rep.score > 25, f"score {rep.score} too low on real text"
    assert garble_frac < 0.5, f"garble blankets {garble_frac:.0%} of confident tiles"
```

- [ ] **Step 2: Run it**

Run: `~/.venvs/plumbline/bin/pytest tests/test_real_ir.py -q`
Expected: either PASS, or FAIL on the score/garble assertions — proceed to tuning.

- [ ] **Step 3: Tune if needed.** If the test fails: print diagnostics with `~/.venvs/plumbline/bin/python -c "..."` (median `band_strength` over confident tiles, the auto-picked tile size). Adjust **one** knob at a time and re-run *both* `tests/test_real_ir.py` and `tests/test_score.py`:
  - `band_thresh` (in `flag_garble`, `score.py`): lower it toward the gap between real-text band contrast (~0.16) and garble (~0.09) — e.g. 0.10 — if real text over-flags; raise it if garble under-flags. Keep it above the garble floor.
  - `k` (in `estimate_scale_and_skew`, `coherence.py`): raise toward 3.0 if the auto tile is too small to span rows on the IR; lower if it clamps to `TILE_MAX` and starves the grid.
  Record the final values in a one-line comment next to each constant. Do **not** weaken the test thresholds (`>25`, `<0.5`) to pass — those encode the redesign's success criteria.

- [ ] **Step 4: Run the full suite**

Run: `~/.venvs/plumbline/bin/pytest -q`
Expected: PASS (real-IR test skips if data absent; passes if present). Confirm no test count regressions vs. Task 5.

- [ ] **Step 5: Document the known gap** — in `README.md`, under interpreting scores, add:

```markdown
**Known limitation:** Plumbline catches rotation/drift (`orient_break`) and
garble (`garble`). A pure *vertical sheet-jump* — rows shifting up/down at a
seam without rotating or changing spacing — is not yet detected. Tile size
auto-adapts to text scale; pass `--tile` to override.
```

Mark the spec implemented: change its `**Status:**` line to `Implemented 2026-06-01 (branch redesign-text-row-core).`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: real-IR text-row regression + tuning; document known gap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** estimate_scale_and_skew (Task 3) ✓; band_strength primitive (Task 3/4) ✓; demoted gated spacing (Task 4) ✓; garble = flipped structure rule (Task 4) ✓; renamed model fields (Task 2) ✓; auto-tile CLI (Task 5) ✓; glyph_rows + real-IR validation rule (Tasks 1, 6) ✓; known-gap doc (Task 6) ✓.
- **Type consistency:** `band_strength`, `pitch_strength`, `garble`, `spacing_break`, `n_garble`, `n_spacing` are used identically across model/score/render/report/cli/dashboard/templates/tests. TileFeatures and IndexRow stay positional with unchanged field order.
- **Tuning is empirical, asserted by the real-IR test** — `band_thresh` and `k` are starting points; the test thresholds (`>25`, `<0.5`) are the contract and must not be relaxed.
