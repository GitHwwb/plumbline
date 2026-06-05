# Coherence View + Tile-Sizing + Spacing-Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Plumbline robust on sparse/loosely-periodic real papyrus text by fixing the coherence *view* (fixed scale + bounded display), the auto *tile-sizing* (stop ballooning on sparse text), and the *spacing* flag (reliability gate) — without changing the `band_contrast` scoring primitive or the garble flag.

**Architecture:** Four code areas, each independently testable. `render.py` gets a bounded display transform + fixed `0–1` color scale (cosmetic only). `coherence.py::row_pitch` gains an optional reliability output (peak prominence), stored on `TileFeatures.pitch_reliability`; `score.py::flag_spacing` gates on it (mirrors the existing orientation reliability gate). `coherence.py::estimate_scale_and_skew` switches its tile-scale proxy from autocorrelation *decay length* (inflates on sparse text) to dominant *row pitch* (the real line spacing), preserving giant-text behavior. `band_contrast`, `flag_garble`, and the orientation path are untouched.

**Tech Stack:** Python 3.11+, numpy, scipy.ndimage, matplotlib (Agg), pytest. Run tools via `~/.venvs/plumbline/bin/...`. Source tree at `/Users/jonathanlopes/Documents/plumbline`; venv at `~/.venvs/plumbline` (outside iCloud — do not relocate).

**Spec:** `docs/superpowers/specs/2026-06-03-plumbline-coherence-view-tiling-spacing-design.md`

**Background facts (already verified, do not re-litigate):**
- `band_contrast = std / max(mean, 0.08)` correctly rejects noise/smear (→0.0–0.03) and rates text high; the garble flag (`band < 0.10`) keys off it and is correct. **Keep it.**
- A bright single band is a *view-only* artifact (garble fires on *low* band, so it never causes a false garble).
- The coherence heatmap currently autoscales `vmax` to the brightest tile, so one outlier repaints everything.
- Auto-tile = `4.5 × autocorrelation-decay-length`; decay is long on sparse text → 1774 px tiles on `gp_20230904`.
- Spacing flags fire on guessed pitch (`row_pitch` returned 159–684 px across `gp_20230904` tiles).
- Pattern to mirror: `dominant_orientation(..., return_reliability=True)` + `TileFeatures.orient_reliability` + `flag_orientation` gating on `rel >= rel_thresh`.

---

## Task 1: Coherence view — bounded display + fixed 0–1 scale (render-only)

**Files:**
- Modify: `plumbline/render.py` (add `_COH_S`, `_coherence_display`; rewrite `heatmap_png` body lines 48–61)
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_render.py`:

```python
def test_coherence_display_bounded_and_compressed():
    from plumbline.render import _coherence_display
    bands = np.array([0.0, 0.05, 0.9, 3.0, 50.0])   # empty, noise, text, sliver, extreme
    v = _coherence_display(bands)
    assert v.min() >= 0.0 and v.max() <= 1.0          # bounded to [0,1]
    assert np.all(np.diff(v) >= 0)                     # monotonic in band
    assert v[1] < 0.1                                  # noise-level -> near 0
    assert v[3] > v[2]                                 # sliver still > text...
    assert (v[3] - v[2]) < (3.0 - 0.9)                 # ...but gap is compressed
    assert (v[4] - v[3]) < 0.05                        # saturates: 3 vs 50 nearly equal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_render.py::test_coherence_display_bounded_and_compressed -v`
Expected: FAIL with `ImportError: cannot import name '_coherence_display'`.

- [ ] **Step 3: Implement the transform + fixed scale**

In `plumbline/render.py`, add near the top (after the imports / `_fig_to_png`):

```python
# Coherence display: squash the unbounded band_contrast into [0,1) for a FIXED
# color scale, so colors mean the same thing in every report and a single bright
# band can't dominate the scale. Display-only -- scoring uses raw band_strength.
_COH_S = 1.0  # calibrated by eye on the demo set (Task 5); text ~0.6-0.8, noise ~0


def _coherence_display(band):
    return np.tanh(np.asarray(band, dtype=float) / _COH_S)
```

Then replace the body of `heatmap_png` (current lines 48–61) with:

```python
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.where(features.confidence, _coherence_display(features.band_strength), np.nan)
    w = max((t.x1 for t in features.tiles), default=data.shape[1])
    h = max((t.y1 for t in features.tiles), default=data.shape[0])
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=[0, w, h, 0], aspect="equal", interpolation="nearest")
    if ink01 is not None:
        a = np.asarray(ink01, dtype=float)
        k = max(1, int(round(max(a.shape) / 2000)))      # downsample for display only
        a = np.clip(a[::k, ::k], 0.0, 1.0)
        marks = np.zeros(a.shape + (4,)); marks[..., :3] = 1.0; marks[..., 3] = a * 0.9
        ax.imshow(marks, extent=[0, w, h, 0], aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row coherence (0–1)")
    ax.set_axis_off()
    return _fig_to_png(fig)
```

(Also update the `heatmap_png` docstring's first sentence to: `"Per-tile row coherence on a FIXED 0-1 scale (bounded display of band_strength) so colors are stable across reports."` Keep the rest of the docstring.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_render.py -v`
Expected: PASS (new test + the existing `test_png_renderers_return_png_bytes` / `test_flagged_regions_lists_pixel_locations`).

- [ ] **Step 5: Commit**

```bash
git add plumbline/render.py tests/test_render.py
git commit -m "Coherence view: bounded tanh display + fixed 0-1 color scale"
```

---

## PIVOT (2026-06-03): Tasks 2 & 3 → a single consensus-gate task (Task 2′)

Implementation surfaced that a *per-tile* pitch reliability does **not** discriminate scattered from real periodic text (median/normalized-prominence saturate to ~1.0; max-others gives overlapping tiny values — clean 0.04–0.08 vs scattered 0.05). The real failure mode is inter-tile pitch **inconsistency**: scattered text has no single line spacing, so neighbors latch onto different pitches (159–684 px) and `flag_spacing` fires on the disagreement. **Tasks 2 & 3 below are SUPERSEDED** by **Task 2′** — a neighborhood pitch-**consensus** gate in `flag_spacing` alone (no new `TileFeatures` field, no `row_pitch` change). Validated: at `consensus_gate=0.70`, the real coarse/fine jump still flags (2) while `gp_20230904` drops 4→0. (Task 2's per-tile infra commit was reverted; branch is back at the Task-1 state.)

---

## Task 2 (SUPERSEDED): Pitch reliability — `row_pitch` output + `TileFeatures` field

**Files:**
- Modify: `plumbline/model.py` (add `pitch_reliability` field to `TileFeatures`)
- Modify: `plumbline/coherence.py::row_pitch` (add `return_reliability` param), `analyze_tiles` (populate it)
- Test: `tests/test_coherence.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_coherence.py`:

```python
def test_row_pitch_reliability_high_for_clean_low_for_noise():
    clean = glyph_rows((512, 512), row_pitch=40)
    pc, sc, rc = row_pitch(clean, 0.0, min_lag=10, max_lag=256, return_reliability=True)
    noise = np.random.default_rng(3).random((512, 512))
    pn, sn, rn = row_pitch(noise, 0.0, min_lag=10, max_lag=256, return_reliability=True)
    assert 0.0 <= rn <= 1.0 and 0.0 <= rc <= 1.0
    assert rc > rn                                  # clean rows are a more prominent peak

def test_row_pitch_default_still_two_tuple():
    out = row_pitch(glyph_rows((512, 512), row_pitch=40), 0.0, min_lag=10, max_lag=256)
    assert len(out) == 2                            # back-compat: (pitch, strength)

def test_analyze_tiles_has_pitch_reliability():
    tf = analyze_tiles(glyph_rows((512, 512), row_pitch=40), tile=256, overlap=0.5)
    assert tf.pitch_reliability is not None
    assert tf.pitch_reliability.shape == (tf.n_rows, tf.n_cols)
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_coherence.py::test_row_pitch_reliability_high_for_clean_low_for_noise tests/test_coherence.py::test_analyze_tiles_has_pitch_reliability -v`
Expected: FAIL — `row_pitch() got an unexpected keyword argument 'return_reliability'` and `AttributeError: 'TileFeatures' object has no attribute 'pitch_reliability'`.

- [ ] **Step 3a: Add the `TileFeatures` field**

In `plumbline/model.py`, in the `TileFeatures` dataclass, add **after** the existing `orient_reliability` block (it must remain the last fields, both defaulted):

```python
    pitch_reliability: Optional[np.ndarray] = None  # (n_rows, n_cols) 0..1: prominence of the
    #   row-pitch autocorrelation peak. Low when several lags compete (scattered text
    #   has no real pitch); gates spacing-break flagging so guessed pitch can't flag.
```

(`Optional` is already imported in `model.py`; `= None` default matches the `orient_reliability` pattern.)

- [ ] **Step 3b: Add reliability to `row_pitch`**

In `plumbline/coherence.py`, replace `row_pitch` (current lines 94–114) with:

```python
def row_pitch(img, theta=0.0, min_lag=8, max_lag=None, return_reliability=False):
    """Row spacing (px) from the profile autocorrelation + its 0..1 peak height.
    Returns (nan, 0.0) when there is no clear periodic peak.

    With return_reliability=True also returns a 0..1 reliability = prominence of the
    chosen peak over the rest of the autocorrelation band (~1 for a single dominant
    peak; ~0 when several lags compete). flag_spacing gates on it so guessed pitch on
    scattered text can't manufacture a spacing break."""
    def _ret(pitch, strength, reliability):
        return (pitch, strength, reliability) if return_reliability else (pitch, strength)
    p = projection_profile(img, theta)
    p = p - p.mean()
    if not np.any(p):
        return _ret(float("nan"), 0.0, 0.0)
    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = p.size // 2 if max_lag is None else int(max_lag)
    lo = max(2, int(min_lag))
    seg = ac[lo:hi]
    if seg.size < 3:
        return _ret(float("nan"), 0.0, 0.0)
    i = np.arange(1, seg.size - 1)
    ismax = (seg[i] > seg[i - 1]) & (seg[i] > seg[i + 1])
    cand = i[ismax]
    if cand.size == 0:
        return _ret(float("nan"), 0.0, 0.0)
    b = int(cand[int(np.argmax(seg[cand]))])
    peak = float(seg[b])
    others = np.delete(seg, b)
    base = float(np.median(others)) if others.size else 0.0
    reliability = max(0.0, min(1.0, (peak - base) / (abs(peak) + 1e-9)))
    return _ret(float(b + lo), peak, reliability)
```

- [ ] **Step 3c: Populate it in `analyze_tiles`**

In `plumbline/coherence.py::analyze_tiles`, add the `prel` array next to `rel` (current line 169):

```python
    rel = np.zeros((nr, nc))
    prel = np.zeros((nr, nc))
```

Change the pitch call inside the confident branch (current lines 183–186) to capture reliability:

```python
        p, s, pr = row_pitch(sub, th, min_lag=max(8, tile // 16),
                             max_lag=sub.shape[0] // 2, return_reliability=True)
        pitch[t.row, t.col] = p
        pstr[t.row, t.col] = s
        prel[t.row, t.col] = pr
        conf[t.row, t.col] = True
```

And the return (current line 188):

```python
    return TileFeatures(nr, nc, theta, band, pitch, pstr, density, conf, tiles, rel, prel)
```

- [ ] **Step 4: Run to verify they pass**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_coherence.py tests/test_smoke.py -v`
Expected: PASS (new tests + existing `test_row_pitch_detects_known_spacing` 2-tuple unpack + `test_smoke` keyword `TileFeatures` construction still work).

- [ ] **Step 5: Commit**

```bash
git add plumbline/model.py plumbline/coherence.py tests/test_coherence.py
git commit -m "Pitch reliability: row_pitch peak prominence + TileFeatures.pitch_reliability"
```

---

## Task 3 (SUPERSEDED — see Task 2′ consensus gate): Spacing reliability gate

**Files:**
- Modify: `plumbline/score.py::flag_spacing` (add `reliability_gate`, gate on `pitch_reliability`)
- Test: `tests/test_score.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_score.py`:

```python
def test_spacing_gate_suppresses_unreliable_pitch():
    from plumbline.model import TileFeatures, Tile
    n = 3
    conf = np.ones((n, n), bool)
    pitch = np.full((n, n), 40.0); pitch[1, 1] = 120.0     # one strongly deviant tile
    pstr = np.full((n, n), 0.5)                            # passes the strength gate
    tiles = [Tile(r, c, r * 10, r * 10 + 10, c * 10, c * 10 + 10)
             for r in range(n) for c in range(n)]
    base = dict(n_rows=n, n_cols=n, theta=np.zeros((n, n)), band_strength=np.ones((n, n)),
                pitch=pitch, pitch_strength=pstr, density=np.ones((n, n)),
                confidence=conf, tiles=tiles)
    reliable = TileFeatures(**base, pitch_reliability=np.full((n, n), 0.9))
    unreliable = TileFeatures(**base, pitch_reliability=np.full((n, n), 0.05))
    assert flag_spacing(reliable).sum() >= 1      # deviant pitch flags when reliable
    assert flag_spacing(unreliable).sum() == 0    # gate suppresses when unreliable
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_score.py::test_spacing_gate_suppresses_unreliable_pitch -v`
Expected: FAIL — `unreliable` case still flags (no gate yet), so `flag_spacing(unreliable).sum() == 0` fails.

- [ ] **Step 3: Add the gate**

In `plumbline/score.py`, replace the `flag_spacing` signature and its `valid` setup (current lines 39–43) with:

```python
def flag_spacing(features, rel_thresh=0.35, radius=2, strength_gate=0.30,
                 reliability_gate=0.30):
    """Row pitch departs from the local median -- ONLY on tiles with a confident
    AND reliably-determined periodic peak. The reliability gate (pitch peak
    prominence) stops scattered text, whose autocorrelation latches onto noise,
    from manufacturing spacing breaks -- mirrors the orientation reliability gate."""
    p = features.pitch
    valid = features.confidence & np.isfinite(p) & (features.pitch_strength >= strength_gate)
    prel = getattr(features, "pitch_reliability", None)
    if prel is not None:
        valid = valid & (prel >= reliability_gate)
```

(Leave everything from `if not valid.any():` onward unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_score.py -v`
Expected: PASS — new test passes, and `test_spacing_flag_fires_on_pitch_change` still passes (the coarse/fine glyph_rows have prominent, reliable pitch peaks so they still flag).

- [ ] **Step 5: Commit**

```bash
git add plumbline/score.py tests/test_score.py
git commit -m "Spacing reliability gate: flag only where row pitch is reliably determined"
```

---

## Task 4: Tile-sizing — row-pitch proxy instead of decay length

**Files:**
- Modify: `plumbline/coherence.py::estimate_scale_and_skew` (current lines 117–144)
- Test: `tests/test_coherence.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_coherence.py`:

```python
def test_tile_tracks_row_pitch_not_sparsity():
    # tile size should follow ROW PITCH (line spacing), not density: a sparse row
    # (big inter-glyph gaps) at a given pitch must NOT get a bigger tile than a
    # dense row at the SAME pitch -- the old decay-length proxy inflated the sparse one.
    dense = glyph_rows((1536, 1536), row_pitch=80, glyph=40, gap=10, seed=1)
    sparse = glyph_rows((1536, 1536), row_pitch=80, glyph=40, gap=180, seed=2)
    td, _ = estimate_scale_and_skew(dense)
    ts, _ = estimate_scale_and_skew(sparse)
    assert 256 <= td <= 2048 and 256 <= ts <= 2048
    assert ts <= td * 1.5            # sparsity at fixed pitch doesn't blow up the tile

def test_tile_bigger_for_bigger_pitch():
    fine = glyph_rows((2048, 2048), row_pitch=60, glyph=28, gap=14, seed=1)
    coarse = glyph_rows((2048, 2048), row_pitch=240, glyph=110, gap=40, seed=2)
    tf, _ = estimate_scale_and_skew(fine)
    tc, _ = estimate_scale_and_skew(coarse)
    assert tc > tf                   # bigger line spacing -> bigger tile
```

- [ ] **Step 2: Run to verify behavior (may pass or fail; this anchors the change)**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_coherence.py::test_tile_tracks_row_pitch_not_sparsity tests/test_coherence.py::test_tile_bigger_for_bigger_pitch -v`
Expected: `test_tile_tracks_row_pitch_not_sparsity` FAILS on the current decay-length proxy (sparse tile inflated). (`test_tile_bigger_for_bigger_pitch` may already pass.)

- [ ] **Step 3: Switch the scale proxy to row pitch (with decay fallback)**

In `plumbline/coherence.py`, replace `estimate_scale_and_skew` (current lines 117–144) with:

```python
def estimate_scale_and_skew(ink, mask=None, k_rows=4.0, target=1000.0):
    """Pick a tile size spanning several text rows + the global skew angle.
    Returns (tile_size:int, theta:float). Scale comes from the dominant ROW PITCH
    (line spacing) of the global profile -- not the autocorrelation decay length,
    which inflates on sparse text and ballooned the tile (see the design spec).
    Falls back to the decay length when no clear pitch peak exists (e.g. a few
    giant rows), so giant-letter fragments still get large tiles."""
    a = to01(ink)
    if mask is not None and mask.shape == a.shape and mask.any():
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(1.0, target / max(a.shape))
    small = _zoom(a, scale, order=1) if scale < 1.0 else a
    theta = dominant_orientation(small, seed=0.0, span=np.radians(25), n=21)
    # Primary: the dominant row pitch (first prominent autocorrelation peak) is the
    # real line spacing; sparsity does not inflate it the way the decay length does.
    pitch_small, pstr = row_pitch(small, theta, min_lag=4,
                                  max_lag=max(5, min(small.shape) // 2))
    if np.isfinite(pitch_small) and pstr > 0.15:
        row_h = pitch_small / max(scale, 1e-9)
    else:
        # Fallback: detrended-profile autocorrelation decay length (the old proxy).
        # Used when there is no clear pitch peak (e.g. a few giant rows) so giant
        # text still yields a large tile.
        det = _detrend(projection_profile(small, theta))
        det = det - det.mean()
        if not np.any(det):
            return TILE_MIN, theta
        ac = np.correlate(det, det, mode="full")[det.size - 1:]
        ac = ac / (ac[0] + 1e-9)
        below = np.where(ac[1:] < 0.2)[0]
        decay_small = int(below[0] + 1) if below.size else max(1, det.size // 4)
        row_h = decay_small / max(scale, 1e-9)
    tile = int(np.clip(round(k_rows * row_h), TILE_MIN, TILE_MAX))
    return tile, theta
```

- [ ] **Step 4: Run the new + guard tests**

Run: `~/.venvs/plumbline/bin/python -m pytest tests/test_coherence.py tests/test_real_ir.py -v`
Expected: PASS — the two new tile tests pass; `test_estimate_scale_and_skew_bounds_and_angle` (tile in 256–2048, theta≈8°) still passes; **`test_real_ir.py` still passes** (frag1-IR giant text still gets a large-enough tile via the row-pitch peak or the decay fallback).

If `test_real_ir` regresses (frag1-IR tile too small → garble_frac or orient_frac breaks), tune in this order and re-run: raise `k_rows` (4.0 → 5.0), then raise the `pstr > 0.15` fallback threshold (so giant-text falls back to the decay proxy). Do not lower `TILE_MAX`.

- [ ] **Step 5: Commit**

```bash
git add plumbline/coherence.py tests/test_coherence.py
git commit -m "Tile-sizing: scale from dominant row pitch, not decay length (fixes sparse-text balloon)"
```

---

## Task 5: Full regression + demo-set verification & calibration

**Files:** none (verification); may revisit `_COH_S` (Task 1) / `k_rows` (Task 4) constants.

- [ ] **Step 1: Run the whole suite**

Run: `~/.venvs/plumbline/bin/python -m pytest -q`
Expected: all tests green (the prior 46 + the new C1/C2/C3 tests).

- [ ] **Step 2: Regenerate the real demo set and check the success criteria**

Run:
```bash
~/.venvs/plumbline/bin/python /tmp/plumbline_real/gen.py            # all 8 real reports
~/.venvs/plumbline/bin/plumbline run /tmp/plumbline_real/newtest/gp_20230904.png \
    -o /tmp/plumbline_demo/newtest_gp_20230904.html --json /tmp/plumbline_real/gp0904.json
```
Then confirm, against `/tmp/plumbline_real/summary.json` + the gp0904 JSON:
- `gp_20230904` uses a **finer** tile grid than before (was 7×4, 1774 px) and its **spacing flags drop** (was 4) toward the genuinely-suspect ones.
- The **model prediction** `pred_s5` is essentially unchanged (was 85, orient 0, garble 43) and the other labels stay 85–100 (no garble/score regression from the view/tiling changes — they shouldn't change scoring at all).
- frag1-IR via `test_real_ir` already green in Step 1.

- [ ] **Step 3: Eyeball the coherence view on the demo server**

Open `localhost:8742/real_s5_seg20241025_sparse.html` and `localhost:8742/newtest_gp_20230904.html`, Coherence tab: colors should be on a stable `0–1` scale, the sliver no longer a lone yellow max, background blank, overlay toggle aligned. If text consistently reads too dim or too bright, adjust `_COH_S` in `render.py` (smaller → brighter) and re-run Step 2; re-commit if changed.

- [ ] **Step 4: Commit any calibration changes**

```bash
git add -A
git commit -m "Calibrate coherence display / tile-sizing constants against the demo set"
```
(Skip if no constants changed.)

---

## Notes for the implementer
- **Do not touch** `band_contrast`, `flag_garble`, `input_warning`, `dominant_orientation`/`flag_orientation`, `io.py`, or `templates/report.html.j2` — out of scope and tested.
- The reliability/`return_*` pattern is deliberately identical between orientation (already shipped) and pitch (this plan) — keep them parallel.
- All four code changes are *additive or display-only* with respect to scoring; if any demo score (esp. the prediction's garble) shifts materially, stop and investigate before proceeding.
