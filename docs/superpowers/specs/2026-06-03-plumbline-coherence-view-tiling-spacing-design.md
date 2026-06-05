# Plumbline — coherence view, tile-sizing & spacing-reliability redesign

**Goal:** Make Plumbline robust on sparse / loosely-periodic real papyrus text by fixing
the three things that actually misbehave there — the coherence *view*, the auto
*tile-sizing*, and the *spacing* flag — **without** changing the scoring/garble
primitive, which testing showed is correct.

**Status:** design approved (brainstorm 2026-06-03). Next: implementation plan.

---

## Background — what the experiments showed

Stress-testing the redesigned core on real GitHub-hosted ink labels + a model
prediction surfaced complaints about the **Coherence** view (a stray bright tile, a
"stretch", apparent over-flaring) and a low score on `gp_20230904` driven by **spacing**
flags. We investigated before designing. Findings (all measured, not assumed):

1. **The scoring metric is right; don't replace it.** `band_contrast = std / max(mean, 0.08)`
   cleanly separates structure from mush — pure noise → **0.03**, uniform smear → **0.00**,
   real text → high — which is exactly what the **garble** flag needs (garble fires on
   *low* band). Candidate replacements failed: a spectral row-band fraction rates pure
   noise **0.50** (same as text) and is inconsistent on single bands, so swapping it in
   would *break* noise rejection. **Decision: `band_contrast` and the garble flag are
   unchanged.**

2. **The "sliver" / yellow-flare is a view-only artifact.** `band_contrast` rates a
   single concentrated band high (e.g. 1.4–3.0) because `std/mean` is unbounded. But
   garble fires on *low* band, so a bright single band **never causes a false garble** —
   it only looks alarming because the heatmap **auto-scales `vmax` to the brightest tile**,
   so one outlier repaints the whole image and lowering the floor (0.05→0.08) *compressed*
   the scale and made mid-value text look more yellow. This is a rendering problem, not a
   scoring problem.

3. **No metric separates dense from sparse text — and that's correct.** Scattered letters
   still project into real horizontal rows, so sparse text *is* legitimately row-like. The
   tool should not penalize sparsity. The real sparse-text pains were elsewhere:

4. **Auto tile-sizing works backwards on sparse text.** `estimate_scale_and_skew` sets
   `tile = clip(4.5 × autocorrelation-decay-length, 256, 2048)`. Sparse/loosely-periodic
   text has a *long* decay length, so the tile balloons (e.g. `gp_20230904`, 4608×7168 →
   **1774 px** tile → a coarse 7×4 grid, 19 confident tiles). The decay proxy conflates
   "giant letters" (which genuinely need big tiles — frag1-IR) with "sparse scatter of
   normal-size letters" (which should get finer tiles).

5. **Spacing flags fire on guessed pitch.** On `gp_20230904`, `row_pitch` returns
   159, 159, 166, 482 px on the flagged tiles, and **159–684 px across all confident
   tiles** — there is no consistent pitch in scattered text, so the autocorrelation latches
   onto noise. The pitches clear the 0.30 strength gate but are meaningless; the spacing
   flag fires on the disagreement. This is the same "unreliable metric trusted anyway"
   failure that orientation had (and which we already fixed with a reliability gate).

---

## Scope

Three coupled components. **Out of scope:** replacing `band_contrast`, the garble flag's
behavior, the orientation reliability gate (already shipped), and any attempt to make the
score distinguish dense from sparse text (rejected as wrong by finding #3).

---

## Component 1 — Coherence view: fixed, bounded display

**Problem:** per-image `vmax` autoscaling makes colors relative/unstable (one outlier
repaints everything) and the underlying value is unbounded.

**Design (render-only, `plumbline/render.py::heatmap_png`):**

- Apply a **bounded display transform** to `band_strength` before rendering:
  `disp = tanh(band_strength / S)`, where `S` is a single calibration constant chosen on
  the demo set so that typical confident text lands ≈ **0.6–0.8**, pure noise ≈ **0**, and
  a single-band sliver **saturates near 1** instead of becoming a lone, scale-dominating
  max. Start point `S ≈ 1.0`; finalize by eye on the demo reports.
- Render with a **fixed scale `vmin=0, vmax=1`** (no `nanmax`), so identical ink produces
  identical colors across every report.
- Unchanged: non-confident tiles blank (NaN), the **overlay-ink** toggle, opaque cells,
  the "rough visual aid — orientation & flags are the actionable signals" caption.
- Colorbar label: `row coherence (0–1)`.

**Explicitly not touched:** `band_contrast`, `flag_garble`, `TileFeatures.band_strength`.
The transform exists only inside the renderer.

*Optional stretch (display only, behind the same transform):* lightly damp obvious
single-band tiles in the *view* — deferred unless calibration alone proves insufficient.

---

## Component 2 — Tile-sizing: a robust row-height estimate

**Problem:** the decay-length proxy grows the tile on sparse text. We need tiles that span
*a few real text rows* regardless of sparsity — large for giant letters, finer for
scattered normal-size letters.

**Design (`plumbline/coherence.py::estimate_scale_and_skew`):**

- Replace the "lag where autocorrelation first drops below 0.2" decay proxy with a
  **row-height / row-pitch estimate** that reflects actual glyph/row size, not sparsity:
  - Primary: the lag of the **first significant local maximum** of the detrended global
    profile's autocorrelation (the dominant row pitch).
  - Fallback when no clear peak exists: the **median vertical extent of ink runs** (typical
    glyph height) from the column-collapsed ink profile, or the existing decay proxy
    clamped — chosen during implementation by which keeps frag1-IR correct.
- `tile = clip(k_rows × row_height, TILE_MIN, TILE_MAX)` with `k_rows` ≈ 3–5 (tile spans a
  few rows). `TILE_MIN/MAX` stay 256/2048.
- **Hard constraint:** frag1-IR (giant text) must still pick a large tile so
  `tests/test_real_ir.py` stays green; `gp_20230904` should drop well below 1774 px to a
  finer grid. Both are calibration targets in the plan.

This is the **riskiest/meatiest** component — the exact estimator and `k_rows` are settled
empirically against frag1-IR + `gp_20230904` + the demo set during implementation, behind
a new unit test.

---

## Component 3 — Spacing reliability gate

> **SUPERSEDED (2026-06-03):** the per-tile pitch-reliability approach described here did not discriminate scattered from real text. Shipped instead as a neighborhood pitch-**consensus** gate in `flag_spacing` — see the PIVOT note in the implementation plan.

**Problem:** `flag_spacing` trusts `row_pitch` even where the autocorrelation peak is
ambiguous, so scattered text manufactures spacing breaks.

**Design (mirror the orientation reliability gate):**

- `plumbline/coherence.py::row_pitch` (or a sibling) computes a **pitch reliability** in
  `[0,1]` = **prominence of the chosen autocorrelation peak** relative to its surroundings
  (e.g. `(peak − local_baseline) / (peak + eps)`, ≈0 when several lags compete or the
  "peak" is noise). This is distinct from the existing peak *height* (`pitch_strength`),
  which scattered text can still pass.
- `plumbline/model.py::TileFeatures` gains `pitch_reliability: Optional[np.ndarray] = None`
  (trailing, defaulted — same pattern as `orient_reliability`). `analyze_tiles` populates it.
- `plumbline/score.py::flag_spacing` gates: flag only where
  `pitch_reliability >= rel_thresh` (≈0.30, calibrated) in addition to the existing
  confidence + `pitch_strength` gate. Strictly *reduces* spacing flags.

---

## Architecture & data flow

```
load_image01 ─► estimate_scale_and_skew ─► analyze_tiles ─► TileFeatures ─► flag_tiles ─► trace_health
                  (C2: row-height tile)      (C3: store         (+pitch_         (C3: spacing
                                              pitch_reliability)  reliability)     reliability gate)
                                                                                 │
ink01, TileFeatures, flags ─► render.heatmap_png (C1: bounded disp + fixed 0–1 scale)
```

Files touched: `coherence.py` (C2 tile-sizing, C3 pitch reliability), `model.py` (C3
field), `score.py` (C3 gate), `render.py` (C1 display). **Unchanged:** `band_contrast`,
`flag_garble`, orientation path, `io.py`, the report template, the CLI.

---

## Testing

- **C1 view:** unit test that `heatmap_png` renders with fixed `vmin=0, vmax=1` and that
  the display value is bounded in `[0,1)` and monotonic in `band_strength`; eyeball the
  regenerated demo set (stable colors, sliver no longer a lone max, noise dark).
- **C2 tiling:** unit test asserting a sparse-scatter synthetic (rows of glyphs with large
  gaps) gets a **finer** grid than the old decay proxy would, while a giant-text synthetic
  (and frag1-IR via `test_real_ir`) still picks a large tile. `test_real_ir` must stay green.
- **C3 spacing:** new `test_score` cases — a synthetic with a genuine row-pitch jump still
  raises a spacing flag; a sparse/random-pitch field raises **none**.
- **Regression:** all existing tests green; demo set re-run shows `gp_20230904` spacing
  flags drop, and the model prediction + other labels are essentially unchanged.

## Risks & mitigations

- **C2 breaks frag1-IR.** Mitigated by the hard constraint + `test_real_ir` as a gate, and
  calibrating the row-height estimator against it explicitly.
- **C1 transform hides real low coherence.** Mitigated by keeping the transform monotonic
  and noise→0; it only compresses the *top*, where the distinction (text vs sliver) is
  cosmetic anyway. Scoring is untouched regardless.
- **C3 over-suppresses real spacing jumps.** Mitigated by the gate being *additive*
  (reliability AND the existing gates) and the synthetic "real jump still flags" test.

## Success criteria

1. Coherence colors are stable across reports (no autoscale); the sliver/yellow-flare no
   longer dominates; `band_contrast`/garble/score unchanged.
2. `gp_20230904`-class sparse images get a finer tile grid; frag1-IR unchanged.
3. `gp_20230904` spacing flags drop to the genuinely-suspect ones (or zero); a synthetic
   real spacing jump still flags.
4. All 46 existing tests green + the new C1/C2/C3 tests; demo-set scores stay sensible.
