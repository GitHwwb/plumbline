# Plumbline — text-row analysis core (redesign)

**Date:** 2026-06-01
**Status:** Implemented 2026-06-02 (branch redesign-text-row-core).
**Supersedes the analysis core of:** `2026-05-31-plumbline-design.md` (tiling, scoring shell,
report, and dashboard are retained unchanged).

## Motivation

Plumbline's original core assumes ink forms **continuous parallel stripes**: it measures
structure-tensor *anisotropy* per tile and flags tiles that have ink but low anisotropy
(`structure_loss`). Tested at last on genuine legible ink — Frag1's IR photo
(`data/frag1/ir.png`, large Greek letters), inverted so ink is high — the tool scored it
**0/100 at every tile size**, dominated by structure-loss, and even warned "not an ink
prediction" on real text.

Root cause is a design flaw, not tuning: real text is **discrete letters** (gaps, strokes in
every direction), so per-tile anisotropy is low everywhere and the structure-loss rule
punishes letters. Synthetic stripes scored ~96 and gave false confidence. The tool rewarded
stripe texture and punished writing — backwards from the goal.

This redesign replaces the per-tile primitive and the flag definitions with **text-row
structure detection**, grounded in measurements on the real IR image rather than synthetic
stripes.

## Evidence (real-data probes, 2026-06-01)

Two throwaway probes on `data/frag1/ir.png` (inverted, cropped to mask bbox) drove the design:

**Probe 1 — is the text periodic?** No. The global horizontal projection profile is dominated
by the fragment's overall ink-density *envelope*, and its autocorrelation decays smoothly with
**no periodic peaks**. Per-tile autocorrelation at 256–1024px reported "banding" only because
it latched onto ~6px fine texture (papyrus/sensor), not text rows — the same false-confidence
trap as anisotropy. Frag1 IR has only ~4–5 huge rows (~1000px+ pitch), so no small tile holds a
full row period.

**Probe 2 — does projection-profile *bandedness* discriminate text from garble?** Yes, when the
tile spans multiple rows. Bandedness = detrended projection-profile contrast (std of the
envelope-detrended row-projection, normalized by mean). Separation between real text tiles and
density-matched random "garble" tiles, in pooled-std units:

| Tile px | text bandedness | garble bandedness | separation |
|--------:|----------------:|------------------:|-----------:|
| 256     | 0.116           | 0.120             | −0.08 (none) |
| 512     | 0.145           | 0.103             | 0.31 (weak) |
| 1024    | 0.158           | 0.092             | 0.68 (moderate) |
| 1536    | 0.163           | 0.089             | **1.16 (good)** |
| 2048    | 0.375           | 0.087             | 0.76 (n=9, noisy) |

**Conclusions that shape the design:**
1. **Bandedness is a viable core primitive** — it separates text from garble — and is far more
   robust than strict periodicity (autocorrelation pitch jumped 85→79→none across scales).
2. **Tile scale must match text scale.** A fixed 256px tile is useless on giant text; Frag1 IR
   registers only at ~1024–1536px. Dense scroll text (the real product target) would register
   at much smaller tiles. ⇒ **tile size must be auto-adapted to the image.**
3. The autocorrelation **decay length** (after detrending) is a usable scale proxy even with no
   clean periodic peak: Frag1 IR's profile decorrelates by lag ~500px ⇒ ~1500px tile, matching
   what worked empirically.

## Goals

- Score and flag a scroll-segment ink prediction by **text-row structure** (band coherence,
  orientation consistency), not stripe anisotropy.
- Score genuine legible text reasonably (Frag1 IR must not score 0 or get blanketed in flags)
  while still flagging garble, drift, and rotation.
- Auto-adapt tile size to text scale so the user need not guess `--tile`.
- CPU-only, no model inference, no GPU, no 3D, no server. Reuse the existing tiling, scoring
  shell, report, dashboard, and JSON sidecar.

## Non-goals

- Model inference or any learned component.
- Detecting a **pure vertical sheet-jump** (rows shift vertically without rotating or changing
  spacing — a band *phase* discontinuity). `orient_break` + `garble` catch rotation and garble;
  phase-only jumps are a documented known gap, deferred to a later pass.
- Changing the report/dashboard layout (UI is reused as-is; only field labels follow renames).

## Architecture

```
ink, mask
   │
   ▼
estimate_scale_and_skew(ink, mask)  ──► (tile_size, global_theta)   [NEW, one downsampled pass]
   │
   ▼
tile_grid(...)                       [unchanged]
   │
   ▼
per-tile row_features(sub, seed_theta) ──► TileFeatures             [REPLACES structure-tensor core]
   │
   ▼
flag_tiles(features) ──► FlagMap     [REWRITTEN: garble / orient_break / spacing_break]
   │
   ▼
trace_health(features, flags) ──► ScoreReport   [shell unchanged]
   │
   ▼
write_report / write_json / dashboard            [unchanged except renamed fields]
```

### Component: `estimate_scale_and_skew(ink, mask) -> (tile_size, global_theta)`

- Downsample `ink` to ~1000px max dimension for speed.
- **Skew:** over a search range of about ±25°, rotate and compute the detrended global
  projection profile; pick the angle maximizing its contrast (std). Use a range wide enough that
  the optimum does not sit at the boundary (Probe 1 hit its ±10° limit).
- **Scale:** detrend the best-angle profile (subtract a broad moving average to remove the
  fragment envelope), take its autocorrelation, and find the **decay length** — the lag at which
  it first falls below a small threshold (≈0.2), or the first genuine periodic peak if present.
  Map back to full-resolution pixels.
- `tile_size = clamp(round(k · scale), [TILE_MIN, TILE_MAX])` with `k` ≈ 3 (several rows per
  tile) and bounds about `[256, 2048]`.
- A user-supplied `--tile` overrides estimation (auto is the default when `--tile` is omitted).

### Component: per-tile `row_features(sub, seed_theta)`

For each confident tile (enough ink + mask coverage; gating logic unchanged):

- **orientation `theta`** — the local angle (searched around `seed_theta`, range ±~25°) that
  maximizes the detrended projection-profile contrast. Replaces the structure-tensor angle.
- **`band_strength` (0..1)** — detrended projection-profile contrast at `theta`, normalized so
  garble ≈ 0 and clean rows ≈ 1. **The core "rowness" signal**, replacing `anisotropy`.
- **`pitch`, `pitch_strength`** — strongest autocorrelation peak of the profile within a
  plausible row-pitch band, and its normalized height. **Secondary and gated**: used only by
  `spacing_break`, never by `garble` or the score directly.
- **`density`, `confidence`** — ink fraction and mask coverage; gating thresholds unchanged.

`analyze_tiles(ink, mask, tile=None, overlap=0.5)` orchestrates: estimate (or accept) tile size
and global skew, build the grid, fill `TileFeatures`.

### Data model (`model.py`)

`TileFeatures`:
- `anisotropy` → **`band_strength`** (0..1 row-band contrast).
- `strength` → **`pitch_strength`** (0..1 autocorrelation-peak height).
- unchanged: `theta`, `pitch`, `density`, `confidence`, `tiles`, `n_rows`, `n_cols`.

`FlagMap`:
- `structure_loss` → **`garble`**.
- `pitch_break` → **`spacing_break`**.
- unchanged: `orient_break`; `any_flag` = `orient_break | spacing_break | garble`.

`ScoreReport`:
- `n_structure` → **`n_garble`**; `n_pitch` → **`n_spacing`**; unchanged: `score`, `n_orient`,
  `low_conf_frac`.

### Flags (`score.py`)

- **`garble`** (replaces `structure_loss`, flipped right-way-round): `confidence &
  (band_strength < band_thresh)` — confident ink but no row-band contrast ⇒ structureless mottle
  (garble or non-text).
- **`orient_break`** (kept): band orientation deviates from the confidence-weighted local
  consensus beyond `deg_thresh` (doubled-angle consensus, reflect padding — existing logic, now
  fed by band orientation).
- **`spacing_break`** (kept, **tightly gated**): row pitch departs from the local median pitch
  beyond `rel_thresh`, **only on tiles whose `pitch_strength` exceeds a confidence gate** so the
  unreliable-pitch problem cannot manufacture noise flags. Expected to be rare; that is intended.
- **`trace_health`** shell unchanged: `score = 100 · (1 − flagged_confident_fraction)`; score 0
  with `low_conf_frac == 1.0` when nothing is analyzable.
- **`input_warning`** re-keyed onto the **`garble`** fraction: most analyzable tiles dense but
  structureless ⇒ "may not be an ink prediction."

### Renders (`render.py`)

- `heatmap_png` plots `band_strength` (label "row coherence") instead of anisotropy.
- `orientation_png` still uses `theta` (now band orientation) — no change.
- `flags_png` / `flagged_regions` / `overlay_png` consume the renamed `FlagMap` fields; colors
  remap structure→garble, pitch→spacing.

## Validation (the hard-learned rule, enforced by tests)

Validate on **text-like** inputs, never continuous stripes (stripes gave false confidence):

- Add `glyph_rows(...)` to `synthetic.py`: rows of discrete glyphs with intra-word letter gaps
  and inter-word spaces, configurable row count/pitch/angle — the dense-text target (the
  OPTION-2 throwaway, made permanent).
- Tests assert:
  1. `glyph_rows` → high score, **no `garble`** flags.
  2. `glyph_rows + garble_patch` → `garble` localizes to the patch.
  3. `glyph_rows + rotate_band` → `orient_break` localizes the rotated band.
  4. pure noise field → low score **and** `input_warning` fires.
  5. **real `data/frag1/ir.png`** with auto-tiling → **non-zero, reasonable score**, and flags
     **do not blanket the legible text** (guards against the exact 0/100 regression). Skipped
     cleanly if the gitignored image is absent.
- Rewrite `test_coherence.py` and `test_score.py` for the new primitives; update field
  references in `test_render.py` / `test_report.py` / any JSON assertions.

## Risks / open items

- **Scale estimation on atypical images.** The decay-length proxy is heuristic. Bounds
  `[TILE_MIN, TILE_MAX]` and the real-IR test guard against pathological picks; thresholds
  (`k`, decay threshold, `band_thresh`) will be tuned against the glyph_rows + Frag1 IR fixtures
  during implementation.
- **Coarse grids on giant text.** At ~1536px tiles Frag1 IR yields few tiles, limiting error
  localization. Acceptable: real autoseg scroll segments are larger and denser; giant single
  fragments are an edge case the score should still get *right* even if coarse.
- **Pure phase-jump sheet errors** are out of scope this pass (see Non-goals).
