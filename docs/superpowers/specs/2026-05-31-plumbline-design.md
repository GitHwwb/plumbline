# Plumbline — Segmentation Trace-Quality Report

**Date:** 2026-05-31
**Status:** Design approved, pre-implementation
**Context:** A contribution to the [Vesuvius Challenge](https://scrollprize.org) targeting the segmentation team's wishlist item — *"integrate ink detection to help see if an area was segmented correctly."* Intended to be genuinely useful to the segmentation team and a candidate monthly **progress-prize** submission.

---

## 1. Problem

Segmenting the Herculaneum scrolls means tracing a single sheet of rolled papyrus through a CT volume. A trace can silently **drift onto an adjacent sheet** ("sheet jump"), producing a flattened segment that looks plausible but is geometrically wrong. Today this is mostly caught by eye.

**Key insight:** ink prediction is a signal for trace *correctness*, not just content. In the flattened frame, a correctly traced sheet yields **continuous, parallel, regularly-spaced text lines**. When the trace drifts, the ink pattern shows the fingerprint of the error:

- an abrupt **orientation** change in the text lines,
- a **line-spacing** (pitch) jump,
- or a **collapse of linear/periodic structure** (garbled region) where ink density says there should be text.

Existing web viewers (Rudolph's Segment browser, Farritor's Scroll Viewer, Geiger's Scroll Sleuth, Hsiao's Segment/Volume Viewer) already *overlay* ink on segments. **Plumbline's differentiator is automatic diagnosis of trace quality**, not display.

## 2. What it is

A command-line tool that takes a segment's ink-prediction image and emits a **self-contained static HTML "trace-quality report"** that flags likely-wrong regions and assigns a global trace-health score.

- **Input:** an ink-prediction PNG + the segment's `flat_mask` (valid-area mask). Accepts a local path, or a `--segment-id` that is fetched via the `vesuvius` Python library / open data server.
- **Output:** one standalone `.html` file per segment (images embedded as base64 — no external assets), plus an optional `--json` sidecar for batch/pipeline use.
- **Runs entirely on CPU on a Mac.** No GPU, no model inference, no server.

### North star

Optimize for *"the segmentation team actually runs it."* Prize-worthiness follows from real utility + strong documentation; the two align.

### Non-goals (MVP)

- No ink-model **inference** (consumes existing predictions only).
- No 3D rendering, no editing/re-tracing — it **diagnoses**, it does not fix.
- No through-stack geometry signal (deferred; see §7).
- No multi-segment batch index page (deferred).

## 3. Segment data shape (verified against scrollprize.org/data_segments)

A segment provides:

- **Surface volume:** 65 flattened layer images `layers/00–64.tif`; **layer 32 = the papyrus surface** (32 above, 32 below along the surface normal).
- **Mesh:** `<id>.obj` + `<id>.mtl` with UV coordinates defining the flattening.
- **Ink predictions:** `<id>_prediction*.png` — 2D images **registered to the same flattened pixel grid** as the layers.
- **Composite/masks:** `composite.jpg`/`<id>.tif`, and `<id>_mask.png` / `<id>_flat_mask.png` / `<id>_flat.png`.
- **Resolution:** scans are ~7.91 µm/voxel, so the flattened grid is ~7.91 µm/pixel. Text reads as roughly parallel **horizontal text lines** within columns.

Plumbline (MVP) consumes only the **ink-prediction PNG** and the **`flat_mask`**.

## 4. Architecture

A small Python package; one focused, independently testable module per job.

```
plumbline/
  io.py          # load ink PNG + flat_mask; --segment-id fetch via vesuvius lib
  tiles.py       # sliding overlapping-tile grid restricted to the masked area
  coherence.py   # per-tile: structure-tensor orientation + anisotropy,
                 #           line-spacing (pitch) via directional FFT
  score.py       # segment-wide consensus, per-tile flagging, global score
  report.py      # render self-contained dashboard HTML (Jinja2, base64 images)
  cli.py         # `plumbline run <ink.png | --segment-id ID> -o report.html`
tests/           # synthetic-perturbation tests + small fixtures
templates/       # report.html.j2
examples/        # committed example report output
```

**Stack:** Python 3.13; `numpy`, `scipy`, `scikit-image` (structure tensor, FFT); `tifffile` + `Pillow` (image IO); `jinja2` (report); `vesuvius` library only on the optional fetch-by-id path. Packaged via `pyproject.toml`.

**Data flow:**
`load ink + mask → tile (within mask) → per-tile {orientation θ, anisotropy, line-pitch} → consensus + flagging → global score → HTML report (+ optional JSON)`

## 5. Diagnostic algorithm (the core)

Over a grid of overlapping tiles (default 256 px, 50% overlap), inside the mask only:

1. **Orientation & strength** — the structure tensor yields each tile's dominant ink-stripe **angle θ** and an **anisotropy/coherence** value (how strongly linear the local texture is — i.e., does it look like text lines).
2. **Line spacing** — project the tile perpendicular to θ, FFT the 1-D profile, extract the dominant **line pitch** and its spectral strength.
3. **Consensus** — across a correctly flattened sheet, θ varies *smoothly* and pitch is *stable*. Compute a robust local consensus field (e.g., median over a neighborhood) for orientation and pitch.
4. **Flag** a tile when any of:
   - θ deviates sharply from its neighborhood consensus (**orientation break** → likely sheet jump),
   - line pitch jumps relative to consensus (**pitch break**),
   - anisotropy/periodicity collapses **where ink density indicates text should exist** (**structure-loss / garbled**).
5. **Confidence gating** — tiles with too little ink are marked **low-confidence** (greyed) and are **never flagged**, so faint/sparse areas do not raise false alarms.
6. **Outputs** — orientation field, continuous **coherence heatmap**, discrete **flagged-region overlay**, and a global **trace-health score (0–100)** with a per-failure-mode breakdown (counts of orientation breaks, pitch breaks, structure-loss regions; % low-confidence area).

Classic, interpretable computer vision — no training.

### Tunable parameters (sensible defaults, CLI-overridable)

tile size, overlap fraction, minimum ink density for confidence, deviation thresholds for orientation/pitch breaks, consensus neighborhood radius.

## 6. Report (dashboard layout)

A single self-contained `.html` file. Layout (chosen direction = **inspector dashboard**):

- **Top bar:** segment id / scroll, and the **trace-health score badge** (color-coded).
- **Main viewer:** large image with **view-toggle tabs** — *Ink · Coherence heatmap · Orientation field · Flags* — switched in place via a few lines of vanilla JS (no server, no dependencies).
- **Side rail:** metrics breakdown (orientation breaks, pitch jumps, structure-loss count, % low-confidence) + legend.
- **Bottom:** **flagged-regions table** (location + failure mode), each row linking/scrolling to the region in the viewer.

All images embedded as base64. Opens by double-click; screenshots cleanly for Discord.

## 7. Deferred / future (explicitly out of MVP)

- **Through-stack geometry signal (#2):** use the full 65-layer stack — a correct trace keeps peak papyrus structure centered near layer 32; deviation/bimodality flags jumps even where there is no ink. Architecture leaves room to add this as a second "diagnostic channel."
- Multi-segment **batch index** dashboard.
- 3D rendering; any model **inference**.

## 8. Validation & testing (build test-first)

Ground truth via **synthetic perturbations** applied to a known-good ink prediction:

| Perturbation | Expected detection |
|---|---|
| Shear/rotate a band | flag an **orientation break** at the band |
| Vertical splice / shift a region | flag a **line-discontinuity** at the seam |
| Noise / garble a patch | flag **structure-loss** in the patch |
| Clean control (no perturbation) | stays at/near **unflagged** |

Tests assert each perturbation is **localized** within tolerance (detection overlaps the perturbed region) and that the **control keeps false positives low**. These synthetic tests are the primary correctness backbone; a real published segment is used for an end-to-end smoke test.

## 9. Definition of done (MVP)

- `plumbline run` works end-to-end on a real published segment on macOS (CPU only) and produces a valid self-contained dashboard `.html`.
- `--json` sidecar emitted with scores and flagged regions.
- Synthetic-perturbation test suite passes (detection + low control false positives).
- `README` includes an install + usage walkthrough and a committed example report (documentation is weighted heavily by progress prizes).

## 10. Open questions (resolve during planning)

- Exact trace-health score formula (how failure-mode counts + flagged-area fraction combine into 0–100). Start simple and interpretable; refine against synthetic + real examples.
- Where ink predictions live on the data server for the chosen smoke-test segment, and the precise `vesuvius`-lib call to fetch ink + `flat_mask` by id.
- Default tile size vs. typical text-line pitch at 7.91 µm/px (pick so a tile spans several text lines).
