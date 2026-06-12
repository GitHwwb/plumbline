# Plumbline

**Trace-quality reports for [Vesuvius Challenge](https://scrollprize.org) scroll segments.**

**▶ Live demo: https://githwwb.github.io/plumbline/** — worked examples: clean text, each failure mode in isolation, and a combined case that fires all four.

Segmenting a Herculaneum scroll means tracing one sheet of rolled papyrus through a
CT volume. A trace can silently **drift onto an adjacent sheet** ("sheet jump"),
producing a flattened segment that looks plausible but is geometrically wrong.

Plumbline reads a segment's **ink-prediction image** and flags where the trace
likely went wrong — using the ink as a signal for *trace correctness*, not content.
The insight: a correctly-traced sheet yields **coherent rows of text**; when the
trace drifts, the ink shows a fingerprint —

- an abrupt **orientation** change in the text rows,
- a **row-spacing** jump,
- a **collapse of row structure** (garble) where ink density says there should be text, or
- a **vertical seam** where rows step up/down at a sheet jump without rotating or re-spacing.

It produces a **self-contained HTML "trace-quality dashboard"** (score badge, a
view-toggle viewer, a metrics rail, and a flagged-regions table) plus an optional
JSON sidecar for batch use. **CPU-only. No model inference, no GPU, no server.**

This tool does **not** run ink inference,
do 3D rendering, or edit traces — it *diagnoses*.

---

## Install

Plumbline is a small Python (3.11+) package. **Run these from the repo root** (cd into the cloned plumbline/ folder first), then create a virtualenv and install editable:

```bash
python3 -m venv ~/.venvs/plumbline
~/.venvs/plumbline/bin/pip install -e ".[dev]"
```

For reading predictions stored as Zarr / OME-Zarr arrays:

```bash
~/.venvs/plumbline/bin/pip install -e ".[dev,zarr]"
~/.venvs/plumbline/bin/pip install -e ".[dev,zarr,remote]"
```

> **macOS / iCloud note:** do **not** put the virtualenv inside an iCloud-synced
> folder (e.g. a repo under `~/Documents` or `~/Desktop` with "Desktop & Documents"
> sync on). iCloud creates ` 2.pth` / ` 2.py` "conflict copies" that corrupt the
> editable install's import machinery, so `import plumbline` and the `plumbline`
> command start failing intermittently. Keep the venv outside iCloud (e.g.
> `~/.venvs/plumbline`, as above). The source tree itself can live anywhere.

Activate it (`source ~/.venvs/plumbline/bin/activate`) or call the binaries by path.

## Usage

```bash
plumbline run path/to/<id>_prediction.png \
    --mask path/to/<id>_flat_mask.png \
    -o report.html \
    --json report.json
```

- `ink` (positional): the ink-prediction PNG/TIF, in the segment's flattened frame.
- `--mask`: the segment's `flat_mask` (valid-area mask). Optional; defaults to the whole image.
- `-o/--output`: HTML report path (default `report.html`).
- `--json`: also write a JSON sidecar for batch pipelines — the score, every
  flagged region (center + full pixel box), and a `params` block recording the
  run configuration (tool version, tile size actually used, grid shape, global
  skew/pitch, overlap) so results are reproducible.
- `--tile` / `--overlap`: analysis tile size (px) and overlap fraction. **`--tile`
  defaults to auto** — Plumbline estimates the text scale and picks a tile that spans
  several rows; pass an explicit value only to override (a tile must span several text
  rows for the row metrics to register).

Open the resulting `report.html` in any browser (it embeds all images as base64 —
no external files), or read the `.json` for scripting.

### Zarr / OME-Zarr input

`plumbline run` accepts a Zarr store anywhere it accepts a PNG/TIF:

    plumbline run prediction.zarr -o report.html

- An **OME-Zarr multiscale** group uses the highest-resolution level by default;
  `--zarr-component 1` picks a different level (or a named array in a plain group).
- A **3-D** array (e.g. `(z, y, x)`) needs an explicit plane: `--zarr-z 12` for one
  slice, or `--zarr-reduce max` / `--zarr-reduce mean` to project the first axis.
  Note: projecting a volume is a convenience, **not** a true flattened render —
  Plumbline analyzes a 2-D ink raster and is happiest with an already-flattened
  prediction.
- Remote stores (`s3://…`, `https://…`) work with the `[remote]` extra installed.

## Batch dashboard (triage a whole scroll)

To rank many segments at once, point `batch` at a folder whose subfolders are
segments (each containing an ink prediction, and optionally a `*_mask` image):

    plumbline batch path/to/segments -o out/

This writes `out/index.html` — a self-contained, sortable, searchable table
(thumbnail · segment · trace-health · flag counts · low-confidence%), sorted
worst-first — plus one `out/<id>.html` report per segment. Click any row to open
its full report. For very large runs, `--no-reports` and `--no-thumbnails` keep
output small and fast, and `--jobs N` evaluates N segments in parallel (results
are identical to a sequential run — segments are independent).

Ink files are auto-detected by name (`*prediction*`, `*result*`, `*inklabels*`);
masks by `*flat_mask*`/`*mask*`. Zarr/OME-Zarr stores (`*.zarr` directories)
are matched by the same name patterns, so a folder of Zarr segments batches the
same way. Segments with no usable image are skipped; a segment that fails to
evaluate is marked "not evaluated" and never aborts the run.

## The four views (plus seam flags)

| View | What it shows |
|------|---------------|
| **Ink + flags** | the ink prediction with flagged tiles outlined in red |
| **Coherence** | per-tile "how row-like is the ink" heatmap (projection-profile band contrast), low-confidence tiles blanked; near-empty single-sliver tiles are dimmed so they don't read brighter than real text; an **overlay ink** toggle shows the text underneath. A rough visual aid — orientation, flags, and seams are the actionable signals |
| **Orientation** | per-tile dominant text-row axis (headless ticks — orientation is measured mod 180°, so reading direction is unknowable) |
| **Flags** | flagged tiles colored by failure mode (orientation / spacing / garble / **seam**) |

## How the score works

Plumbline first estimates the text scale and skew, then auto-sizes a grid of
overlapping tiles to span several rows. Per tile it works from the **projection
profile** (mean ink per row along the writing direction) and measures: dominant
**row orientation** (the angle maximizing row sharpness), **band strength** (how
strongly ink separates into rows vs. gaps), **row pitch**, and **ink density**. A
tile with too little ink/coverage is marked **low-confidence** and never flagged
(so faint areas don't false-alarm).

A tile is flagged when it breaks from a smooth, confidence-weighted local **consensus**:
orientation deviates sharply (drift/rotation) — but only on tiles whose row direction is
*reliably* determined, so a sparse letter-fragment with no real direction can't
manufacture a break — or row spacing jumps (only where a periodic peak is confident), or
band structure collapses where ink says text should exist (**garble**), or text rows step
vertically at a column **seam** without rotating or re-spacing (a sheet jump — see below).
The global **trace-health score** is `100 × (1 − flagged_confident_fraction)`,
rounded to 0–100. An all-low-confidence (unanalyzable) scan scores 0, not 100 — check
`low_conf_frac`.

The **seam** check is a separate one-pass scan: it slices the image into adjacent
full-height vertical strips (one row-pitch wide), reads each strip's vertical row offset
relative to its left neighbour by cross-correlation, and flags a column where that offset
jumps sharply *and* a vertical shift substantially improves the strips' alignment (so a
real step, not a wrap-lag coincidence on continuous text). It catches the pure vertical
sheet-jump that orientation/spacing/garble structurally cannot.

Because a seam flags only the few tiles straddling one column, it barely moves the
area-based score — a seamed segment can still score 95+. It is therefore called out
*categorically* instead: a **⚠ sheet-jump seam** banner on the report header, and a
**seam badge plus top-tier placement** in the batch dashboard's default ordering, so
a sheet jump can't hide behind a healthy-looking number.

## Interpreting scores — what a low score does (and doesn't) mean

Plumbline is a **heuristic triage signal, not a verdict.** It assumes ink appears
as **rows of text** (ink banded into lines separated by gaps), so a low score means
"these tiles don't look like coherent text rows — worth a human look," not
"this segment is definitely bad." Two ways to be misled:

- **Give it a real ink *prediction* — and confirm you can see letters.** Plumbline
  is calibrated for ink-detection probability maps. A bare **surface render**
  (papyrus fibers, no ink) can score *high* if the fibers happen to band like rows —
  because Plumbline measures row-banding, not "is this actually ink." It prints a
  warning when an input looks dense-but-structureless, but a fiber-aligned surface
  can slip past, so eyeball that letters are present. Rowness is measured on the
  ink *above* each tile's background level (the darkest quartile), so a photograph's
  gray papyrus glow — or a model's diffuse probability floor — no longer drowns
  legible rows; the flip side is that the dense-but-structureless warning is *less*
  likely to fire on photo-like mottle, so the eyeball check still matters.
- **Very large, sparse lettering is near the edge of regime.** The garble check
  needs several text rows per tile; tiles wholly inside one giant stroke or gap are
  genuinely featureless. Auto tile-sizing plus background-relative rowness handle
  the giant-letter fragments we have (the Fragment 1 infrared, once a heavy
  garble over-flagger, now scores 98/100) — but a few enormous letters remain a
  stress case, not the design center.
- **Input scaling matters.** Integer images are normalized by their full bit-depth
  range, so 8-bit data saved in a 16-bit container reads as nearly black (Plumbline
  warns at load when an image uses <1% of its range). Float images with values above
  1 are divided by their own per-image maximum, so scores from unusually-scaled
  float inputs aren't strictly comparable across files — feed predictions already
  in [0, 1] for apples-to-apples ranking.

Bottom line: trust Plumbline to *rank and surface* suspect regions for review;
don't treat the number as a final quality judgement.

**Detection modes:** Plumbline flags rotation/drift (orientation breaks), row-spacing jumps,
garble, and a **pure vertical sheet-jump** (`seam`: text rows stepping up/down at a column
without rotating or changing spacing). The seam detector's thresholds are calibrated on a
handful of real images; its clean-text false-positive margin is real but narrow, so treat a
lone seam flag as "look here", not proof. Tile size auto-adapts to text scale; pass
`--tile` to override.

**Rotated segments:** a full-range orientation reconnaissance runs on every image. If the
writing direction lies beyond the standard ±25° skew range *and* periodic text lines exist
at that angle, the **analyzer aligns itself to the text**: per-tile metrics, tile sizing
and the seam scan all run in the text's frame (the seam scan derotates an internal view
only — the input is never resampled, and all flags stay in original pixel coordinates).
A notice in the report and CLI states the detected angle. Rotated-input support is newer
than upright; eyeball the flags.

## Validation

Correctness is anchored by **synthetic text** — `glyph_rows` in
`plumbline/synthetic.py` makes rows of discrete glyph blocks (real text, *not*
stripes — stripes gave false confidence in an earlier design). Rotating a band must
raise an *orientation* flag, a noise patch must raise a *garble* flag, pure noise must
trigger the input warning, and clean text must stay (near) unflagged
(`tests/test_score.py`). It is also validated on **real IR text**
(`tests/test_real_ir.py`): genuine legible writing must score reasonably and not be
blanketed in flags — the regression that motivated the text-row redesign. Run the suite:

```bash
~/.venvs/plumbline/bin/pytest -q
```

## Example

Generate a synthetic segment with three known defects (skew + splice seam + garble)
and produce a report:

```bash
python examples/make_synthetic_example.py
# -> examples/synthetic_report.html, examples/synthetic_report.json
```

`examples/synthetic_report.json` is committed so you can see the kind of output
(score + per-mode flag counts + flagged-region coordinates) without running anything.

## Trying it on real fragment data (and an important caveat)

You can point Plumbline at downloaded fragment images, but heed the caveat above.
The Kaggle fragment volpkgs under `working/54keV_exposed_surface/` contain **surface
renders** (`result.png`) and **hand-drawn ink labels** (`inklabels.png`) — *not*
ink-model predictions. Running on them is instructive but is **not** a real
quality measurement:

- a fiber-coherent surface (e.g. Frag1) can score deceptively high — that reflects
  papyrus-fiber banding, not ink;
- a mottled surface (Frag2/Frag3) or the blobby `inklabels.png` scores low and
  triggers the "may not be an ink prediction" warning.

```bash
bash examples/fetch_frag1.sh        # downloads result.png (a SURFACE render) + mask.png
plumbline run data/frag1/result.png --mask data/frag1/mask.png \
    -o data/frag1/report.html --tile 1024
```

**Validated on a genuine ink-model prediction.** Community GitHub repos publish real
flattened ink imagery with no data-server credentials needed:
[`Bodillium/Herculaneum-Scroll-Labels`](https://github.com/Bodillium/Herculaneum-Scroll-Labels)
ships a Scroll 5 model **prediction** (`Images/predictions_*.png`) alongside hand-drawn
labels, and [`hendrikschilling/Vesuvius-Grandprize-Winner`](https://github.com/hendrikschilling/Vesuvius-Grandprize-Winner)
has the Grand-Prize banner ink **labels** (`all_labels/*.png`). Plumbline scores the
Scroll 5 prediction **99/100** (1 orientation break, seam 0; the model's diffuse
probability floor no longer reads as garble) and the real labels **85–100**
across two scrolls — the redesigned core holds up on real papyrus. Large flattened
segments (hundreds of megapixels) load fine. The `--segment-id` fetch path
(`plumbline/io.py:fetch_segment`) remains a stub for the data-server `vesuvius` API.

## Data access & attribution

Scroll/fragment data comes from the Vesuvius Challenge. Accept the data agreement at
<https://scrollprize.org/data> to obtain access credentials — **do not share those
credentials publicly** (per the agreement). The fetch script reads them from
`VESUVIUS_USER` / `VESUVIUS_PASS` in your environment rather than embedding them.
This repo redistributes no scroll imagery; downloaded data stays in the gitignored
`data/`, and only numeric metrics (`examples/scroll5_prediction_result.json` — the
score + flagged-region modes from a real Scroll-5 ink-model prediction) are committed.

Work using this data should cite the **EduceLab-Scrolls** dataset and:
Parsons et al. (2023), *EduceLab-Scrolls: Verifiable Recovery of Text from
Herculaneum Papyri using X-ray CT*. arXiv:2304.02084.

## Non-goals

No ink-model inference · no 3D rendering · no trace editing · no server. Plumbline
consumes existing ink predictions and reports on them.
