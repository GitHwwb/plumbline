# Plumbline Batch Dashboard — Design

**Date:** 2026-06-01
**Status:** Design approved, pre-implementation
**Builds on:** the shipped Plumbline single-segment tool (`docs/superpowers/specs/2026-05-31-plumbline-design.md`).

---

## 1. Problem

Auto-segmentation produces *many* segments per scroll. Plumbline today reports on **one** segment at a time, so finding the bad ones across a whole scroll means running it repeatedly and opening reports by hand. The community currently tracks segment quality as a manual, subjective list — which doesn't scale.

**Goal:** a `plumbline batch` command that runs the existing pipeline over a folder of segments and produces a **single self-contained `index.html` dashboard** — a sortable, searchable, thumbnail table that surfaces the worst (lowest-trace-health) segments first, with each row linking to that segment's full report. This turns Plumbline from "check one segment" into "triage/search a whole scroll's segments."

## 2. Command & scope

```
plumbline batch <segments_dir> -o <out_dir> [--tile 256] [--overlap 0.5] [--no-reports] [--no-thumbnails]
```

- Produces `<out_dir>/index.html` (the dashboard) plus one `<out_dir>/<id>.html` per segment.
- `--no-reports`: skip per-segment reports (dashboard only) — for very large runs.
- `--no-thumbnails`: skip row thumbnails — for very large runs.

**Non-goals (YAGNI):** no pagination, no server, no database, no cross-scroll aggregation, no ink inference, no 3D. Static files only. Per-segment reports reuse the existing renderer unchanged.

## 3. Input model — folder of segment subfolders

Each immediate subfolder of `<segments_dir>` is one segment (folder name = segment id). Within a subfolder:

- **ink prediction:** first file matching, in order, `*prediction*`, `*result*`, `*inklabels*` (case-insensitive, `.png/.tif/.tiff`); if none match but exactly one image exists, use it.
- **mask:** first file matching `*flat_mask*` then `*mask*`; optional (if absent, the whole image is treated as valid, as in `run`).

Subfolders with no usable image are **skipped with a warning**, not errors.

## 4. Architecture

Two new focused modules + a thin CLI subcommand. The existing pipeline (`coherence.analyze_tiles` → `score.flag_tiles` → `score.trace_health`) and `report.write_report` are reused unchanged.

```
plumbline/
  discover.py     # find_segments(root) -> list[SegmentInputs]
  dashboard.py    # thumbnail_png(...), render_index(rows, meta) -> html
  model.py        # + SegmentInputs, IndexRow dataclasses
  cli.py          # + `batch` subcommand (_cmd_batch)
  templates/
    index.html.j2 # the dashboard template
```

**New data types (`model.py`):**

```python
@dataclass
class SegmentInputs:
    seg_id: str
    ink_path: str
    mask_path: str | None

@dataclass
class IndexRow:
    seg_id: str
    score: int
    n_orient: int
    n_pitch: int
    n_structure: int
    low_conf_frac: float
    report_filename: str | None   # "<id>.html", or None if --no-reports
    thumb_b64: str | None         # base64 PNG, or None if --no-thumbnails
    error: str | None             # set if the segment could not be evaluated
```

**`discover.find_segments(root) -> list[SegmentInputs]`** — directory scan + glob matching described in §3. Deterministic ordering (sorted by seg_id).

**`dashboard.thumbnail_png(ink01, features, flags, max_px=160) -> bytes`** — a small downscaled ink-overlay PNG (the existing overlay, rendered at low resolution / small figsize) for the row thumbnail.

**`dashboard.render_index(rows, meta) -> str`** — render `index.html.j2` with the rows (sorted worst-health first server-side as the default; client JS can re-sort).

**Orchestration (`cli._cmd_batch`)** — for each discovered segment: load ink+mask → `analyze_tiles` → `flag_tiles` → `trace_health`; unless `--no-reports`, `write_report(out/<id>.html, …)`; unless `--no-thumbnails`, `thumbnail_png(…)`; build an `IndexRow`. On any per-segment exception, build an `IndexRow` with `error` set (score 0) and continue. Finally `render_index` → `out/index.html`, and print `processed N, skipped M, errored K`.

**Data flow:**
`discover → [per segment: load → analyze → flag → score → (write_report) + (thumbnail) → IndexRow] → render_index → out/index.html`

## 5. Dashboard UI (`index.html.j2`)

Dark theme matching the single-segment report. Self-contained: thumbnails inlined as base64; the only external references are relative links to sibling `<id>.html` reports.

- **Header:** title + segment count + flagged count.
- **Search box:** filters rows by `seg_id` substring, client-side, instant.
- **Columns:** thumbnail · segment · **health** (color badge: ≥85 green, 60–84 amber, <60 red — same `_score_color` thresholds as reports) · orient · pitch · struct · low-conf%.
- **Sorting:** click a column header to sort (vanilla JS; numeric for score/counts/low-conf, string for id; toggles asc/desc). **Default: health ascending (worst first).**
- **Row click:** navigates to `report_filename` (if present). Error rows are visually marked ("not evaluated") and not clickable.
- `--no-thumbnails` omits the thumbnail column; `--no-reports` makes rows non-links.

## 6. Error handling

- A segment that fails to load or evaluate never aborts the run: it becomes an `IndexRow` with `error` set, score 0, shown as a marked row, and the batch continues.
- Subfolders with no usable image are skipped (counted as "skipped", with a warning), distinct from "errored".
- Final stdout line: `processed N, skipped M, errored K -> <out_dir>/index.html`.

## 7. Testing (TDD)

- **`discover`:** a temp dir with two segment subfolders (synthetic ink+mask PNGs) → returns both with correct ink/mask paths, sorted; a subfolder with no image is skipped.
- **`thumbnail_png`:** returns valid PNG bytes (magic header) at small dimensions.
- **`render_index`:** given sample `IndexRow`s → HTML contains the search box, every `seg_id`, base64 thumbnails, links to each `<id>.html`, sortable headers, and rows default to worst-health-first.
- **End-to-end (`cli.main(["batch", …])`):** a temp dir with synthetic segments (one clean `striped_field`, one `garble_patch`) → produces `out/index.html` + per-segment `<id>.html`; the garbled segment scores lower and appears first in the default order. `--no-reports` produces only `index.html`; `--no-thumbnails` omits thumbnails.

## 8. Definition of done

- `plumbline batch <dir> -o out/` works end-to-end on a folder of real downloaded segments (e.g., a few Frag/segment folders), producing a working dashboard whose rows open the right reports.
- Full test suite (existing + new) passes.
- README gains a short "Batch dashboard" section with usage and a note that it's the triage/search view over many segments.

## 9. Open questions (resolve in planning)

- Thumbnail rendering path: reuse `render.overlay_png` at a small figsize, or a dedicated lighter draw? Start by reusing the overlay at low dpi; optimize only if batch is slow.
- Whether per-segment report filenames need sanitizing (segment ids are numeric/safe today, but guard against odd folder names).
