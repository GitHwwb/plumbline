# Plumbline Batch Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `plumbline batch <segments_dir> -o out/` command that runs the existing pipeline over a folder of segment subfolders and emits a self-contained, sortable/searchable `index.html` dashboard (one thumbnail + score + flag-count row per segment, worst-first, rows link to per-segment reports) plus the per-segment reports.

**Architecture:** Two new focused modules — `discover.py` (find segments + their ink/mask) and `dashboard.py` (thumbnail + index HTML) — plus a `batch` CLI subcommand that orchestrates them. Reuses the existing pipeline (`coherence.analyze_tiles` → `score.flag_tiles` → `score.trace_health`) and `report.write_report` unchanged. Static files only.

**Tech Stack:** Python 3.11+, numpy/scipy/scikit-image, Pillow, matplotlib (Agg), jinja2, argparse, pytest. **Important:** the virtualenv lives OUTSIDE the repo at `~/.venvs/plumbline` (the repo is in an iCloud-synced folder, which corrupts an in-repo venv). Run tests with `~/.venvs/plumbline/bin/pytest`.

---

## File Structure

```
plumbline/
  model.py        # MODIFY: add SegmentInputs, IndexRow dataclasses
  discover.py     # CREATE: find_segments(root) -> list[SegmentInputs]
  render.py       # MODIFY: overlay_png gains a figsize param (for thumbnails)
  dashboard.py    # CREATE: thumbnail_png(...), render_index(rows, meta) -> html
  cli.py          # MODIFY: add `batch` subcommand (_cmd_batch + parser)
  templates/
    index.html.j2 # CREATE: the dashboard template
tests/
  test_smoke.py       # MODIFY: append model dataclass test
  test_discover.py    # CREATE
  test_dashboard.py   # CREATE
  test_batch.py       # CREATE (end-to-end CLI)
README.md             # MODIFY: add "Batch dashboard" section
```

Work from `/Users/jonathanlopes/Documents/plumbline` on branch `main` (the prior feature already merged). Activate: `. ~/.venvs/plumbline/bin/activate` or call binaries by path. Current baseline: 33 tests passing.

---

## Task 1: Data model (SegmentInputs, IndexRow)

**Files:**
- Modify: `plumbline/model.py`
- Test: `tests/test_smoke.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_smoke.py`)

```python
def test_segment_inputs_and_index_row():
    from plumbline.model import SegmentInputs, IndexRow
    s = SegmentInputs(seg_id="abc", ink_path="/a/ink.png", mask_path=None)
    assert s.seg_id == "abc" and s.mask_path is None
    r = IndexRow(seg_id="abc", score=73, n_orient=0, n_pitch=2, n_structure=1,
                 low_conf_frac=0.3, report_filename="abc.html", thumb_b64="xx", error=None)
    assert r.score == 73 and r.report_filename == "abc.html" and r.error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_smoke.py::test_segment_inputs_and_index_row -v`
Expected: FAIL with `ImportError: cannot import name 'SegmentInputs'`

- [ ] **Step 3: Add the dataclasses to `plumbline/model.py`**

Change the import line at the top from `from typing import List` to:
```python
from typing import List, Optional
```
Then append at the end of the file:
```python
@dataclass
class SegmentInputs:
    seg_id: str
    ink_path: str
    mask_path: Optional[str]


@dataclass
class IndexRow:
    seg_id: str
    score: int
    n_orient: int
    n_pitch: int
    n_structure: int
    low_conf_frac: float
    report_filename: Optional[str]   # "<id>.html", or None when --no-reports
    thumb_b64: Optional[str]         # base64 PNG, or None when --no-thumbnails
    error: Optional[str] = None      # set when the segment could not be evaluated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plumbline/model.py tests/test_smoke.py
git commit -m "feat: add SegmentInputs and IndexRow dataclasses"
```

---

## Task 2: Segment discovery

**Files:**
- Create: `plumbline/discover.py`
- Test: `tests/test_discover.py`

- [ ] **Step 1: Write the failing test** in `tests/test_discover.py`

```python
import numpy as np
from PIL import Image
from plumbline.discover import find_segments


def _png(path, val=200, size=(32, 32)):
    Image.fromarray(np.full(size, val, dtype=np.uint8)).save(path)


def test_find_segments_picks_ink_and_mask(tmp_path):
    s1 = tmp_path / "seg1"; s1.mkdir()
    _png(s1 / "seg1_prediction.png"); _png(s1 / "seg1_mask.png", 255)
    s2 = tmp_path / "seg2"; s2.mkdir()
    _png(s2 / "result.png")                      # no mask
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "readme.txt").write_text("hi")      # no image -> skipped

    segs = find_segments(str(tmp_path))
    assert [s.seg_id for s in segs] == ["seg1", "seg2"]   # sorted, notes skipped
    assert segs[0].ink_path.endswith("seg1_prediction.png")
    assert segs[0].mask_path.endswith("seg1_mask.png")
    assert segs[1].ink_path.endswith("result.png")
    assert segs[1].mask_path is None


def test_find_segments_skips_folder_with_only_mask(tmp_path):
    s = tmp_path / "onlymask"; s.mkdir()
    _png(s / "x_mask.png", 255)
    assert find_segments(str(tmp_path)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_discover.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.discover'`

- [ ] **Step 3: Write `plumbline/discover.py`**

```python
import os
import glob
import fnmatch
from typing import List
from plumbline.model import SegmentInputs

_INK_PATTERNS = ["*prediction*", "*result*", "*inklabels*"]
_MASK_PATTERNS = ["*flat_mask*", "*mask*"]
_IMG_EXTS = (".png", ".tif", ".tiff")


def _images_in(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if os.path.isfile(f) and f.lower().endswith(_IMG_EXTS))


def _matches_any(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _first_match(images, patterns):
    for pat in patterns:
        for img in images:
            if fnmatch.fnmatch(os.path.basename(img).lower(), pat):
                return img
    return None


def find_segments(root) -> List[SegmentInputs]:
    """Each immediate subfolder of `root` is one segment (folder name = id).
    Pick an ink image (prediction/result/inklabels, else the lone non-mask
    image) and an optional mask. Subfolders with no usable ink image are
    skipped."""
    segments: List[SegmentInputs] = []
    for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        images = _images_in(d)
        if not images:
            continue
        masks = [m for m in images if _matches_any(os.path.basename(m).lower(), _MASK_PATTERNS)]
        candidates = [i for i in images if i not in masks]
        ink = _first_match(candidates, _INK_PATTERNS)
        if ink is None:
            if len(candidates) == 1:
                ink = candidates[0]
            else:
                continue  # no ink-like name and 0 or >1 plain images -> skip
        mask = masks[0] if masks else None
        segments.append(SegmentInputs(seg_id=os.path.basename(d), ink_path=ink, mask_path=mask))
    return segments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_discover.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plumbline/discover.py tests/test_discover.py
git commit -m "feat: add segment discovery for batch mode"
```

---

## Task 3: Thumbnail rendering (reuse overlay at small size)

**Files:**
- Modify: `plumbline/render.py` (add `figsize` param to `overlay_png`)
- Create: `plumbline/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test** in `tests/test_dashboard.py`

```python
import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles
from plumbline.dashboard import thumbnail_png


def _bundle():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    return f, feats, flag_tiles(feats)


def test_thumbnail_png_is_small_png():
    f, feats, flags = _bundle()
    png = thumbnail_png(f, feats, flags)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) < 200_000      # a thumbnail, not a full render
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plumbline.dashboard'`

- [ ] **Step 3: Add a `figsize` param to `overlay_png` in `plumbline/render.py`**

Find this function:
```python
def overlay_png(ink01, features, flags) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ink01, cmap="gray", origin="upper")
    for (x, y, w, h) in _flag_extent(features, flags):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    ax.set_axis_off()
    return _fig_to_png(fig)
```
Change only the signature and the `subplots` call:
```python
def overlay_png(ink01, features, flags, figsize=(6, 6)) -> bytes:
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(ink01, cmap="gray", origin="upper")
    for (x, y, w, h) in _flag_extent(features, flags):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    ax.set_axis_off()
    return _fig_to_png(fig)
```
(Existing callers pass no `figsize`, so `report.py` is unaffected.)

- [ ] **Step 4: Write `plumbline/dashboard.py`** (thumbnail only for now)

```python
import base64
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plumbline.render import overlay_png
from plumbline.report import _score_color

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def thumbnail_png(ink01, features, flags) -> bytes:
    """A small ink-overlay PNG (flagged tiles outlined) for a dashboard row.
    Reuses the report overlay renderer at a small figure size."""
    return overlay_png(ink01, features, flags, figsize=(1.6, 1.6))


def thumbnail_b64(ink01, features, flags) -> str:
    return base64.b64encode(thumbnail_png(ink01, features, flags)).decode("ascii")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_dashboard.py -v`
Expected: PASS. Then `~/.venvs/plumbline/bin/pytest tests/test_render.py -q` to confirm the `overlay_png` change didn't break existing renders (Expected: PASS).

- [ ] **Step 6: Commit**

```bash
git add plumbline/render.py plumbline/dashboard.py tests/test_dashboard.py
git commit -m "feat: add dashboard thumbnail rendering (overlay at small size)"
```

---

## Task 4: Index HTML rendering

**Files:**
- Modify: `plumbline/dashboard.py` (add `render_index`)
- Create: `plumbline/templates/index.html.j2`
- Test: `tests/test_dashboard.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_dashboard.py`)

```python
from plumbline.dashboard import render_index
from plumbline.model import IndexRow


def test_render_index_search_links_thumbs_worst_first():
    rows = [
        IndexRow("good", 95, 0, 0, 0, 0.05, "good.html", "AAAA", None),
        IndexRow("bad", 30, 3, 9, 5, 0.40, "bad.html", "BBBB", None),
    ]
    html = render_index(rows, {"scroll": "Scroll1"})
    assert 'id="q"' in html                                   # search box
    assert "good.html" in html and "bad.html" in html         # row links
    assert "data:image/png;base64,AAAA" in html               # thumbnail inlined
    assert "<html" in html.lower()
    # worst-first default: the low-score row comes first in source order
    assert html.index("bad.html") < html.index("good.html")


def test_render_index_marks_error_rows():
    rows = [IndexRow("oops", 0, 0, 0, 0, 1.0, None, None, "boom")]
    html = render_index(rows, {"scroll": "S"})
    assert "not evaluated" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_dashboard.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_index'`

- [ ] **Step 3: Add `render_index` to `plumbline/dashboard.py`**

```python
def render_index(rows, meta) -> str:
    """Render the sortable/searchable dashboard. Rows are sorted worst-health
    first by default (client JS can re-sort)."""
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR),
                      autoescape=select_autoescape(["html"]))
    tmpl = env.get_template("index.html.j2")
    ordered = sorted(rows, key=lambda r: (r.score, r.seg_id))
    n_flagged = sum(1 for r in ordered
                    if (r.n_orient + r.n_pitch + r.n_structure) > 0)
    return tmpl.render(rows=ordered, meta=meta, n_total=len(ordered),
                       n_flagged=n_flagged, score_color=_score_color)
```

- [ ] **Step 4: Write `plumbline/templates/index.html.j2`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plumbline — {{ meta.scroll }}</title>
<style>
  body { margin:0; background:#0a1521; color:#cfe0ee; font:14px/1.5 ui-monospace,Menlo,monospace; }
  header { display:flex; justify-content:space-between; align-items:center;
           padding:12px 18px; border-bottom:1px solid #22354a; }
  .title { font:700 18px Georgia,serif; }
  .sub { color:#5a7a98; font-size:12px; }
  .toolbar { padding:10px 18px; border-bottom:1px solid #22354a; }
  #q { width:50%; padding:6px 10px; background:#0a1521; color:#cfe0ee;
       border:1px solid #2a3f55; border-radius:4px; font:inherit; }
  table { width:100%; border-collapse:collapse; }
  th,td { text-align:left; padding:7px 12px; font-size:13px; }
  thead th { color:#7fb0e0; font-size:11px; letter-spacing:.5px; text-transform:uppercase;
             background:#13243a; cursor:pointer; user-select:none; }
  tbody tr { border-top:1px solid #16263a; }
  tr.clickable { cursor:pointer; }
  tr.clickable:hover { background:#13243a; }
  .thumb { width:48px; height:30px; object-fit:cover; border:1px solid #22354a; border-radius:3px; display:block; }
  .badge { color:#0a1521; padding:1px 7px; border-radius:3px; font-weight:700; }
  .err { color:#ff8c8c; font-size:11px; }
  .muted { color:#5a7a98; }
</style>
</head>
<body>
<header>
  <span class="title">Plumbline · {{ meta.scroll }}</span>
  <span class="sub">{{ n_total }} segments · {{ n_flagged }} flagged</span>
</header>
<div class="toolbar"><input id="q" placeholder="search segments…" oninput="filt()"></div>
<table id="t">
<thead><tr>
  <th>view</th>
  <th onclick="sortBy(1,'s')">segment</th>
  <th onclick="sortBy(2,'n')">health</th>
  <th onclick="sortBy(3,'n')">orient</th>
  <th onclick="sortBy(4,'n')">pitch</th>
  <th onclick="sortBy(5,'n')">struct</th>
  <th onclick="sortBy(6,'n')">low-conf</th>
</tr></thead>
<tbody>
{% for r in rows %}
<tr data-id="{{ r.seg_id }}"{% if r.report_filename and not r.error %} class="clickable" onclick="location.href='{{ r.report_filename }}'"{% endif %}>
  <td>{% if r.thumb_b64 %}<img class="thumb" src="data:image/png;base64,{{ r.thumb_b64 }}">{% endif %}</td>
  <td>{{ r.seg_id }}{% if r.error %} <span class="err">not evaluated</span>{% endif %}</td>
  <td data-v="{{ r.score }}"><span class="badge" style="background:{{ score_color(r.score) }}">{{ r.score }}</span></td>
  <td data-v="{{ r.n_orient }}" class="{{ '' if r.n_orient else 'muted' }}">{{ r.n_orient }}</td>
  <td data-v="{{ r.n_pitch }}" class="{{ '' if r.n_pitch else 'muted' }}">{{ r.n_pitch }}</td>
  <td data-v="{{ r.n_structure }}" class="{{ '' if r.n_structure else 'muted' }}">{{ r.n_structure }}</td>
  <td data-v="{{ r.low_conf_frac }}" class="muted">{{ '%.0f' % (r.low_conf_frac * 100) }}%</td>
</tr>
{% endfor %}
</tbody>
</table>
<script>
function filt(){
  var q = document.getElementById('q').value.toLowerCase();
  document.querySelectorAll('#t tbody tr').forEach(function(tr){
    tr.style.display = tr.getAttribute('data-id').toLowerCase().indexOf(q) >= 0 ? '' : 'none';
  });
}
var asc = {};
function sortBy(col, type){
  var tb = document.querySelector('#t tbody');
  var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'));
  asc[col] = !asc[col];
  var dir = asc[col] ? 1 : -1;
  rows.sort(function(a, b){
    var av, bv;
    if (type === 'n'){
      av = parseFloat(a.children[col].getAttribute('data-v'));
      bv = parseFloat(b.children[col].getAttribute('data-v'));
    } else {
      av = a.children[col].textContent.trim();
      bv = b.children[col].textContent.trim();
    }
    return av < bv ? -dir : av > bv ? dir : 0;
  });
  rows.forEach(function(r){ tb.appendChild(r); });
}
</script>
</body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/.venvs/plumbline/bin/pytest tests/test_dashboard.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add plumbline/dashboard.py plumbline/templates/index.html.j2 tests/test_dashboard.py
git commit -m "feat: add sortable/searchable index dashboard rendering"
```

---

## Task 5: `batch` CLI subcommand

**Files:**
- Modify: `plumbline/cli.py`
- Test: `tests/test_batch.py`

- [ ] **Step 1: Write the failing test** in `tests/test_batch.py`

```python
import numpy as np
from PIL import Image
from plumbline.synthetic import striped_field, garble_patch
from plumbline.cli import main


def _save(path, field):
    Image.fromarray((field * 255).astype("uint8")).save(path)


def test_batch_builds_dashboard_worst_first(tmp_path):
    segs = tmp_path / "segs"; segs.mkdir()
    clean = segs / "clean"; clean.mkdir()
    _save(clean / "prediction.png", striped_field((512, 512), pitch=24, angle=0.0))
    bad = segs / "bad"; bad.mkdir()
    _save(bad / "prediction.png",
          garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 384, 128, 384))
    out = tmp_path / "out"

    rc = main(["batch", str(segs), "-o", str(out), "--tile", "128"])
    assert rc == 0
    assert (out / "index.html").exists()
    assert (out / "clean.html").exists() and (out / "bad.html").exists()
    html = (out / "index.html").read_text()
    assert html.index("bad.html") < html.index("clean.html")   # worst-first


def test_batch_no_reports_no_thumbnails(tmp_path):
    segs = tmp_path / "segs"; segs.mkdir()
    s = segs / "s1"; s.mkdir()
    _save(s / "prediction.png", striped_field((256, 256), pitch=24, angle=0.0))
    out = tmp_path / "out"

    rc = main(["batch", str(segs), "-o", str(out), "--tile", "128",
               "--no-reports", "--no-thumbnails"])
    assert rc == 0
    assert (out / "index.html").exists()
    assert not (out / "s1.html").exists()
    assert "data:image/png;base64," not in (out / "index.html").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_batch.py -v`
Expected: FAIL with `SystemExit` / argparse error "invalid choice: 'batch'" (the subcommand doesn't exist yet).

- [ ] **Step 3: Add the `batch` command to `plumbline/cli.py`**

Add these imports at the top (next to the existing imports):
```python
import base64
import glob
from plumbline import discover
from plumbline.dashboard import render_index, thumbnail_b64
from plumbline.model import IndexRow
```
Add this function (above `build_parser`):
```python
def _cmd_batch(args) -> int:
    segs = discover.find_segments(args.segments_dir)
    n_subdirs = sum(1 for p in glob.glob(os.path.join(args.segments_dir, "*"))
                    if os.path.isdir(p))
    skipped = n_subdirs - len(segs)
    os.makedirs(args.output, exist_ok=True)

    rows = []
    n_err = 0
    for s in segs:
        try:
            ink = pio.load_image01(s.ink_path)
            mask = pio.load_mask(s.mask_path) if s.mask_path else np.ones(ink.shape, dtype=bool)
            feats = analyze_tiles(ink, mask, tile=args.tile, overlap=args.overlap)
            flags = flag_tiles(feats)
            rep = trace_health(feats, flags)
            report_filename = None
            if not args.no_reports:
                report_filename = f"{s.seg_id}.html"
                write_report(os.path.join(args.output, report_filename),
                             {"segment_id": s.seg_id, "scroll": args.scroll},
                             ink, feats, flags, rep)
            thumb = None if args.no_thumbnails else thumbnail_b64(ink, feats, flags)
            rows.append(IndexRow(s.seg_id, rep.score, rep.n_orient, rep.n_pitch,
                                 rep.n_structure, rep.low_conf_frac,
                                 report_filename, thumb, None))
        except Exception as e:  # one bad segment must not abort the whole batch
            n_err += 1
            rows.append(IndexRow(s.seg_id, 0, 0, 0, 0, 1.0, None, None, str(e)))

    index_path = os.path.join(args.output, "index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(render_index(rows, {"scroll": args.scroll}))
    print(f"processed {len(rows) - n_err}, skipped {skipped}, errored {n_err} "
          f"-> {index_path}")
    return 0
```
Register the subcommand inside `build_parser`, right after the `run.set_defaults(func=_cmd_run)` line:
```python
    batch = sub.add_parser("batch", help="analyze a folder of segments into a dashboard")
    batch.add_argument("segments_dir", help="folder whose subfolders are segments")
    batch.add_argument("-o", "--output", default="plumbline_out",
                       help="output dir for index.html + per-segment reports")
    batch.add_argument("--scroll", default="Scroll1")
    batch.add_argument("--tile", type=int, default=256)
    batch.add_argument("--overlap", type=float, default=0.5)
    batch.add_argument("--no-reports", action="store_true",
                       help="skip per-segment reports (dashboard only)")
    batch.add_argument("--no-thumbnails", action="store_true",
                       help="skip row thumbnails")
    batch.set_defaults(func=_cmd_batch)
```
Note: `os`, `np`, `pio`, `analyze_tiles`, `flag_tiles`, `trace_health`, `write_report` are already imported at the top of `cli.py` from the `run` command — do not re-import them. The existing `main()` already dispatches via `args.func(args)`; the `run`-specific guard in `main()` stays unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_batch.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Verify the CLI + full suite**

Run:
```bash
~/.venvs/plumbline/bin/plumbline batch --help
~/.venvs/plumbline/bin/pytest -q
```
Expected: `--help` prints the batch usage; whole suite PASSES.

- [ ] **Step 6: Commit**

```bash
git add plumbline/cli.py tests/test_batch.py
git commit -m "feat: add plumbline batch dashboard command"
```

---

## Task 6: README + real-folder check

**Files:**
- Modify: `README.md`
- (No new code; integration + docs.)

- [ ] **Step 1: Build a real multi-segment dashboard**

Make a folder with at least two real segments and run batch. Reuse the Frag1 data already in `data/frag1/` (gitignored): create a segment subfolder layout and run.
```bash
mkdir -p data/scroll1demo/frag1
cp data/frag1/result.png data/scroll1demo/frag1/prediction.png
cp data/frag1/mask.png   data/scroll1demo/frag1/mask.png
# (add a second segment folder if you have one; one is enough to smoke-test)
~/.venvs/plumbline/bin/plumbline batch data/scroll1demo -o data/scroll1demo_out --tile 1024
```
Expected: prints `processed 1, skipped 0, errored 0 -> data/scroll1demo_out/index.html`. Open `data/scroll1demo_out/index.html` in a browser; confirm the row shows a thumbnail + score and clicking it opens `frag1.html`.

- [ ] **Step 2: Add a "Batch dashboard" section to `README.md`**

Insert this section immediately AFTER the existing "## Usage" section:
```markdown
## Batch dashboard (triage a whole scroll)

To rank many segments at once, point `batch` at a folder whose subfolders are
segments (each containing an ink prediction, and optionally a `*_mask` image):

    plumbline batch path/to/segments -o out/ --tile 256

This writes `out/index.html` — a self-contained, sortable, searchable table
(thumbnail · segment · trace-health · flag counts · low-confidence%), sorted
worst-first — plus one `out/<id>.html` report per segment. Click any row to open
its full report. For very large runs, `--no-reports` and `--no-thumbnails` keep
output small and fast.

Ink files are auto-detected by name (`*prediction*`, `*result*`, `*inklabels*`);
masks by `*flat_mask*`/`*mask*`. Segments with no usable image are skipped; a
segment that fails to evaluate is marked "not evaluated" and never aborts the run.
```

- [ ] **Step 3: Confirm the suite is green and commit**

Run: `~/.venvs/plumbline/bin/pytest -q`
Expected: all PASS.
```bash
git add README.md
git commit -m "docs: document the batch dashboard command"
```
Note: `data/` is gitignored, so the demo outputs are not committed.

---

## Self-Review notes

- **Spec coverage:** command + flags (Task 5), folder-of-subfolders discovery with ink/mask globs (Task 2), `SegmentInputs`/`IndexRow` (Task 1), thumbnails reusing the overlay (Task 3), sortable/searchable worst-first index with row links + error rows + score-color thresholds reused from reports (Task 4), per-segment report reuse + `--no-reports`/`--no-thumbnails` + error handling + summary line (Task 5), README + real-folder run (Task 6). Synthetic clean-vs-garbled end-to-end is Task 5.
- **Type consistency:** `find_segments -> list[SegmentInputs(seg_id, ink_path, mask_path)]`; `IndexRow(seg_id, score, n_orient, n_pitch, n_structure, low_conf_frac, report_filename, thumb_b64, error)` used identically in `dashboard.render_index`, the template, and `cli._cmd_batch`. `thumbnail_b64` (used by cli) and `thumbnail_png` (tested) both live in `dashboard.py`. `overlay_png` gains an optional `figsize` with the same default, so existing callers are unaffected.
- **Reuse/DRY:** `_score_color` imported from `report.py`; `overlay_png` reused for thumbnails; pipeline + `write_report` reused unchanged.
- **Known minor:** sort directions are tracked per-column in `asc{}`; clicking "health" once sorts ascending again (already worst-first), which is harmless. Not worth extra code.
