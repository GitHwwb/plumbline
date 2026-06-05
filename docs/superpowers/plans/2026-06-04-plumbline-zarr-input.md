# Plumbline Zarr/OME-Zarr Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Handover note (other machine):** This is a *separate* task from `docs/HANDOVER_NEXT.md`
> (the spacing-harmonic + e_all_modes fixes). It is a **pure I/O addition** — a new way for the
> 2-D ink prediction to *enter* the tool. Same analysis, new intake pipe. When done, hand the code
> back; the origin machine does all GitHub deployment.

**Goal:** Let Plumbline read a 2-D ink prediction directly from a Zarr / OME-Zarr store (local path or remote URL), in addition to PNG/TIF, so it satisfies the Monthly-Progress-Prize "accept standard community formats" criterion and slots into the data pipeline without a manual PNG export.

**Architecture:** Add `load_zarr01()` and a path-dispatching `load_input01()` to `plumbline/io.py`; the dispatcher routes Zarr paths to the new loader and everything else to the existing `load_image01()`. The Zarr loader resolves a store down to one 2-D array (OME multiscale level → highest-res by default; 3-D volume → explicit z-slice or projection) and feeds it through the existing `util.to01()` normalizer. The CLI `run` command routes its inputs through `load_input01()` and gains three optional `--zarr-*` flags. **No change to the analysis core** (`coherence.py`, `score.py`, `band_contrast`, the seam detector) — they receive the same `(float64 image in [0,1], bool mask)` they get today.

**Tech Stack:** Python, NumPy, the `zarr` library (v2, added as an **optional** extra `plumbline[zarr]`), existing `plumbline.util.to01`. Remote (`s3://`/`https://`) stores work when `fsspec`/`s3fs` are also installed (`plumbline[remote]`).

---

## Design decisions (resolved — do not re-litigate)

1. **Detection:** a path is Zarr if it ends in `.zarr`, OR is a remote URL (`s3://`/`gs://`/`http(s)://`) ending in `.zarr`, OR is a local directory containing `.zarray` / `.zgroup` / `zarr.json`. Everything else → existing image loader.
2. **Group vs array:** a bare Zarr **array** is used directly. A Zarr **group** with OME-NGFF `multiscales` metadata → open the highest-resolution level (`datasets[0].path`, usually `"0"`). `component="<name>"` overrides (e.g. `"1"` for a lower-res level, or a named array in a plain group). A group with neither `multiscales` nor a `component` → `ValueError` (don't guess).
3. **Dimensionality:** 2-D array → use as-is. >2-D (e.g. a `(z, y, x)` volume) → the caller MUST pick a plane: `z=<index>` for one slice, or `reduce="max"|"mean"` to project the first axis. Neither given → `ValueError`. Document that projecting a volume is a *convenience, not a true flattened render*.
4. **Normalization:** the extracted 2-D array goes through the existing `to01()` unchanged — float [0,1] passes through, ints scale by dtype range. **Do not modify `to01`.**
5. **Dependency hygiene:** `zarr` is an **optional** extra. `load_zarr01` raises a clear "install `plumbline[zarr]`" message if `import zarr` fails (mirror the existing `fetch_segment` pattern in `io.py`). Pin `zarr>=2.16,<3` (the v3 API differs; v3 is out of scope here).
6. **Scope:** `run` only. `batch` Zarr discovery is out of scope (its `discover.find_segments` expects image files). No mesh support (Plumbline analyzes the 2-D raster, not 3-D geometry).

## File structure

- **Modify `plumbline/io.py`** — add `import os`; add `_is_zarr(path)`, `load_zarr01(path, component, z, reduce)`, and `load_input01(path, component, z, reduce)`. Leave `load_image01`, `load_mask`, `fetch_segment` untouched.
- **Modify `plumbline/cli.py`** — route `_load_inputs` through `pio.load_input01`; add `--zarr-component`, `--zarr-z`, `--zarr-reduce` to the `run` subparser.
- **Modify `pyproject.toml`** — add optional extras `zarr` and `remote`.
- **Create `tests/test_zarr_io.py`** — synthetic tiny-array tests (`pytest.importorskip("zarr")`; no scroll data).
- **Modify `README.md`** — document Zarr input, the optional extra, and the 3-D caveat.

Environment: tests run with `~/.venvs/plumbline/bin/pytest -q` (venv is **outside** the iCloud-synced repo). First: `~/.venvs/plumbline/bin/pip install 'zarr>=2.16,<3'`.

---

### Task 1: Add the optional `zarr` dependency + import guard

**Files:**
- Modify: `pyproject.toml:16-18` (the `[project.optional-dependencies]` block)
- Modify: `plumbline/io.py` (top: add `import os`)

- [ ] **Step 1: Add the extras to `pyproject.toml`**

Replace the existing optional-dependencies block:

```toml
[project.optional-dependencies]
fetch = ["vesuvius"]
zarr = ["zarr>=2.16,<3"]
remote = ["fsspec", "s3fs"]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Install the extra into the dev venv**

Run: `~/.venvs/plumbline/bin/pip install 'zarr>=2.16,<3'`
Expected: installs zarr + numcodecs; `~/.venvs/plumbline/bin/python -c "import zarr; print(zarr.__version__)"` prints a 2.x version.

- [ ] **Step 3: Add `import os` to `plumbline/io.py`**

At the top of `plumbline/io.py`, change the imports to:

```python
import os
import numpy as np
from plumbline.util import to01
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml plumbline/io.py
git commit -m "build: add optional zarr/remote extras; import os in io"
```

---

### Task 2: `load_zarr01` — read a 2-D array store

**Files:**
- Modify: `plumbline/io.py` (add `load_zarr01`)
- Test: `tests/test_zarr_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_io.py`:

```python
import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
from plumbline.io import load_zarr01


def _write_array(path, data):
    z = zarr.open(str(path), mode="w", shape=data.shape,
                  chunks=True, dtype=data.dtype)
    z[:] = data
    return z


def test_load_zarr01_2d_float_passthrough(tmp_path):
    data = np.linspace(0.0, 1.0, 16 * 20, dtype="float32").reshape(16, 20)
    _write_array(tmp_path / "pred.zarr", data)
    out = load_zarr01(tmp_path / "pred.zarr")
    assert out.shape == (16, 20)
    assert out.dtype == np.float64
    assert np.allclose(out, data, atol=1e-6)   # already in [0,1] -> passthrough


def test_load_zarr01_2d_uint8_scaled(tmp_path):
    data = np.array([[0, 128, 255]], dtype="uint8")
    _write_array(tmp_path / "u8.zarr", data)
    out = load_zarr01(tmp_path / "u8.zarr")
    assert np.allclose(out, [[0.0, 128 / 255, 1.0]], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_zarr01'`.

- [ ] **Step 3: Implement `load_zarr01`**

Add to `plumbline/io.py` (after `load_mask`):

```python
def load_zarr01(path, component=None, z=None, reduce=None) -> np.ndarray:
    """Load a 2-D ink prediction from a (local or remote) Zarr / OME-Zarr store.

    - A bare array store is used directly.
    - An OME-Zarr multiscale group resolves to its highest-resolution level
      (datasets[0]); pass `component` (e.g. "1") to pick another level or a
      named array in a plain group.
    - A >2-D array (e.g. a (z, y, x) volume) needs an explicit plane: pass
      `z=<index>` for one slice, or `reduce="max"|"mean"` to project the first
      axis. (Projecting a volume is a convenience, NOT a true flattened render.)

    Returns a grayscale float64 image in [0, 1] (via to01).
    """
    try:
        import zarr
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Zarr input needs the optional extra: pip install 'plumbline[zarr]'"
        ) from e

    obj = zarr.open(str(path), mode="r")

    # A group (no .shape) is an OME-Zarr multiscale pyramid or a container of
    # arrays; a bare array has .shape and is used directly.
    if getattr(obj, "shape", None) is None:
        if component is not None:
            obj = obj[component]
        else:
            multiscales = obj.attrs.get("multiscales")
            if multiscales:
                obj = obj[multiscales[0]["datasets"][0]["path"]]
            else:
                raise ValueError(
                    f"{path} is a Zarr group with no 'multiscales' metadata; "
                    "pass component=<array name> to choose an array"
                )

    arr = np.asarray(obj[:])
    if arr.ndim > 2:
        if z is not None:
            arr = arr[z]
        elif reduce == "max":
            arr = arr.max(axis=0)
        elif reduce == "mean":
            arr = arr.mean(axis=0)
        else:
            raise ValueError(
                f"{path} is {arr.ndim}-D {arr.shape}; pass z=<index> for one "
                "plane or reduce='max'|'mean' to project the first axis"
            )
    if arr.ndim > 2:  # e.g. a 4-D store; one selection still leaves it >2-D
        raise ValueError(f"selected data is still {arr.ndim}-D: {arr.shape}")
    return to01(arr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add plumbline/io.py tests/test_zarr_io.py
git commit -m "feat: load_zarr01 reads a 2-D prediction from a Zarr store"
```

---

### Task 3: `load_input01` dispatcher + `_is_zarr` detection

**Files:**
- Modify: `plumbline/io.py` (add `_is_zarr`, `load_input01`)
- Test: `tests/test_zarr_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_io.py`:

```python
from plumbline.io import load_input01, _is_zarr


def test_is_zarr_detection(tmp_path):
    _write_array(tmp_path / "x.zarr", np.zeros((8, 8), dtype="float32"))
    _write_array(tmp_path / "noext", np.zeros((8, 8), dtype="float32"))
    assert _is_zarr(tmp_path / "x.zarr") is True       # by suffix
    assert _is_zarr(tmp_path / "noext") is True         # by .zarray inside dir
    assert _is_zarr("s3://bucket/pred.zarr") is True     # remote suffix
    assert _is_zarr("prediction.png") is False
    assert _is_zarr("prediction.tif") is False


def test_load_input01_dispatch(tmp_path):
    data = np.full((8, 8), 0.5, dtype="float32")
    _write_array(tmp_path / "x.zarr", data)
    out = load_input01(tmp_path / "x.zarr")
    assert out.shape == (8, 8)
    assert np.allclose(out, 0.5, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py::test_is_zarr_detection -v`
Expected: FAIL with `ImportError: cannot import name '_is_zarr'`.

- [ ] **Step 3: Implement `_is_zarr` and `load_input01`**

Add to `plumbline/io.py` (after `load_zarr01`):

```python
def _is_zarr(path) -> bool:
    """True if `path` looks like a Zarr/OME-Zarr store (local dir or remote URL)."""
    s = str(path).rstrip("/")
    if s.endswith(".zarr"):
        return True
    if s.startswith(("s3://", "gs://", "http://", "https://")):
        return s.endswith(".zarr")
    if os.path.isdir(s):
        return any(os.path.exists(os.path.join(s, m))
                   for m in (".zarray", ".zgroup", "zarr.json"))
    return False


def load_input01(path, component=None, z=None, reduce=None) -> np.ndarray:
    """Load a 2-D image in [0, 1] from either an image file (PNG/TIF) or a
    Zarr/OME-Zarr store, dispatching on the path. Zarr-only options
    (component/z/reduce) are ignored for image files."""
    if _is_zarr(path):
        return load_zarr01(path, component=component, z=z, reduce=reduce)
    return load_image01(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py -v`
Expected: PASS (all zarr tests).

- [ ] **Step 5: Commit**

```bash
git add plumbline/io.py tests/test_zarr_io.py
git commit -m "feat: load_input01 dispatches Zarr vs image by path"
```

---

### Task 4: 3-D volume handling (z-slice + projection)

**Files:**
- Test: `tests/test_zarr_io.py` (behavior already implemented in Task 2 — this task is the regression test that locks it in)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_io.py`:

```python
def test_load_zarr01_3d_requires_explicit_plane(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.random((4, 8, 8), dtype=np.float32)
    _write_array(tmp_path / "vol.zarr", vol)
    with pytest.raises(ValueError):
        load_zarr01(tmp_path / "vol.zarr")                       # ambiguous -> error
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", z=2), vol[2], atol=1e-6)
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", reduce="max"),
                       vol.max(axis=0), atol=1e-6)
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", reduce="mean"),
                       vol.mean(axis=0), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 2)**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py::test_load_zarr01_3d_requires_explicit_plane -v`
Expected: PASS. (If it FAILS, the Task-2 `load_zarr01` ndim logic is wrong — fix it there, do not special-case here.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_io.py
git commit -m "test: lock in 3-D Zarr plane selection + projection"
```

---

### Task 5: OME-Zarr multiscale group resolution

**Files:**
- Test: `tests/test_zarr_io.py` (behavior implemented in Task 2 — regression test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_io.py`:

```python
def test_load_zarr01_ome_multiscale_default_and_override(tmp_path):
    rng = np.random.default_rng(1)
    full = rng.random((16, 16), dtype=np.float32)
    half = np.ascontiguousarray(full[::2, ::2])
    g = zarr.open_group(str(tmp_path / "img.zarr"), mode="w")
    g.create_dataset("0", data=full)
    g.create_dataset("1", data=half)
    g.attrs["multiscales"] = [
        {"version": "0.4", "datasets": [{"path": "0"}, {"path": "1"}]}
    ]
    assert load_zarr01(tmp_path / "img.zarr").shape == (16, 16)             # default -> level 0
    assert load_zarr01(tmp_path / "img.zarr", component="1").shape == (8, 8)  # override


def test_load_zarr01_group_without_multiscales_needs_component(tmp_path):
    g = zarr.open_group(str(tmp_path / "bare.zarr"), mode="w")
    g.create_dataset("data", data=np.zeros((4, 4), dtype="float32"))
    with pytest.raises(ValueError):
        load_zarr01(tmp_path / "bare.zarr")
    assert load_zarr01(tmp_path / "bare.zarr", component="data").shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it passes (already implemented in Task 2)**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py -k multiscale -v`
Expected: PASS. (If `getattr(obj, "shape", None) is None` doesn't detect the group on your zarr version, adjust the group-detection line in `load_zarr01` and re-run — do not add a parallel code path.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_io.py
git commit -m "test: lock in OME-Zarr multiscale level resolution"
```

---

### Task 6: CLI — route `run` through `load_input01` + add `--zarr-*` flags

**Files:**
- Modify: `plumbline/cli.py:14-23` (`_load_inputs`)
- Modify: `plumbline/cli.py:85-94` (the `run` subparser)
- Test: `tests/test_zarr_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_io.py`:

```python
def test_cli_run_accepts_zarr(tmp_path):
    from plumbline.cli import main
    rng = np.random.default_rng(2)
    data = (rng.random((64, 64)) > 0.5).astype("float32")
    _write_array(tmp_path / "pred.zarr", data)
    out = tmp_path / "report.html"
    rc = main(["run", str(tmp_path / "pred.zarr"), "-o", str(out), "--tile", "32"])
    assert rc == 0
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py::test_cli_run_accepts_zarr -v`
Expected: FAIL — `load_image01` raises (it tries to open the `.zarr` directory as an image).

- [ ] **Step 3: Implement the CLI wiring**

In `plumbline/cli.py`, replace `_load_inputs` (lines 14-23) with:

```python
def _load_inputs(args):
    if args.segment_id:
        ink, mask = pio.fetch_segment(args.segment_id, scroll=args.scroll)
        meta = {"segment_id": args.segment_id, "scroll": args.scroll}
        return ink, mask, meta
    zopts = dict(component=args.zarr_component, z=args.zarr_z, reduce=args.zarr_reduce)
    ink = pio.load_input01(args.ink, **zopts)
    if args.mask:
        mask = pio.load_input01(args.mask, **zopts) > 0.0
    else:
        mask = np.ones(ink.shape, dtype=bool)
    meta = {"segment_id": os.path.splitext(os.path.basename(str(args.ink).rstrip("/")))[0],
            "scroll": args.scroll}
    return ink, mask, meta
```

In the `run` subparser (after line 89, `run.add_argument("--mask", ...)`), add:

```python
    run.add_argument("--zarr-component",
                     help="array name / multiscale level within a Zarr group "
                          "(default: highest-resolution level)")
    run.add_argument("--zarr-z", type=int, default=None,
                     help="for a 3-D Zarr, the plane index to analyze")
    run.add_argument("--zarr-reduce", choices=["max", "mean"], default=None,
                     help="for a 3-D Zarr, project the first axis instead of "
                          "picking one plane")
```

(`os` is already imported in `cli.py`. argparse maps `--zarr-component`→`args.zarr_component`, `--zarr-z`→`args.zarr_z`, `--zarr-reduce`→`args.zarr_reduce`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.venvs/plumbline/bin/pytest tests/test_zarr_io.py::test_cli_run_accepts_zarr -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL suite (nothing else regressed)**

Run: `~/.venvs/plumbline/bin/pytest -q`
Expected: all prior tests still pass + the new `test_zarr_io.py` tests (the PNG/TIF `batch` path is unchanged because only `_cmd_run` was rewired; `_cmd_batch` still calls `load_image01`/`load_mask`).

- [ ] **Step 6: Commit**

```bash
git add plumbline/cli.py tests/test_zarr_io.py
git commit -m "feat: plumbline run accepts Zarr/OME-Zarr input via --zarr-* flags"
```

---

### Task 7: Document Zarr input

**Files:**
- Modify: `README.md` (the install + usage sections)

- [ ] **Step 1: Add an install line for the extra**

Under the install instructions in `README.md`, add:

```markdown
For reading predictions stored as Zarr / OME-Zarr arrays:

    pip install 'plumbline[zarr]'          # local stores
    pip install 'plumbline[zarr,remote]'   # also s3:// / https:// stores
```

- [ ] **Step 2: Add a Zarr usage subsection**

Add to the usage section of `README.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document Zarr/OME-Zarr input + the 3-D caveat"
```

---

## Constraints / do-not-touch (carry over from the project)

- **Do NOT modify** `band_contrast`, `flag_*`/`score.py`, the seam detector, or `util.to01`. This is intake only; the analysis receives the same `(float64 [0,1] image, bool mask)` it does today.
- **No scroll data in the repo or tests** — all fixtures are tiny synthetic arrays written to `tmp_path`.
- **Keep core dependency-light** — `zarr` stays an optional extra; never `import zarr` at module top level (only inside `load_zarr01`).
- **Public commits** are authored as `GitHwwb <132339858+GitHwwb@users.noreply.github.com>` on the origin machine's clean snapshot — you just commit locally and hand the code back.

## Self-review (done by plan author)

- **Spec coverage:** detection (Task 3) · 2-D read + int/float normalization (Task 2) · 3-D plane/projection (Tasks 2+4) · OME multiscale + override + bare-group error (Tasks 2+5) · CLI wiring + flags (Task 6) · optional dependency + guard (Task 1) · docs incl. 3-D caveat (Task 7) · "don't touch the core" (Constraints). No gaps.
- **Placeholder scan:** none — every code step shows complete code and an exact command with expected output.
- **Type consistency:** `_is_zarr` / `load_zarr01` / `load_input01` signatures `(path, component=None, z=None, reduce=None)` are identical across Tasks 2, 3, 6; CLI passes them as `component`/`z`/`reduce`; `to01` is the single normalizer everywhere. Group-detection (`getattr(obj, "shape", None) is None`) is the same in impl and tests.
