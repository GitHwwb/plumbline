import os
import warnings
import numpy as np
from plumbline.util import to01

# Flattened scroll segments are legitimately huge (hundreds of megapixels), so
# raise PIL's decompression-bomb ceiling -- otherwise load_image01 raises
# DecompressionBombError on real inputs (e.g. 215/237 MP GP/Scroll5 segments).
# A large FINITE cap (2 gigapixels, ~10x the largest segment seen) keeps the
# bomb protection itself, which None would disable process-wide.
try:
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = 2_000_000_000
except Exception:  # pragma: no cover - PIL always present in practice
    pass


def _warn_underexposed_range(path, dtype, amax):
    """Core of the underexposure warning, for callers that already know the
    raw max (e.g. the chunked Zarr reader, which never holds the raw array)."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer) and dtype.itemsize > 1:
        vmax = int(np.iinfo(dtype).max)
        if 0 < amax <= vmax // 100:
            warnings.warn(
                f"{path}: pixel values use <1% of the {dtype} range "
                f"(max {amax} of {vmax}); after normalization everything is "
                "near 0 and the segment may score 0 / all-low-confidence. "
                "If this is 8-bit data in a 16-bit container, rescale or "
                "convert it first."
            )


def _warn_if_underexposed(arr, path):
    """8-bit data saved in a 16-bit container (a common pipeline artifact):
    to01 scales by the DTYPE range, so the image lands in a sliver of [0, 1],
    every tile fails the ink-density gate, and the segment silently scores 0
    all-low-confidence with no hint why. Warn at load time, once."""
    _warn_underexposed_range(path, arr.dtype, int(arr.max()) if arr.size else 0)


def load_image01(path) -> np.ndarray:
    """Load a PNG/TIF as a grayscale float32 image in [0, 1]."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        with Image.open(path) as img:
            # Palette/RGB images store indices or channels, not a single brightness;
            # convert those to 8-bit luminance. Keep integer/float "intensity" modes
            # (e.g. 16-bit "I;16") as-is so to01 can scale them by their bit depth.
            if img.mode not in ("L", "I", "I;16", "I;16B", "I;16L", "F"):
                img = img.convert("L")
            arr = np.asarray(img)
    _warn_if_underexposed(arr, path)
    return to01(arr)


def load_mask(path) -> np.ndarray:
    """Load a mask (image file or Zarr store) as a boolean array (any nonzero
    pixel is True)."""
    img = load_input01(path)
    return img > 0.0


def load_zarr01(path, component=None, z=None, reduce=None) -> np.ndarray:
    """Load a 2-D ink prediction from a (local or remote) Zarr / OME-Zarr store.

    - A bare array store is used directly.
    - An OME-Zarr multiscale group resolves to its highest-resolution level
      (datasets[0]); pass `component` (e.g. "1") to pick another level or a
      named array in a plain group.
    - A >2-D array (e.g. a (z, y, x) volume) needs an explicit plane: pass
      `z=<index>` for one slice, or `reduce="max"|"mean"` to project the first
      axis. (Projecting a volume is a convenience, NOT a true flattened render.)

    Returns a grayscale float32 image in [0, 1] (via to01).
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

    return _read_zarr01(obj, z=z, reduce=reduce, path=path)


def _read_zarr01(obj, z=None, reduce=None, path="", slab_rows=None):
    """Read a (component-resolved) Zarr ARRAY into a float32 [0,1] image
    without materializing more than necessary -- chunked access is the point
    of Zarr, and `np.asarray(obj[:])` was throwing it away:

      - ``z=<i>``      loads only that plane's chunks;
      - ``reduce=...`` streams the first axis in chunk-sized slabs into a 2-D
        accumulator (running max, or float64 sum / depth for the mean) -- the
        (z, y, x) volume is never held at once;
      - 2-D arrays fill the float32 output in row slabs (~8 MB or one chunk,
        whichever is larger), so peak memory is the float32 image plus one
        slab instead of raw-dtype array + converted copy.

    Normalization matches to01 exactly: integer dtypes scale by the dtype
    range, float data with max > 1 scales by that max, everything clips to
    [0, 1]. `slab_rows` exists for tests."""
    shape = tuple(obj.shape)
    nd = len(shape)
    if nd > 2:
        if z is not None:
            arr = np.asarray(obj[z])
        elif reduce in ("max", "mean"):
            chunks = getattr(obj, "chunks", None)
            step = max(1, int(chunks[0])) if chunks else 1
            acc = None
            for i in range(0, shape[0], step):
                blk = np.asarray(obj[i:i + step])
                part = (blk.max(axis=0) if reduce == "max"
                        else blk.sum(axis=0, dtype=np.float64))
                if acc is None:
                    acc = part
                else:
                    acc = np.maximum(acc, part) if reduce == "max" else acc + part
            arr = acc if reduce == "max" else acc / shape[0]
        else:
            raise ValueError(
                f"{path} is {nd}-D {shape}; pass z=<index> for one "
                "plane or reduce='max'|'mean' to project the first axis"
            )
        if arr.ndim > 2:  # e.g. a 4-D store; one selection still leaves it >2-D
            raise ValueError(f"selected data is still {arr.ndim}-D: {arr.shape}")
        _warn_if_underexposed(arr, path)
        return to01(arr)

    # 2-D: stream row slabs straight into the float32 output.
    dt = np.dtype(obj.dtype)
    if slab_rows is None:
        chunks = getattr(obj, "chunks", None)
        row_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize)
        slab_rows = max(int(chunks[0]) if chunks else 1,
                        int(np.ceil(8 * 2**20 / row_bytes)))
    out = np.empty(shape, dtype=np.float32)
    amax = None
    for i in range(0, shape[0], max(1, int(slab_rows))):
        blk = np.asarray(obj[i:i + int(slab_rows)])
        if blk.size:
            bmax = blk.max()
            amax = bmax if amax is None else max(amax, bmax)
        out[i:i + blk.shape[0]] = blk
    if np.issubdtype(dt, np.integer):
        vmax = float(np.iinfo(dt).max)
        if vmax > 0:
            out /= np.float32(vmax)
        _warn_underexposed_range(path, dt, int(amax) if amax is not None else 0)
    elif amax is not None and float(amax) > 1.0:
        out /= np.float32(amax)
    np.clip(out, 0.0, 1.0, out=out)
    return out


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
    if str(path).startswith(("s3://", "gs://", "http://", "https://")):
        # Without this, a remote non-Zarr path falls through to PIL, which
        # cannot open URLs and fails with a baffling FileNotFoundError.
        raise ValueError(
            f"{path}: remote inputs must be Zarr/OME-Zarr stores (*.zarr); "
            "download image files locally first"
        )
    return load_image01(path)


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
        "fetch_segment is a stub: wire to the vesuvius API during Task 17 "
        "once the exact ink/flat_mask accessor for the chosen segment is known."
    )
