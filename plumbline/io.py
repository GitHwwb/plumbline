import os
import numpy as np
from plumbline.util import to01

# Flattened scroll segments are legitimately huge (hundreds of megapixels), so
# lift PIL's decompression-bomb ceiling -- otherwise load_image01 raises
# DecompressionBombError on real inputs (e.g. 215/237 MP GP/Scroll5 segments).
try:
    from PIL import Image as _Image
    _Image.MAX_IMAGE_PIXELS = None
except Exception:  # pragma: no cover - PIL always present in practice
    pass


def load_image01(path) -> np.ndarray:
    """Load a PNG/TIF as a grayscale float64 image in [0, 1]."""
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(path)
    else:
        from PIL import Image
        img = Image.open(path)
        # Palette/RGB images store indices or channels, not a single brightness;
        # convert those to 8-bit luminance. Keep integer/float "intensity" modes
        # (e.g. 16-bit "I;16") as-is so to01 can scale them by their bit depth.
        if img.mode not in ("L", "I", "I;16", "I;16B", "I;16L", "F"):
            img = img.convert("L")
        arr = np.asarray(img)
    return to01(arr)


def load_mask(path) -> np.ndarray:
    """Load a mask image as a boolean array (any nonzero pixel is True)."""
    img = load_image01(path)
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
