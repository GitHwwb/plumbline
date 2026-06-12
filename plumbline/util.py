import numpy as np


def to01(img) -> np.ndarray:
    """Grayscale float32 image in [0, 1].

    Integer images are scaled by their full type range, so 8-bit (/255) and
    16-bit (/65535) inputs both land in [0, 1]. A 16-bit ink prediction divided
    by 255 would otherwise overflow and clip almost entirely to 1.0.

    float32, not float64: full precision for [0,1] intensities at HALF the
    memory -- flattened segments run to hundreds of megapixels (a 237 MP image
    is ~0.9 GB instead of ~1.9 GB before any tile copies/rotations).
    """
    raw = np.asarray(img)
    a = raw.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if np.issubdtype(raw.dtype, np.integer):
        vmax = float(np.iinfo(raw.dtype).max)
        if vmax > 0:
            a = a / vmax
    elif a.size and a.max() > 1.0:
        a = a / a.max()
    return np.clip(a, 0.0, 1.0)


def wrap_angle(a):
    """Wrap an orientation angle (mod pi) into [-pi/2, pi/2)."""
    return (a + np.pi / 2) % np.pi - np.pi / 2
