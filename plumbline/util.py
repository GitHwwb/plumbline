import numpy as np


def to01(img) -> np.ndarray:
    """Grayscale float64 image in [0, 1].

    Integer images are scaled by their full type range, so 8-bit (/255) and
    16-bit (/65535) inputs both land in [0, 1]. A 16-bit ink prediction divided
    by 255 would otherwise overflow and clip almost entirely to 1.0.
    """
    raw = np.asarray(img)
    a = raw.astype(np.float64)
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
