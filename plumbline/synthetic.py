import numpy as np
from scipy.ndimage import rotate as _ndrotate


def striped_field(shape=(512, 512), pitch=20, angle=0.0, sharpness=0.85, noise=0.03, seed=0):
    """A clean 'ink-like' field of parallel text lines running along `angle`
    (radians), with the given line `pitch` (pixels). Square-ish stripes."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    # coordinate perpendicular to lines running along `angle`
    s = yy * np.cos(angle) - xx * np.sin(angle)
    stripes = (np.sin(2 * np.pi * s / pitch) > 0).astype(np.float64)
    rng = np.random.default_rng(seed)
    img = stripes * sharpness + rng.normal(0.0, noise, shape)
    return np.clip(img, 0.0, 1.0)


def glyph_rows(shape=(512, 512), row_pitch=40, glyph=18, gap=8, angle=0.0,
               fill=0.85, sharpness=0.9, noise=0.03, seed=0):
    """Rows of discrete glyph blocks separated by inter-row gaps and
    intra-row letter/word spaces -- a stand-in for real text (NOT stripes).
    `angle` rotates the writing direction (radians)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float64)
    gh = max(1, int(glyph * 0.8))
    y = row_pitch // 2
    while y < h:
        x = gap
        while x < w:
            gw = max(1, glyph + int(rng.integers(-3, 4)))
            word_gap = gap * (3 if rng.random() < 0.2 else 1)
            if rng.random() < fill:                       # leave letter-ish gaps
                img[y:min(y + gh, h), x:min(x + gw, w)] = sharpness
            x += gw + word_gap
        y += row_pitch
    img = img + rng.normal(0.0, noise, shape)
    if angle:
        img = _ndrotate(img, np.degrees(angle), reshape=False, order=1,
                        mode="constant", cval=0.0)
    return np.clip(img, 0.0, 1.0)


def rotate_band(field, y0, y1, ddeg=25):
    """Rotate a horizontal band in place -> local orientation discontinuity."""
    out = field.copy()
    out[y0:y1] = _ndrotate(field[y0:y1], ddeg, reshape=False, order=1, mode="reflect")
    return out


def splice_shift(field, x_split, dy=15):
    """Vertically shift everything right of x_split -> seam / line break."""
    out = field.copy()
    out[:, x_split:] = np.roll(field[:, x_split:], dy, axis=0)
    return out


def garble_patch(field, y0, y1, x0, x1, seed=1):
    """Replace a patch with noise -> high ink density, no line structure."""
    out = field.copy()
    rng = np.random.default_rng(seed)
    out[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0))
    return out
