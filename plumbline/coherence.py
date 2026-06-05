import numpy as np
from scipy.ndimage import rotate as _ndrotate, zoom as _zoom, uniform_filter1d
from plumbline.util import to01, wrap_angle
from plumbline.model import TileFeatures
from plumbline.tiles import tile_grid

TILE_MIN, TILE_MAX = 256, 2048


def projection_profile(img, theta=0.0, mode="constant"):
    """Mean ink per text-row: rotate so rows are horizontal, average along them.
    `mode` is the rotation fill: 'constant' (zero pad) for the band/pitch metrics;
    the orientation search passes 'reflect' so injected zeros don't add spurious
    edge transitions that would bias the objective toward large rotation angles."""
    a = to01(img)
    if abs(theta) > 1e-6:
        a = _ndrotate(a, -np.degrees(theta), reshape=False, order=1,
                      mode=mode, cval=0.0)
    return a.mean(axis=1)


def _detrend(profile):
    """Remove the broad density envelope, keep row-scale oscillation."""
    win = max(3, (profile.size // 2) | 1)
    return profile - uniform_filter1d(profile, size=win, mode="nearest")


def _linear_detrend(profile):
    """Remove only DC + linear slope, preserving row oscillation at ANY scale --
    including 1-2 GIANT rows, which the moving-average `_detrend` (window ~ half
    the tile) would erase, making giant-letter text read as structureless mush."""
    n = profile.size
    if n < 2:
        return profile - float(profile.mean())
    x = np.arange(n)
    a, b = np.polyfit(x, profile, 1)
    return profile - (a * x + b)


def band_contrast(img, theta=0.0, profile=None):
    """>=0 'rowness' (~0 for noise, ~0.1-1.5+ for text; unbounded): linearly-
    detrended projection-profile contrast. High when ink forms rows separated by
    gaps -- at ANY scale, from many fine rows to a few giant ones; ~0 for
    structureless mottle. Linear (not moving-average) detrend so low-frequency
    giant rows survive while the broad envelope/gradient is still removed.

    `profile` lets a caller that ALREADY computed projection_profile(img, theta)
    pass it in to avoid a second rotation (the dominant cost on large skewed
    strips); when None it is computed as before, so default behaviour is identical."""
    p = projection_profile(img, theta) if profile is None else np.asarray(profile)
    det = _linear_detrend(p)
    # Floor the denominator at a minimum mean-ink level. std/mean explodes as
    # mean->0, so near-empty / sparse single-band tiles (e.g. one catching only the
    # bottoms of letters bled in from the row above) otherwise read as HIGHER
    # 'rowness' than real dense text. max() leaves confident dense text
    # (mean >> 0.08) unchanged, so the tuned garble threshold holds; only
    # thin/sparse tiles get pulled back into line.
    return float(det.std() / max(float(p.mean()), 0.08))


def _row_sharpness(profile):
    """Energy of the projection profile's first difference, normalized by mean^2.
    High when ink forms crisp horizontal rows. Differencing removes the slow ink
    envelope, so -- unlike band contrast -- it is not fooled into the fragment's
    diagonal on sparse text (where envelope dominates over row structure)."""
    d = np.diff(profile)
    if d.size == 0:                       # length<2 profile: no rows to score
        return 0.0
    return float((d * d).mean() / (float(profile.mean()) ** 2 + 1e-9))


def dominant_orientation(img, seed=0.0, span=np.radians(25), n=13,
                         return_reliability=False):
    """Writing-direction angle (radians, mod pi) that maximizes row sharpness over
    a seed-centred sweep. Uses reflect-padded profiles so the objective tracks
    real text rows rather than rotation artifacts. (Was orientation_by_contrast;
    a band-contrast objective saturated at the search boundary on sparse text,
    spraying false orient_break flags -- see tests/test_real_ir.py.)

    With return_reliability=True also returns a 0..1 reliability: how decisively
    the sweep peaks at an INTERIOR angle. A sparse fragment or structureless tile
    gives a flat or boundary-railed objective (no real writing direction) -> ~0,
    which flag_orientation uses to refuse manufacturing a break from noise."""
    angles = float(np.asarray(seed)) + np.linspace(-span, span, n)
    vs = np.array([_row_sharpness(projection_profile(img, float(t), mode="reflect"))
                   for t in angles])
    ib = int(np.argmax(vs))
    theta = float(wrap_angle(float(angles[ib])))
    if not return_reliability:
        return theta
    best = float(vs[ib]); med = float(np.median(vs))
    peak = (best - med) / (best + 1e-12)        # 0..1: flat objective -> ~0
    railed = (ib == 0 or ib == n - 1)           # max at the search boundary
    reliability = 0.0 if railed else max(0.0, min(1.0, peak))
    return theta, reliability


def row_pitch(img, theta=0.0, min_lag=8, max_lag=None):
    """Row spacing (px) from the profile autocorrelation + its 0..1 peak height.
    Returns (nan, 0.0) when there is no clear periodic peak."""
    p = projection_profile(img, theta)
    p = p - p.mean()
    if not np.any(p):
        return float("nan"), 0.0
    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = p.size // 2 if max_lag is None else int(max_lag)
    lo = max(2, int(min_lag))
    seg = ac[lo:hi]
    if seg.size < 3:
        return float("nan"), 0.0
    i = np.arange(1, seg.size - 1)
    ismax = (seg[i] > seg[i - 1]) & (seg[i] > seg[i + 1])
    cand = i[ismax]
    if cand.size == 0:
        return float("nan"), 0.0
    b = int(cand[int(np.argmax(seg[cand]))])
    return float(b + lo), float(seg[b])


def seam_offset_profile(ink, theta, pitch, strip_px=None):
    """Per-strip vertical row-offset profile for the pure-vertical sheet-jump
    detector. Slices the image into adjacent full-height vertical strips one row
    pitch wide, takes each strip's row projection profile, and cross-correlates
    each strip against its left neighbour to read the vertical row-OFFSET between
    them. Continuous text -> offset ~0 everywhere; a sheet-jump seam -> one sharp
    offset spike of ~dy (mod pitch) at the seam column, while pitch and orientation
    are unchanged (which is why orientation/spacing/garble all miss it).

    Returns (cx, m, rowness, corr, zero_corr): per-strip left edge x, the wrapped
    half-pitch offset residual m=min(|off|, pitch-|off|), the strip rowness
    (band_contrast), the cross-correlation peak, and the correlation at lag 0 (how
    aligned the two strips already are WITHOUT shifting). Empty arrays when pitch is
    unusable or there are too few strips. The cross-correlation is OVERLAP-RESTRICTED
    (>=70% overlap, normalized by overlapping-segment norms only) so partial-overlap
    edge lags cannot win the argmax and wrap-around aliasing can't manufacture a spike.

    zero_corr is the discriminator that real giant-pitch label text forced into the
    design (the synthetic glyph_rows tournament never exposed it): a TRUE seam means
    the unshifted rows are poorly aligned and a vertical shift dramatically improves
    the match (corr >> zero_corr), whereas on continuous text whose strips happen to
    correlate at a half-pitch wrap lag, corr ~= zero_corr -- the shift buys nothing.
    flag_seam gates on the SHIFT GAIN (corr - zero_corr)."""
    a = to01(ink)
    h, w = a.shape
    empty = (np.array([]),) * 5
    if not np.isfinite(pitch) or pitch < 4:
        return empty
    sp = strip_px or int(np.clip(round(pitch), 8, max(8, w // 2)))
    edges = list(range(0, w - sp + 1, sp))
    if len(edges) < 3:
        return empty
    profs, rown, cx = [], [], []
    for x0 in edges:
        sub = a[:, x0:x0 + sp]
        p = projection_profile(sub, theta)          # rotate once; reuse for both
        rown.append(band_contrast(sub, theta, profile=p))
        profs.append(p - p.mean())
        cx.append(x0)
    maxlag = max(1, int(round(pitch / 2)))
    off = np.zeros(len(profs))
    corr = np.zeros(len(profs))
    zero_corr = np.zeros(len(profs))
    n = profs[0].size
    min_ov = int(0.70 * n)
    for i in range(1, len(profs)):
        p0, p1 = profs[i - 1], profs[i]
        m = min(p0.size, p1.size)
        p0, p1 = p0[:m], p1[:m]
        best_c, best_L = -1e9, 0
        z0 = 0.0
        for L in range(-maxlag, maxlag + 1):     # overlap-restricted x-corr
            ov = m - abs(L)
            if ov < min_ov:
                continue
            if L >= 0:
                x_, y_ = p0[L:], p1[:m - L]
            else:
                x_, y_ = p0[:m + L], p1[-L:]
            d = np.sqrt(np.dot(x_, x_) * np.dot(y_, y_)) + 1e-9
            c = float(np.dot(x_, y_) / d)
            if L == 0:
                z0 = c
            if c > best_c + 1e-6:                 # ties keep the smaller |lag|
                best_c, best_L = c, L
        off[i] = best_L
        corr[i] = best_c
        zero_corr[i] = z0
    mres = np.minimum(np.abs(off), pitch - np.abs(off))
    return np.asarray(cx, dtype=float), mres, np.asarray(rown), corr, zero_corr


def estimate_scale_and_skew(ink, mask=None, k_rows=4.0, target=1000.0):
    """Pick a tile size spanning several text rows + the global skew angle.
    Returns (tile_size:int, theta:float). Scale comes from the dominant ROW PITCH
    (line spacing) of the global profile -- not the autocorrelation decay length,
    which inflates on sparse text and ballooned the tile (see the design spec:
    docs/superpowers/specs/2026-06-03-plumbline-coherence-view-tiling-spacing-design.md).
    Falls back to the decay length when no clear pitch peak exists (e.g. a few
    giant rows), so giant-letter fragments still get large tiles."""
    a = to01(ink)
    if mask is not None and mask.shape == a.shape and mask.any():
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(1.0, target / max(a.shape))
    small = _zoom(a, scale, order=1) if scale < 1.0 else a
    theta = dominant_orientation(small, seed=0.0, span=np.radians(25), n=21)
    # Primary: the dominant row pitch (first prominent autocorrelation peak) is the
    # real line spacing; sparsity does not inflate it the way the decay length does.
    pitch_small, pstr = row_pitch(small, theta, min_lag=4,
                                  max_lag=max(5, min(small.shape) // 2))
    # Guard against picking a harmonic (2×, 3×) instead of the fundamental: if a
    # strong sub-harmonic exists at ~half the detected pitch, prefer it (it is the
    # true line spacing). This avoids the sparse-glyph case where the 2× harmonic
    # is marginally higher than the fundamental due to inter-glyph gaps.
    if np.isfinite(pitch_small) and pstr > 0.15 and pitch_small > 8:
        half_max = int(pitch_small * 0.65)  # 0.65 of pitch comfortably brackets pitch/2 (the 2x harmonic)
        if half_max > 4:
            sub_pitch, sub_pstr = row_pitch(small, theta, min_lag=4, max_lag=half_max)
            if np.isfinite(sub_pitch) and sub_pstr > 0.6 * pstr:  # only swap if the sub-harmonic is nearly as strong as the detected peak
                pitch_small, pstr = sub_pitch, sub_pstr
    # pstr > 0.15: empirical pitch-peak floor; below this the peak is likely noise, so use the decay-length fallback instead of trusting a spurious pitch.
    if np.isfinite(pitch_small) and pstr > 0.15:
        row_h = pitch_small / max(scale, 1e-9)
    else:
        # Fallback: detrended-profile autocorrelation decay length (the old proxy).
        # Used when there is no clear pitch peak (e.g. a few giant rows) so giant
        # text still yields a large tile.
        det = _detrend(projection_profile(small, theta))
        det = det - det.mean()
        if not np.any(det):
            return TILE_MIN, theta
        ac = np.correlate(det, det, mode="full")[det.size - 1:]
        ac = ac / (ac[0] + 1e-9)
        below = np.where(ac[1:] < 0.2)[0]
        decay_small = int(below[0] + 1) if below.size else max(1, det.size // 4)
        row_h = decay_small / max(scale, 1e-9)
    # k_rows=4.0: tile spans ~4 text rows; validated against frag1-IR (test_real_ir.py).
    tile = int(np.clip(round(k_rows * row_h), TILE_MIN, TILE_MAX))
    return tile, theta


def ink_density(img, thresh=0.25):
    """Fraction of pixels above an ink threshold."""
    a = to01(img)
    return float((a > thresh).mean())


def analyze_tiles(ink, mask=None, tile=None, overlap=0.5,
                  min_density=0.02, min_coverage=0.5):
    """Per-tile band features over the grid. Tile size auto-adapts to text
    scale when `tile` is None. Low-coverage / low-ink tiles stay low-confidence
    (orientation + band_strength still recorded; pitch left NaN)."""
    a = to01(ink)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    # always need gtheta to seed per-tile orientation; auto_tile only used when tile is None
    auto_tile, gtheta = estimate_scale_and_skew(a, mask)
    if tile is None:
        tile = auto_tile
    tiles, nr, nc = tile_grid(a.shape, tile, overlap)
    theta = np.zeros((nr, nc)); band = np.zeros((nr, nc))
    pitch = np.full((nr, nc), np.nan); pstr = np.zeros((nr, nc))
    density = np.zeros((nr, nc)); conf = np.zeros((nr, nc), dtype=bool)
    rel = np.zeros((nr, nc))
    for t in tiles:
        sub = a[t.y0:t.y1, t.x0:t.x1]
        if min(sub.shape) < 8:
            continue
        cov = float(mask[t.y0:t.y1, t.x0:t.x1].mean())
        d = ink_density(sub)
        th, rl = dominant_orientation(sub, seed=gtheta, return_reliability=True)
        theta[t.row, t.col] = th
        rel[t.row, t.col] = rl
        band[t.row, t.col] = band_contrast(sub, th)
        density[t.row, t.col] = d
        if cov < min_coverage or d < min_density:
            continue
        p, s = row_pitch(sub, th, min_lag=max(8, tile // 16),
                         max_lag=sub.shape[0] // 2)
        pitch[t.row, t.col] = p
        pstr[t.row, t.col] = s
        conf[t.row, t.col] = True
    # GRAFT 1: the seam detector's reference spacing is the per-tile MEDIAN pitch,
    # not a fresh global single-profile pitch -- a sheet-jump corrupts the global
    # profile's autocorrelation, but most tiles still measure the true local pitch.
    # A confident tile can still have NaN pitch (dense but aperiodic), so guard on
    # "any finite confident pitch" -- np.nanmedian over an all-NaN slice would both
    # warn and return NaN.
    conf_pitch = pitch[conf]
    gpitch = (float(np.nanmedian(conf_pitch)) if np.isfinite(conf_pitch).any()
              else float("nan"))
    return TileFeatures(nr, nc, theta, band, pitch, pstr, density, conf, tiles,
                        rel, gtheta=float(gtheta), gpitch=gpitch)
