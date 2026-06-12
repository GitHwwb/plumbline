import io as _io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_png(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# Coherence display: squash the unbounded band_strength into [0,1] for a FIXED
# color scale, so colors mean the same thing in every report and a single bright
# band can't dominate the scale. DISPLAY-ONLY -- scoring uses raw band_strength.
#
# Calibrated on the real demo set (gp_20230904 / gp_20230827 ink labels +
# s5_prediction). Measured confident band_strength: noise ~0.03-0.05, dense real
# text median ~0.156 (s5 auto grid), clean synthetic/label text 0.6-1.5+.
# The plain tanh(band) curve cannot land text mid-range AND push noise to ~0 (the
# two are only ~3x apart), so we subtract a small noise floor _COH_B0 before the
# tanh slope _COH_S: disp_band = tanh(max(band - _COH_B0, 0) / _COH_S). With
# _COH_B0=0.06, _COH_S=0.12 the s5 dense-text band 0.156 maps to ~0.66 (mid-range)
# while band<=0.06 (noise) maps to exactly 0.
_COH_B0 = 0.06   # band noise floor: rowness at/below this reads as "no structure"
_COH_S  = 0.12   # tanh slope above the floor; sets where text lands mid-range

# Density gate: near-EMPTY tiles are the rendering bug -- a lone ink sliver clears
# the 2% confidence gate, forms ONE concentrated band, and band_contrast rates it
# moderately-to-very high, so the heatmap paints background-with-a-sliver BRIGHTER
# than real dense text. Such tiles are exactly the low-density ones, so we DIM (not
# black out) the DISPLAYED coherence by a smooth function of features.density.
#
# IMPORTANT (the dimming has a FLOOR, not 0): density alone cannot distinguish a
# sliver artifact from GENUINELY-banded sparse text -- e.g. giant Greek text, which
# the scorer correctly counts healthy (flag_garble keeps band>=0.10) and which can
# sit at density ~0.06 with high band. An earlier version dimmed such tiles to ~0,
# which BLACKED OUT exactly the segments the rest of the pipeline treats as healthy
# (opposite-direction false reassurance). The floor `_COH_D_FLOOR` keeps low-density
# tiles VISIBLE (dimmed, never black): a sliver no longer reads brighter than real
# dense text (item A fixed) while genuine sparse text stays legible on the heatmap.
# Measured: real slivers/sparse sit at density ~0.03-0.06; dense text >=0.15 (s5
# dense text is all >=0.35, untouched). DISPLAY-ONLY -- density is not used in any
# flag/score/JSON path.
_COH_D_LO = 0.06     # below this density -> dimmed to the floor
_COH_D_HI = 0.15     # at/above this density -> no dimming (real dense text)
_COH_D_FLOOR = 0.35  # minimum density multiplier: dim low-density tiles, never to 0


def _smoothstep(x, lo, hi):
    """Hermite ramp: 0 below lo, 1 at/above hi, smooth (C1) in between."""
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _downsample_mean(a, k):
    """k x k block-mean downsample for the DISPLAY underlays. Striding
    (a[::k, ::k]) keeps every k-th row/column, so a 1-px stroke survives only
    when its index happens to be a multiple of k -- thin strokes flickered in
    and out of the previews. Averaging keeps them visible (proportionally dim)."""
    if k <= 1:
        return a
    h, w = a.shape
    a = a[:h - h % k, :w - w % k]
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _coherence_display(band, density=None):
    """Map raw band_strength -> a stable [0,1] display value (DISPLAY-ONLY).

    Two stages, both display-only:
      1. band curve  : tanh(max(band - _COH_B0, 0) / _COH_S) -- bounds + calibrates
                       so noise -> 0 and typical confident text lands mid-range.
      2. density dim : x [_COH_D_FLOOR + (1-_COH_D_FLOOR)·smoothstep(density, LO, HI)]
                       -- dims low-density single-sliver tiles toward the FLOOR (not 0)
                       so they stop reading as MORE coherent than real text, WITHOUT
                       blacking out genuinely-banded sparse text. Omitted (factor 1.0)
                       when `density` is None, so the bare band curve is unchanged."""
    disp = np.tanh(np.clip(np.asarray(band, dtype=float) - _COH_B0, 0.0, None) / _COH_S)
    if density is not None:
        factor = _COH_D_FLOOR + (1.0 - _COH_D_FLOOR) * _smoothstep(density, _COH_D_LO, _COH_D_HI)
        disp = disp * factor
    return disp


def _flag_extent(features, flags):
    rects = []
    fm = flags.any_flag
    for t in features.tiles:
        if fm[t.row, t.col]:
            rects.append((t.x0, t.y0, t.x1 - t.x0, t.y1 - t.y0))
    return rects


def ink_png(ink01, max_px=2000) -> bytes:
    """The raw ink image as an EXACT-extent grayscale PNG (pure PIL): pixel
    (0,0) IS array (0,0) and the PNG spans precisely the array, so the report
    can position flag boxes over it in percent coordinates and they line up by
    construction -- matplotlib's bbox_inches='tight' margins would skew them.
    Block-mean downsampled to <= max_px on the long side (trimming at most
    k-1 px ~ 0.1% at the right/bottom edge, far below a tile)."""
    from PIL import Image
    a = np.asarray(ink01, dtype=float)
    k = max(1, int(np.ceil(max(a.shape) / max_px)))
    a = np.clip(_downsample_mean(a, k), 0.0, 1.0)
    img = Image.fromarray((a * 255.0 + 0.5).astype("uint8"), mode="L")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_png(ink01, features, flags, figsize=(6, 6)) -> bytes:
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(ink01, cmap="gray", origin="upper")
    for (x, y, w, h) in _flag_extent(features, flags):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.5))
    ax.set_axis_off()
    return _fig_to_png(fig)


def heatmap_png(features, ink01=None) -> bytes:
    """Per-tile row coherence on a FIXED 0-1 scale (bounded display of
    band_strength) so colors are stable across reports. Non-confident tiles are left blank (NaN) so bare
    background reads as 'not assessed' rather than as signal.

    Plain and overlay render in the SAME pixel extent/aspect (taken from the tile
    grid) so the 'overlay ink' toggle does not shift or restretch the plot. With
    ink01 the heatmap cells stay fully opaque (true colours, matching the plain
    view and the colourbar) and the ink is drawn ON TOP as white marks whose
    opacity follows ink strength -- so the gradient is unchanged off the letters
    while the text shows through to locate it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.where(features.confidence,
                    _coherence_display(features.band_strength, features.density),
                    np.nan)
    w = max((t.x1 for t in features.tiles), default=data.shape[1])
    h = max((t.y1 for t in features.tiles), default=data.shape[0])
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=[0, w, h, 0], aspect="equal", interpolation="nearest")
    if ink01 is not None:
        a = np.asarray(ink01, dtype=float)
        k = max(1, int(round(max(a.shape) / 2000)))      # downsample for display only
        a = np.clip(_downsample_mean(a, k), 0.0, 1.0)
        marks = np.zeros(a.shape + (4,)); marks[..., :3] = 1.0; marks[..., 3] = a * 0.9
        ax.imshow(marks, extent=[0, w, h, 0], aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row coherence (0-1)")
    ax.set_axis_off()
    return _fig_to_png(fig)


def _row_direction(theta):
    """Display-coordinate (x, y-down) unit direction of text rows for codebase
    theta: (cos t, -SIN t). The minus sign is the convention pinned by the
    rotated seam-geometry test -- with +sin the drawn arrows are mirrored
    about the horizontal (error = 2*theta: invisible on upright text, a
    glaring NNE-vs-NNW flip on a -74deg segment, user-caught)."""
    return float(np.cos(theta)), float(-np.sin(theta))


def orientation_png(features, ink01=None) -> bytes:
    """Per-confident-tile dominant text-row direction (quiver). Arrows sit at true
    tile-CENTER pixel positions in the SAME pixel extent as the heatmap, so an
    optional ink underlay (ink01) lines the arrows up against the text they describe
    -- pass ink01 to draw that faint gray underlay, omit it for the plain field."""
    fig, ax = plt.subplots(figsize=(6, 6))
    w = max((t.x1 for t in features.tiles), default=features.n_cols)
    h = max((t.y1 for t in features.tiles), default=features.n_rows)
    if ink01 is not None:
        a = np.asarray(ink01, dtype=float)
        k = max(1, int(round(max(a.shape) / 2000)))      # downsample for display only
        a = np.clip(_downsample_mean(a, k), 0.0, 1.0)
        ax.imshow(a, cmap="gray", extent=[0, w, h, 0], aspect="equal",
                  interpolation="nearest")
    # arrow length ~0.4 tile so arrows read as row-direction ticks, not a dense field
    tile_w = np.median([t.x1 - t.x0 for t in features.tiles]) if features.tiles else 1.0
    L = 0.4 * float(tile_w)
    xs, ys, us, vs = [], [], [], []
    for t in features.tiles:
        if not features.confidence[t.row, t.col]:
            continue
        th = features.theta[t.row, t.col]
        xs.append((t.x0 + t.x1) / 2.0); ys.append((t.y0 + t.y1) / 2.0)
        ux, vy = _row_direction(th)                # (cos t, -sin t): see helper
        us.append(ux * L); vs.append(vy * L)
    if xs:
        ax.quiver(xs, ys, us, vs, pivot="mid", angles="xy",
                  scale_units="xy", scale=1.0,
                  color=("#ffce5c" if ink01 is not None else "black"), width=0.004)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)               # pixel extent, y down (no invert)
    ax.set_aspect("equal"); ax.set_axis_off()
    return _fig_to_png(fig)


def _flag_layers(flags):
    """(mode, grid) pairs for every flag mode, including seam when present."""
    layers = [("orientation", flags.orient_break), ("spacing", flags.spacing_break),
              ("garble", flags.garble)]
    if getattr(flags, "seam_break", None) is not None:
        layers.append(("seam", flags.seam_break))
    return layers


def flags_png(ink01, features, flags) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ink01, cmap="gray", origin="upper", alpha=0.6)
    colors = {"orientation": "#ff5c5c", "spacing": "#ffce5c", "garble": "#7fb0e0",
              "seam": "#c77dff"}
    for mode, grid in _flag_layers(flags):
        for t in features.tiles:
            if grid[t.row, t.col]:
                ax.add_patch(plt.Rectangle((t.x0, t.y0), t.x1 - t.x0, t.y1 - t.y0,
                                           fill=True, alpha=0.35, color=colors[mode]))
    ax.set_axis_off()
    return _fig_to_png(fig)


def _tile_detail(features, t):
    """One-line human summary of a tile's measurements, for hover tooltips and
    the JSON sidecar: writing-direction angle, row pitch (or an em-dash when no
    periodic peak exists), rowness (band_strength) and ink fraction."""
    th = float(np.degrees(features.theta[t.row, t.col]))
    p = float(features.pitch[t.row, t.col])
    band = float(features.band_strength[t.row, t.col])
    d = float(features.density[t.row, t.col])
    pitch_s = f"{p:.0f}px" if np.isfinite(p) else "—"
    return f"angle {th:.0f}° · row pitch {pitch_s} · rowness {band:.2f} · ink {d:.0%}"


def flagged_regions(features, flags):
    """Flat list of flagged cells in pixel coordinates: center {x, y}, the full
    tile box {x0, y0, x1, y1} (so a JSON consumer can draw/crop the flagged
    area without knowing the tile size), the flag mode, and a human-readable
    `detail` line with the tile's measurements."""
    out = []
    for mode, grid in _flag_layers(flags):
        for t in features.tiles:
            if grid[t.row, t.col]:
                out.append({"x": int((t.x0 + t.x1) // 2), "y": int((t.y0 + t.y1) // 2),
                            "x0": int(t.x0), "y0": int(t.y0),
                            "x1": int(t.x1), "y1": int(t.y1), "mode": mode,
                            "detail": _tile_detail(features, t)})
    return out
