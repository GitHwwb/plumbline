import numpy as np
from typing import Optional
from scipy.ndimage import uniform_filter
from plumbline.model import FlagMap, ScoreReport


def orientation_deviation(theta, confidence, radius=2):
    """Angular deviation (radians, mod pi) of each tile's orientation from a
    confidence-weighted local consensus, using doubled-angle vectors."""
    u = np.cos(2 * theta)
    v = np.sin(2 * theta)
    w = confidence.astype(float)
    size = 2 * radius + 1
    # reflect (not nearest) so border padding mirrors interior tiles rather than
    # repeating a possibly-anomalous edge tile into its own consensus window,
    # which otherwise raises the effective flag threshold at corners/edges.
    su = uniform_filter(u * w, size=size, mode="reflect")
    sv = uniform_filter(v * w, size=size, mode="reflect")
    sw = uniform_filter(w, size=size, mode="reflect") + 1e-9
    consensus = 0.5 * np.arctan2(sv / sw, su / sw)
    dev = np.abs(((theta - consensus) + np.pi / 2) % np.pi - np.pi / 2)
    return dev


def flag_orientation(features, deg_thresh=15.0, radius=2, rel_thresh=0.30):
    """Orientation departs from the local consensus -- but ONLY on tiles whose
    orientation is reliably determined. Sparse fragments / structureless tiles
    yield a flat or boundary-railed sweep (orient_reliability ~0) and a
    meaningless angle; without this gate they spray false orientation breaks on
    clear-but-sparse label text (see the s5/gp segments)."""
    dev = orientation_deviation(features.theta, features.confidence, radius)
    flag = (dev > np.radians(deg_thresh)) & features.confidence
    rel = getattr(features, "orient_reliability", None)
    if rel is not None:
        flag = flag & (rel >= rel_thresh)
    return flag


def flag_spacing(features, rel_thresh=0.35, radius=2, strength_gate=0.30,
                 consensus_gate=0.70, harmonic_consensus=0.85):
    """Row pitch departs from the local consensus -- but ONLY where a consensus
    pitch exists to depart from. Scattered text has no single line spacing, so
    neighboring tiles latch onto different pitches; without a consensus check the
    disagreement between them manufactures false spacing breaks. A tile flags only
    when (a) enough of its confident neighbors agree on a pitch (consensus) and
    (b) this tile's pitch deviates from that consensus by more than rel_thresh.
    consensus_gate=0.70 was calibrated so scattered text (gp_20230904) drops to
    0 spacing flags while a genuine coarse/fine pitch jump still flags.

    2x AUTOCORRELATION HARMONIC GUARD: a single tile can pick the SECOND
    autocorrelation peak (pitch = 2x the true line spacing) inside an otherwise
    perfectly uniform field. That is not a spacing change at all -- it's a
    period-doubling artifact of the per-tile pitch estimator -- yet it deviates
    from the neighbour median by exactly +100% and would flag. The signature is
    sharp: the deviating tile sits in a NEAR-UNANIMOUS neighbourhood (consensus
    ~1.0, every neighbour agrees on the base pitch) and its pitch is an (almost)
    exact integer multiple -- specifically 2x, the dominant harmonic. We suppress
    only that narrow case: consensus >= harmonic_consensus AND round(ratio) == 2
    AND the ratio is within rel_thresh of that integer.

    Why round(ratio) == 2 ONLY, and why this does not hide real jumps: a GENUINE
    line-spacing change is a REGION BOUNDARY between two stretches of differently
    spaced text. At a boundary the deviating tile's neighbourhood is SPLIT (some
    neighbours on the coarse side, some on the fine side), so its consensus sits
    well below harmonic_consensus -- e.g. a real 20->60 boundary flags a tile at
    consensus ~0.73, ratio ~3.0. Both AND-clauses (consensus and round==2) must
    hold to suppress, so a real boundary survives on EITHER its split consensus
    OR its non-2 ratio. The harmonic, by contrast, is unanimous AND exactly 2x.
    Restricting to round(ratio) == 2 (rather than any integer >= 2) keeps the
    guard to the one harmonic actually observed -- the 2x second peak -- and
    leaves 3x/4x deviations (which in practice only arise at split-consensus
    region boundaries) free to flag. This guard is DECISION-LEVEL ONLY: it never
    touches the stored per-tile pitch, so the seam detector's per-tile-median
    pitch (gpitch) strip width is unaffected -- the documented coupling trap."""
    p = features.pitch
    valid = features.confidence & np.isfinite(p) & (features.pitch_strength >= strength_gate)
    flags = np.zeros_like(valid)
    if not valid.any():
        return flags
    nr, nc = valid.shape
    # small grids (tens of tiles): a clear per-tile loop, intentionally not vectorized
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            rs = slice(max(0, r - radius), min(nr, r + radius + 1))
            cs = slice(max(0, c - radius), min(nc, c + radius + 1))
            mask_win = valid[rs, cs].copy()
            mask_win[r - rs.start, c - cs.start] = False    # exclude self from consensus
            win = p[rs, cs][mask_win]
            if win.size < 3:  # need >=3 valid neighbors for a stable median
                continue
            med = float(np.median(win))
            if med <= 0:  # degenerate: pitch is a pixel distance, always > 0
                continue
            consensus = float(np.mean(np.abs(win - med) <= rel_thresh * med))
            if consensus < consensus_gate:
                continue                       # no agreed pitch here -> don't flag
            if abs(p[r, c] - med) / med <= rel_thresh:
                continue                       # within tolerance -> not a break
            # 2x autocorrelation harmonic: unanimous neighbourhood + exact-2x pitch
            # -> period doubling, not a real spacing change. (Split-consensus region
            # boundaries fail the consensus AND-clause; 3x/4x fail the round==2 clause.)
            ratio = max(p[r, c] / med, med / p[r, c])
            if (consensus >= harmonic_consensus and round(ratio) == 2
                    and abs(ratio - round(ratio)) <= rel_thresh):
                continue
            flags[r, c] = True
    return flags


def flag_seam(features, ink=None, theta=None, pitch=None, seam_frac=0.30,
              row_gate=0.10, corr_gate=0.35, gain_gate=0.18):
    """A PURE VERTICAL SHEET-JUMP -- text rows step up/down at a vertical seam
    WITHOUT rotating or changing spacing -- is invisible to the orientation,
    spacing, and garble flags (each side still looks like good periodic text).
    This detects it: it scans adjacent full-height strips' row offsets
    (seam_offset_profile) and flags the confident tiles straddling a column where
    the vertical row offset jumps sharply.

    Five gates keep clean text quiet (the dominant design goal -- a detector that
    false-alarms on good text is worse than none):
      (a) the wrapped offset residual m >= seam_frac * pitch (a real step, not wobble);
      (b) BOTH flanking strips are rowful (band_contrast >= row_gate) -- so a seam
          can only be claimed between two stretches of actual text;
      (c) the post-shift cross-correlation >= corr_gate (the two sides ARE the same
          text, just offset -- not unrelated noise);
      (d) SHIFT GAIN (corr - zero_corr) >= gain_gate -- the vertical shift must
          SUBSTANTIALLY improve the match over the unshifted alignment. This is the
          gate real giant-pitch label text forced in: on continuous text whose coarse
          strips happen to correlate at a half-pitch wrap lag, corr ~= zero_corr so
          the shift buys nothing and it stays quiet; a true seam has corr >> zero_corr.
          (The synthetic glyph_rows tournament never exposed this -- full-width rows
          always align at lag 0 -- so it surfaced only on real data.)
      (e) ISOLATION: if many boundaries trip at once it's a noisy/folded texture, not
          a single seam -> suppress all; survivors must also be a local offset max.
    `ink` is the source image (needed for the strip scan); theta/pitch default to the
    per-image skew + per-tile-median pitch stashed on `features`. Returns an all-False
    grid when ink is None (graceful no-op for callers that don't pass it)."""
    grid = np.zeros((features.n_rows, features.n_cols), dtype=bool)
    if ink is None:
        return grid
    if theta is None:
        theta = getattr(features, "gtheta", 0.0)
    if pitch is None:
        pitch = getattr(features, "gpitch", float("nan"))
    if not np.isfinite(pitch):
        return grid
    from plumbline.coherence import seam_offset_profile
    cx, m, rown, corr, zero_corr = seam_offset_profile(ink, theta, pitch)
    if m.size < 3:
        return grid
    thr = seam_frac * pitch
    gain = corr - zero_corr                                  # shift-vs-no-shift improvement
    cand = (m >= thr) & (corr >= corr_gate) & (gain >= gain_gate)
    cand[1:] &= (rown[:-1] >= row_gate) & (rown[1:] >= row_gate)
    if cand.sum() > max(2, int(round(0.05 * cand.size))):   # wall of spikes -> noise/fold
        return grid
    for i in np.where(cand)[0]:                              # local-max + tile straddle
        if m[i] < m[max(0, i - 1):i + 2].max() - 1e-9:
            continue
        seam_x = cx[i]
        for t in features.tiles:
            if t.x0 <= seam_x <= t.x1 and features.confidence[t.row, t.col]:
                grid[t.row, t.col] = True
    return grid


def flag_garble(features, band_thresh=0.10):
    """Confident ink but no row-band contrast -> structureless mottle (garble
    or non-text). The corrected, right-way-round structure rule.
    Threshold validated against frag1 IR (giant sparse Greek text, ~1800px auto
    tile) with the linear-detrend band_contrast: real-text band_strength median
    ~0.17 (min ~0.065; only ~30% of tiles dip below 0.10 -- those sit wholly
    inside one giant stroke/gap and are irreducibly featureless), clean dense
    text >1.1, synthetic noise median ~0.05. So 0.10 leaves real text mostly
    unflagged (garble_frac ~0.3, well under the 0.5 contract) while still
    flagging >=half of noise tiles so input_warning fires.

    2026-06-11 photograph guard: those frag1 'irreducibly featureless' tiles
    turned out to be GLOW victims -- the papyrus background contributed ~97% of
    the tile mean, drowning detected rows (pitch 199px!) in the std/mean ratio.
    band_contrast now subtracts the tile's darkest-quartile background first
    (predictions with black background are byte-identical), so frag1 garble_frac
    dropped ~0.3 -> ~0 and the s5 prediction's diffuse-probability-floor tiles
    (garble 75 -> 0) stopped flagging. The 0.10 threshold itself is unchanged
    and still flags structureless noise (guarded by tests)."""
    return features.confidence & (features.band_strength < band_thresh)


def flag_tiles(features, ink=None, theta=None, pitch=None) -> FlagMap:
    """All flag modes for the grid. Pass `ink` (the source image) to also run the
    seam (vertical sheet-jump) detector; theta/pitch default to the per-image skew
    and per-tile-median pitch stashed on `features`. Omitting `ink` leaves
    seam_break all-False -- a graceful no-op that keeps existing callers (which pass
    only `features`) working unchanged."""
    return FlagMap(
        orient_break=flag_orientation(features),
        spacing_break=flag_spacing(features),
        garble=flag_garble(features),
        seam_break=flag_seam(features, ink=ink, theta=theta, pitch=pitch),
    )


def trace_health(features, flags) -> ScoreReport:
    """0..100 health = 100 * (1 - flagged_confident_fraction)."""
    conf = features.confidence
    n_conf = int(conf.sum())
    flagged = int((flags.any_flag & conf).sum())
    if n_conf == 0:
        # Nothing was analyzable -> not a healthy trace. Score 0; the real
        # signal is low_conf_frac == 1.0, which callers should check.
        score = 0
    else:
        frac_bad = flagged / n_conf
        score = max(0, min(100, int(round(100 * (1.0 - frac_bad)))))
    total = features.confidence.size
    low_conf = 1.0 - (n_conf / total) if total else 1.0
    n_seam = int(flags.seam_break.sum()) if flags.seam_break is not None else 0
    return ScoreReport(
        score=score,
        n_orient=int(flags.orient_break.sum()),
        n_spacing=int(flags.spacing_break.sum()),
        n_garble=int(flags.garble.sum()),
        low_conf_frac=float(low_conf),
        n_seam=n_seam,
    )


def input_warning(features, flags) -> Optional[str]:
    """Heuristic warnings when an input is outside the tool's regime.

    1. ROTATION OUT OF RANGE (checked first -- more specific and actionable):
       the GLOBAL skew search (estimate_scale_and_skew) sweeps only +-25deg,
       so text rotated beyond that pegs gtheta at EXACTLY the boundary --
       measured: 60deg-rotated glyph text -> gtheta = -25.0deg railed, while
       10deg text -> 10.0deg, upright -> 0.0deg, noise -> 0.0deg, all interior.
       (Per-tile sweeps do NOT rail in this regime -- they re-seed from the
       railed gtheta and find spurious interior maxima -- so the global rail
       is the only reliable fingerprint.) The structured-fraction conjunct
       keeps an unlucky flat-objective boundary argmax on noise from
       misdiagnosing as rotation. This matters because every downstream
       detector then measures across the wrong axis -- a real ~60deg-rotated
       GP-banner label set collected 50 bogus 'seam' flags from the
       vertical-strip scan (user-reported). Confess, don't emit confident flags.
    2. DENSE-BUT-STRUCTURELESS: most analyzable tiles are dense but have no
       text-line structure -- the signature of a bare surface render or a
       label mask, not detected ink.

    Returns a message, or None when the input looks like real upright ink."""
    n_conf = int(features.confidence.sum())
    if n_conf == 0:
        return None
    gtheta = float(getattr(features, "gtheta", 0.0))
    span = np.radians(25.0)                      # mirrors estimate_scale_and_skew's sweep
    structured_frac = float((features.confidence
                             & (features.band_strength >= 0.10)).sum()) / n_conf
    if abs(gtheta) >= span - 1e-6 and structured_frac >= 0.5:
        return ("text orientation appears to lie OUTSIDE the supported ±25° skew "
                "range (the global skew search pegged its boundary). "
                "Orientation/spacing/seam flags are unreliable on rotated "
                "input -- rotate the image upright and re-run.")
    garble_frac = int(flags.garble.sum()) / n_conf
    if garble_frac >= 0.5:
        return ("input may not be an ink prediction: most analyzable tiles have "
                "ink-like density but no text-line structure (looks like a surface "
                "render or label mask) -- treat the score with caution.")
    return None
