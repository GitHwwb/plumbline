import numpy as np
from plumbline.synthetic import glyph_rows, rotate_band, garble_patch, splice_shift
from plumbline.coherence import analyze_tiles, estimate_scale_and_skew
from plumbline.score import (flag_orientation, flag_spacing, flag_garble,
                             flag_seam, flag_tiles, trace_health, input_warning)
from plumbline.model import ScoreReport


def _spacing_decision_trace(feats, rel_thresh=0.35, radius=2, strength_gate=0.30,
                            consensus_gate=0.70):
    """Re-walk flag_spacing's decision loop and return, for every tile that PASSES
    the consensus gate AND deviates past rel_thresh (i.e. every tile the bare
    deviation rule would flag, BEFORE the harmonic guard), a dict of its
    (r, c, pitch, med, consensus, ratio). Lets the jump test assert it is
    exercising a GENUINE split-consensus boundary jump -- not a unanimous 2x
    autocorrelation harmonic the guard now suppresses."""
    p = feats.pitch
    valid = feats.confidence & np.isfinite(p) & (feats.pitch_strength >= strength_gate)
    nr, nc = valid.shape
    out = []
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            rs = slice(max(0, r - radius), min(nr, r + radius + 1))
            cs = slice(max(0, c - radius), min(nc, c + radius + 1))
            mask_win = valid[rs, cs].copy()
            mask_win[r - rs.start, c - cs.start] = False
            win = p[rs, cs][mask_win]
            if win.size < 3:
                continue
            med = float(np.median(win))
            if med <= 0:
                continue
            consensus = float(np.mean(np.abs(win - med) <= rel_thresh * med))
            if consensus < consensus_gate:
                continue
            if abs(p[r, c] - med) / med <= rel_thresh:
                continue
            ratio = max(p[r, c] / med, med / p[r, c])
            out.append(dict(r=r, c=c, pitch=float(p[r, c]), med=med,
                            consensus=consensus, ratio=ratio))
    return out


def _synthetic_feats(pitch):
    """Build a TileFeatures whose only meaningful field is `pitch` (square grid,
    all confident, strong, finite) -- a hand-controlled fixture for pinning the
    spacing decision logic directly, WITHOUT going through analyze_tiles. This is
    what lets the harmonic-guard tests construct an exact 2x harmonic; clean
    synthetic glyph_rows never manufactures one (no tile clears the consensus gate
    at a guardable harmonic), so a glyph-based negative test would be vacuous."""
    from plumbline.model import TileFeatures, Tile
    n = pitch.shape[0]
    tiles = [Tile(r, c, r * 10, r * 10 + 10, c * 10, c * 10 + 10)
             for r in range(n) for c in range(n)]
    return TileFeatures(n, n, np.zeros((n, n)), np.ones((n, n)), pitch.astype(float),
                        np.full((n, n), 0.5), np.ones((n, n)),
                        np.ones((n, n), bool), tiles)


def test_spacing_flag_fires_on_pitch_change():
    # GENUINE region-boundary spacing jump (honest version). A tall 20|60 field
    # (two 512x768 glyph_rows stacked) produces a real coarse/fine boundary. The
    # surviving flag is a SPLIT-CONSENSUS jump -- a boundary tile whose neighbour-
    # hood straddles both the fine (20) and coarse (60) regions, so its consensus
    # sits below the harmonic_consensus AND-clause AND its pitch ratio is ~3x (not
    # the 2x autocorrelation harmonic). That is what makes it a real jump rather
    # than the period-doubling artifact the guard suppresses.
    #
    # The OLD 384|384 field flagged ONLY via a 2x harmonic tile up in its uniform
    # pitch-20 region (consensus 1.0, ratio 2.0) -- never the real 20->60 boundary
    # -- so its assertion was dishonest (it asserted on the exact artifact the
    # harmonic guard now, correctly, removes). This taller field exercises the
    # actual boundary.
    top = glyph_rows((512, 768), row_pitch=20, seed=1)
    bot = glyph_rows((512, 768), row_pitch=60, seed=2)
    f = np.vstack([top, bot])
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    # The decision must still produce at least one flag after the harmonic guard.
    assert flag_spacing(feats).sum() >= 1
    # ...and that surviving flag must be a HONEST split-consensus jump, not a 2x
    # harmonic: at least one passing tile has split consensus (< harmonic_consensus
    # 0.85) and a non-2x ratio. (The unanimous 2x harmonics in the uniform region
    # are exactly what the guard now removes, so they must NOT be the only signal.)
    passing = _spacing_decision_trace(feats)
    assert passing, "expected at least one tile past the consensus+deviation gate"
    genuine = [t for t in passing if t["consensus"] < 0.85 and round(t["ratio"]) != 2]
    assert genuine, (
        "jump test is not honest: every passing tile is a unanimous 2x harmonic; "
        f"passing tiles = {passing}")


def test_spacing_2x_harmonic_in_uniform_text_does_not_flag():
    # NEGATIVE guard pin (the demo false-positive signature): a lone tile reading
    # EXACTLY 2x the surrounding pitch inside a perfectly uniform field is a 2x
    # autocorrelation harmonic (period doubling), NOT a spacing change. Its
    # neighbourhood is unanimous (consensus ~1.0) and its ratio rounds to 2 -> the
    # harmonic guard must suppress it. This is precisely the demo a/c/d false
    # positive (tile (6,2): pitch 80 in a field of 40, consensus 1.000, ratio
    # 2.000). NOTE: this is NOT vacuous -- WITHOUT the guard this field flags 1
    # tile (the bare deviation rule fires on +100%); the guard takes it to 0.
    n = 6
    uniform = np.full((n, n), 40.0)
    uniform[3, 3] = 80.0                              # exact 2x second-peak harmonic
    assert flag_spacing(_synthetic_feats(uniform)).sum() == 0


def test_spacing_genuine_2x_split_consensus_jump_still_flags():
    # POSITIVE guard pin: a REAL spacing jump survives the guard EVEN when the
    # pitch ratio is exactly 2x, because its consensus is SPLIT (below the
    # harmonic_consensus 0.85 AND-clause). The guard keys on BOTH unanimity AND
    # round(ratio)==2; a split-consensus 2x jump fails the unanimity clause and so
    # must still flag. Center tile (3,3) reads 80 in a field of 40, but a handful
    # of other 80s sit inside its 5x5 window -> its consensus drops below 0.85
    # while the median stays 40 and the ratio is 2.0. It must flag; the corner 80s,
    # whose windows ARE near-unanimous, are suppressed as harmonics.
    n = 7
    field = np.full((n, n), 40.0)
    field[3, 3] = 80.0
    for cell in [(1, 1), (1, 5), (5, 1), (5, 5), (2, 4)]:
        field[cell] = 80.0                            # break unanimity around (3,3)
    flags = flag_spacing(_synthetic_feats(field))
    assert flags[3, 3], "split-consensus 2x jump at (3,3) must still flag"
    # confirm the survivor really is a split-consensus 2x case (not unanimous):
    passing = _spacing_decision_trace(_synthetic_feats(field))
    center = [t for t in passing if (t["r"], t["c"]) == (3, 3)]
    assert center and center[0]["consensus"] < 0.85 and round(center[0]["ratio"]) == 2, \
        f"center should be split-consensus 2x, got {center}"
    # and the near-unanimous 2x corners must stay suppressed as harmonics:
    flagged = list(zip(*np.where(flags)))
    assert (1, 1) not in flagged and (5, 5) not in flagged, \
        f"near-unanimous 2x harmonics must stay suppressed, got {flagged}"


def _feats(field):
    return analyze_tiles(field, np.ones(field.shape, bool), tile=128, overlap=0.5)


def test_garble_quiet_on_clean_text():
    feats = _feats(glyph_rows((512, 512), row_pitch=40))
    assert flag_garble(feats).sum() == 0


def test_garble_fires_on_garbled_patch():
    f = garble_patch(glyph_rows((512, 512), row_pitch=40), 192, 384, 192, 384)
    feats = _feats(f)
    assert flag_garble(feats).sum() >= 1


def test_orientation_flag_quiet_on_clean_text():
    feats = _feats(glyph_rows((512, 512), row_pitch=40))
    assert flag_orientation(feats).sum() <= 1


def test_orientation_flag_fires_on_rotated_band():
    # 20deg break: within the per-tile orientation search range (+-25deg around
    # the local seed). A larger break would exceed the bounded search by design
    # -- text orientation varies smoothly across a real trace, not +-40deg.
    f = rotate_band(glyph_rows((512, 512), row_pitch=40), 200, 320, ddeg=20)
    feats = _feats(f)
    assert flag_orientation(feats).sum() >= 1


def test_trace_health_high_on_text_low_on_garbled():
    clean = _feats(glyph_rows((512, 512), row_pitch=40))
    clean_rep = trace_health(clean, flag_tiles(clean))
    assert isinstance(clean_rep, ScoreReport)
    assert clean_rep.score >= 80
    bad = _feats(garble_patch(glyph_rows((512, 512), row_pitch=40), 128, 384, 128, 384))
    bad_rep = trace_health(bad, flag_tiles(bad))
    assert bad_rep.score < clean_rep.score
    assert 0 <= bad_rep.score <= 100


def test_trace_health_all_low_confidence_is_not_healthy():
    feats = _feats(np.zeros((512, 512)))
    assert not feats.confidence.any()
    rep = trace_health(feats, flag_tiles(feats))
    assert rep.low_conf_frac == 1.0
    assert rep.score == 0


def test_input_warning_flags_noise_not_text():
    noise = np.random.default_rng(0).random((512, 512))
    nf = analyze_tiles(noise, np.ones(noise.shape, bool), tile=128, overlap=0.5)
    assert input_warning(nf, flag_tiles(nf)) is not None
    cf = _feats(glyph_rows((512, 512), row_pitch=40))
    assert input_warning(cf, flag_tiles(cf)) is None


def test_input_warning_fires_on_heavily_rotated_text():
    # REGIME VIOLATION (user-reported on a real GP-banner label set): the
    # orientation search sweeps only +-25deg, so text rotated far beyond that
    # pegs the search boundary tile after tile, and every downstream detector
    # (pitch, spacing, the VERTICAL-strip seam scan) measures across the wrong
    # axis -- the real segment collected 50 'seam' flags that were artifacts.
    # The tool must confess instead of emitting confident flags: structured
    # tiles railing the sweep => loud input warning naming rotation.
    f = glyph_rows((768, 768), row_pitch=40, angle=np.radians(60), seed=4)
    feats = analyze_tiles(f)
    w = input_warning(feats, flag_tiles(feats))
    assert w is not None and "rotat" in w.lower()


def test_input_warning_quiet_on_modest_skew():
    # 10deg is comfortably inside the supported range: no rotation warning.
    f = glyph_rows((768, 768), row_pitch=40, angle=np.radians(10), seed=4)
    feats = analyze_tiles(f)
    w = input_warning(feats, flag_tiles(feats))
    assert w is None or "rotat" not in w.lower()


def test_input_warning_fires_on_rotation_with_spurious_interior_skew():
    # THE REAL-WORLD FAILURE (the rotated GP-banner label set): on sparse
    # rotated text the +-25deg skew search does NOT rail -- it finds a
    # spurious INTERIOR angle (the real segment read 7.5deg; this fixture
    # reads -15deg), so a boundary-rail fingerprint misses it entirely. The
    # reconnaissance must sweep the FULL range: the true angle (75deg) wins
    # there WITH strong row periodicity (pitch strength ~0.87) -- while an
    # upright fragment whose full-range sweep is fooled by its own outline
    # (frag1: 84deg 'winner') shows NO periodicity at that angle (0.00).
    f = glyph_rows((1024, 1024), row_pitch=60, glyph=24, gap=12, fill=0.5,
                   angle=np.radians(75), seed=5)
    feats = analyze_tiles(f)
    w = input_warning(feats, flag_tiles(feats))
    assert w is not None and "rotat" in w.lower()


def test_spacing_consensus_gate_keeps_real_jump_drops_scatter():
    from plumbline.model import TileFeatures, Tile
    rng = np.random.default_rng(0)
    n = 6
    tiles = [Tile(r, c, r * 10, r * 10 + 10, c * 10, c * 10 + 10)
             for r in range(n) for c in range(n)]
    def feats(pitch):
        return TileFeatures(n, n, np.zeros((n, n)), np.ones((n, n)), pitch,
                            np.full((n, n), 0.5), np.ones((n, n)),
                            np.ones((n, n), bool), tiles)
    consistent = np.full((n, n), 40.0)
    consistent[3, 3] = 120.0   # deviant in an agreeing field
    assert flag_spacing(feats(consistent)).sum() >= 1               # consensus exists -> flag
    # Checkerboard of two well-separated pitch bands: every window is ~50% low /
    # ~50% high -> neighbor consensus always ~0.5, well below the 0.70 gate.
    # This is more robust than a uniform-random draw (which can accidentally
    # produce a cluster of similar-pitch neighbors around a low outlier).
    rng = np.random.default_rng(42)
    lo = rng.uniform(150.0, 250.0, size=(n, n))
    hi = rng.uniform(500.0, 600.0, size=(n, n))
    checker = (np.arange(n)[:, None] + np.arange(n)[None, :]) % 2
    scattered = np.where(checker, lo, hi)                          # no agreed pitch
    assert flag_spacing(feats(scattered)).sum() == 0               # no consensus -> no flag


# --- Seam (pure vertical sheet-jump) detector -------------------------------
# A splice_shift shifts everything right of x_split vertically by dy WITHOUT
# rotating or changing row pitch, so orientation/spacing/garble all miss it. The
# seam detector must FIRE on that and stay quiet on every clean/perturbed-but-
# not-seam field. Thresholds (module defaults): seam_frac=0.30, row_gate=0.10,
# corr_gate=0.35, isolation_cap=max(2, round(0.05*n_boundaries)), overlap=0.70.

def _seam(field):
    """Run the real analysis path and return (feats, theta, pitch) for flag_seam.
    pitch is the per-tile MEDIAN (GRAFT 1) -- a sheet-jump corrupts a global
    single-profile pitch, so the per-tile median is what keeps dy~pitch/2 firing."""
    mask = np.ones(field.shape, bool)
    feats = analyze_tiles(field, mask)                 # auto tile (real path)
    _, theta = estimate_scale_and_skew(field, mask)
    pitch = (float(np.nanmedian(feats.pitch[feats.confidence]))
             if feats.confidence.any() else float("nan"))
    return feats, theta, pitch


def _column_warp(field, amp, n_periods):
    """Smooth column-wise vertical wander (NOT a seam): each column is rolled by a
    sinusoid of amplitude `amp` px over `n_periods` cycles across the width. Models
    bowed/curved papyrus rows -- the false-positive vector that kills phase designs."""
    h, w = field.shape
    out = np.empty_like(field)
    shifts = (amp * np.sin(2 * np.pi * n_periods * np.arange(w) / w)).round().astype(int)
    for x in range(w):
        out[:, x] = np.roll(field[:, x], int(shifts[x]))
    return out


def test_seam_fires_on_splice():
    for dy in (14, 18):
        f = splice_shift(glyph_rows((512, 768), row_pitch=40, seed=3), x_split=384, dy=dy)
        feats, th, p = _seam(f)
        g = flag_seam(feats, f, th, p)
        assert g.sum() >= 1, f"seam missed splice dy={dy}"
        # a flagged tile must straddle the seam column (~x=384)
        straddle = [t for t in feats.tiles
                    if g[t.row, t.col] and t.x0 <= 384 <= t.x1]
        assert straddle, f"flag not at the seam for dy={dy}"


def test_seam_fires_on_half_pitch_jump():
    # THE GRAFT REGRESSION GUARD: dy == pitch/2 is the case that MISSES if pitch is
    # taken from a global single profile (the splice collapses that estimate). The
    # per-tile-median pitch keeps it firing.
    for pitch, dy in ((40, 20), (60, 30)):
        f = splice_shift(glyph_rows((512, 768), row_pitch=pitch, seed=3), x_split=384, dy=dy)
        feats, th, p = _seam(f)
        assert flag_seam(feats, f, th, p).sum() >= 1, f"seam missed dy=pitch/2 (pitch={pitch})"


def test_seam_quiet_on_clean_glyph_rows():
    for seed in (0, 1, 2, 5, 7, 11):
        for pitch in (28, 40, 52, 64):
            f = glyph_rows((512, 768), row_pitch=pitch, seed=seed)
            feats, th, p = _seam(f)
            assert flag_seam(feats, f, th, p).sum() == 0, \
                f"false seam on clean text seed={seed} pitch={pitch}"


def test_seam_quiet_on_smooth_warp():
    # Bowed rows (curvature) must NOT read as a seam -- this is the FP vector that
    # killed the phase-based designs. Relative offsets stay small under smooth warp.
    base = glyph_rows((512, 768), row_pitch=40, seed=3)
    for amp_frac in (0.30, 0.50):
        f = _column_warp(base, amp=amp_frac * 40, n_periods=1)
        feats, th, p = _seam(f)
        assert flag_seam(feats, f, th, p).sum() == 0, \
            f"false seam on smooth warp amp={amp_frac}p"


def test_seam_quiet_on_sharp_fold():
    # High-frequency wander = a wall of offset spikes -> isolation cap suppresses all.
    base = glyph_rows((512, 768), row_pitch=40, seed=3)
    f = _column_warp(base, amp=40, n_periods=4)
    feats, th, p = _seam(f)
    assert flag_seam(feats, f, th, p).sum() == 0


def test_seam_quiet_on_rotate_band():
    f = rotate_band(glyph_rows((512, 768), row_pitch=40, seed=3), 200, 320, ddeg=20)
    feats, th, p = _seam(f)
    assert flag_seam(feats, f, th, p).sum() == 0


def test_seam_quiet_on_garble():
    f = garble_patch(glyph_rows((512, 768), row_pitch=40, seed=3), 192, 384, 192, 384)
    feats, th, p = _seam(f)
    assert flag_seam(feats, f, th, p).sum() == 0


def test_seam_quiet_on_noise():
    f = np.random.default_rng(0).random((512, 768))
    feats, th, p = _seam(f)
    assert flag_seam(feats, f, th, p).sum() == 0


def test_seam_quiet_when_no_ink():
    f = glyph_rows((512, 768), row_pitch=40, seed=3)
    feats, th, p = _seam(f)
    assert flag_seam(feats, None, th, p).sum() == 0        # no ink -> no-op
    z = np.zeros((512, 768))
    fz, thz, pz = _seam(z)
    assert flag_seam(fz, z, thz, pz).sum() == 0            # blank -> nan pitch -> no-op


def test_seam_quiet_on_skewed_clean():
    f = glyph_rows((512, 768), row_pitch=40, angle=np.radians(6), seed=3)
    feats, th, p = _seam(f)
    assert flag_seam(feats, f, th, p).sum() == 0


def test_flag_tiles_includes_seam_and_is_backcompat():
    f = splice_shift(glyph_rows((512, 768), row_pitch=40, seed=3), x_split=384, dy=14)
    feats, th, p = _seam(f)
    flags = flag_tiles(feats, ink=f, theta=th, pitch=p)
    assert flags.seam_break is not None and flags.seam_break.any()
    # seam adds genuinely NEW flagged tiles beyond the other three modes
    other = (flag_orientation(feats) | flag_spacing(feats) | flag_garble(feats))
    assert (flags.any_flag & feats.confidence).sum() > (other & feats.confidence).sum()
    # back-compat: no ink arg -> seam grid all-False (guards existing positional callers)
    plain = flag_tiles(feats)
    assert not plain.seam_break.any()


def test_trace_health_counts_seam():
    spliced = splice_shift(glyph_rows((512, 768), row_pitch=40, seed=3), x_split=384, dy=14)
    feats, th, p = _seam(spliced)
    rep = trace_health(feats, flag_tiles(feats, ink=spliced, theta=th, pitch=p))
    assert rep.n_seam > 0
    clean = glyph_rows((512, 768), row_pitch=40, seed=3)
    cf, cth, cp = _seam(clean)
    crep = trace_health(cf, flag_tiles(cf, ink=clean, theta=cth, pitch=cp))
    assert crep.n_seam == 0
    assert rep.score < crep.score        # a detected seam lowers the score
