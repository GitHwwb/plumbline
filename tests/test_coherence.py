import numpy as np
from plumbline.util import to01, wrap_angle


def test_to01_normalizes_uint8():
    a = np.array([[0, 255], [128, 255]], dtype=np.uint8)
    out = to01(a)
    # float32: full precision for [0,1] intensities at HALF the memory of
    # float64 -- a 237 MP segment is ~0.9 GB instead of ~1.9 GB.
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0


def test_to01_collapses_rgb():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to01(rgb).shape == (4, 4)


def test_analyze_tiles_adopts_decisive_rotated_text_direction():
    # When the full-range reconnaissance finds the writing direction beyond
    # the +-25deg sweep WITH real row periodicity there, analyze_tiles adopts
    # it as gtheta -- per-tile sweeps then seed at the true angle and every
    # downstream detector works in the text's frame (user direction: rotate
    # the analyzer, not the image). In-regime inputs keep the old estimate
    # byte-identically.
    from plumbline.synthetic import glyph_rows
    from plumbline.coherence import analyze_tiles
    f = glyph_rows((1024, 1024), row_pitch=60, glyph=24, gap=12, fill=0.5,
                   angle=np.radians(75), seed=5)
    feats = analyze_tiles(f)
    assert abs(abs(np.degrees(feats.gtheta)) - 75) < 8, \
        f"gtheta {np.degrees(feats.gtheta):.1f} should adopt ~±75°"
    # upright text: gtheta stays the in-regime estimate (near 0)
    up = glyph_rows((512, 512), row_pitch=40, seed=2)
    assert abs(np.degrees(analyze_tiles(up).gtheta)) < 5


def test_band_contrast_robust_to_photo_background_glow():
    # A PHOTOGRAPH (e.g. the frag1 infrared) has the papyrus itself glowing
    # mid-gray, contributing most of the tile mean; std/mean then reads CLEAR
    # letter rows as structureless (measured on frag1: rows detected at pitch
    # 199px, yet band 0.071 < 0.10 -> garble flags on legible Greek). Rowness
    # must judge the ink ABOVE the background glow, not the glow itself.
    from plumbline.synthetic import glyph_rows
    from plumbline.coherence import band_contrast
    rng = np.random.default_rng(0)
    text = glyph_rows((512, 512), row_pitch=40, seed=2)
    glow_text = np.clip(0.40 + 0.15 * text + rng.normal(0, 0.02, text.shape), 0, 1)
    assert band_contrast(glow_text) >= 0.10, \
        "legible rows over a photo glow must not read as garble"


def test_band_contrast_glow_does_not_unmask_noise():
    # The background subtraction must NOT rescue structureless mottle: dense
    # noise over the same glow stays below the garble line (both the garble
    # detector and the 'may not be an ink prediction' warning depend on this).
    from plumbline.coherence import band_contrast
    rng = np.random.default_rng(1)
    glow_noise = np.clip(0.40 + 0.45 * rng.random((512, 512)), 0, 1)
    assert band_contrast(glow_noise) < 0.10


def test_ink_density_counts_ink_above_parchment_not_parchment():
    # On a PHOTOGRAPH the parchment glow itself clears the fixed 0.25 cut, so
    # the tooltip read 'ink 76%' on a frag1 tile that is mostly bare parchment
    # -- the number meant '76% of the tile is not void'. Density must count
    # the layer ABOVE the background: letters a notch brighter than parchment.
    from plumbline.synthetic import glyph_rows
    from plumbline.coherence import ink_density
    rng = np.random.default_rng(0)
    text = glyph_rows((512, 512), row_pitch=40, seed=2)
    true_ink = ink_density(text)                      # ground truth on the prediction
    photo = np.clip(0.40 + 0.25 * text + rng.normal(0, 0.02, text.shape), 0, 1)
    d = ink_density(photo)
    assert abs(d - true_ink) < 0.10, f"photo ink {d:.2f} should track true ink {true_ink:.2f}"
    assert d < 0.5, "parchment glow must not be counted as ink"


def test_ink_density_featureless_parchment_reads_near_zero():
    # Bare parchment with no letters: there is no brighter layer, so density
    # must be LOW (such tiles should fall toward low-confidence, the honest
    # 'too little ink to judge' outcome) -- not ~100% because gray > 0.25.
    from plumbline.coherence import ink_density
    rng = np.random.default_rng(1)
    parchment = np.clip(0.40 + rng.normal(0, 0.02, (512, 512)), 0, 1)
    assert ink_density(parchment) < 0.10


def test_ink_density_prediction_path_is_byte_identical():
    # Predictions (near-black background) keep the EXACT fixed-threshold
    # behaviour: the 0.02 confidence gate and the display-dimming curve
    # (D_LO/D_HI) were calibrated on it.
    from plumbline.synthetic import glyph_rows
    from plumbline.coherence import ink_density
    text = glyph_rows((512, 512), row_pitch=40, seed=2)
    assert ink_density(text) == float((text > 0.25).mean())


def test_band_contrast_prediction_path_is_byte_identical():
    # Ink predictions (background ~ 0) must take the EXACT old code path:
    # every calibrated threshold (garble 0.10, the seam detector's row_gate,
    # the coherence display curve) was tuned on this definition -- including
    # the profile= fast path the seam scan relies on.
    from plumbline.synthetic import glyph_rows
    from plumbline.coherence import band_contrast, projection_profile, _linear_detrend
    text = glyph_rows((512, 512), row_pitch=40, seed=2)
    p = projection_profile(text, 0.0)
    old = float(_linear_detrend(p).std() / max(float(p.mean()), 0.08))
    assert abs(band_contrast(text) - old) < 1e-12
    assert abs(band_contrast(text, 0.0, profile=p) - old) < 1e-12


def test_to01_normalizes_uint16():
    # 16-bit inputs (real ink predictions) must scale by 65535, not 255,
    # or bright pixels overflow and clip to 1.0 (everything reads as ink).
    a = np.array([[0, 65535], [32768, 65535]], dtype=np.uint16)
    out = to01(a)
    assert out.max() == 1.0
    assert abs(out[1, 0] - 0.5) < 0.01


def test_wrap_angle_mod_pi():
    assert abs(wrap_angle(np.pi)) < 1e-9
    assert abs(wrap_angle(np.pi / 2) - np.pi / 2) < 1e-9 or \
           abs(wrap_angle(np.pi / 2) + np.pi / 2) < 1e-9


from plumbline.synthetic import glyph_rows
from plumbline.coherence import ink_density, analyze_tiles
from plumbline.model import TileFeatures
from plumbline.coherence import (projection_profile, band_contrast,
                                 dominant_orientation, row_pitch,
                                 estimate_scale_and_skew)


def test_ink_density_blank_vs_dense():
    assert ink_density(np.zeros((64, 64))) < 0.05
    assert ink_density(np.ones((64, 64))) > 0.9


def test_analyze_tiles_text_is_banded_noise_is_not():
    text = glyph_rows((512, 512), row_pitch=40)
    tf = analyze_tiles(text, tile=256, overlap=0.5)
    assert isinstance(tf, TileFeatures)
    assert tf.theta.shape == (tf.n_rows, tf.n_cols)
    text_band = np.median(tf.band_strength[tf.confidence])

    noise = np.random.default_rng(7).random((512, 512))
    nf = analyze_tiles(noise, tile=256, overlap=0.5)
    noise_band = np.median(nf.band_strength[nf.confidence])
    assert text_band > 1.5 * noise_band


def test_analyze_tiles_auto_picks_tile_in_bounds():
    text = glyph_rows((1024, 1024), row_pitch=48)
    tf = analyze_tiles(text, tile=None)      # auto-adapt
    assert tf.confidence.any()


def test_band_contrast_text_beats_noise():
    text = glyph_rows((512, 512), row_pitch=40)
    noise = np.random.default_rng(1).random((512, 512))
    assert band_contrast(text, 0.0) > 2 * band_contrast(noise, 0.0)


def test_dominant_orientation_recovers_angle():
    for deg in (0, 12, -20):
        f = glyph_rows((512, 512), row_pitch=40, angle=np.radians(deg))
        est = np.degrees(dominant_orientation(f, seed=0.0))
        assert abs(est - deg) < 6, f"deg={deg} -> est={est:.1f}"


def test_row_pitch_detects_known_spacing():
    f = glyph_rows((512, 512), row_pitch=40)
    pitch, strength = row_pitch(f, 0.0, min_lag=10, max_lag=256)
    assert abs(pitch - 40) < 8
    assert strength > 0.0


def test_estimate_scale_and_skew_bounds_and_angle():
    f = glyph_rows((1024, 1024), row_pitch=48, angle=np.radians(8))
    tile, theta = estimate_scale_and_skew(f)
    assert 256 <= tile <= 2048
    assert abs(np.degrees(theta) - 8) < 8


def test_tile_tracks_row_pitch_not_sparsity():
    # tile size should follow ROW PITCH (line spacing), not density: a sparse row
    # (big inter-glyph gaps) at a given pitch must NOT get a bigger tile than a
    # dense row at the SAME pitch -- the old decay-length proxy inflated the sparse one.
    dense = glyph_rows((1536, 1536), row_pitch=80, glyph=40, gap=10, seed=1)
    sparse = glyph_rows((1536, 1536), row_pitch=80, glyph=40, gap=180, seed=2)
    td, _ = estimate_scale_and_skew(dense)
    ts, _ = estimate_scale_and_skew(sparse)
    assert 256 <= td <= 2048 and 256 <= ts <= 2048
    assert td / 1.5 <= ts <= td * 1.5     # sparse tracks the dense tile (same pitch), neither way


def test_tile_bigger_for_bigger_pitch():
    fine = glyph_rows((2048, 2048), row_pitch=60, glyph=28, gap=14, seed=1)
    coarse = glyph_rows((2048, 2048), row_pitch=240, glyph=110, gap=40, seed=2)
    tf, _ = estimate_scale_and_skew(fine)
    tc, _ = estimate_scale_and_skew(coarse)
    assert tc > tf                   # bigger line spacing -> bigger tile


def test_tile_giant_rows_stay_large():
    # a few GIANT rows must still yield a LARGE tile (via the row-pitch primary
    # path or the decay-length fallback) -- guards the giant-text case synthetically,
    # independent of the frag1-IR image fixture.
    giant = glyph_rows((2048, 2048), row_pitch=600, glyph=320, gap=120, seed=3)
    tile, _ = estimate_scale_and_skew(giant)
    assert tile >= 1024
