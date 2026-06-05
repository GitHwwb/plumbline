import numpy as np
from plumbline.util import to01, wrap_angle


def test_to01_normalizes_uint8():
    a = np.array([[0, 255], [128, 255]], dtype=np.uint8)
    out = to01(a)
    assert out.dtype == np.float64
    assert out.min() == 0.0 and out.max() == 1.0


def test_to01_collapses_rgb():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to01(rgb).shape == (4, 4)


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
