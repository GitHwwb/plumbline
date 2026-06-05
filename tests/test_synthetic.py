import numpy as np
from plumbline.synthetic import striped_field, rotate_band, splice_shift, garble_patch
from plumbline.synthetic import glyph_rows


def test_striped_field_shape_and_range():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    assert f.shape == (256, 256)
    assert 0.0 <= f.min() and f.max() <= 1.0


def test_striped_field_is_periodic_along_rows_for_zero_angle():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    profile = f.mean(axis=1)            # average across columns -> vary along rows
    profile = profile - profile.mean()
    spec = np.abs(np.fft.rfft(profile))
    freqs = np.fft.rfftfreq(len(profile))
    peak_pitch = 1.0 / freqs[1:][np.argmax(spec[1:])]
    assert abs(peak_pitch - 20) < 3


def test_perturbations_preserve_shape_and_change_content():
    f = striped_field((256, 256), pitch=20, angle=0.0)
    assert rotate_band(f, 100, 160, ddeg=30).shape == f.shape
    assert splice_shift(f, x_split=128, dy=15).shape == f.shape
    g = garble_patch(f, 80, 140, 80, 140)
    assert g.shape == f.shape
    assert not np.allclose(g[80:140, 80:140], f[80:140, 80:140])


def test_glyph_rows_shape_and_range():
    f = glyph_rows((512, 512), row_pitch=40)
    assert f.shape == (512, 512)
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_glyph_rows_has_row_banding_unlike_noise():
    f = glyph_rows((512, 512), row_pitch=40)
    rows_profile = f.mean(axis=1)            # mean ink per image row
    noise = np.random.default_rng(0).random((512, 512)).mean(axis=1)
    # discrete rows + gaps -> projection profile oscillates far more than noise
    assert rows_profile.std() > 5 * noise.std()
