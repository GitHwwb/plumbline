import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles
from plumbline.render import overlay_png, heatmap_png, orientation_png, flags_png, flagged_regions


def _setup():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    return f, feats, flag_tiles(feats)


def test_png_renderers_return_png_bytes():
    f, feats, flags = _setup()
    for png in (overlay_png(f, feats, flags), heatmap_png(feats),
                orientation_png(feats), flags_png(f, feats, flags)):
        assert isinstance(png, (bytes, bytearray))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_flagged_regions_lists_pixel_locations():
    f, feats, flags = _setup()
    regions = flagged_regions(feats, flags)
    assert isinstance(regions, list)
    assert regions, "garbled patch should produce at least one region"
    r = regions[0]
    assert {"x", "y", "mode"} <= set(r.keys())
    assert r["mode"] in {"orientation", "spacing", "garble"}


def test_coherence_display_bounded_and_compressed():
    from plumbline.render import _coherence_display
    bands = np.array([0.0, 0.05, 0.9, 3.0, 50.0])   # empty, noise, text, sliver, extreme
    v = _coherence_display(bands)                      # bare band curve (no density gate)
    assert v.min() >= 0.0 and v.max() <= 1.0          # bounded to [0,1]
    assert np.all(np.diff(v) >= 0)                     # monotonic in band
    assert v[1] < 0.1                                  # noise-level -> near 0
    assert v[3] >= v[2]                                 # sliver still >= text...
    assert (v[3] - v[2]) < (3.0 - 0.9)                 # ...but gap is compressed
    assert (v[4] - v[3]) < 0.05                        # saturates: 3 vs 50 nearly equal


def test_coherence_display_density_gate_dims_slivers():
    # OPEN ITEM A: a near-empty single-band sliver tile (low density) must be DIMMED
    # in the display so it stops reading as MORE coherent than real dense text. The
    # gate keys off density, not band: identical band, different density -> different
    # displayed coherence. (Display-only; band_strength/score are untouched.)
    from plumbline.render import _coherence_display
    band = np.array([1.2, 1.2])                        # SAME rowness on both tiles
    density = np.array([0.03, 0.30])                   # sliver vs real dense text
    v = _coherence_display(band, density)
    assert v[0] < v[1]                                 # sliver dimmer than dense text
    assert v[1] > 0.9                                  # dense text -> bright, untouched
    # density=None path leaves the bare band curve unchanged (what test_render exercises)
    assert np.allclose(_coherence_display(band), _coherence_display(band, None))


def test_coherence_display_does_not_black_out_sparse_text():
    # REGRESSION (adversarial review): density alone cannot tell a sliver artifact
    # from GENUINELY-banded sparse text (giant Greek text sits ~density 0.06 with
    # high band, and the SCORER counts it healthy -- flag_garble keeps band>=0.10).
    # The density dim must have a FLOOR so such tiles stay VISIBLE on the heatmap,
    # not blacked out (the opposite-direction false reassurance an earlier hard gate
    # produced). Item A still holds (sliver < dense) but sparse text must read clearly.
    from plumbline.render import _coherence_display, _COH_D_FLOOR
    band = np.array([0.9, 0.75, 1.2])                  # genuinely banded
    density = np.array([0.06, 0.03, 0.30])             # sparse, sliver, dense
    v = _coherence_display(band, density)
    assert v[0] > 0.3, "genuinely-banded sparse text must not be blacked out"
    assert v[1] > 0.3, "even a low-density tile stays visible (floor), never ~0"
    assert v[0] < v[2] and v[1] < v[2]                 # still dimmer than dense text
    # the dimming floor bounds how dark any banded low-density tile can get
    assert v[1] >= _COH_D_FLOOR * np.tanh((0.75 - 0.06) / 0.12) - 1e-9


def test_coherence_display_is_display_only():
    # The density gate must not feed back into band_strength / flags / score.
    from plumbline.render import _coherence_display, heatmap_png
    from plumbline.score import flag_tiles, trace_health
    f, feats, flags = _setup()
    band_before = feats.band_strength.copy()
    rep_before = trace_health(feats, flag_tiles(feats))
    _ = heatmap_png(feats)                              # render uses the density gate
    rep_after = trace_health(feats, flag_tiles(feats))
    assert np.array_equal(band_before, feats.band_strength)   # band untouched
    assert rep_before.score == rep_after.score                # score untouched
