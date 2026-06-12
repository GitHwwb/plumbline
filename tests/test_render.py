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


def test_overlay_outlines_use_mode_colors():
    # Dashboard thumbnails reuse overlay_png, which outlined every flagged
    # tile in red regardless of mode (user-caught on the demo dashboard).
    # Outlines must use the mode's palette colour: this fixture flags GARBLE,
    # so sky #5acbfa pixels must appear and the old all-red must not.
    from PIL import Image as PILImage
    import io as _io
    f, feats, flags = _setup()
    assert flags.garble.any() and not flags.orient_break.any()
    png = overlay_png(f, feats, flags)
    a = np.asarray(PILImage.open(_io.BytesIO(png)).convert("RGB"), dtype=int)

    def present(hexc, tol=40):
        rgb = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)])
        return bool((np.abs(a - rgb).sum(axis=2) < tol).any())

    assert present("#5acbfa"), "garble outlines must be sky"
    assert not present("#e23227"), "no red outlines on non-orientation flags"


def test_flagged_regions_lists_pixel_locations():
    f, feats, flags = _setup()
    regions = flagged_regions(feats, flags)
    assert isinstance(regions, list)
    assert regions, "garbled patch should produce at least one region"
    r = regions[0]
    assert {"x", "y", "mode"} <= set(r.keys())
    assert r["mode"] in {"orientation", "spacing", "garble"}


def test_flagged_regions_include_human_detail():
    # Hover tooltips need per-tile measurements in words: every region carries
    # a "detail" string with the tile's angle / row pitch / rowness / ink
    # fraction, so a reviewer can see WHY a box exists without reading JSON.
    f, feats, flags = _setup()
    for r in flagged_regions(feats, flags):
        assert "detail" in r and isinstance(r["detail"], str)
        assert "angle" in r["detail"] and "rowness" in r["detail"]


def test_ink_png_exact_extent():
    # The interactive overlay positions flag boxes in PERCENT coordinates, so
    # the base PNG must span exactly the array extent -- matplotlib's
    # bbox_inches='tight' adds pad_inches margins, which would skew every box.
    # ink_png is a pure-PIL render: pixel (0,0) IS array (0,0).
    from PIL import Image
    import io as _io
    from plumbline.render import ink_png
    a = np.zeros((100, 50)); a[10, :] = 1.0
    png = ink_png(a)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(_io.BytesIO(png))
    assert img.size == (50, 100)                  # PIL size is (w, h): exact extent
    big = np.zeros((4000, 1000))
    img2 = Image.open(_io.BytesIO(ink_png(big, max_px=2000)))
    assert max(img2.size) <= 2000                 # downsampled for display


def test_flagged_regions_include_tile_extents():
    # A JSON consumer needs the flagged BOX, not just its center: without
    # x0/y0/x1/y1 a downstream pipeline cannot draw or crop the flagged area
    # (the tile size isn't in the sidecar either).
    f, feats, flags = _setup()
    for r in flagged_regions(feats, flags):
        assert {"x0", "y0", "x1", "y1"} <= set(r.keys())
        assert r["x0"] <= r["x"] <= r["x1"]
        assert r["y0"] <= r["y"] <= r["y1"]
        assert r["x1"] - r["x0"] > 0 and r["y1"] - r["y0"] > 0


def test_quiver_row_direction_matches_display_convention():
    # User-caught on the -74deg segment: quiver arrows pointed NNE while the
    # actual text lines run NNW. The display-coordinate row direction for
    # codebase theta is (cos t, -SIN t) -- pinned analytically by the rotated
    # seam-geometry test -- but the quiver drew (cos t, +sin t), a mirror
    # about the horizontal (error = 2*theta: invisible upright, glaring when
    # rotated). The orientation view must use the same convention as the
    # seam mapping.
    from plumbline.render import _row_direction
    th = np.radians(-74.0)
    ux, vy = _row_direction(th)
    assert ux > 0 and vy > 0, "for theta=-74deg the on-screen line is NNW/SSE: +x, +y(down)"
    assert np.isclose(ux, np.cos(th)) and np.isclose(vy, -np.sin(th))


def test_display_downsample_keeps_thin_strokes():
    # Stride downsampling (a[::k, ::k]) keeps every k-th row, so a 1-px stroke
    # survives only if its row index happens to be a multiple of k -- strokes
    # flicker in and out of the report previews. Block-mean pooling keeps them
    # visible (dimmed in proportion, which is the honest rendering).
    from plumbline.render import _downsample_mean
    a = np.zeros((100, 100)); a[3, :] = 1.0       # stroke on a non-multiple row
    d = _downsample_mean(a, 4)
    assert d.shape == (25, 25)
    assert d.max() > 0, "thin stroke must survive display downsampling"
    assert np.allclose(_downsample_mean(a, 1), a)  # k<=1 is the identity


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
