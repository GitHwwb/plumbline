import os
import numpy as np
import pytest

IR = "data/frag1/ir.png"
MASK = "data/frag1/mask.png"


@pytest.mark.skipif(not os.path.exists(IR), reason="gitignored real IR image absent")
def test_real_ir_text_scores_and_is_not_blanketed():
    from PIL import Image
    from plumbline.coherence import analyze_tiles
    from plumbline.score import flag_tiles, trace_health
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(IR).convert("L"), dtype=np.float64) / 255.0
    ink = 1.0 - img                              # IR photo: invert so ink=high
    mask = None
    if os.path.exists(MASK):
        m = np.asarray(Image.open(MASK).convert("L")) > 127
        if m.shape == ink.shape:
            mask = m
    feats = analyze_tiles(ink, mask=mask, tile=None)   # auto-adapt scale
    flags = flag_tiles(feats)
    rep = trace_health(feats, flags)
    n_conf = int(feats.confidence.sum())
    assert n_conf > 0, "no analyzable tiles on real text"
    garble_frac = int((flags.garble & feats.confidence).sum()) / n_conf
    orient_frac = int((flags.orient_break & feats.confidence).sum()) / n_conf
    # The redesign's whole point: real legible text must NOT score ~0 nor be
    # blanketed in garble (the stripe core scored this exact image 0/100).
    assert rep.score > 25, f"score {rep.score} too low on real text"
    assert garble_frac < 0.5, f"garble blankets {garble_frac:.0%} of confident tiles"
    # ...and the orientation axis must not blanket it either: legible roughly-
    # horizontal text should yield a consistent orientation field, not a spray of
    # orient_break flags from the skew search saturating at its search boundary.
    assert orient_frac < 0.15, f"orient_break blankets {orient_frac:.0%} of confident tiles"
