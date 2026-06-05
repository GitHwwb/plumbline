import plumbline
import numpy as np
from plumbline.model import Tile, TileFeatures, FlagMap, ScoreReport


def test_version_present():
    assert plumbline.__version__ == "0.1.0"


def test_model_dataclasses_construct():
    t = Tile(row=0, col=1, y0=0, y1=256, x0=256, x1=512)
    assert (t.row, t.col, t.x0) == (0, 1, 256)

    z = np.zeros((2, 2))
    feats = TileFeatures(
        n_rows=2, n_cols=2, theta=z.copy(), band_strength=z.copy(),
        pitch=z.copy(), pitch_strength=z.copy(), density=z.copy(),
        confidence=z.astype(bool), tiles=[t],
    )
    assert feats.n_rows == 2 and feats.tiles[0] is t

    flags = FlagMap(
        orient_break=z.astype(bool), spacing_break=z.astype(bool),
        garble=z.astype(bool),
    )
    assert flags.any_flag.shape == (2, 2)
    assert not flags.any_flag.any()

    rep = ScoreReport(score=100, n_orient=0, n_spacing=0, n_garble=0, low_conf_frac=0.0)
    assert rep.score == 100


def test_segment_inputs_and_index_row():
    from plumbline.model import SegmentInputs, IndexRow
    s = SegmentInputs(seg_id="abc", ink_path="/a/ink.png", mask_path=None)
    assert s.seg_id == "abc" and s.mask_path is None
    r = IndexRow(seg_id="abc", score=73, n_orient=0, n_spacing=2, n_garble=1,
                 low_conf_frac=0.3, report_filename="abc.html", thumb_b64="xx", error=None)
    assert r.score == 73 and r.report_filename == "abc.html" and r.error is None
