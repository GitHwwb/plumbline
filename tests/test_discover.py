import numpy as np
from PIL import Image
from plumbline.discover import find_segments


def _png(path, val=200, size=(32, 32)):
    Image.fromarray(np.full(size, val, dtype=np.uint8)).save(path)


def test_find_segments_picks_ink_and_mask(tmp_path):
    s1 = tmp_path / "seg1"; s1.mkdir()
    _png(s1 / "seg1_prediction.png"); _png(s1 / "seg1_mask.png", 255)
    s2 = tmp_path / "seg2"; s2.mkdir()
    _png(s2 / "result.png")                      # no mask
    notes = tmp_path / "notes"; notes.mkdir()
    (notes / "readme.txt").write_text("hi")      # no image -> skipped

    segs = find_segments(str(tmp_path))
    assert [s.seg_id for s in segs] == ["seg1", "seg2"]   # sorted, notes skipped
    assert segs[0].ink_path.endswith("seg1_prediction.png")
    assert segs[0].mask_path.endswith("seg1_mask.png")
    assert segs[1].ink_path.endswith("result.png")
    assert segs[1].mask_path is None


def test_find_segments_skips_folder_with_only_mask(tmp_path):
    s = tmp_path / "onlymask"; s.mkdir()
    _png(s / "x_mask.png", 255)
    assert find_segments(str(tmp_path)) == []
