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


def test_find_segments_picks_zarr_store_as_ink(tmp_path):
    # `run` accepts Zarr anywhere it accepts a PNG, but batch discovery only
    # globbed image-file extensions -- a segments folder of Zarr stores was
    # silently skipped. Stores join the same name-pattern matching as images.
    import pytest
    zarr = pytest.importorskip("zarr")
    s = tmp_path / "segz"; s.mkdir()
    z = zarr.open(str(s / "prediction.zarr"), mode="w", shape=(64, 64),
                  dtype="uint8")
    z[:] = 100
    _png(s / "segz_mask.png", 255)
    segs = find_segments(str(tmp_path))
    assert [x.seg_id for x in segs] == ["segz"]
    assert segs[0].ink_path.endswith("prediction.zarr")
    assert segs[0].mask_path.endswith("segz_mask.png")
