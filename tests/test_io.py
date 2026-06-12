import numpy as np
import pytest
from PIL import Image
from plumbline.io import load_image01, load_input01, load_mask


def test_decompression_bomb_cap_is_finite_but_huge():
    # MAX_IMAGE_PIXELS=None disabled PIL's decompression-bomb guard for EVERY
    # consumer in the process; a large finite cap admits real scroll segments
    # (largest seen ~237 MP) with ~10x margin while keeping the protection.
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS >= 2_000_000_000


def test_load_image01_warns_when_16bit_data_is_underexposed(tmp_path):
    # 8-bit data saved in a 16-bit container (common pipeline artifact): to01
    # divides by 65535, the image lands in [0, 0.004], every tile fails the
    # ink-density gate, and the segment silently scores 0 with no hint why.
    import tifffile
    arr = np.full((64, 64), 200, dtype=np.uint16)
    p = tmp_path / "pred.tif"
    tifffile.imwrite(str(p), arr)
    with pytest.warns(UserWarning, match="1%"):
        load_image01(str(p))


def test_load_input01_rejects_remote_non_zarr():
    # A remote URL that isn't a Zarr store fell through to PIL, which cannot
    # open URLs -> confusing PIL error. Fail with a clear message instead.
    with pytest.raises(ValueError, match="[Rr]emote"):
        load_input01("https://example.com/prediction.png")


def test_load_image01_roundtrip(tmp_path):
    arr = (np.linspace(0, 255, 64 * 64).reshape(64, 64)).astype(np.uint8)
    p = tmp_path / "ink.png"
    Image.fromarray(arr).save(p)
    out = load_image01(str(p))
    assert out.shape == (64, 64)
    assert out.dtype == np.float32   # half the memory of float64 on huge segments
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_load_mask_is_bool(tmp_path):
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 8:24] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(arr).save(p)
    m = load_mask(str(p))
    assert m.dtype == bool
    assert m.sum() == 16 * 16
