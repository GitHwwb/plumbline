import numpy as np
from PIL import Image
from plumbline.io import load_image01, load_mask


def test_load_image01_roundtrip(tmp_path):
    arr = (np.linspace(0, 255, 64 * 64).reshape(64, 64)).astype(np.uint8)
    p = tmp_path / "ink.png"
    Image.fromarray(arr).save(p)
    out = load_image01(str(p))
    assert out.shape == (64, 64)
    assert out.dtype == np.float64
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_load_mask_is_bool(tmp_path):
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 8:24] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(arr).save(p)
    m = load_mask(str(p))
    assert m.dtype == bool
    assert m.sum() == 16 * 16
