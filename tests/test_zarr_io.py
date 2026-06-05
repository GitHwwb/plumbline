import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
from plumbline.io import load_zarr01


def _write_array(path, data):
    z = zarr.open(str(path), mode="w", shape=data.shape,
                  chunks=True, dtype=data.dtype)
    z[:] = data
    return z


def test_load_zarr01_2d_float_passthrough(tmp_path):
    data = np.linspace(0.0, 1.0, 16 * 20, dtype="float32").reshape(16, 20)
    _write_array(tmp_path / "pred.zarr", data)
    out = load_zarr01(tmp_path / "pred.zarr")
    assert out.shape == (16, 20)
    assert out.dtype == np.float64
    assert np.allclose(out, data, atol=1e-6)   # already in [0,1] -> passthrough


def test_load_zarr01_2d_uint8_scaled(tmp_path):
    data = np.array([[0, 128, 255]], dtype="uint8")
    _write_array(tmp_path / "u8.zarr", data)
    out = load_zarr01(tmp_path / "u8.zarr")
    assert np.allclose(out, [[0.0, 128 / 255, 1.0]], atol=1e-6)


# --- Task 3: dispatcher + detection -----------------------------------------
from plumbline.io import load_input01, _is_zarr


def test_is_zarr_detection(tmp_path):
    _write_array(tmp_path / "x.zarr", np.zeros((8, 8), dtype="float32"))
    _write_array(tmp_path / "noext", np.zeros((8, 8), dtype="float32"))
    assert _is_zarr(tmp_path / "x.zarr") is True       # by suffix
    assert _is_zarr(tmp_path / "noext") is True         # by .zarray inside dir
    assert _is_zarr("s3://bucket/pred.zarr") is True     # remote suffix
    assert _is_zarr("prediction.png") is False
    assert _is_zarr("prediction.tif") is False


def test_load_input01_dispatch(tmp_path):
    data = np.full((8, 8), 0.5, dtype="float32")
    _write_array(tmp_path / "x.zarr", data)
    out = load_input01(tmp_path / "x.zarr")
    assert out.shape == (8, 8)
    assert np.allclose(out, 0.5, atol=1e-6)


# --- Task 4: 3-D volume handling --------------------------------------------
def test_load_zarr01_3d_requires_explicit_plane(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.random((4, 8, 8), dtype=np.float32)
    _write_array(tmp_path / "vol.zarr", vol)
    with pytest.raises(ValueError):
        load_zarr01(tmp_path / "vol.zarr")                       # ambiguous -> error
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", z=2), vol[2], atol=1e-6)
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", reduce="max"),
                       vol.max(axis=0), atol=1e-6)
    assert np.allclose(load_zarr01(tmp_path / "vol.zarr", reduce="mean"),
                       vol.mean(axis=0), atol=1e-6)


# --- Task 5: OME-Zarr multiscale group resolution ---------------------------
def test_load_zarr01_ome_multiscale_default_and_override(tmp_path):
    rng = np.random.default_rng(1)
    full = rng.random((16, 16), dtype=np.float32)
    half = np.ascontiguousarray(full[::2, ::2])
    g = zarr.open_group(str(tmp_path / "img.zarr"), mode="w")
    g.create_dataset("0", data=full)
    g.create_dataset("1", data=half)
    g.attrs["multiscales"] = [
        {"version": "0.4", "datasets": [{"path": "0"}, {"path": "1"}]}
    ]
    assert load_zarr01(tmp_path / "img.zarr").shape == (16, 16)             # default -> level 0
    assert load_zarr01(tmp_path / "img.zarr", component="1").shape == (8, 8)  # override


def test_load_zarr01_group_without_multiscales_needs_component(tmp_path):
    g = zarr.open_group(str(tmp_path / "bare.zarr"), mode="w")
    g.create_dataset("data", data=np.zeros((4, 4), dtype="float32"))
    with pytest.raises(ValueError):
        load_zarr01(tmp_path / "bare.zarr")
    assert load_zarr01(tmp_path / "bare.zarr", component="data").shape == (4, 4)


# --- Task 6: CLI routes run through load_input01 ----------------------------
def test_cli_run_accepts_zarr(tmp_path):
    from plumbline.cli import main
    rng = np.random.default_rng(2)
    data = (rng.random((64, 64)) > 0.5).astype("float32")
    _write_array(tmp_path / "pred.zarr", data)
    out = tmp_path / "report.html"
    rc = main(["run", str(tmp_path / "pred.zarr"), "-o", str(out), "--tile", "32"])
    assert rc == 0
    assert out.exists()
