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
    assert out.dtype == np.float32   # half the memory of float64 on huge segments
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


# --- Chunked reading: never materialize more than needed ---------------------
class _RecordingArray:
    """Minimal zarr-array stand-in that records every __getitem__ request, so
    tests can assert the loader streams chunk-sized slabs instead of loading
    the whole store (the point of Zarr)."""
    def __init__(self, data, chunks):
        self._d = np.asarray(data)
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.chunks = chunks
        self.requests = []

    def __getitem__(self, key):
        self.requests.append(key)
        return self._d[key]


def _z_extent(key, depth):
    """How many planes of the first axis one request touches."""
    k = key[0] if isinstance(key, tuple) else key
    if isinstance(k, slice):
        return len(range(*k.indices(depth)))
    return 1


def test_read_zarr01_z_slice_reads_one_plane_not_the_volume():
    from plumbline.io import _read_zarr01
    rng = np.random.default_rng(0)
    vol = rng.random((40, 8, 8), dtype=np.float32)
    a = _RecordingArray(vol, chunks=(5, 8, 8))
    out = _read_zarr01(a, z=7)
    assert np.allclose(out, vol[7], atol=1e-6)
    assert max(_z_extent(k, 40) for k in a.requests) == 1   # never the whole volume


def test_read_zarr01_reduce_streams_chunk_slabs():
    from plumbline.io import _read_zarr01
    rng = np.random.default_rng(1)
    vol = rng.random((40, 8, 8), dtype=np.float32)
    for red, ref in (("max", vol.max(axis=0)), ("mean", vol.mean(axis=0))):
        a = _RecordingArray(vol, chunks=(5, 8, 8))
        out = _read_zarr01(a, reduce=red)
        assert np.allclose(out, ref, atol=1e-6), red
        assert max(_z_extent(k, 40) for k in a.requests) <= 5, red


def test_read_zarr01_2d_streams_row_slabs():
    from plumbline.io import _read_zarr01
    data = ((np.arange(64 * 8).reshape(64, 8) % 251) * 257).astype("uint16")
    a = _RecordingArray(data, chunks=(8, 8))
    out = _read_zarr01(a, slab_rows=8)
    assert out.dtype == np.float32
    assert np.allclose(out, data / 65535.0, atol=1e-6)
    assert max(_z_extent(k, 64) for k in a.requests) <= 8


def test_read_zarr01_2d_float_above_one_normalized_by_max():
    from plumbline.io import _read_zarr01
    data = np.array([[0.0, 2.0], [4.0, 1.0]], dtype="float32")
    a = _RecordingArray(data, chunks=(1, 2))
    out = _read_zarr01(a, slab_rows=1)
    assert np.allclose(out, data / 4.0, atol=1e-6)


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
