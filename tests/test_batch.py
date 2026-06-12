import numpy as np
from PIL import Image
from plumbline.synthetic import striped_field, garble_patch
from plumbline.cli import main


def _save(path, field):
    Image.fromarray((field * 255).astype("uint8")).save(path)


def test_batch_builds_dashboard_worst_first(tmp_path):
    segs = tmp_path / "segs"; segs.mkdir()
    clean = segs / "clean"; clean.mkdir()
    _save(clean / "prediction.png", striped_field((512, 512), pitch=24, angle=0.0))
    bad = segs / "bad"; bad.mkdir()
    _save(bad / "prediction.png",
          garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 384, 128, 384))
    out = tmp_path / "out"

    rc = main(["batch", str(segs), "-o", str(out), "--tile", "128"])
    assert rc == 0
    assert (out / "index.html").exists()
    assert (out / "clean.html").exists() and (out / "bad.html").exists()
    html = (out / "index.html").read_text()
    assert html.index("bad.html") < html.index("clean.html")   # worst-first


def test_batch_no_reports_no_thumbnails(tmp_path):
    segs = tmp_path / "segs"; segs.mkdir()
    s = segs / "s1"; s.mkdir()
    _save(s / "prediction.png", striped_field((256, 256), pitch=24, angle=0.0))
    out = tmp_path / "out"

    rc = main(["batch", str(segs), "-o", str(out), "--tile", "128",
               "--no-reports", "--no-thumbnails"])
    assert rc == 0
    assert (out / "index.html").exists()
    assert not (out / "s1.html").exists()
    assert "data:image/png;base64," not in (out / "index.html").read_text()


def test_batch_jobs_parallel_matches_sequential(tmp_path):
    # Segments are independent, so --jobs N must give the SAME dashboard as the
    # sequential path (same rows, same order, same scores) -- parallelism is a
    # throughput knob, never a results knob.
    import re
    segs = tmp_path / "segs"; segs.mkdir()
    clean = segs / "clean"; clean.mkdir()
    _save(clean / "prediction.png", striped_field((256, 256), pitch=24, angle=0.0))
    bad = segs / "bad"; bad.mkdir()
    _save(bad / "prediction.png",
          garble_patch(striped_field((256, 256), pitch=24, angle=0.0), 64, 192, 64, 192))

    out_seq = tmp_path / "seq"; out_par = tmp_path / "par"
    assert main(["batch", str(segs), "-o", str(out_seq), "--tile", "128"]) == 0
    assert main(["batch", str(segs), "-o", str(out_par), "--tile", "128",
                 "--jobs", "2"]) == 0

    def ids_and_scores(p):
        html = (p / "index.html").read_text()
        return (re.findall(r'data-id="([^"]+)"', html),
                re.findall(r'class="badge"[^>]*>(\d+)</span>', html))

    assert ids_and_scores(out_seq) == ids_and_scores(out_par)
    assert (out_par / "clean.html").exists() and (out_par / "bad.html").exists()


def test_batch_evaluates_zarr_segment(tmp_path):
    # End-to-end: a segment whose ink lives in a Zarr store must be discovered
    # AND loaded (batch previously loaded via load_image01, which can't open a
    # store even if discovery had found it) -- not marked "not evaluated".
    import pytest
    zarr = pytest.importorskip("zarr")
    segs = tmp_path / "segs"; segs.mkdir()
    s = segs / "zseg"; s.mkdir()
    field = (striped_field((256, 256), pitch=24, angle=0.0) * 255).astype("uint8")
    z = zarr.open(str(s / "prediction.zarr"), mode="w", shape=field.shape,
                  dtype="uint8")
    z[:] = field
    out = tmp_path / "out"

    rc = main(["batch", str(segs), "-o", str(out), "--tile", "128"])
    assert rc == 0
    html = (out / "index.html").read_text()
    assert "zseg" in html
    # the error badge (<span class="err">not evaluated</span>) must be absent --
    # NB the bare string "not evaluated" also appears in a template JS comment,
    # so assert on the markup, and on the per-segment report actually existing
    assert 'class="err"' not in html
    assert (out / "zseg.html").exists()
