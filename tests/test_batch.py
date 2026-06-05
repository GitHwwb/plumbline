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
