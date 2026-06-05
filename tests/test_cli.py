import json
import numpy as np
from PIL import Image
from plumbline.synthetic import striped_field, garble_patch
from plumbline.cli import main


def test_cli_run_produces_html_and_json(tmp_path):
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    ink_path = tmp_path / "ink.png"
    Image.fromarray((f * 255).astype("uint8")).save(ink_path)
    out_html = tmp_path / "out.html"
    out_json = tmp_path / "out.json"

    rc = main(["run", str(ink_path), "-o", str(out_html),
               "--json", str(out_json), "--tile", "128"])
    assert rc == 0
    assert out_html.exists() and out_html.stat().st_size > 1000
    data = json.loads(out_json.read_text())
    assert 0 <= data["score"] <= 100


def test_run_auto_tile(tmp_path):
    import numpy as np
    from PIL import Image
    from plumbline.cli import main
    from plumbline.synthetic import glyph_rows
    ink = (glyph_rows((640, 640), row_pitch=40) * 255).astype("uint8")
    p = tmp_path / "ink.png"; Image.fromarray(ink).save(p)
    out = tmp_path / "r.html"
    rc = main(["run", str(p), "-o", str(out)])     # no --tile -> auto
    assert rc == 0 and out.exists()
