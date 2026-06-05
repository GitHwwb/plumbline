import json
import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles, trace_health
from plumbline.report import render_report, write_report, write_json


def _bundle():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    flags = flag_tiles(feats)
    return f, feats, flags, trace_health(feats, flags)


def test_render_report_is_self_contained_html():
    f, feats, flags, rep = _bundle()
    html = render_report({"segment_id": "seg-test", "scroll": "Scroll1"},
                         f, feats, flags, rep)
    assert "<html" in html.lower()
    assert str(rep.score) in html
    assert "data:image/png;base64," in html        # images embedded, no external files
    assert "http://" not in html and "https://" not in html


def test_write_report_and_json(tmp_path):
    f, feats, flags, rep = _bundle()
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"
    write_report(str(html_path), {"segment_id": "seg-test", "scroll": "Scroll1"},
                 f, feats, flags, rep)
    write_json(str(json_path), {"segment_id": "seg-test"}, feats, flags, rep)
    assert html_path.exists() and html_path.stat().st_size > 1000
    data = json.loads(json_path.read_text())
    assert data["score"] == rep.score
    assert "regions" in data and isinstance(data["regions"], list)
