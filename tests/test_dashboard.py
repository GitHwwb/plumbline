import numpy as np
from plumbline.synthetic import striped_field, garble_patch
from plumbline.coherence import analyze_tiles
from plumbline.score import flag_tiles
from plumbline.dashboard import thumbnail_png, render_index
from plumbline.model import IndexRow


def _bundle():
    f = garble_patch(striped_field((512, 512), pitch=24, angle=0.0), 128, 320, 128, 320)
    feats = analyze_tiles(f, np.ones(f.shape, bool), tile=128, overlap=0.5)
    return f, feats, flag_tiles(feats)


def test_thumbnail_png_is_small_png():
    f, feats, flags = _bundle()
    png = thumbnail_png(f, feats, flags)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) < 200_000


def test_render_index_search_links_thumbs_worst_first():
    rows = [
        IndexRow("good", 95, 0, 0, 0, 0.05, "good.html", "AAAA", None),
        IndexRow("bad", 30, 3, 9, 5, 0.40, "bad.html", "BBBB", None),
    ]
    html = render_index(rows, {"scroll": "Scroll1"})
    assert 'id="q"' in html
    assert "good.html" in html and "bad.html" in html
    assert "data:image/png;base64,AAAA" in html
    assert "<html" in html.lower()
    assert html.index("bad.html") < html.index("good.html")   # worst-first


def test_render_index_marks_error_rows():
    rows = [IndexRow("oops", 0, 0, 0, 0, 1.0, None, None, "boom"),
            IndexRow("realbad", 20, 2, 5, 3, 0.3, "realbad.html", None, None)]
    html = render_index(rows, {"scroll": "S"})
    assert "not evaluated" in html
    # an un-evaluatable segment sorts BELOW a genuinely low-scoring one
    assert html.index("realbad") < html.index("oops")


def test_render_index_escapes_html():
    # autoescape must be on for the .j2 template, or segment ids / meta render raw.
    rows = [IndexRow("<script>x</script>", 50, 0, 0, 0, 0.1, None, None, None)]
    html = render_index(rows, {"scroll": "S<b>"})
    assert "<script>x</script>" not in html      # injected seg_id is escaped
    assert "&lt;script&gt;" in html
    assert "S<b>" not in html                     # meta is escaped too
