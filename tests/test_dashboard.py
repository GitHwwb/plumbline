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


def test_render_index_seam_segments_sort_first_with_badge():
    # A sheet jump is TOPOLOGICAL damage: it flags only the few tiles straddling
    # one column, so a seamed segment can score 95+ and would sort BELOW merely
    # noisy segments in the worst-first order -- burying exactly the defect the
    # tool exists to surface. Seam segments form their own top tier (worst-first
    # within it) and carry a visible badge. The score itself stays area-based.
    rows = [
        IndexRow("noisy", 40, 5, 4, 6, 0.2, "noisy.html", None, None),
        IndexRow("seamed", 95, 0, 0, 0, 0.05, "seamed.html", None, None, n_seam=2),
        IndexRow("clean", 100, 0, 0, 0, 0.05, "clean.html", None, None),
    ]
    html = render_index(rows, {"scroll": "S"})
    assert html.index("seamed.html") < html.index("noisy.html") < html.index("clean.html")
    assert "seam-badge" in html                  # visible marker on the seam row
    assert html.count("seam-badge") >= 1


def test_render_index_escapes_html():
    # autoescape must be on for the .j2 template, or segment ids / meta render raw.
    rows = [IndexRow("<script>x</script>", 50, 0, 0, 0, 0.1, None, None, None)]
    html = render_index(rows, {"scroll": "S<b>"})
    assert "<script>x</script>" not in html      # injected seg_id is escaped
    assert "&lt;script&gt;" in html
    assert "S<b>" not in html                     # meta is escaped too


def test_render_index_headers_explain_flag_types():
    # User request: hovering a flag-type column header on the dashboard must
    # explain what the flag means (same wording family as the report rail).
    rows = [IndexRow("s", 95, 1, 0, 0, 0.1, "s.html", None, None)]
    html = render_index(rows, {"scroll": "S"})
    for phrase in ("text-row direction departs", "row spacing departs",
                   "no row structure", "fingerprint of a sheet jump",
                   "too little ink to assess"):
        assert phrase in html, f"missing header tooltip: {phrase}"


def test_render_index_headers_have_sort_indicators():
    # Sortable headers carry a neutral indicator; the active column shows an
    # asc/desc arrow (state classes toggled by sortBy, arrows drawn in CSS).
    rows = [IndexRow("s", 95, 1, 0, 0, 0.1, "s.html", None, None)]
    html = render_index(rows, {"scroll": "S"})
    assert html.count('class="sortable') >= 7, "all data columns sortable-marked"
    assert "▲" in html and "▼" in html
