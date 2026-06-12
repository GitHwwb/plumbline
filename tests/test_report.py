import json
import numpy as np
import pytest
from plumbline import __version__
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


def test_render_report_interactive_flag_boxes():
    # The default view draws flag boxes as live positioned elements over an
    # exact-extent ink PNG: each box carries a hover tooltip with the tile's
    # measurements, and each row of the flagged-regions table links to its box.
    from plumbline.render import flagged_regions
    f, feats, flags, rep = _bundle()
    regions = flagged_regions(feats, flags)
    assert regions, "bundle must produce flagged regions"
    html = render_report({"segment_id": "s", "scroll": "S"}, f, feats, flags, rep)
    assert html.count('class="flagbox') == len(regions)   # one live box per region
    assert 'class="tip"' in html                          # hover tooltip markup
    assert html.count('class="regrow"') == len(regions)   # table rows wired up
    assert "data-idx" in html
    # hovering a table row glows its box (click remains the locate action)
    assert "mouseenter" in html and "mouseleave" in html
    assert ".glow" in html


def test_render_report_shows_seam_banner_only_when_seam_found():
    # The report header must call out a detected sheet-jump seam explicitly --
    # the score barely moves (few tiles straddle the seam), so without a banner
    # the report reads healthy.
    import dataclasses
    f, feats, flags, rep = _bundle()
    # NB assert on the rendered MARKUP (class="seam-banner"), not the bare
    # string -- the .seam-banner CSS rule is in <style> on every report.
    html_no = render_report({"segment_id": "s", "scroll": "S"}, f, feats, flags, rep)
    assert 'class="seam-banner"' not in html_no  # this bundle has no seam
    rep_seam = dataclasses.replace(rep, n_seam=3)
    html_yes = render_report({"segment_id": "s", "scroll": "S"}, f, feats, flags, rep_seam)
    assert 'class="seam-banner"' in html_yes


def test_write_json_records_run_params(tmp_path):
    # Reproducibility: the sidecar must record HOW the run was configured --
    # tool version, the tile size actually used (auto-chosen sizes differ per
    # image), grid shape, global skew + pitch -- plus caller-known settings
    # (overlap) via the optional `params` argument. Without these, two runs
    # that disagree cannot be diagnosed and a pipeline cannot re-run one.
    f, feats, flags, rep = _bundle()
    json_path = tmp_path / "report.json"
    write_json(str(json_path), {"segment_id": "seg-test"}, feats, flags, rep,
               params={"overlap": 0.5})
    data = json.loads(json_path.read_text())
    p = data["params"]
    assert p["plumbline_version"] == __version__
    assert p["tile_px"] == 128                     # _bundle analyzes at tile=128
    assert p["grid"] == [feats.n_rows, feats.n_cols]
    assert p["overlap"] == 0.5
    assert isinstance(p["gtheta_rad"], float)
    # regions carry the full flagged box, not just its center
    assert data["regions"], "garbled bundle should flag at least one region"
    assert {"x", "y", "x0", "y0", "x1", "y1", "mode"} <= set(data["regions"][0].keys())


def test_write_json_is_strict_json_when_gpitch_is_nan(tmp_path):
    # An aperiodic input leaves gpitch = NaN; json.dump would happily emit the
    # token NaN, which strict JSON parsers (jq, browsers, most languages)
    # reject. It must serialize as null instead.
    f, feats, flags, rep = _bundle()
    feats.gpitch = float("nan")
    json_path = tmp_path / "report.json"
    write_json(str(json_path), {"segment_id": "seg-test"}, feats, flags, rep)
    data = json.loads(json_path.read_text(),
                      parse_constant=lambda c: pytest.fail(f"non-strict JSON constant: {c}"))
    assert data["params"]["gpitch_px"] is None
