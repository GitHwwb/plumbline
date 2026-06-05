import base64
import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plumbline.render import (overlay_png, heatmap_png, orientation_png,
                              flags_png, flagged_regions)
from plumbline.score import input_warning

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _score_color(score: int) -> str:
    if score >= 85:
        return "#2bd47a"
    if score >= 60:
        return "#ffce5c"
    return "#ff5c5c"


def render_report(meta, ink01, features, flags, report) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR),
                      autoescape=select_autoescape(["html", "j2"]))
    tmpl = env.get_template("report.html.j2")
    return tmpl.render(
        meta=meta, report=report,
        score_color=_score_color(report.score),
        regions=flagged_regions(features, flags),
        img_overlay=_b64(overlay_png(ink01, features, flags)),
        img_heat=_b64(heatmap_png(features)),
        img_heat_over=_b64(heatmap_png(features, ink01)),
        img_orient=_b64(orientation_png(features)),
        img_orient_over=_b64(orientation_png(features, ink01)),
        img_flags=_b64(flags_png(ink01, features, flags)),
        warning=input_warning(features, flags),
    )


def write_report(path, meta, ink01, features, flags, report):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(meta, ink01, features, flags, report))


def write_json(path, meta, features, flags, report):
    payload = {
        "segment_id": meta.get("segment_id"),
        "score": report.score,
        "n_orient": report.n_orient,
        "n_spacing": report.n_spacing,
        "n_garble": report.n_garble,
        "n_seam": report.n_seam,
        "low_conf_frac": report.low_conf_frac,
        "regions": flagged_regions(features, flags),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
