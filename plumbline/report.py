import base64
import datetime as _dt
import json
import math
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plumbline import __version__
from plumbline.render import (ink_png, heatmap_png, orientation_png,
                              flags_png, flagged_regions)
from plumbline.score import input_warning

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _score_color(score: int) -> str:
    # neon health palette (user-approved d_merged_neon mockup): lime / lemon / red
    if score >= 85:
        return "#8efc4e"
    if score >= 60:
        return "#fcf151"
    return "#e23227"


def render_report(meta, ink01, features, flags, report) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR),
                      autoescape=select_autoescape(["html", "j2"]))
    tmpl = env.get_template("report.html.j2")
    tiles = features.tiles
    tile_px = (max(max(t.y1 - t.y0 for t in tiles),
                   max(t.x1 - t.x0 for t in tiles)) if tiles else 0)
    gpitch = float(getattr(features, "gpitch", float("nan")))
    return tmpl.render(
        meta=meta, report=report,
        score_color=_score_color(report.score),
        regions=flagged_regions(features, flags),
        # iw/ih: ORIGINAL pixel extent -- flag boxes are positioned in percent
        # of these, over the exact-extent ink PNG (no matplotlib margins).
        iw=int(ink01.shape[1]), ih=int(ink01.shape[0]),
        # header-band + params-footer metadata
        analyzed=_dt.date.today().strftime("%d %b %Y"),
        tile_px=int(tile_px),
        gpitch_px=(gpitch if math.isfinite(gpitch) else None),
        gtheta_deg=math.degrees(float(getattr(features, "gtheta", 0.0))),
        version=__version__,
        img_ink=_b64(ink_png(ink01)),
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


def write_json(path, meta, features, flags, report, params=None):
    """Write the JSON sidecar. Besides the score/flag counts/regions it records
    the run configuration needed to reproduce or diagnose a result: tool
    version, the tile size actually used (auto-sizing picks a different tile
    per image), grid shape, and the global skew/pitch estimates. `params` lets
    the caller add settings only it knows (e.g. the CLI's overlap)."""
    tiles = features.tiles
    tile_px = (max(max(t.y1 - t.y0 for t in tiles),
                   max(t.x1 - t.x0 for t in tiles)) if tiles else None)
    gpitch = float(getattr(features, "gpitch", float("nan")))
    run_params = {
        "plumbline_version": __version__,
        "tile_px": tile_px,
        "grid": [features.n_rows, features.n_cols],
        "gtheta_rad": float(getattr(features, "gtheta", 0.0)),
        # NaN (aperiodic input, no per-tile median pitch) must serialize as
        # null: the bare NaN token json.dump would emit is not strict JSON.
        "gpitch_px": gpitch if math.isfinite(gpitch) else None,
    }
    if params:
        run_params.update(params)
    payload = {
        "segment_id": meta.get("segment_id"),
        "score": report.score,
        "n_orient": report.n_orient,
        "n_spacing": report.n_spacing,
        "n_garble": report.n_garble,
        "n_seam": report.n_seam,
        "low_conf_frac": report.low_conf_frac,
        "params": run_params,
        "regions": flagged_regions(features, flags),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
