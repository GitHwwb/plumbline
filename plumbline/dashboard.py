import base64
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plumbline.render import overlay_png
from plumbline.report import _score_color

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def thumbnail_png(ink01, features, flags) -> bytes:
    """A small ink-overlay PNG (flagged tiles outlined) for a dashboard row.
    Reuses the report overlay renderer at a small figure size."""
    return overlay_png(ink01, features, flags, figsize=(1.6, 1.6))


def thumbnail_b64(ink01, features, flags) -> str:
    return base64.b64encode(thumbnail_png(ink01, features, flags)).decode("ascii")


def render_index(rows, meta) -> str:
    """Render the sortable/searchable dashboard. Rows are sorted worst-health
    first by default (client JS can re-sort)."""
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR),
                      autoescape=select_autoescape(["html", "j2"]))
    tmpl = env.get_template("index.html.j2")
    # Worst-health first, but: (a) segments with a detected SEAM form their own
    # TOP tier -- a sheet jump flags only the few tiles straddling one column,
    # so a seamed segment can score 95+ and would otherwise sort below merely
    # noisy ones, burying the defect the tool exists to surface (the score
    # itself stays area-based by design); (b) un-evaluatable (error) rows float
    # to the bottom so a failed segment isn't mistaken for the lowest-quality one.
    ordered = sorted(rows, key=lambda r: (r.error is not None, r.n_seam == 0,
                                          r.score, r.seg_id))
    n_flagged = sum(1 for r in ordered
                    if r.error is None
                    and (r.n_orient + r.n_spacing + r.n_garble + r.n_seam) > 0)
    n_errors = sum(1 for r in ordered if r.error)
    return tmpl.render(rows=ordered, meta=meta, n_total=len(ordered),
                       n_flagged=n_flagged, n_errors=n_errors, score_color=_score_color)
