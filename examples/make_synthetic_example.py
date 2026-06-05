"""Generate a synthetic Plumbline example with known, localized defects.

Builds a clean field of discrete-glyph "text rows" (the same `glyph_rows` model the
test suite is anchored on -- real text-like rows, NOT stripes; stripes gave false
confidence in an earlier design) and injects three faults a real mis-traced segment
would show, then runs the full pipeline and writes a trace-quality report. A runnable
demo with no scroll data needed.

The three injected defects, each meant to trip a different flag mode:
  * a rotated band      -> ORIENTATION break (text rows turn at the band)
  * a garbled patch     -> GARBLE (confident ink, no row structure)
  * a vertical splice   -> SEAM break (rows step up/down at a sheet-jump seam)

Usage:
    python examples/make_synthetic_example.py
Produces (next to this file):
    synthetic_ink.png         the fake ink prediction
    synthetic_report.html     the self-contained dashboard
    synthetic_report.json     the machine-readable sidecar
"""
import os
import numpy as np
from PIL import Image

from plumbline.synthetic import glyph_rows, rotate_band, garble_patch, splice_shift
from plumbline.coherence import analyze_tiles, estimate_scale_and_skew
from plumbline.score import flag_tiles, trace_health
from plumbline.report import write_report, write_json

HERE = os.path.dirname(os.path.abspath(__file__))


def build_field():
    # Clean rows of glyphs (row pitch 40 px), then three localized, known defects.
    f = glyph_rows((1024, 1024), row_pitch=40, glyph=18, gap=8, seed=7)
    f = rotate_band(f, 120, 280, ddeg=20)        # -> orientation break in that band
    f = garble_patch(f, 620, 820, 250, 480, seed=3)   # -> structure loss (garble)
    f = splice_shift(f, x_split=512, dy=18)      # -> vertical sheet-jump seam at x=512
    return f


def main():
    f = build_field()
    Image.fromarray((f * 255).astype("uint8")).save(os.path.join(HERE, "synthetic_ink.png"))

    mask = np.ones(f.shape, dtype=bool)
    feats = analyze_tiles(f, mask, tile=128, overlap=0.5)
    # ink enables the seam (vertical sheet-jump) detector; theta/pitch come from features.
    flags = flag_tiles(feats, ink=f)
    report = trace_health(feats, flags)
    meta = {"segment_id": "synthetic-demo", "scroll": "(synthetic)"}

    write_report(os.path.join(HERE, "synthetic_report.html"), meta, f, feats, flags, report)
    write_json(os.path.join(HERE, "synthetic_report.json"), meta, feats, flags, report)
    print(f"trace health {report.score}/100 "
          f"(orient {report.n_orient}, spacing {report.n_spacing}, "
          f"garble {report.n_garble}, seam {report.n_seam})")


if __name__ == "__main__":
    main()
