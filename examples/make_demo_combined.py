"""Regenerate the combined-modes demo image (demo_pack/e_all_modes_combined).

The "all modes combined" showcase must HONESTLY fire all FOUR flag modes —
orientation, spacing, garble, and seam — at the AUTO tile size the analyzer
actually picks. The earlier combined image diluted: its rotated band and noise
patch were small relative to one auto tile, so no tile was dominated and only the
(inherently full-height) seam fired.

This recipe fixes that by giving each defect its OWN large, well-SEPARATED region,
each big enough to dominate whole tiles at the auto tile size:

  * ORIENTATION : a ~one-tile-tall rotated band across the top. Sized to ~one tile
                  row on purpose — a taller band would fill its own consensus window
                  and dilute the break below threshold (this is the dilution bug).
  * SPACING     : a coarse pitch-100 glyph block in the left-middle gap. 100/40 =
                  ratio 2.5 is NON-INTEGER, so the boundary tiles read a GENUINE
                  line-spacing jump — never the 2x autocorrelation harmonic the
                  round-4 guard suppresses (that guard only fires on
                  consensus >= 0.85 AND round(ratio) == 2). The flags ride the
                  region boundary where the neighbourhood is split (consensus
                  ~0.76), so this honestly demonstrates the spacing mode AND that
                  the harmonic guard correctly lets a real jump through.
  * SEAM        : a pure vertical sheet-jump on the right half (full-height by nature).
  * GARBLE      : a large, tile-aligned structureless box in the bottom-left.

seed=1 base is harmonic-free (no false-positive spacing flags), so the ONLY spacing
flags come from the injected coarse block. Uses ONLY the documented synthetic
primitives in plumbline/synthetic.py — no scroll data, safe to ship.

Usage:
    python examples/make_demo_combined.py
Writes (relative to the repo root):
    demo_pack/e_all_modes_combined/prediction.png
Then verify the AUTO-tile path fires all four modes:
    plumbline run demo_pack/e_all_modes_combined/prediction.png -o /tmp/e.html
"""
import os
import numpy as np
from PIL import Image

from plumbline.synthetic import glyph_rows, rotate_band, garble_patch, splice_shift

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "demo_pack", "e_all_modes_combined", "prediction.png")


def build_field():
    H, W = 1536, 1024
    # Clean rows of glyphs across the whole sheet (row pitch 40 px).
    f = glyph_rows((H, W), row_pitch=40, glyph=18, gap=8, seed=1)
    # ORIENTATION: rotate a ~one-tile-tall top band (~256px) at 20deg. Sized so the
    #   surrounding clean text keeps the local consensus near 0 -> band tiles read the
    #   full ~20deg deviation (>15deg threshold). A taller band would dominate its own
    #   consensus window and dilute the break to ~half-angle (<threshold) -- the bug.
    #   ddeg=20 stays interior to the +/-25deg orientation search so reliability stays
    #   above the gate (ddeg=25 rails the search boundary -> reliability 0 -> gated off).
    f = rotate_band(f, 128, 384, ddeg=20)
    # SPACING: a coarse pitch-100 glyph block in the open left-middle gap (y 448-704,
    #   x 0-384). 100/40 = ratio 2.5 is NON-INTEGER -> an unambiguous GENUINE
    #   line-spacing jump, NOT the 2x autocorrelation harmonic the round-4 guard
    #   suppresses. The yellow flags ride the region boundary (the tiles straddling
    #   the pitch-40/pitch-100 edge), where the neighbourhood is split (consensus
    #   ~0.76 < 0.85), so they survive the guard on BOTH counts (non-integer ratio
    #   AND split consensus). Disjoint from orient (y<384), garble (y>=1152), and the
    #   seam corridor (x>=640): the block stays in column ~1 of the tile grid, well
    #   left of the seam, so the other three modes are byte-identical (orient 7,
    #   garble 6, seam 33). A wider/taller block makes its interior tiles adopt
    #   pitch-100 as their OWN neighbour-median (dev=0 -> they stop flagging), so it
    #   is kept compact -- the flag lives on the boundary ring, which is correct.
    coarse = glyph_rows((704 - 448, 384), row_pitch=100, glyph=40, gap=16, seed=7)
    f[448:704, 0:384] = coarse
    # SEAM: pure vertical sheet-jump on the right half. splice_shift is full-height;
    #   split on the right so the left-half garble box is untouched.
    f = splice_shift(f, x_split=640, dy=18)
    # GARBLE: large structureless box at the bottom-left, tile-aligned (auto tile 256 /
    #   step 128 -> row edges ...1152,1280). >=400px each side so it fills whole tiles
    #   (a fully-covered tile reads as confident-ink-without-rows = garble, rather than
    #   a half-covered tile whose corrupted pitch would trip a spacing flag).
    f = garble_patch(f, 1152, 1536, 0, 512, seed=3)
    return f


def main():
    f = build_field()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Image.fromarray((f * 255).astype("uint8")).save(OUT)
    print(f"wrote {OUT}  ({f.shape[1]}x{f.shape[0]})")


if __name__ == "__main__":
    main()
