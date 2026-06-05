from typing import List, Tuple
from plumbline.model import Tile


def tile_grid(shape, tile: int = 256, overlap: float = 0.5) -> Tuple[List[Tile], int, int]:
    """Build a full rectangular grid of (possibly overlapping) tile boxes.

    Returns (tiles, n_rows, n_cols). Boxes are clamped to image bounds. The
    grid always covers the whole image; mask/ink emptiness is handled later
    via per-tile confidence, not by dropping tiles here.
    """
    h, w = shape[0], shape[1]
    step = max(1, int(round(tile * (1.0 - overlap))))
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    tiles: List[Tile] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tiles.append(Tile(r, c, y0, min(y0 + tile, h), x0, min(x0 + tile, w)))
    return tiles, len(ys), len(xs)
