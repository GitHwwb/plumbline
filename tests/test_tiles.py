from plumbline.tiles import tile_grid
from plumbline.model import Tile


def test_tile_grid_counts_and_coords():
    tiles, n_rows, n_cols = tile_grid((512, 512), tile=256, overlap=0.5)
    assert (n_rows, n_cols) == (3, 3)
    assert len(tiles) == 9
    assert all(isinstance(t, Tile) for t in tiles)
    first = tiles[0]
    assert (first.row, first.col, first.y0, first.x0) == (0, 0, 0, 0)
    assert all(t.y1 <= 512 and t.x1 <= 512 for t in tiles)


def test_tile_grid_small_image_yields_single_tile():
    tiles, n_rows, n_cols = tile_grid((100, 80), tile=256, overlap=0.5)
    assert (n_rows, n_cols) == (1, 1)
    assert len(tiles) == 1
    assert (tiles[0].y1, tiles[0].x1) == (100, 80)
