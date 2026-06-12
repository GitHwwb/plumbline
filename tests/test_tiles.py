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


def test_tile_grid_covers_image_edges_for_non_round_sizes():
    # REGRESSION: the strided ranges stop the last tile short of the edge
    # whenever (dim - tile) % step != 0, leaving up to step-1 px of image in
    # NO tile (1000x1000 / tile 256 / overlap .5 left a 104px blind strip at
    # the right+bottom; a defect there could never flag, and flag_seam's
    # seam-column -> tile mapping silently dropped edge seams).
    for shape, tile in (((1000, 1000), 256), ((5000, 3000), 2048), ((517, 261), 128)):
        tiles, n_rows, n_cols = tile_grid(shape, tile=tile, overlap=0.5)
        assert max(t.y1 for t in tiles) == shape[0], (shape, tile)
        assert max(t.x1 for t in tiles) == shape[1], (shape, tile)
        # edge-anchored tiles keep the uniform size the per-tile stats assume
        assert all((t.y1 - t.y0) == tile and (t.x1 - t.x0) == tile for t in tiles)
        # grid bookkeeping stays consistent with the returned shape
        assert max(t.row for t in tiles) == n_rows - 1
        assert max(t.col for t in tiles) == n_cols - 1
        assert len(tiles) == n_rows * n_cols
