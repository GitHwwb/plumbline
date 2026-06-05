from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class TileFeatures:
    n_rows: int
    n_cols: int
    theta: np.ndarray         # (n_rows, n_cols) radians, text-line orientation, mod pi
    band_strength: np.ndarray # (n_rows, n_cols) >=0 band contrast (rows vs gaps; ~0 noise, .1-1.5+ text)
    pitch: np.ndarray         # (n_rows, n_cols) pixels between lines, NaN if none
    pitch_strength: np.ndarray # (n_rows, n_cols) 0..1 autocorrelation peak height
    density: np.ndarray      # (n_rows, n_cols) 0..1 ink fraction
    confidence: np.ndarray   # (n_rows, n_cols) bool, enough ink+coverage to judge
    tiles: List[Tile]        # maps grid cell -> pixel box
    orient_reliability: Optional[np.ndarray] = None  # (n_rows, n_cols) 0..1: how
    #   decisively the orientation sweep peaks at an interior angle. Low for sparse
    #   fragments / rail-pegged fits with no real writing direction; gates
    #   orientation-break flagging so a noisy angle can't manufacture a flag.
    gtheta: float = 0.0           # global skew angle (radians) used for the seam scan
    gpitch: float = float("nan")  # per-tile MEDIAN row pitch (px): the seam detector's
    #   reference spacing. The per-tile median (not a fresh global single-profile pitch)
    #   is load-bearing -- a sheet-jump corrupts the global profile's autocorrelation
    #   and would drive the seam threshold down, missing the dy~pitch/2 case.


@dataclass
class FlagMap:
    orient_break: np.ndarray     # bool (n_rows, n_cols)
    spacing_break: np.ndarray    # bool (n_rows, n_cols)
    garble: np.ndarray           # bool (n_rows, n_cols)
    seam_break: Optional[np.ndarray] = None  # bool (n_rows, n_cols): rows step
    #   vertically at a sheet-jump seam (no rotation/spacing change). Trailing &
    #   defaulted (same pattern as the Optional reliability fields) so old callers
    #   that build a FlagMap without it -- e.g. tests -- keep working.

    @property
    def any_flag(self) -> np.ndarray:
        base = self.orient_break | self.spacing_break | self.garble
        if self.seam_break is not None:
            base = base | self.seam_break
        return base


@dataclass
class ScoreReport:
    score: int            # 0..100 trace health
    n_orient: int
    n_spacing: int
    n_garble: int
    low_conf_frac: float  # fraction of grid that was low-confidence
    n_seam: int = 0       # seam (pure vertical sheet-jump) flagged tiles


@dataclass
class SegmentInputs:
    seg_id: str
    ink_path: str
    mask_path: Optional[str]


@dataclass
class IndexRow:
    seg_id: str
    score: int
    n_orient: int
    n_spacing: int
    n_garble: int
    low_conf_frac: float
    report_filename: Optional[str]   # "<id>.html", or None when --no-reports
    thumb_b64: Optional[str]         # base64 PNG, or None when --no-thumbnails
    error: Optional[str] = None      # set when the segment could not be evaluated
    n_seam: int = 0                  # seam (vertical sheet-jump) flag count
