import os
import glob
import fnmatch
import warnings
from typing import List
from plumbline.model import SegmentInputs

_INK_PATTERNS = ["*prediction*", "*result*", "*inklabels*"]
_MASK_PATTERNS = ["*flat_mask*", "*mask*"]
_IMG_EXTS = (".png", ".tif", ".tiff")


def _images_in(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if os.path.isfile(f) and f.lower().endswith(_IMG_EXTS))


def _matches_any(name, patterns):
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _first_match(images, patterns):
    for pat in patterns:
        for img in images:
            if fnmatch.fnmatch(os.path.basename(img).lower(), pat):
                return img
    return None


def find_segments(root) -> List[SegmentInputs]:
    """Each immediate subfolder of `root` is one segment (folder name = id).
    Pick an ink image (prediction/result/inklabels, else the lone non-mask
    image) and an optional mask. Subfolders with no usable ink image are
    skipped."""
    segments: List[SegmentInputs] = []
    for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        images = _images_in(d)
        if not images:
            continue
        masks = [m for m in images if _matches_any(os.path.basename(m).lower(), _MASK_PATTERNS)]
        candidates = [i for i in images if i not in masks]
        ink = _first_match(candidates, _INK_PATTERNS)
        if ink is None:
            if len(candidates) == 1:
                ink = candidates[0]
            else:
                warnings.warn(
                    f"skipping {os.path.basename(d)!r}: cannot auto-detect an ink "
                    f"image ({len(candidates)} non-mask image(s), none named "
                    f"prediction/result/inklabels)"
                )
                continue
        mask = masks[0] if masks else None
        segments.append(SegmentInputs(seg_id=os.path.basename(d), ink_path=ink, mask_path=mask))
    return segments
