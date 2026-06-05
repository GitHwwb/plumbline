#!/usr/bin/env bash
# Download example fragment images from the Vesuvius Challenge data server for
# Kaggle fragment Frag1 (PHercParis2Fr47).
#
# NOTE: result.png here is the fragment's SURFACE render, not an ink-model
# prediction (see the README "Trying it on real fragment data" caveat). It's a
# convenient sample to exercise the tool, not a real quality measurement.
#
# ACCESS: you must accept the data agreement at https://scrollprize.org/data .
# That page gives you the data-server credentials. Per the agreement, those
# credentials must NOT be shared publicly, so this script reads them from your
# environment rather than embedding them:
#
#     export VESUVIUS_USER=...   # from https://scrollprize.org/data
#     export VESUVIUS_PASS=...
#     bash examples/fetch_frag1.sh
set -euo pipefail

: "${VESUVIUS_USER:?Set VESUVIUS_USER (data-server username from https://scrollprize.org/data)}"
: "${VESUVIUS_PASS:?Set VESUVIUS_PASS (data-server password from https://scrollprize.org/data)}"

BASE="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface"
OUT="data/frag1"
mkdir -p "$OUT"

echo "Downloading surface render (~49 MB, 16-bit) and mask ..."
curl -fL -u "$VESUVIUS_USER:$VESUVIUS_PASS" "$BASE/result.png" -o "$OUT/result.png"
curl -fL -u "$VESUVIUS_USER:$VESUVIUS_PASS" "$BASE/mask.png"   -o "$OUT/mask.png"

echo
echo "Done. Now run Plumbline:"
echo "  plumbline run $OUT/result.png --mask $OUT/mask.png \\"
echo "      -o $OUT/report.html --json $OUT/report.json --tile 1024"
