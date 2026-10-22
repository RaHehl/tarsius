#!/usr/bin/env bash
# Rebuild the bundled meter models from the training outputs.
#
# The wheel and dial readers ship as exported by the training (numpy .npz),
# the two finders are converted from Keras to ncnn by keras_to_ncnn.py.
# Inputs and outputs are pinned by sha256, so a rebuild reproduces the
# committed files byte for byte or fails. The training outputs are not
# published; this script records how the committed files were made and
# checks them. Needs uv.
#
#   scripts/build_meter_models.sh digits.npz dials.npz finder.keras dialfinder.keras
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 digits.npz dials.npz finder.keras dialfinder.keras" >&2
    exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/src/tarsius/meter/models"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp "$1" "$WORK/digits.npz"
cp "$2" "$WORK/dials.npz"
cp "$3" "$WORK/finder.keras"
cp "$4" "$WORK/dialfinder.keras"
(
    cd "$WORK"
    sha256sum -c - <<SUMS
d1c8c0f14af680b97f923048e9672bcaaf72232cde3bbc0c625a91141041940d  digits.npz
9253a90d99d15f749dfecc6f8a4ec012cb9061ebcbbfbcd091264f0971aefce6  dials.npz
f8224ab932c0c459c2c884d13ea6dcbce7bd214bc65a14d88995081a38e1ded0  finder.keras
ce84562e94c2930ca6b9670433835dc51572d1904918a24876ac9cf9fabcfdfb  dialfinder.keras
SUMS
)
for model in finder dialfinder; do
    uv run --quiet --no-project --python 3.12 --with h5py==3.14.0 --with numpy==2.5.4 \
        python "$ROOT/scripts/keras_to_ncnn.py" "$WORK/$model.keras" "$WORK/$model"
done
(cd "$WORK" && sha256sum -c "$OUT/SHA256SUMS")
cp "$WORK"/{digits.npz,dials.npz,finder.param,finder.bin,dialfinder.param,dialfinder.bin} "$OUT/"
