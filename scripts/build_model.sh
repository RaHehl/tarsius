#!/usr/bin/env bash
# Build the bundled SqueezeNet 1.1 ncnn model from the original Caffe release.
#
# Sources and outputs are pinned by sha256, so CI and a local run produce the
# same files. Needs x86_64 Linux with curl, unzip, libgomp1 and, for
# caffe2ncnn, libprotobuf32t64 (Ubuntu 24.04).
set -euo pipefail

SQUEEZENET_COMMIT=51147dc0681638c28aad593b9e923f3500117125
SQUEEZENET_URL="https://raw.githubusercontent.com/forresti/SqueezeNet/${SQUEEZENET_COMMIT}/SqueezeNet_v1.1"
NCNN_TAG=20260526
NCNN_ZIP="ncnn-${NCNN_TAG}-ubuntu-2404"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_TAG}/${NCNN_ZIP}.zip"

OUT="$(cd "$(dirname "$0")/.." && pwd)/src/tarsius/models"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

curl -sfL -o "$WORK/deploy.prototxt" "$SQUEEZENET_URL/deploy.prototxt"
curl -sfL -o "$WORK/squeezenet_v1.1.caffemodel" "$SQUEEZENET_URL/squeezenet_v1.1.caffemodel"
curl -sfL -o "$WORK/ncnn.zip" "$NCNN_URL"
(
    cd "$WORK"
    sha256sum -c - <<SUMS
d041bfb2ab4b32fda4ff6c6966684132f2924e329916aa5bfe9285c6b23e3d1c  deploy.prototxt
72b912ace512e8621f8ff168a7d72af55910d3c7c9445af8dfbff4c2ee960142  squeezenet_v1.1.caffemodel
c6daf0c8fb2546de420c7b320e2ce62ba764163ace0ba63a2ea0cdfca70d684d  ncnn.zip
SUMS
    unzip -q -j ncnn.zip "${NCNN_ZIP}/bin/caffe2ncnn" "${NCNN_ZIP}/bin/ncnnoptimize" -d bin
    ./bin/caffe2ncnn deploy.prototxt squeezenet_v1.1.caffemodel raw.param raw.bin
    # 1 = store weights as fp16, half the size with no measurable accuracy loss.
    ./bin/ncnnoptimize raw.param raw.bin squeezenet_v1.1.param squeezenet_v1.1.bin 1
)
(
    cd "$WORK"
    sha256sum -c - <<SUMS
ba925b0bbc9aeaea34baa2a58056990da92ba406250209bce0382f5c66e2725b  squeezenet_v1.1.bin
ca5585520ab1281a44fa27eb06e9851a3dfe882b217978d6cdadddd58356477c  squeezenet_v1.1.param
SUMS
)
mkdir -p "$OUT"
cp "$WORK/squeezenet_v1.1.param" "$WORK/squeezenet_v1.1.bin" "$OUT/"
