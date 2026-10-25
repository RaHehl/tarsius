# Meter models

Trained by Raphael Hehl for tarsius, released with it under Apache-2.0.

| File | Purpose | Parameters | Runtime | Input |
|---|---|---|---|---|
| `digits.npz` | wheel reader: 100 value classes, 5 fault classes, orientation | 475k | numpy engine | 32×20 RGB cell |
| `dials.npz` | pointer dial reader: 100 value classes | 643k | numpy engine | 32×32 RGB crop |
| `finder.param`, `finder.bin` | ALPHA: counter window segmentation | 181k | ncnn, fp16 weights | 256×256 RGB frame |
| `dialfinder.param`, `dialfinder.bin` | ALPHA: dial segmentation | 676k | ncnn, fp16 weights | 256×256 grey frame |

`SHA256SUMS` pins all six files; a test checks them.

## Training data

- Synthetic wheels, dials and meter faces drawn by the training generator,
  with open-licensed fonts.
- The training split of the Word-Wheel Water Meter Dataset (Zhao, S., Gao,
  Y., Zhang, Y., Li, J., Liu, F. (2026). *A comprehensive dataset for
  word-wheel water meter reading under challenging conditions.* Dryad.
  CC0-1.0. [doi:10.5061/dryad.7d7wm3860](https://doi.org/10.5061/dryad.7d7wm3860)).
  Its test split was used only to measure.

No other third-party model, dataset or code went into training. Real
photographs of meters are not part of tarsius: they show serial numbers.

## Provenance

| File | Training run |
|---|---|
| `digits.npz` | `jit020_s3`: synthetic plus 50 % Dryad cells, framing jitter 0.20, seed 3 |
| `dials.npz` | `soft32-s38`: width 2.0, synthetic only, dial faces in open colours, plate continued around the face, numerals at the rim, crops down to 0.94 times the ring, half of them softened, seed 38 |
| `finder.*` | `finder_faces`: width 1.0, Dryad detection split plus 30 % synthetic faces |
| `dialfinder.*` | `dialfinder_w2`: width 2.0, synthetic faces, cut-off dials not labelled |

The readers ship as exported by the training. The finders are converted
from the Keras files by `scripts/keras_to_ncnn.py`, which folds batch
normalization and the input scaling into the convolutions and stores
weights as fp16. `scripts/build_meter_models.sh` reruns the conversion from
the training outputs and checks inputs and outputs by sha256.
`scripts/meter_finder_parity.py` compares ncnn with Keras: with fp16 weights
the logits differ by at most 0.17 and 5e-5 of the mask pixels flip, with
fp32 weights by 5e-4 and none.
