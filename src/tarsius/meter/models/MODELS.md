# Meter models

Trained by Raphael Hehl for tarsius, released with it under Apache-2.0.

| File | Purpose | Parameters | Runtime | Input |
|---|---|---|---|---|
| `digits.npz` | wheel reader: 100 value classes, 5 fault classes, orientation | 475k | numpy engine | 32×20 RGB cell |
| `dials.npz` | pointer dial reader: 100 value classes | 168k | numpy engine | 32×32 RGB crop |

`SHA256SUMS` pins the files; a test checks them.

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
| `dials.npz` | dial reader release of 2026-08-09, synthetic only |

The readers ship as exported by the training, with batch normalization
folded into the convolutions.
