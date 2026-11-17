# Constrained decoding for overnight N1 duration

Full selects a shared N1 decoding rule for U-Sleep and DeepResNet using locally scored nights. The analysis compares Full with the original model outputs and two Bias controls.

## Run the example

Use Python 3.12 and run both commands from the repository root.

```sh
python -m pip install -r requirements.txt
python run.py demo
```

The synthetic example selects `consensus_half24` from 20 calibration nights. The fitted offsets and scores for six evaluation nights are written to `demo_output.json`.

## Run on frozen predictions

Use one `.npz` file per model per night. Give the two files for each night the same basename and index them against the same time origin.

| Array | Contents |
| --- | --- |
| `epoch_indices` | Strictly increasing integer indices of valid 30-second epochs |
| `probabilities` | An array with one row per epoch and columns for W, N1, N2, N3, REM |
| `original` | One original stage code (0–4, in the same order) per epoch |
| `y_true` | One reference stage code (0–4) per epoch |

Use 20 subjects for initial calibration and one night per subject for evaluation. List each night in a JSON manifest. Relative paths start from the working directory.

```json
[
  {
    "subject_id": "example_001",
    "usleep": "predictions/usleep/night_001.npz",
    "deepresnet": "predictions/deepresnet/night_001.npz"
  }
]
```

```sh
python run.py evaluate --calibration20 calibration.json --evaluation evaluation.json
```

Add `--calibration40 expanded.json` to allow a second calibration pass if no rule qualifies. The expanded manifest must include 21–40 subjects and retain every initial calibration night. Keep evaluation subjects and recordings separate from both calibration sets.

The command fits both Bias controls on the same nights used to select the Full rule, then writes `results.json`.

## PHYS2018 results

[Figs. 5 and 6 estimates](results/phys2018_primary.csv) include MAE in minutes and the percentage of nights whose absolute N1-duration error increased by at least 10 minutes relative to Original. Paired bootstrap intervals resample evaluation subjects.

Reproducing the estimates for the 914 evaluation nights requires the study's frozen predictions, calibration split, and evaluation manifest in its original order. [Analysis settings](config/analysis.json) give the random seeds and bootstrap draw count.

## Data and weights

[Model provenance](config/base_models.json) lists checkpoint filenames, checksums, and the DeepResNet inference image digest.

- [PHYS2018](https://doi.org/10.13026/6phb-r450)
- [U-Sleep NSRR-2024 ONNX distribution](https://github.com/paulzerr/nidra)
- [DeepResNet NSRR-2024](https://github.com/biomedical-signal-processing/sleepyland) and [inference code](https://github.com/biomedical-signal-processing/uSLEEPYLAND)
