# ABC2027 UWB Fall Detection

Detecting and temporally localizing falls in continuous ultra-wideband (UWB) radar streams, with saved detector models, inference code, experiment reports, and the ABC2027 training/validation package.

The task is to identify **when a fall starts and ends** while distinguishing falls from everyday activities. Input is radar channel impulse response (CIR) data rather than camera footage. Outputs are scored fall intervals in a CSV file.

## What is included

This project preserves detector versions **V4, V5, V7, and V12**, along with the supplied dataset, baseline implementation, notebook, labels, and evaluation tools. V12 is the latest packaged candidate. V7 and earlier versions remain available for comparison.

The original challenge documentation is preserved in [DATASET_README.md](DATASET_README.md), including recording details, challenge rules, dataset limitations, attribution, and citations. This repository documents a project built on that package; it is not an official challenge release.

## How V12 works

1. Read each continuous radar stream independently and extract radar/motion features and paired I/Q features.
2. Obtain frame-level fall probabilities from saved tree models and temporal convolutional networks (TCNs).
3. Blend **50% short paired-I/Q future-context histogram gradient boosting** with **50% of the V7 ensemble**.
4. Decode the probabilities into fall intervals using the frozen thresholds, smoothing, gap handling, and boundary adjustment in `best_detector_v12/selected.json`.
5. Write predictions and run the supplied prediction-format validator. Validation mode also compares probabilities with the packaged reference and evaluates event-level performance.

The complete blend weights are:

| Component | Weight |
|---|---:|
| Short I/Q future-context histogram boosting | 50% |
| XGBoost on motion features | 25% |
| Histogram boosting on motion features | 16.25% |
| TCN seed 2028 | 4.375% |
| TCN seed 2029 | 4.375% |

V12 uses up to **0.5 seconds of future context**; its alarm timestamps include that delay. The selected V12 model does **not** use hard-negative weighting. Hard-negative training was explored, but the short-reference blend was selected. Inference loads frozen models and does not fit them. Package hashes are checked before prediction.

This is a batch inference implementation. Reported signal-time latency is not a measured end-to-end live-system response time.

## Recorded validation results

These values come from the included [V12 report](best_detector_v12/REPORT.md); they were not newly measured for this repository preparation.

| Metric | V7 | V12 |
|---|---:|---:|
| Ranking F1 (mean at tIoU 0.1, 0.3, 0.5) | 0.8284 | 0.8537 |
| True positives at tIoU 0.3 | 57 | 62 |
| False positives at tIoU 0.3 | 12 | 10 |
| False negatives at tIoU 0.3 | 10 | 5 |
| Start-time mean absolute error | 0.571 s | 0.547 s |
| End-time mean absolute error | 0.322 s | 0.472 s |
| Median alarm latency | 0.380 s | 0.800 s |

V12 improves the recorded ranking and detection counts, with worse end-time error and latency. Public validation was repeatedly used for selection; the reused diagnostic split is not an independent test set. **Hidden-test performance is unknown.** A validator PASS confirms submission format, not detection accuracy.

## Dataset

The supplied training/validation package contains 13 training streams and 5 validation streams, approximately 6.18 and 2.05 hours respectively. Its event CSV contains 478 annotated falls across those streams. HDF5 files contain `cir` arrays of shape `(3, T, 368)`, timestamps `t`, and radar serial numbers, on a 50 Hz grid.

The data and models make the complete project approximately **12.6 GB** before Git storage. Binary datasets, checkpoints, and probability arrays are configured for **Git LFS**. Install Git LFS before cloning, then run `git lfs install` and `git lfs pull` in your checkout if needed. A source ZIP may contain LFS pointers rather than the actual binary data.

## Setup

Use a virtual environment. The base package declares Python 3.10 or later; the pinned detector dependencies may require a newer compatible Python version. Dependency availability and a fresh installation have not been verified as part of repository preparation.

For V12 inference, install its own dependency set from the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r best_detector_v12/requirements.txt
```

For the original challenge baselines/notebook, use a separate environment with the root requirements and editable package installation described in [DATASET_README.md](DATASET_README.md). The root and detector requirements pin different versions; avoid installing both sets into the same environment blindly.

## Run V12 on validation

From the repository root:

```bash
python best_detector_v12/predict.py --data . --split val --out best_detector_v12/results
```

The command writes `val_submission.csv` and reports into the output directory, checks the frozen package, validates prediction format, compares probabilities with the saved validation reference, and scores the predictions. Use a different `--out` directory to preserve the included results.

## Run on separately supplied test streams

```bash
python best_detector_v12/predict.py --data /path/to/test-package --split test --evaluator-package . --out test_predictions
```

The test package must contain `streams/test/stream_*.h5`. `--evaluator-package .` supplies the evaluator from this training package. Predictions go to `test_predictions/test_submission.csv`. Test mode does not compute an official test score without ground truth.

The portable Python commands above are preferred. Some historical PowerShell launchers retain machine-specific interpreter paths from the original work environment.

## Prediction format and evaluation

Prediction CSV fields are `stream_id`, `t_start`, `t_end`, and `score`, with optional `t_alarm`. Times are in seconds; a start must precede its end. Stream identifiers use names such as `val_00` and `test_00`.

Events are matched by temporal intersection-over-union (tIoU), measuring overlap between predicted and annotated intervals. Ranking averages event F1 at tIoU thresholds 0.1, 0.3, and 0.5. The supplied reports also include timing errors, false alarms, and latency. The recorded detector comparison ignores flagged events using the challenge evaluator's rules.

## Repository layout

| Path | Contents |
|---|---|
| `best_detector_v12/` | Latest frozen blend, models, inference, configuration, reports, and saved results |
| `best_detector_v7/` | Earlier tree/TCN ensemble and research artifacts |
| `best_detector_v5/`, `best_detector_v4/` | Earlier detector packages and experiments |
| `src/abc2027/` | Challenge evaluator, baselines, stream and annotation utilities |
| `streams/train/`, `streams/val/` | Continuous HDF5 radar streams |
| `segments/` | Supplied aligned source arrays used by the stream generator |
| `configs/`, `scripts/` | Baseline/stream settings and stream-building script |
| `notebooks/` | Starter notebook |
| `data/`, `reports/` | Supporting artifacts and figures |
| `streams_events.csv` | Fall annotations |
| `streams_composition.csv` | Supplied training/validation segment composition |

Read [V12 research](best_detector_v12/RESEARCH.md), [V12 results](best_detector_v12/REPORT.md), and [V7 documentation](best_detector_v7/README.md) for experimental details. Historical reports may reference research scripts outside this supplied folder; the folder does not necessarily provide complete retraining reproduction for every recorded experiment.

## Limitations and challenge rules

Falls were simulated on a mat at fixed positions. Streams are synthetic concatenations; room context, recording conditions, and temporal annotation uncertainty can affect generalization. Some events have low-quality or review flags. Reported validation performance does not establish real-world deployment reliability.

The original challenge documentation prohibits using original LIARA recordings or external copies for training and inference. The packaged detector documentation states that the detectors use released continuous streams, without original external recordings or external pretrained weights. Preserve and follow the challenge rules in [DATASET_README.md](DATASET_README.md).

## Attribution and license

The supplied package carries **Creative Commons Attribution-NonCommercial 4.0**, with copyright attributed to LIARA Laboratory, Université du Québec à Chicoutimi. The original [LICENSE](LICENSE) is preserved; this repository preparation does not grant a different license. See the original documentation for the complete attribution and usage context.

Citations supplied with the dataset:

- Maitre, Bouchard and Gaboury, IEEE JBHI 2021: https://doi.org/10.1109/JBHI.2020.3027967
- Imbeault-Nepton et al., GoodIT 2023: https://doi.org/10.1145/3582515.3609535
- Abudalfa and Bouchard, Transportation Research Procedia 2025: https://doi.org/10.1016/j.trpro.2025.03.039
- Abudalfa and Bouchard, Applied Intelligence 2025: https://doi.org/10.1007/s10489-024-06156-9
