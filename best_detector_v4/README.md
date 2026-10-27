# Frozen UWB detector

Selected model: top_two

Validation mean event F1: 0.810552; at tIoU 0.3: TP 58, FP 14, FN 9.
Diagnostic mean F1: 0.793651 (only nine evaluable falls).
Start MAE: 0.7155s; end MAE: 0.4138s.
Median signal-time alarm latency: 0.2400s. CPU runtime is not included in this latency.
Configured additional signal delay: 0.0s.

Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Best.ps1` from this folder.
When official test streams arrive, use `-Split test -Data "PATH_TO_PACKAGE"`.
The output is `search_v4/val_submission.csv` or `search_v4/test_submission.csv`.
The val CSV is for validation, not an official test submission.

The first run caches features. Keep the dataset streams unchanged. Only continuous HDF5 radar
streams are inference inputs. Ground truth is read only for reporting with `-Split val`;
test inference does not read ground truth. No source corpora or segments are used.
All weights were fit on training labels only; model/decoder selection used val_00..02.
val_03..04 were already examined and serve only as a rejection diagnostic, not an independent test.
Many validation comparisons were made. Validation scores do not guarantee test performance or a win.
Ignored low-quality/review events are excluded using the unchanged official evaluator.
The full official reports include other overlap thresholds, boundary tolerances and false alarms/hour.

The predictor runs in batches. Alarm times simulate signal availability, including any declared delay;
this is not a deployed live-monitoring service. Boundary revisions happen after the coarse event closes.
Raw static room context remains in most models; the README warns that fixed mat locations can inflate
generalization. No claim of real-world safety validation is made.

All model paths are relative. On another computer install requirements.txt in a Python environment
and pass its interpreter with `-Python`. The official package provides src/abc2027/evaluate.py.
Training/search source is included in uwb_search_v4.py and uwb_temporal_v2.py; the original feature
and boosting source is uwb_detector.py. Search reports document retained and rejected candidates.
Dataset use remains subject to the challenge CC BY-NC 4.0 license and source-recording prohibition.
