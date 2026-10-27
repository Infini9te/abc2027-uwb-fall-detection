# UWB detector V7 — best measured ranking candidate

Run Run-Best.ps1. Default data path is the requested abc2027_uwb_train_v1 (1) package.
Use -Split test -Data "PATH_TO_OFFICIAL_PACKAGE" for released test streams.
Validation predictions: results/val_submission.csv. Test predictions: results/test_submission.csv.
On another computer, install requirements.txt and pass -Python with the interpreter path.

Model: 50% XGBoost motion features, 32.5% histogram gradient boosting motion features,
8.75% causal TCN seed 2028 and 8.75% causal TCN seed 2029.
Full validation, flagged events ignored: ranking mean F1 0.828431; at tIoU .3,
57 true positives, 12 false alarms, 10 missed falls; F1 .838235.
Start MAE .570526 s; end MAE .321754 s; median signal-time latency .38 s.
V5 has lower ranking F1 .823529 but better primary detection: 59 TP, 10 FP, 8 FN,
and lower median latency .32 s. Neither dominates all criteria. V5 is preserved separately.
The reused nine-fall diagnostic scored .857143 versus V5 .833333.

This package does not guarantee a competition win. Hidden test data were unavailable.
Repeated validation selection can overfit. Read RESEARCH.md and full evaluator reports.
The official evaluator was not edited (SHA256
df23749689ac2a41861f400059b18a575ada32ca91ea49733d9d219c3d9d4daa).
Training and inference did not use original LIARA recordings, external copies, or source
segment/composition metadata. Predictions use continuous released radar streams.
Validation labels are used for selection/reporting, not inference; test mode reads no labels.
First-alarm times represent causal signal availability, excluding computation time.
Final intervals are available after event completion; this batch runner is not a live service.
The neural features retain static room context, which may limit generalization.
The challenge dataset remains subject to CC BY-NC 4.0 and its competition rules.
