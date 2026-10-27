# UWB detector V5

Run Run-Best.ps1 in PowerShell. It defaults to the requested dataset folder and validation split.
For official test data, use -Split test -Data "PATH_TO_PACKAGE". Output is results/test_submission.csv.
No official test streams were available or evaluated. Validation output is results/val_submission.csv.

Selected blend: 65% motion-only boosted trees, 17.5% temporal network seed 2028,
17.5% temporal network seed 2029. All weights and normalization fit training data only.
Nine predefined mixtures were compared with 48 decoder settings each, using val_00..02.
The previously seen val_03..04 served as a rejection diagnostic, not an untouched test set.
V4 was preserved. No additional model weights were trained for this V5 experiment.

Official validation with flagged events ignored: mean F1 0.823529, TP 59, FP 10, FN 8 at tIoU .3.
Diagnostic mean F1 0.833333, TP 9, FP 2, FN 0. This small reused diagnostic is not hidden-test evidence.
Start MAE 0.723729 s, end MAE 0.425932 s, median signal-time latency 0.32 s.
Compared with V4, ranking and false alarms improved; mean boundary errors and latency slightly worsened.
The full reports contain all evaluator metrics. Validation has been repeatedly examined;
this result cannot guarantee a competition win or perfect detection.

Inference uses continuous radar HDF5 streams only. No prohibited source recordings, source segments,
composition files or labels are used for predictions. Validation labels are read for reporting only;
the test prediction path does not read labels. The official evaluator is unchanged.
Alarm times simulate signal availability, excluding CPU runtime; this batch runner is not a live service.
Most neural features include static room context, a generalization limitation noted in the competition README.

On another computer, install requirements.txt and pass -Python with the Python interpreter path.
Dataset licensing remains CC BY-NC 4.0 and the challenge source-recording prohibition applies.
