# V12: short paired-I/Q future-context blend

Frozen blend: 50% short paired-I/Q histogram boosting + 50% V7. The selected new model does NOT use hard-negative weighting. That method improved provisional grouped results but was not the strongest continuous validation candidate.

Validation ranking F1 0.8537; at tIoU 0.3: TP62, FP10, FN5. Reused diagnostic F1 0.9474. Median alarm latency 0.80 s, start MAE 0.547 s, end MAE 0.472 s. Compared with V7, detection/ranking improve but end-time MAE and latency worsen. Public validation has been repeatedly tuned; hidden-test performance is unknown.

Each stream is processed independently. The new model uses up to 0.5 s future context; reported alarms include that delay. No repeated-content matching. No original LIARA recordings or external pretrained weights.

Run Run-Best.ps1 for validation. For test, pass -Data pointing to the extracted test package, -Split test and -EvaluatorPackage pointing to the training package (which supplies the unchanged official evaluator). Model fitting is never performed by predict.py. Hashes are checked before prediction.

V7 is preserved separately. Use SUBMISSION_DECLARATION.txt with the competition form where requested. A validator PASS confirms format, not test accuracy. See REPORT.md and RESEARCH.md for all experiment results and sources.
