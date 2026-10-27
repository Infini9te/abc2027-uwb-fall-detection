# Experiment results

Eight new models were trained: five tree variants, two 16-epoch temporal networks with different seeds, and one temporal network with a declared one-second delay. Eighteen single-model/ensemble configurations were compared, each with an 48-setting decoder search; 27 additional boundary refinements were tried on the selected configuration. Selection used val_00..02. The previously seen val_03..04 were a final rejection diagnostic, not an untouched test.

| Candidate | Tuning mean F1 |
|---|---:|
| top_two | 0.813559 |
| motion_only | 0.805797 |
| tcn16_delay1_plus_v3 | 0.800000 |
| tcn16_seed2029_plus_v3 | 0.793103 |
| motion_only_plus_v3 | 0.792023 |
| blend_v3 | 0.783333 |
| tcn16_seed2029 | 0.778711 |
| boost_long_plus_v3 | 0.774011 |
| boost_long | 0.773109 |
| tcn16_delay1 | 0.769648 |
| hard_negatives_plus_v3 | 0.764228 |
| tcn16_seed2028_plus_v3 | 0.761905 |
| tcn16_seed2028 | 0.761111 |
| boost_context_plus_v3 | 0.759563 |
| hard_negatives | 0.759420 |
| extra_trees_plus_v3 | 0.726776 |
| extra_trees | 0.726287 |
| boost_context | 0.697819 |

Selected: 50% motion-only boosted trees + 50% temporal network seed 2029. No extra boundary refinement or added one-second delay was selected.

Verification: unchanged official evaluator SHA256; exact saved-model probability replay for all five validation streams; packaged relative-path model replay; unlabeled synthetic test-stream inference; causal feature-prefix checks; temporal network prefix/chunk equivalence and finite gradients; boundary alarm preservation; explicit delayed-alarm accounting.

The synthetic test verifies the code path only. It is not an estimate of performance on the withheld competition test. Validation has been repeatedly examined; no winning or perfect-detection claim is justified.