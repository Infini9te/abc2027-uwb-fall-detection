# Future context plus hard-negative training: results

This batch trained nine models: three inner mining models, three outer held-out-group models, one long-context full model, and a matched short-context reference/hard-negative pair. It reused and exactly replayed the prior V11 group reference models. All model fitting and negative mining used the released training streams. No official test data or external training data was used.

## Provisional group comparison

Same 186 evaluable falls, same guarded segment interiors, same fixed decoder. These are not V7 scores and not certified subject-disjoint results.

| Model | Ranking F1 | TP at .3 | FP | FN | Median latency |
|---|---:|---:|---:|---:|---:|
| Future-context reference | 0.6536 | 140 | 81 | 46 | 0.830 |
| Future context + hard negatives | 0.7110 | 141 | 49 | 45 | 0.840 |

| Fold | Reference F1 | Hard-negative F1 | Mined training frames |
|---|---:|---:|---:|
| 0 | 0.5892 | 0.7020 | 1069 |
| 1 | 0.7420 | 0.8162 | 1124 |
| 2 | 0.6521 | 0.6375 | 1017 |

Predeclared group support gate passed: **True**. The gate requires improved pooled F1, improvement in at least two folds, and no more than two additional missed falls.

## Continuous public validation

Decoder selection uses val_00..02. Val_03..04 has been reused and is only a diagnostic. All scores use the unchanged official evaluator with flagged events ignored.

| Model | Ranking F1 | TP at .3 | FP | FN | Start MAE | End MAE | Median latency | Diagnostic F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 | 0.8284 | 57 | 12 | 10 | 0.571 | 0.322 | 0.380 | 0.8571 |
| V11 future reference (1.0 new weight) | 0.8099 | 56 | 12 | 11 | 0.601 | 0.363 | 1.030 | 0.9123 |
| V11 future reference (0.5 new weight) | 0.8137 | 57 | 12 | 10 | 0.721 | 0.362 | 0.840 | 0.8772 |
| alone | 0.8039 | 56 | 13 | 11 | 0.606 | 0.318 | 0.880 | 0.9123 |
| blend_v7 | 0.8247 | 57 | 11 | 10 | 0.515 | 0.362 | 0.920 | 0.9123 |
| short_reference/alone | 0.8102 | 59 | 18 | 8 | 0.511 | 0.402 | 0.880 | 0.8421 |
| short_reference/blend_v7 | 0.8537 | 62 | 10 | 5 | 0.547 | 0.472 | 0.800 | 0.9474 |
| short_hard/alone | 0.8169 | 60 | 15 | 7 | 0.425 | 0.456 | 1.070 | 0.8889 |
| short_hard/blend_v7 | 0.8333 | 59 | 10 | 8 | 0.550 | 0.432 | 0.960 | 0.9474 |

## Decision and checks

Candidates passing all gates: **2**. V7 has not been replaced.
The short-reference control is assessed against its own prior V11 future-versus-raw group evidence; it is not credited with gains from hard-negative training. Any selection of that control must be described as short future-context blending, not as a hard-negative success.
- Negative mining excluded the outer held-out group. The final model used out-of-group ADL predictions for mining.
- All prior reference group scores and primary metrics were reproduced exactly before comparison.
- Saved full-model replay maximum probability difference: 0.0.
- Frozen V7, evaluator and original test-submission hash checks passed: True.
- Alarm times include the 0.5 s look-ahead. No hidden test scores can be computed without ground truth.
- Fixed thresholds for mining/weights were not searched on outer-group outcomes. Full validation decoder tuning still carries selection optimism.

## Research sources and implementation

See [RESEARCH.md](RESEARCH.md) for primary-paper links and the distinction between radar classification and continuous temporal localization. The implemented method is an offline cross-fitted hard-negative adaptation; it is not a claimed reproduction of any paper architecture. Models, mined indices, probability arrays, event-level errors and configurations are retained in this directory.

Reproduce with `outputs/Run-Future-Hard-V12.ps1`. Completed model fits are reused when present.
