# Research rationale for future context plus hard-negative training

Reviewed primary publications/author records on 2026-10-25. Published classification results are not comparable to ABC2027 event F1 on unsegmented streams. No outside recordings, pretrained weights or paper code were added to the detector.

| Source | Relevant idea | Applicability and limit |
|---|---|---|
| [Abudalfa and Bouchard, 2025: hybrid vision transformer and residual network](https://pure.kfupm.edu.sa/en/publications/a-hybrid-vision-transformer-and-residual-neural-network-model-for/) | UWB fall classification with three radars; leave-one-subject-out evaluation | Especially relevant sensing setting and cited by the challenge. Its binary classification accuracy does not establish continuous temporal-localization F1. We reviewed the author institution's abstract, not an accessible full implementation. |
| [Room-Level Fall Detection with UWB and ConvLSTM, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7071514/) | Spatiotemporal features from radar ranging sequences | Supports modeling how radar motion evolves over time. This is an architectural direction, not proof it beats the current ensemble on ABC2027. Full-page access was inconsistent; indexed paper text was available. |
| [Huang et al., 2024: UWB adaptive weighted fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359866/) | Fuse range-time, time-frequency and range-frequency representations; suppress static clutter | Suggests complementary signal representations. Their classification protocol and radar settings differ. Spectral windows must be redesigned for this package's 50 Hz complex CIR; paper windows cannot be copied blindly. |
| [Shrivastava et al., 2016: online hard example mining](https://arxiv.org/abs/1604.03540) | Concentrate training on difficult examples in an imbalanced detection task | General detection principle, not a radar-specific result. Our method uses offline cross-fitted ADL negative selection and sample weights, not the paper's online CNN algorithm. |
| [MS-TCN, 2019](https://arxiv.org/abs/1903.01945), [author code](https://github.com/yabufarha/ms-tcn) | Multi-stage temporal refinement and a loss discouraging over-segmentation | Relevant to continuous labeling and fragmented predictions. It was developed for video, so any radar version needs new training/evaluation. Reviewed as a possible subsequent experiment, not implemented in this batch. |

## Why this experiment comes first

The previous paired-I/Q future-context group model improved F1 from 0.6026 to 0.6536 while reducing false alarms from 111 to 81 at unchanged 140 detected falls. Earlier hard-negative training helped a different short-history model. The interaction between these two changes was untested. Testing the combination gives a controlled comparison with existing saved reference models and requires no extra labels or external recordings.

## Fixed protocol

- Features: paired complex-motion statistics plus 0.5 s of future context, independently within each stream.
- Group checks: same V11 scopes, three provisional filename folds, 0.8 s guards at both segment edges, fixed decoder.
- Mining: ADL frames with negative labels and probability at least 0.2 from a model trained without that group; sample weight 5, versus weight 3 for positives.
- Nested isolation: outer held-out groups are excluded from both inner model fitting and negative mining. Three unique single-fold models cover the six inner directions without redundant fitting.
- Full continuous model: long-context feature version trained on the 13 training streams. Negative selection uses held-out-group predictions from V11 reference models. Validation is used only to select the decoder, not to fit models or choose negative examples.
- Compare standalone and a declared 50% V7 blend. No arbitrary blend search.
- Group support requires improved pooled F1, improvements in at least two folds and at most two additional missed falls. Promotion additionally requires higher full-validation mean F1 and no drop on the reused validation diagnostic subset.
- Group identities remain provisional. Previously examined validation streams are not an unbiased hidden-test estimate. This is a bounded experiment, not an exhaustive search or guarantee of winning.

README v1.1 explicitly permits non-causal independent-stream processing. This experiment does not match repeated signal content or use original LIARA source recordings. Future-context alarm timestamps include the 0.5 s look-ahead.
