# Additional model experiments — October 25, 2026

Eleven new models were fitted: three XGBoost, three LightGBM, two CatBoost, CNN-LSTM, causal local attention, and a boundary-supervised causal TCN. Each was compared alone and in a fixed 50:50 V5 blend (22 configurations). Each configuration used 48 decoder candidates. Six further boundary refinements were rejected. Earlier experiments also covered histogram gradient boosting, Extra Trees, Random Forests and several temporal CNN seeds. This is a finite experiment budget, not every possible model or hyperparameter.

| Candidate | Selection mean F1 (val_00..02) |
|---|---:|
| xgb_motion4_plus_v5 | 0.823188 |
| xgb_base6_plus_v5 | 0.818713 |
| cat_motion6_plus_v5 | 0.814159 |
| boundary_tcn_plus_v5 | 0.807910 |
| cat_base6_plus_v5 | 0.807018 |
| lgb_motion31_plus_v5 | 0.804598 |
| lgb_motion15_plus_v5 | 0.803419 |
| xgb_motion4 | 0.803279 |
| xgb_motion6_plus_v5 | 0.800000 |
| xgb_motion6 | 0.793388 |
| cat_base6 | 0.792023 |
| cat_motion6 | 0.788889 |
| cnn_lstm_plus_v5 | 0.787356 |
| lgb_base31_plus_v5 | 0.785714 |
| lgb_motion31 | 0.783333 |
| lgb_motion15 | 0.776860 |
| local_attention_plus_v5 | 0.774929 |
| xgb_base6 | 0.769231 |
| lgb_base31 | 0.760234 |
| cnn_lstm | 0.757835 |
| boundary_tcn | 0.733333 |
| local_attention | 0.715847 |

V5 selection mean F1: 0.821839. Winner: XGBoost motion depth 4 plus V5. Full validation mean F1: 0.828431; diagnostic: 0.857143. Full and diagnostic values are not the selection-column values above.

Training weights and normalizers used training data only. New boosting iteration counts and neural epoch counts used internal train_00..09 / train_10..12 selection, followed by refitting all 13 training streams. Internal streams are not guaranteed subject-disjoint. External validation was used for decoder/model selection and has been repeatedly examined; val_03..04 is a reused nine-fall diagnostic, not an independent test. No hidden-test score or significance claim is available. The 0.004902 full-validation gain is small and may not generalize.

## Research consulted
- [Original UWB CNN-LSTM study](https://pubmed.ncbi.nlm.nih.gov/33017299/): motivated the compact recurrent experiment. Our feature-based model is not a reproduction.
- [UWB transfer-learning study](https://www.sciencedirect.com/science/article/pii/S2352146525000870) and [hybrid ViT/ResNet study](https://link.springer.com/article/10.1007/s10489-024-06156-9): publisher abstracts describe classification results, which cannot be compared directly to continuous event-localization F1. No pretrained vision backbone or prohibited source corpus was used.
- [ActionFormer](https://arxiv.org/abs/2202.07925): local attention and temporal boundary regression inspired compact adaptations; this is not an ActionFormer implementation.
- [MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html): motivated temporal smoothing regularization, not a full multi-stage reproduction.
- Official implementation documentation: [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).

Training scripts and individual manifests are included for inspection. Re-running the whole study requires the original outputs/run cache and V5 experiment artifacts in the workspace; the prediction package itself is self-contained apart from Python dependencies and the official challenge package. No external dataset was used.
