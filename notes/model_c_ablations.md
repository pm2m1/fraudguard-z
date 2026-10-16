# Model C ablations (development experiment)

Run on Mac from the repository root:

```bash
source .venv/bin/activate
python -m pytest tests -q
python -m src.model_c_ablations
cat reports/model_c_ablations_validation.json
```

The script trains four separate HistGradientBoostingClassifier models using the **same hyperparameters and original 1–238 training / 239–306 validation split**:

- C0: Original eight Model B features, nonlinear learner.
- C1: Those eight plus six past-only sender/recipient history features. It excludes explicit current amount/balance relationship features but still includes current amount and balance, so the learner may infer their relationship.
- C2: Five transaction type indicators plus six past-only history features. Excludes all current amount and current sender-balance features.
- C3: All 17 Model C features.

It also includes the original logistic Model B ranking as reference and reports the positive exact-balance rule separately. The rule is not quota-constrained or a deployed streaming policy.

The output is **exploratory validation evidence only**. Repeated model comparisons use an already-observed validation set; the original final holdout has also been inspected. Do not claim new independent generalization evidence, do not deploy uncalibrated `predict_proba` scores, and do not retune the frozen Model B threshold. No model artifacts are overwritten; the script only writes `reports/model_c_ablations_validation.json`.

This full-data experiment fits four models on 3,177,938 rows and may require substantial memory and time. If terminated, no production changes occur.
