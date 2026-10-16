# Provisional calibration gate — frozen before executing

The 21-feature state engine remains the complete contextual data contract.
The initial scored model is deliberately narrower: the existing
`logistic_type_history_network.joblib` from
`data/processed/full_context_no_current_shortcut_v1`, using exactly
16 columns (indices 1,2,3,4,5,9,10,11,13,14,15,16,17,18,19,20).

Reason: N2 dominates the historical network improvement, and the
nonlinear transaction-based models exhibit severe PaySim-specific
amount/balance shortcuts. The *logistic* network model is a transparent
provisional comparison, not a confirmed generalizable model.
N2 is not a causal explanation or a reason to block payments.

Source training rows 0:3,177,938 and validation
3,177,938:4,263,279 were previously inspected. This gate **does not
refit** the base model or use validation labels. Fit Platt/sigmoid
calibration on calibration rows 4,263,279:4,759,371 (286 frauds)
only, using uncalibrated model `decision_function` margins.
Reject reversed/non-finite calibration. Report Brier, log loss, AP,
positive prevalence, and mean prediction on calibration as *fit-period
diagnostics*, not held-out calibration quality. No policy scores or
policy labels are inspected in this gate.

Output a new isolated artifact directory under ignored
`data/processed/full_context_calibrated_logistic_v1`.
Record feature positions, model SHA256, calibration object SHA256,
source manifest SHA256 and schema SHA256. Never overwrite.
Verify the local source artifact checksum against the archived
shortcut-sensitivity comparison. Treat joblib as trusted local data only.

A later, separately frozen policy step will use policy partition
4,759,371:5,285,888; choose review and (if justifiable) conservative
simulated block thresholds with explicit costs and false-positive
bounds, report a causal sequential review budget, and allow zero BLOCK.
No real-world blocking and no LinuxONE until full local acceptance.
The original final holdout is already inspected and is not used.
