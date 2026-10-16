# FDH replication: amended-01 exploratory recovery

The original pre-registered experiment was started under commit d233ce7. It failed on seed_104729__default due to unsigned NumPy subtraction during paired outcome reporting. The warning implies a negative paired difference, but the exact value was not saved. The original lock, local start marker and empty run directory must be preserved.

This amendment changes the output namespace to reports/fdh_amended_01_* and uses the incident-01 corrected evaluator. The same three original seeds and three regimes are retained, including the first touched condition. These are **exploratory amended results**, not untouched confirmatory evidence. No settings, models, seeds, thresholds or fraud rules are changed in response to the observed failure. The original run's start marker remains on disk.

Procedure: execute the existing tests and new amended-driver tests; prepare the amended lock without generating events; inspect, commit and push that lock; verify exact committed source/lock/environment and original incident markers. Only after review consider running amended one-time execution. The runner must never delete or overwrite original evidence. Preserve amended start marker if anything fails.

The simplified independently written FDH-inspired generator is not equivalent to the upstream Handbook notebook. Upstream source code is GPLv3; an appropriate license review remains required before release. No real-bank, cross-simulator or production deployment claims are authorized.