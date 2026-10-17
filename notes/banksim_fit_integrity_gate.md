# BankSim policy diagnostic acceptance gate

Use only the already fitted BankSim model and calibrator. Verify recorded feature, label and artifact SHA-256 hashes before reading policy labels. Keep final-test labels sealed. Apply the frozen global review cutoff 0.41746818058367946 with 1% prefix capacity; never enable BLOCK. Write one immutable diagnostic report and do not use policy outcomes to change the trained model, calibration or threshold.

This diagnostic can evaluate the frozen policy on BankSim steps 144–161 but is not independent validation of the original PaySim model.
