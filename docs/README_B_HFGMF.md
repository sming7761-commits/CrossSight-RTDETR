# B Innovation: HF-GMF (MSFF-FE based)

This version keeps the effective MSFF-FE/MFFF frequency-enhanced fusion branch from UAV-DETR and adds a lightweight high-frequency residual gate. The gate is initialized close to 1.0, so the module starts very close to original MSFF-FE behavior while allowing adaptive high-frequency residual selection during training.

Paper wording suggestion: Inspired by UAV-DETR, we design a High-Frequency Gated Multi-scale Fusion (HF-GMF) module by introducing a lightweight residual gate into the frequency-enhanced multi-scale fusion branch for UAV small object detection. Please cite UAV-DETR as the source of the frequency-enhanced fusion idea.

Main files:
- ultralytics/nn/uav_modules/block.py: original MFFF is retained; new HFGMF is added.
- ultralytics/cfg/models/rt-detr/rtdetr-r18-hfgmf.yaml: B model used for formal training.
- run_hfgmf_train.sh: train B.
- run_hfgmf_val.sh: validate B.
- run_hfgmf_ab_oldstyle.sh: evaluate A+B with IS-A640-GLF.
