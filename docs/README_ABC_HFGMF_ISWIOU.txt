RT-DETR A+B+C merged version

A: IS-A640-GLF
- Effective code: val_a640_native_oldstyle.py
- Scripts: run_a640_oldstyle_adaptive.sh / run_a640_oldstyle_fixed.sh
- Meaning: input-space 640 global-local fusion module.

B: HF-GMF
- YAML: ultralytics/cfg/models/rt-detr/rtdetr-r18-hfgmf.yaml
- Modules: ultralytics/nn/uav_modules/block.py
- Scripts: run_hfgmf_train.sh, run_hfgmf_val.sh, run_hfgmf_ab_oldstyle.sh

C: IS-WIoU-Lite
- Code: ultralytics/models/utils/loss.py
- Script: run_iswiou_train.sh / run_iswiou_val.sh / run_iswiou_ac_oldstyle.sh
- Inspired by Wise-IoU, with input-space small-object gate.

B+C formal training from scratch:
cd /root/RTDETR
TORCH_CUDNN_V8_API_DISABLED=1 nohup bash run_bc_hfgmf_iswiou_train.sh none 200 8 rtdetr_r18_hfgmf_iswiou_200_b8 0.70 1 > train_bc_hfgmf_iswiou_200_b8.log 2>&1 &
tail -f train_bc_hfgmf_iswiou_200_b8.log

If batch 8 OOM:
TORCH_CUDNN_V8_API_DISABLED=1 nohup bash run_bc_hfgmf_iswiou_train.sh none 200 4 rtdetr_r18_hfgmf_iswiou_200_b4 0.70 1 > train_bc_hfgmf_iswiou_200_b4.log 2>&1 &

BC native validation:
bash run_bc_hfgmf_iswiou_val.sh runs/train/rtdetr_r18_hfgmf_iswiou_200_b8/weights/best.pt test BC_HFGMF_ISWIoU_native_test

ABC validation with A oldstyle:
bash run_abc_hfgmf_iswiou_oldstyle.sh runs/train/rtdetr_r18_hfgmf_iswiou_200_b8/weights/best.pt test ABC_HFGMF_ISWIoU_IS_A640_GLF_test

Keep these output dirs:
- runs/val_bc_hfgmf_iswiou/
- runs/val_abc_hfgmf_iswiou_a640_oldstyle/
