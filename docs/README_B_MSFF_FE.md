# B innovation patch: RT-DETR + MSFF-FE

This package keeps the existing A innovation code:

- `val_a640_native_oldstyle.py`
- `run_a640_oldstyle_fixed.sh`
- `run_a640_oldstyle_adaptive.sh`

and adds the B candidate copied/adapted from UAV-DETR:

- `ultralytics/nn/uav_modules/block.py`
- `ultralytics/nn/uav_modules/__init__.py`
- `ultralytics/cfg/models/rt-detr/rtdetr-r18-msff-fe.yaml`

## What was added

The B module is MSFF-FE / MFFF from UAV-DETR. In this implementation, only the MSFF-FE-related P2/P3/Y4 fusion and MFFF frequency enhancement are enabled. UAV-DETR's FD and SAC are **not** enabled in the main MSFF-FE YAML, so the test is cleaner and focuses on one B module.

A full UAV-DETR reference YAML is also included as:

- `ultralytics/cfg/models/rt-detr/uavdetr-r18-full-reference.yaml`

Do not use it for the B-only experiment unless you intentionally want to test full UAV-DETR.

## Step 1: Train B only

Formal training from scratch:

```bash
cd /root/RTDETR
nohup bash run_msff_fe_train.sh none 200 4 rtdetr_r18_msff_fe_200 > train_msff_fe.log 2>&1 &
tail -f train_msff_fe.log
```

Quick warm-start debugging from baseline weight, not recommended for final paper unless you clearly report it:

```bash
cd /root/RTDETR
nohup bash run_msff_fe_train.sh /root/best.pt 50 4 rtdetr_r18_msff_fe_debug > train_msff_fe_debug.log 2>&1 &
```

## Step 2: Validate B only

After training finishes, run native validation:

```bash
cd /root/RTDETR
bash run_msff_fe_val.sh runs/train/rtdetr_r18_msff_fe_200/weights/best.pt test B_MSFF_FE_native_test
```

This gives B-only paper data and plots under:

```text
runs/val_msff_fe/B_MSFF_FE_native_test/
```

## Step 3: Validate A+B

Use the trained B weight with the existing A oldstyle input-space slicing module:

```bash
cd /root/RTDETR
bash run_msff_fe_ab_oldstyle.sh runs/train/rtdetr_r18_msff_fe_200/weights/best.pt test AB_MSFF_FE_IS_A640_GLF_test
```

This gives A+B results under:

```text
runs/val_ab_msff_a640_oldstyle/AB_MSFF_FE_IS_A640_GLF_test/
```

## Recommended paper table

- Baseline RT-DETR: existing `runs/val_native/native_baseline_test`
- A: existing `runs/val_a640_oldstyle/A640_oldstyle_adaptive_test`
- B: `runs/val_msff_fe/B_MSFF_FE_native_test`
- A+B: `runs/val_ab_msff_a640_oldstyle/AB_MSFF_FE_IS_A640_GLF_test`

## Notes

- A is still IS-A640-GLF, not adaptive slicing.
- B is a true network structure modification, because the model YAML and `ultralytics/nn/uav_modules` add MSFF-FE/MFFF inside the RT-DETR neck/fusion stage.
- I did not delete old A files or old archived scripts, because the current priority is keeping the project runnable on the server.


## If training reports an FFT/half precision error

The copied MSFF-FE module uses FFT operations. I added float32 casting inside the FFT path, so AMP should usually work. If your server still reports a `torch.fft` or `half` precision error, rerun with the fifth argument set to `1` to disable AMP:

```bash
cd /root/RTDETR
nohup bash run_msff_fe_train.sh none 200 4 rtdetr_r18_msff_fe_200_noamp 1 > train_msff_fe_noamp.log 2>&1 &
```
