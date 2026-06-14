C innovation: IS-WIoU-Lite (Input-Space aware Wise-IoU Lite Loss)

Source basis:
- Wise-IoU v3: Bounding Box Regression Loss with Dynamic Focusing Mechanism
- Official code: https://github.com/Instinct323/Wise-IoU

Our adaptation:
1) Keep original RT-DETR L1 regression loss and Hungarian matcher unchanged.
2) Replace the matched GIoU term with a conservative mixture:
   L_reg = (1 - mix) * L_GIoU + mix * L_WIoU
3) Add normalized input-space small-object gate, so WIoU contributes more to UAV small objects.
4) No network structure change and no inference-time cost.

Recommended quick test:
TORCH_CUDNN_V8_API_DISABLED=1 nohup bash run_iswiou_train.sh /root/best.pt 20 8 rtdetr_r18_iswiou_finetune20 0.70 1 > train_iswiou_finetune20.log 2>&1 &

Test C:
bash run_iswiou_val.sh runs/train/rtdetr_r18_iswiou_finetune20/weights/best.pt test C_ISWIoU_finetune20_test

Test A+C:
bash run_iswiou_ac_oldstyle.sh runs/train/rtdetr_r18_iswiou_finetune20/weights/best.pt test AC_ISWIoU_finetune20_IS_A640_GLF_test

If it works, run 200 epochs from scratch or from baseline depending on final experiment plan.
