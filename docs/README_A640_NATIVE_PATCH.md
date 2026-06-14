# A640-GLF Native Validator Patch

把这些文件复制到你的 `/root/RTDETR/` 根目录即可，不需要替换 `val.py`，也不需要重训权重。

新增文件：

- `val_a640_native.py`：A640-GLF 的 Ultralytics 原生指标口径验证脚本。
- `run_a640_native_adaptive.sh`：跑最终 A 创新点：全图 + 自适应 640 切图 + 融合。
- `run_a640_native_fixed.sh`：跑全图 + 固定 640 切图 + 融合消融。
- `run_a640_native_slice_only.sh`：跑仅 640 切图消融。
- `run_a640_native_no_slice.sh`：用 A640 native 脚本检查 no-slice 口径。
- `run_all_a640_native_ablation.sh`：一键跑原生 baseline + A640 native 消融。

## 推荐运行

```bash
cd /root/RTDETR
unzip -o /root/a640_native_patch.zip
chmod +x run_a640_native_*.sh run_all_a640_native_ablation.sh

# 先验证集
nohup bash run_all_a640_native_ablation.sh /root/best.pt val > a640_native_val.log 2>&1 &
tail -f a640_native_val.log

# 再测试集
nohup bash run_all_a640_native_ablation.sh /root/best.pt test > a640_native_test.log 2>&1 &
tail -f a640_native_test.log
```

## 只跑最终 A 创新点

```bash
cd /root/RTDETR
bash run_a640_native_adaptive.sh /root/best.pt test A640_GLF_adaptive_native_test
```

输出在：

```text
runs/val_a640_native/A640_GLF_adaptive_native_test/paper_data.txt
runs/val_a640_native/A640_GLF_adaptive_native_test/metrics_summary_native.json
```

## 注意

论文主表的原始 RT-DETR baseline 仍建议使用你原来的：

```bash
bash run_val_native_plots.sh /root/best.pt test native_baseline_test
```

A640-GLF 则用：

```bash
bash run_a640_native_adaptive.sh /root/best.pt test A640_GLF_adaptive_native_test
```

这样两边都使用 Ultralytics 原生 DetMetrics/AP/绘图逻辑。
