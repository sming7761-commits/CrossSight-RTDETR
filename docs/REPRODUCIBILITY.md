# Reproducibility Guide

This checklist records the minimum information required to reproduce the experiments reported for CrossSight-RTDETR.

## Environment

- Operating system: Ubuntu 22.04.3 LTS
- Python: 3.10.16
- PyTorch: 2.2.2
- CUDA: 12.1
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- CPU: 16 cores
- Memory: 48 GB

## Core settings

- Detector: RT-DETR-R18
- Full-view input: 960 × 960
- Local-view size: 640 × 640
- Main training duration: 200 epochs
- VisDrone classes: 10
- Main local-view evaluator: `val_a640_native_oldstyle.py`

## Component mapping

| Paper component | Code identifier |
|---|---|
| InSight-640 | A640 |
| DetailGate | HF-GMF |
| TinyLoc | ISWIoU |
| DetailGate + TinyLoc | HF-GMF + ISWIoU |
| Full framework | A640 + HF-GMF + ISWIoU |

## Main full-evaluation settings

The current full script uses:

```text
--mode adaptive
--tile 640
--overlap 0.20
--conf 0.001
--max-det 1000
--merge-iou 0.55
```

Do not describe the main result as fixed-view inference unless the provenance of the reported checkpoint and result has been independently confirmed.

## Dataset protocol

For every dataset, record:

1. Official dataset source and version.
2. Training, validation, and test split definitions.
3. Class names and class-index mapping.
4. Annotation conversion procedure.
5. Input size and preprocessing.
6. Random seed.
7. Checkpoint selection rule.
8. Evaluation split and thresholds.
9. Whether each model was trained independently on that dataset.

The experiments in the manuscript train and evaluate models separately for each dataset. They should be described as multi-dataset applicability or cross-domain adaptability, not zero-shot domain generalisation.

## Result provenance checklist

Before reporting a number, record:

- Git commit hash
- Script name
- Model configuration
- Checkpoint path and SHA256
- Dataset configuration
- Evaluation split
- Input size
- Local-view mode
- Tile size and overlap
- Confidence threshold
- Prediction IoU threshold, if used
- Merge IoU threshold
- Maximum detections
- GPU and batch size
- Output directory

## Clean-room verification

Before creating a release:

```bash
git clone https://github.com/sming7761-commits/CrossSight-RTDETR.git
cd CrossSight-RTDETR
conda env create -f environment.yml
conda activate crosssight-rtdetr
pip install -e .
```

Then run at least:

1. Import smoke test.
2. Native baseline validation.
3. DetailGate + TinyLoc validation.
4. Full InSight-640 evaluation on a small subset.
5. One complete test-split reproduction using a released checkpoint.

## Release checklist

- [ ] README commands tested from a fresh clone
- [ ] All shell scripts use LF line endings
- [ ] No local absolute paths remain
- [ ] No datasets, checkpoints, caches, or `runs/` directories are tracked
- [ ] Main result checkpoint uploaded to GitHub Releases
- [ ] SHA256 checksums published
- [ ] Dataset configurations for all reported datasets included
- [ ] Final paper citation added
- [ ] Version tag created
