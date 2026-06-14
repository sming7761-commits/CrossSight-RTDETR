# Experiment scripts

The shell entry points are grouped by purpose.

## `train/`

Official training workflows:

- `run_baseline_200_clean.sh`
- `run_hfgmf_train.sh` — DetailGate implementation
- `run_iswiou_train.sh` — TinyLoc implementation
- `run_bc_hfgmf_iswiou_train.sh` — DetailGate + TinyLoc

## `eval/`

Official and component evaluation workflows:

- native module validation scripts
- AB and AC evaluations
- full A+B+C evaluation
- native plot generation

## `ablation/`

InSight-640 variants and metric-control experiments:

- adaptive and fixed local views
- native and historical evaluators
- slice-only variants
- no-slice controls
- batch ablation launchers

## `legacy/`

Historical MSFF-FE scripts retained for provenance. They are not the recommended entry points for the current paper.

## Working-directory handling

Every script resolves the repository root before running. Scripts can therefore be launched from the repository root using paths such as:

```bash
bash scripts/train/run_baseline_200_clean.sh
```

or from another directory using an absolute script path.
