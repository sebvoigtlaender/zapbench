# Janelia and time-series run commands

The shell scripts in this directory are the operational reference for how the
current dataset-aware time-series code is trained and evaluated. In particular:

- `train.sh` selects individual `subject_NN` entries from
  `zapbench/constants.py`.
- `train_zapbench.sh` selects the `240930_traces` ZAPBench entry.
- `pre-train.sh` selects the combined `janelia_pretrain` entry.
- `infer.sh` passes each completed training directory to inference through
  `exp_workdir` and writes results to a separate inference directory.

The scripts were written for a machine with repositories under
`/home/sebastian`, storage under `/mnt/storage`, and a usable GPU at index 7.
Those literal paths and the GPU index are historical. They are not portable
RunPod commands and should not be executed there verbatim.

## Minimal command pattern

Run from the repository root in the `zapbench` Conda environment. On the fixed
RunPod, initialize the persistent environment first:

```bash
cd /space/git/zapbench
source /space/conda/miniforge3/etc/profile.d/conda.sh
conda activate zapbench
```

Train one selected model, registered dataset, and context:

```bash
python zapbench/ts_forecasting/main_train.py \
  --config zapbench/ts_forecasting/configs/<model>.py:dataset_name=<dataset>,runlocal=False,timesteps_input=<context> \
  --workdir <persistent-run-directory>/training
```

Run inference from that exact training directory:

```bash
python zapbench/ts_forecasting/main_infer.py \
  --config zapbench/ts_forecasting/configs/infer.py:exp_workdir=<persistent-run-directory>/training \
  --workdir <persistent-run-directory>/inference
```

`<model>` is a config name under `zapbench/ts_forecasting/configs/`, such as
`mean`, `linear`, `timemix`, `tsmixer`, or `tide`. `<dataset>` must be an exact
key in `zapbench/constants.py`. The selected registry entry is the authority
for data paths and shapes; check it before running. For the `subject_NN`
entries, set `ROOT_PATH` to the directory containing the required `ts_files/`
subdirectory.

For RunPod experiments, replace `<persistent-run-directory>` with the approved
path under `/space/vault/zapbench/experiments/<experiment-id>/runs/<run-id>` and
put the resulting literal commands in the run card. Use one command for the
atomic run instead of modifying these historical batch scripts or adding a new
launcher.
