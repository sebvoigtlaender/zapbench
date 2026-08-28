# Time-series forecasting

For details on the time-series forecasting models included in ZAPBench, refer to [our ICLR paper](https://openreview.net/pdf?id=oCHsDpyawq).

## Setup

Run setup from the repository root:

```bash
git clone https://github.com/google-research/zapbench
cd zapbench

conda env create -f zapbench-env.yaml
conda activate zapbench
```

Use `conda env update -f zapbench-env.yaml` from the repository root when the
committed environment definition changes.

Dataset selection is registry-based. The `dataset_name` override must be an
exact key in `zapbench/constants.py`; that entry supplies the data locations,
shapes, conditions, timeseries, and covariates. Local Janelia entries resolve
their `ts_files/` paths relative to `ROOT_PATH`.

The scripts under `process_janelia/` are the operational reference for the
current time-series training and inference command structure. Their absolute
paths and GPU index are specific to the machine on which they were written;
replace those values for the current host rather than running the scripts
verbatim. See `process_janelia/README.md` for concise templates.

## Training

```bash
python zapbench/ts_forecasting/main_train.py \
  --config zapbench/ts_forecasting/configs/mean.py:dataset_name=240930_traces,runlocal=False,timesteps_input=4 \
  --workdir /persistent/run/directory/training
```

Replace `mean`, the registered dataset key, and supported config overrides for
the exact run. `runlocal` changes the configured training schedule; it does not
select whether the command runs on a laptop or RunPod.

## Inference

```bash
python zapbench/ts_forecasting/main_infer.py \
  --config zapbench/ts_forecasting/configs/infer.py:exp_workdir=/persistent/run/directory/training \
  --workdir /persistent/run/directory/inference
```

Metrics are written to a subdirectory within the inference work directory as JSON files,
according to the `infer_prefix` setting in the config.

The json-files can be turned into a pandas dataframe using a utility function:

```python
from zapbench.ts_forecasting import util

df = util.get_per_step_metrics_from_directory(
  '/persistent/run/directory/inference/subdir/with/metrics',
  metric='MAE')  # or: MSE
```
