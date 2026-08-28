# Video forecasting

For details on the video forecasting models included in ZAPBench, refer to [our ICLR paper](https://openreview.net/pdf?id=oCHsDpyawq). Extensive model selection and pretraining results with video forecasting models are in [Immer et al. (2025)](https://arxiv.org/abs/2503.00073).

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

## Training

```bash
# See configs/ for available models and options.
python zapbench/video_forecasting/main_train.py \
  --config zapbench/video_forecasting/configs/unet_test.py \
  --workdir /dir/for/training
```

## Inference

```bash
# See configs/infer.py for additional options.
python zapbench/video_forecasting/main_infer.py \
  --config zapbench/video_forecasting/configs/infer.py:/dir/for/inference \
  --config.exp_workdir=/dir/for/training \
  --workdir /dir/for/inference
```

Metrics are written to a subdirectory within `/dir/for/inference` as JSON files,
according to the `json_path_prefix` setting in the config.

The json-files can be turned into a pandas dataframe using a utility function:

```python
from zapbench.ts_forecasting import util

df = util.get_per_step_metrics_from_directory(
  '/dir/for/inference/subdir/with/metrics',
  metric='trace_step_mae')  # or: trace_step_mse
```
