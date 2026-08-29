# ZAPBench project instructions

## Authority and scope

- This file is the canonical source for the generic ZAPBench development, experiment, and RunPod workflow. Do not put transferable workflow rules in experiment notes.
- Before designing or executing an experiment, read the relevant notes under `/Users/s/vault/zapbench` when they exist. Notes provide experiment-specific scientific and execution detail, but their absence must not break the generic workflow and they do not authorize execution.
- Instruction precedence is: the user's current request, this file, then the relevant experiment note. Surface conflicts instead of choosing silently.
- The user defines each atomic experiment. Agree with the user on a short semantic experiment ID before implementation or dispatch; do not invent one silently.
- This workflow applies to every current and future model, not only TiDE or POCO, and must scale to dozens or hundreds of experiments without a central run registry.

## Execution target

- Treat `run locally` and `run on pod` as explicit, mutually exclusive execution targets.
- `Run locally` means execute the requested work entirely on the laptop and do not use SSH, `runpodctl`, or RunPod. It explicitly permits the requested model or data operations locally.
- `Run on pod` means execute the workload on the fixed RunPod. Local commands may control the pod and transfer code or artifacts, but model computation occurs remotely and follows the approved run contract below.
- If neither target is specified, perform local development only within the lightweight boundary below: do not execute a model and do not touch RunPod.
- Configuration fields such as `runlocal=True` do not select the physical execution target.
- If `run on pod` is requested before the exact job is settled, prepare the short run card and wait for its approval before changing pod state.

## Local development and review

- Local development may include reading and exploring code, configs, documentation, schemas, and lightweight metadata; implementing and refactoring code; reviewing diffs and tracing interfaces; formatting, linting, type checking, and syntax checks; small unit tests of pure utilities with synthetic inputs; and config parsing or serialization that does not open real data or initialize a model.
- Unless the user explicitly says `run locally`, do not initialize or execute any model; run forward or backward passes, training, inference, evaluation, or model smoke tests; exercise representative model data pipelines; load, scan, transform, or validate full datasets; or run accelerator, performance, or memory tests on the laptop.
- User-run notebook exploration is informal context, not a mandatory workflow gate. Do not create a notebook-validation checklist unless the user asks for one.
- Before proposing a run, review the final diff, configuration plumbing, paths, outputs, and failure behavior; run only the allowed local checks; and report both verified and execution-dependent behavior. This review does not require a separate approval from the run card.

## Source and configuration handoff

- Every repository state used by a run must be pinned to an exact commit that is available from its shared Git remote, and the pod must execute a clean checkout of that commit. A typical POCO-to-ZAPBench pipeline should use separate producer and consumer runs; list multiple repositories only when one process genuinely executes code from both.
- Never transfer or dispatch dirty or untracked source files. Unrelated local working-tree changes are not part of the run and need not be discarded; report relevant local differences instead. Do not create run-specific branches or tags automatically. Pushing is a separate action and is not implied by approval to use RunPod.
- Keep substantive model and configuration logic in reviewed, committed files that follow the repository's existing conventions. Supported concise selections and overrides such as dataset, context, seed, ablations, run ID, and output path may remain in the exact command.
- Do not build a generic experiment launcher, workflow engine, generated preflight system, manifest framework, separate requirements lock, or per-experiment bootstrap layer. Prefer the minimum-complexity working path through existing entry points.

## Time-series training and inference contract

- Treat `process_janelia/train.sh`, `process_janelia/train_zapbench.sh`, `process_janelia/pre-train.sh`, and `process_janelia/infer.sh` as the source of truth for the current time-series invocation pattern. They use repository-relative entry points and require an explicit `OUTPUT_ROOT` for their local batch layout. They do not replace the literal one-run commands in an approved RunPod run card; select a GPU externally only when the exact run requires it.
- Run the time-series entry points from the repository root. Training uses `zapbench/ts_forecasting/main_train.py` with a model config and concise overrides; inference uses `zapbench/ts_forecasting/main_infer.py`, with the completed training directory supplied as `exp_workdir`:

  ```bash
  python zapbench/ts_forecasting/main_train.py \
    --config zapbench/ts_forecasting/configs/<model>.py:dataset_name=<dataset>,runlocal=False,timesteps_input=<context> \
    --workdir <persistent-run-directory>/training

  python zapbench/ts_forecasting/main_infer.py \
    --config zapbench/ts_forecasting/configs/infer.py:exp_workdir=<persistent-run-directory>/training \
    --workdir <persistent-run-directory>/inference
  ```

- `dataset_name` must be an exact key in `zapbench/constants.py`. That registry, not the shell command, determines the timeseries, covariates, shapes, conditions, and data locations. Inspect the selected registry entry before approving a run. For local-file entries, set `ROOT_PATH` to the directory whose `ts_files/` child contains the registered stores; keep the variable and data under `/space` on RunPod.
- `runlocal=False` controls the training schedule inside the config; it does not authorize RunPod or select the physical execution target. Omit historical GPU indices on a one-GPU pod unless an exact run card requires an explicit `CUDA_VISIBLE_DEVICES` value.
- Keep training and inference as separate commands and directories. Inference reads checkpoints from `exp_workdir` and writes metrics beneath its own work directory. Do not infer a checkpoint from another run or from a directory named `latest`.
- For a future run, the minimal handoff is: choose the agreed experiment ID, model config, registered dataset key, context and other supported overrides; verify the corresponding `constants.py` entry and persistent input path; choose the exact training and optional inference work directories; then put the two literal commands in the run card. Do not wrap them in a new generic launcher.

## Fixed RunPod infrastructure

- Reuse only pod `hzqt2j4fi0av1z` and network volume `vqz96r3qy0`. Never create, substitute, or replace ZAPBench infrastructure, and never terminate or delete the fixed pod or volume; start and stop the fixed pod as authorized by the approved run lifecycle.
- Begin authorized pod discovery with `runpodctl pod list --all` so stopped pods are included. Start only pod `hzqt2j4fi0av1z` when the approved job requires it.
- Connect with `ssh hzqt2j4fi0av1z-64411adc@ssh.runpod.io -i ~/.ssh/id_ed25519`. If the recorded pod or endpoint cannot be verified, stop and ask instead of substituting infrastructure.
- The persistent root is `/space`; repositories live under `/space/git`, including `/space/git/zapbench` and `/space/git/POCO`; persistent data and outputs live under `/space/vault`.
- Treat `/workspace`, `/root`, `/opt`, and `/tmp` as ephemeral. Do not place any reusable environment, dependency, data, checkpoint, config, command, log, metric, or other retained artifact there.
- Use Conda to create, activate, and update ZAPBench environments. `zapbench-env.yaml` at the repository root is the operational environment definition; its editable `.[dev]` entry may in turn read package dependencies from `pyproject.toml`, but do not use a direct `pip install` as the environment setup workflow.
- The persistent Conda distribution lives at `/space/conda/miniforge3`; initialize it with `source /space/conda/miniforge3/etc/profile.d/conda.sh` before `conda activate zapbench`. Conda environments live under `/space/conda/envs`, and dependency caches live under `/space/.cache`.
- For every normal approved experiment run, reuse the installed environment without recreating, updating, or reinstalling it. After connecting to the fixed pod, the complete environment handoff is:

  ```bash
  source /space/conda/miniforge3/etc/profile.d/conda.sh
  conda activate zapbench
  cd /space/git/zapbench
  ```

  Then execute the exact approved command from the run card. The environment already stores `PIP_CACHE_DIR=/space/.cache/pip`; normal runs do not need to set it or install anything. If activation fails or an approved command reports a missing or incompatible dependency, stop the job and return to local development; do not repair the environment inside a normal experiment run.
- Creating, updating, or activating the environment does not require a GPU. Do not start or wait for GPU capacity solely for environment maintenance; accelerator requirements belong to an explicitly approved model run or environment change.
- Never silently fall back to CPU for a run that requires a GPU. If the existing environment has not already been validated for the run's required accelerator backend, treat that preparation and validation as an environment change requiring its own approved run card; do not improvise it inside a normal experiment run.
- If an approved GPU run cannot obtain capacity, keep the same run ID, commits, command, outputs, and approval, and leave its status `planned`. Capacity waiting alone does not require a revised run card.
- Attempt to start only pod `hzqt2j4fi0av1z`, and accept it only when it reports the GPU required by the run card. If startup fails for lack of capacity or the pod starts with `gpuCount: 0`, do not connect, activate the environment, or launch the job; stop a GPU-less running pod, report that the run is waiting for capacity, and retry the same pod every 10 minutes while the user has asked to wait. Never substitute infrastructure or fall back to CPU.
- Once the required GPU is present, continue the already approved lifecycle. If the user cancels while waiting, stop the pod if necessary and mark the run `cancelled`.
- Reuse the single existing Conda environment named `zapbench`. For initial setup or an approved dependency change, run `conda env create -f zapbench-env.yaml` or `conda env update -f zapbench-env.yaml` from the repository root, then `conda activate zapbench`. Make dependency changes in the committed environment or package metadata first; do not make undocumented one-off installs.
- On RunPod, the Conda installation, `zapbench` environment prefix, package cache, and every installed dependency must live under `/space`. Reuse the existing persistent environment before considering an update, and record the reason plus the before/after environment state for approved changes.
- A normal run assumes its environment and data are already prepared. Do not add a generic pod preflight or smoke-test layer. Any unusual setup or environment mutation must be written into and approved with that job's run card.

## Experiment, data, and run layout

- An experiment is one user-designed scientific question or comparison. A run is one exact execution of committed code and a selected configuration. One experiment may have multiple runs.
- Use `/space/vault/zapbench/experiments/<experiment-id>/runs/<run-id>/` for each persistent run. Generate run IDs from a UTC timestamp, short commit SHA, and an optional concise label such as a seed.
- Use `/space/vault/zapbench/data/<dataset-id>/` for new reusable datasets. Register datasets consumed by ZAPBench through a reviewed, committed entry in `zapbench/constants.py`; do not introduce a generic metadata-sidecar system unless a future experiment specifically needs one.
- Cross-experiment inputs must identify the producer experiment and run, exact persistent path, and checksum. Never infer `latest`. Do not duplicate large upstream artifacts merely to make a downstream run self-contained.
- Keep existing persistent datasets in place; do not reorganize them merely to match the new layout.

## Run card and approval

- Before dispatch, present one short, human-readable run card containing: experiment ID; generated run ID; exact commit for every participating repository; existing environment and input-data paths; exact command and config/overrides; persistent output directory; and expected terminal artifact or metric.
- The user's approval of that exact run card authorizes the complete lifecycle for that job only: start the fixed pod if needed, fetch and check out the approved commits, activate the existing environment, run the exact command, monitor it, collect results, and stop the pod after a verified terminal outcome.
- Any substantive change to code, configuration, command, data, environment, or outputs requires a revised run card and renewed approval.
- Store the approved card as `run.md` in the persistent run directory with initial status `planned`. Update it with `running`, then `succeeded`, `failed`, or `cancelled`, plus timestamps, exit status, key metrics, and artifact locations.

## Execution, failure, and return handoff

- Keep routine execution to the few commands required to fetch and check out the exact commits, activate the existing environment, and invoke the repository's existing entry point. Run long jobs under `tmux` and write stdout/stderr to `stdout.log` in the persistent run directory.
- Do not edit code, tweak configs, repair dependencies, or improvise reruns on the pod. A scientific or runtime failure returns to local development and requires a new reviewed commit, run ID, and approved run card. Relaunch the same command only when it demonstrably never started; do not retry automatically.
- Do not stop the pod while the approved job is still running merely because the controlling task is interrupted. Recover state from remote `run.md`, `tmux`, and logs. If the user explicitly cancels, terminate the job, preserve its partial status and log, then stop the pod.
- At terminal success or failure, verify that the job ended and required artifacts are readable under `/space`. Record key output paths and checksums where relevant.
- Copy a disposable inspection bundle to `/Users/s/git/zapbench/.runpod-runs/<experiment-id>/<run-id>/` before stopping. It should contain `run.md`, final metrics, and the full log or a useful log summary; large checkpoints, predictions, and datasets may remain on `/space` but must be listed in `run.md` with paths, sizes, and checksums where practical.
- The local inspection bundle is not canonical. Never delete it automatically, and never make future workflow depend on its existence; the user may remove it after inspection and keep only selected notes.
- After terminal verification and local handoff, stop idle compute with `runpodctl pod stop hzqt2j4fi0av1z`. Never delete network volume `vqz96r3qy0`.
- Never overwrite or automatically delete persistent experiment data or run artifacts. The user may clean them up manually, and future work must tolerate old notes or experiments being absent unless they are explicitly declared inputs.
