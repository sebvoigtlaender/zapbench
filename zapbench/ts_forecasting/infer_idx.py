# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods for inference, janelia."""

import os
from collections.abc import Sequence

import connectomics.jax.metrics as metrics_lib
import flax.jax_utils as flax_utils
import flax.linen as nn
import grain.python as grain
import jax
import ml_collections
import numpy as np
from absl import logging
from clu import platform  # pylint: disable=unused-import
from clu import metric_writers, parameter_overview, periodic_actions
from connectomics.common import ts_utils
from connectomics.jax import checkpoint, training
from etils import epath
from orbax import checkpoint as ocp

import zapbench.models.util as model_util
from zapbench.ts_forecasting import heads, input_pipeline, train


def _get_checkpoint_step(
    checkpoint_manager: ocp.CheckpointManager,
    selection_strategy: str,
) -> int | None:
  """Returns the checkpoint step to use given a selection strategy.

  Args:
    checkpoint_manager: Checkpoint manager.
    selection_strategy: Checkpoint selection strategy, can be 'early_stopping',
      'best_val_loss', or 'latest'.

  Returns:
    Checkpoint step.
  """
  if selection_strategy == 'early_stopping':
    checkpointed_state = dict(
        early_stop=None,
    )
    checkpointed_state = checkpoint.restore_checkpoint(
        checkpoint_manager,
        state=checkpointed_state,
        step=checkpoint_manager.latest_step(),
    )
    return checkpointed_state['early_stop']['best_step']
  elif selection_strategy == 'best_val_loss':
    checkpointed_state = dict(
        track_best_val_loss_step=None,
    )
    checkpointed_state = checkpoint.restore_checkpoint(
        checkpoint_manager,
        state=checkpointed_state,
        step=checkpoint_manager.latest_step(),
    )
    return checkpointed_state['track_best_val_loss_step']['best_step']
  elif selection_strategy == 'latest':
    return checkpoint_manager.latest_step()
  else:
    raise ValueError(f'Unknown checkpoint selection: {selection_strategy}')


def infer_single_step(
    model: nn.Module,
    head: heads.Head,
    train_state: train.TrainState,
    data_source: grain.RandomAccessDataSource,
    idx: int,
    infer_key: jax.Array,  # pylint: disable=unused-argument
    covariates: Sequence[str] = (),
    covariates_static: jax.Array | None = None,
    with_carry: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Runs independent inference on each index in the test set.

  Returns:
    prediction: prediction array
    target: target array
  """
  carry = None

  batch = data_source[idx]
  if 'covariates_static' in covariates:
    batch['covariates_static'] = covariates_static

  out = train.pred_step(
      model,
      train_state,
      batch,
      covariates,
      initial_carry=carry,
      return_carry=with_carry,
  )

  if not with_carry:
    dist = head.get_distribution(out)
  else:
    carry, dist = out[0], head.get_distribution(out[1])

  prediction = dist.mode()
  target = batch['timeseries_output']

  return prediction, target


def inference(infer_config: ml_collections.ConfigDict, workdir: epath.PathLike):
  """Runs inference.

  Args:
    infer_config: Inference configuration to use.
    workdir: Working directory for inference.

  Returns:
    Checkpoint dictionary.
  """
  infer_workdir = epath.Path(workdir)
  infer_workdir.mkdir(parents=True, exist_ok=True)
  logging.info('Inference workdir %r', infer_workdir)

  # Get experiment workdir and config.
  exp_workdir = (
      infer_config.exp_workdir if infer_config.exp_workdir else infer_workdir
  )
  exp_config = model_util.load_config(os.path.join(exp_workdir, 'config.json'))
  logging.info('Experiment workdir %r', exp_workdir)
  logging.info('Experiment config %r', exp_config)

  # Update experiment config with inference config.
  config = exp_config.copy_and_resolve_references()
  config.update(infer_config)

  # Seeding.
  rng = training.get_rng(config.seed)
  logging.info('Using random seed %s.', rng)
  rng, infer_rng = jax.random.split(rng)  # pylint: disable=unused-variable

  # Model.
  model = model_util.model_from_config(config)

  # Static covariates.
  covariates_static = input_pipeline.get_static_covariates(config)

  # Initialize head.
  head = heads.create_head(config)

  # Set up checkpointing.
  checkpoint_manager = checkpoint.get_checkpoint_manager(
      exp_workdir,
      item_names=(
          'early_stop',
          'train_state',
          'track_best_val_loss_step',
      ),
  )

  # Determine checkpoint to use.
  # TODO(jan-matthis): Support numeric checkpoint selection.
  step = _get_checkpoint_step(checkpoint_manager, config.checkpoint_selection)
  inference_message = 'Selected checkpoint at step %d for inference.' % step
  logging.info(inference_message)

  # Restore checkpoint.
  checkpointed_state = dict(
      train_state=None,
  )
  checkpointed_state = checkpoint.restore_checkpoint(
      checkpoint_manager,
      state=checkpointed_state,
      step=step,
  )
  train_state = checkpointed_state['train_state']
  train_state = train.TrainState(
      **train_state
  )  # TODO(jan-matthis): Via checkpointed_state.
  train_state = flax_utils.replicate(train_state)

  # Load data source for inference.
  infer_source = input_pipeline.create_inference_source_with_transforms(config)
  infer_prefix = config.infer_prefix.format(workdir=infer_workdir, step=step)
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY,
      infer_prefix,
      'Path prefix for inference results',
  )

  writer = metric_writers.create_default_writer(
      infer_workdir, just_logging=jax.process_index() > 0
  )
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=None, writer=writer
  )
  with metric_writers.ensure_flushes(writer):
    parameter_overview.log_parameter_overview(train_state.params)

    # Inference.
    logging.info('Starting inference with checkpoint from step %d', step)
    infer_key = jax.random.fold_in(key=infer_rng, data=step)
    for infer_idx_set in config.infer_idx_sets:
      name, idx_list = (infer_idx_set[k] for k in ('name', 'idx_list'))
      infer_metrics = None
      with report_progress.timed(f'infer_{name}'):
        train_state = train.merge_batch_stats(train_state)
        for i, idx in enumerate(idx_list):
          logging.log_first_n(
              logging.INFO, 'infer step %i of %i', 1, i + 1, len(idx_list)
          )
          prediction, target = infer_single_step(
              model,
              head,
              flax_utils.unreplicate(train_state),
              infer_source,
              idx,
              infer_key=infer_key,
              covariates=tuple(config.covariates),
              covariates_static=covariates_static,
              with_carry=config.infer_with_carry,
          )

          if f'infer_{name}' in head.metrics:
            metrics_update = head.metrics[
                f'infer_{name}'
            ].single_from_model_output(predictions=prediction, targets=target)
            infer_metrics = (
                metrics_update
                if infer_metrics is None
                else infer_metrics.merge(metrics_update)
            )

        if infer_metrics is not None:
          infer_metrics_cpu = jax.tree.map(np.array, infer_metrics.compute())
          writer.write_scalars(
              step, metrics_lib.make_dict_of_scalars(infer_metrics_cpu)
          )
          if config.infer_save_json:
            ts_utils.write_json(
                to_write={
                    k: float(v)
                    for k, v in metrics_lib.make_dict_of_scalars(
                        infer_metrics_cpu
                    ).items()
                },
                kvstore=os.path.join(infer_prefix, f'{name}.json'),
            )

  logging.info('Finished.')
  return checkpointed_state
