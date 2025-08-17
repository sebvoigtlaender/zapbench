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

"""Data utilities."""

import copy
import json
import logging
from typing import Any, Optional, Sequence

from connectomics.common import file
import numpy as np
import pandas as pd
import tensorstore as ts
from zapbench import constants

logger = logging.getLogger(__name__)


def restrict_specs_to_somas(spec: dict[str, Any], soma_ids: Sequence[int] = tuple()) -> dict[str, Any]:
  """Updates a spec to restrict it to specific soma IDs."

  Args:
    spec: Spec in dictionary form.
    soma_ids: Ids of somas to load.

  Returns:
    Updated spec.
  """
  if not soma_ids:
    return spec

  spec = copy.deepcopy(spec)
  spec['transform']['output'] = [
    {'input_dimension': 0},
    {'index_array': [soma_ids]}
  ]
  spec['transform']['input_exclusive_max'][1] = len(soma_ids)
  return spec


def get_spec(spec_name: str, soma_ids: Sequence[int] = tuple(), dataset_name: str = constants.DEFAULT_DATASET) -> ts.Spec:
  """Gets TensorStore spec from dataset-specific SPECS.

  Args:
    spec_name: Key in dataset's specs.
    soma_ids: Ids of somas for which data will be loaded.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    TensorStore Spec.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  if spec_name not in dataset_config['specs']:
    raise ValueError(f'{spec_name} not in dataset {dataset_name} specs.')
  spec_dict = dataset_config['specs'][spec_name]

  return ts.Spec(restrict_specs_to_somas(spec_dict, soma_ids))


def get_covariate_spec(spec_name: str, dataset_name: str = constants.DEFAULT_DATASET) -> ts.Spec:
  """Gets TensorStore spec from dataset-specific COVARIATE_SPECS.

  Args:
    spec_name: Key in dataset's covariate specs.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    TensorStore Spec.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  if spec_name not in dataset_config['covariate_specs']:
    raise ValueError(f'{spec_name} not in dataset {dataset_name} covariate specs.')
  spec_dict = dataset_config['covariate_specs'][spec_name]

  return ts.Spec(spec_dict)


def get_position_embedding_spec(spec_name: str, dataset_name: str = constants.DEFAULT_DATASET) -> ts.Spec:
  """Gets TensorStore spec from dataset-specific POSITION_EMBEDDING_SPECS with fallback.

  Args:
    spec_name: Key in dataset's position embedding specs.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    TensorStore Spec.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  if spec_name not in dataset_config['position_embedding_specs']:
    raise ValueError(f'{spec_name} not in dataset {dataset_name} position embedding specs.')
  spec_dict = dataset_config['position_embedding_specs'][spec_name]

  return ts.Spec(spec_dict)


def get_rastermap_spec(spec_name: str, dataset_name: str = constants.DEFAULT_DATASET) -> ts.Spec:
  """Gets TensorStore spec from dataset-specific RASTERMAP_SPECS with fallback.

  Args:
    spec_name: Key in dataset's rastermap specs.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    TensorStore Spec.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  if spec_name not in dataset_config['rastermap_specs']:
    raise ValueError(f'{spec_name} not in dataset {dataset_name} rastermap specs.')
  spec_dict = dataset_config['rastermap_specs'][spec_name]

  return ts.Spec(spec_dict)


def get_segmentation_dataframe(df_name: str, dataset_name: str = constants.DEFAULT_DATASET) -> pd.DataFrame:
  """Gets segmentation dataframe from dataset-specific SEGMENTATION_DATAFRAMES with fallback.

  Can be mapped to segmentations through the label-column and to indices in
  associated trace matrices though `df.loc[trace_idx]` (or label minus 1).

  Args:
    df_name: Key in dataset's segmentation dataframes.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    Dataframe.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  if df_name not in dataset_config['segmentation_dataframes']:
    raise ValueError(f'{df_name} not in dataset {dataset_name} segmentation dataframes.')
  path = dataset_config['segmentation_dataframes'][df_name]
  with file.Path(path).open('r') as f:
    df = pd.DataFrame(json.load(f))
  return df.sort_values('label').reset_index(drop=True)


def get_condition_bounds(condition: int, dataset_name: str = constants.DEFAULT_DATASET) -> tuple[int, int]:
  """Get bounds of a condition's temporal indices with dataset-specific offsets.

  Args:
    condition: Condition number, starting from zero.
    dataset_name: Dataset to use. If None, uses default dataset.

  Returns:
    (inclusive_min, exclusive_max)-tuple of boundaries.
  """
  dataset_config = constants.get_dataset_config(dataset_name)
  condition_offsets = dataset_config['condition_offsets']
  condition_padding = constants.CONDITION_PADDING  # Still global (dataset-agnostic)

  if condition < 0 or condition >= len(condition_offsets) - 1:
    raise ValueError(f'condition must be in [0, {len(condition_offsets)-1}]')

  return (
      condition_offsets[condition] + condition_padding,
      condition_offsets[condition + 1] - condition_padding,
  )


def adjust_condition_bounds_for_split(
    split: str,
    inclusive_min: int,
    exclusive_max: int,
    num_timesteps_context: int,
) -> tuple[int, int]:
  """Adjust condition bounds for a split.

  Args:
    split: Requested split, 'train' (training set), 'val' (validation set),
      'test' (test set), 'test_holdout' (test set for a condition that was
      held-out from training), 'train_val' (train and validation set), or
      None (in which case the condition is not split).
    inclusive_min: Lower bound.
    exclusive_max: Upper bound.
    num_timesteps_context: Number of timesteps for context.

  Returns:
    (inclusive_min, exclusive_max)-tuple of boundaries.
  """
  num_timesteps_total = exclusive_max - inclusive_min
  num_timesteps_test = int(num_timesteps_total * constants.TEST_FRACTION)
  num_timesteps_val = int(num_timesteps_total * constants.VAL_FRACTION)
  num_timesteps_train = (
      num_timesteps_total - num_timesteps_test - num_timesteps_val
  )

  if split == 'train':
    exclusive_max = inclusive_min + num_timesteps_train
  elif split == 'val':
    exclusive_max = inclusive_min + num_timesteps_train + num_timesteps_val
    inclusive_min += num_timesteps_train - num_timesteps_context
  elif split == 'train_val':
    exclusive_max = inclusive_min + num_timesteps_train + num_timesteps_val
  elif split == 'test':
    inclusive_min += (
        num_timesteps_train + num_timesteps_val - num_timesteps_context
    )
  elif split == 'test_holdout':
    inclusive_min += constants.MAX_CONTEXT_LENGTH - num_timesteps_context
  elif split is None:
    pass
  else:
    raise ValueError(
        f'split must be train, val, test, test_holdout or None but is {split}.'
    )

  assert (
      inclusive_min >= 0
  ), f'inclusive_min must be >= 0 but is {inclusive_min}.'
  assert inclusive_min < exclusive_max, 'inclusive_min must be < exclusive_max.'

  return inclusive_min, exclusive_max


def get_num_windows(
    inclusive_min: int, exclusive_max: int, num_timesteps_context: int
) -> int:
  """Given condition bounds, calculate number of prediction windows at stride 1."""
  return (
      exclusive_max
      - inclusive_min
      - constants.PREDICTION_WINDOW_LENGTH
      - num_timesteps_context
      + 1
  )


def get_condition_intervals(condition: int, dataset_name: str = constants.DEFAULT_DATASET) -> tuple[tuple[int, int], ...]:
  """Get padded intervals for a condition."""
  dataset_config = constants.get_dataset_config(dataset_name)
  intervals = dataset_config['condition_intervals'][condition]

  padded_intervals = []
  for start, end in intervals:
    padded_start = start + constants.CONDITION_PADDING
    padded_end = end - constants.CONDITION_PADDING
    if not (padded_start < padded_end):
      logger.warning(f"Padded interval [{padded_start}, {padded_end}) is not valid")
      continue
    padded_intervals.append((padded_start, padded_end))

  return tuple(padded_intervals)


def calculate_window_size(num_timesteps_context: int) -> int:
  """Calculate window size with safety checks."""
  if num_timesteps_context <= 0:
    raise ValueError(f"num_timesteps_context must be > 0, got {num_timesteps_context}")

  if num_timesteps_context > constants.MAX_CONTEXT_LENGTH:
    raise ValueError(f"num_timesteps_context {num_timesteps_context} exceeds MAX_CONTEXT_LENGTH {constants.MAX_CONTEXT_LENGTH}")

  window_size = num_timesteps_context + constants.PREDICTION_WINDOW_LENGTH
  return window_size


def build_valid_timesteps(intervals: tuple[tuple[int, int], ...], window_size: int) -> list[int]:
  """Build timesteps that can start complete windows within intervals."""
  valid_timesteps = []

  for start, end in intervals:
    interval_size = end - start

    if interval_size >= window_size:
      valid_timesteps.extend(range(start, end - window_size + 1))
    else:
      logger.warning(f"Interval [{start}, {end}) too small for window_size {window_size}")

  if not valid_timesteps:
    raise ValueError(f"No intervals large enough for window_size={window_size}.")

  return sorted(valid_timesteps)


def adjust_spec_for_condition_and_split(
    spec: ts.Spec,
    condition: int,
    split: Optional[str],
    num_timesteps_context: int,
    dataset_name: str = constants.DEFAULT_DATASET,
) -> ts.Spec:
  """Adjust spec for multi-interval conditions with gap-aware windowing.

  For example, to get the training timeseries for the first condition for an
  algorithm that uses 32 timesteps as context:
    spec = get_spec('timeseries')
    ds = ts.open(adjust_spec_for_condition_and_split(
        spec, 0, 'train', 32)).result()

  Args:
    spec: TensorStore spec with dimension `t`.
    condition: Condition number, starting from zero.
    split: Requested split, 'train' (training set), 'val' (validation set),
      'test' (test set), 'test_holdout' (test set for a condition that was
      held-out from training), or None (in which case the condition is not
      split).
    num_timesteps_context: Number of additional timesteps for context.
    dataset_name: Dataset to use for condition bounds.

  Returns:
    TensorStore Spec.
  """
  if 't' not in spec.domain.labels:
    raise ValueError('Required dimension label `t` not found in spec.')

  intervals = get_condition_intervals(condition, dataset_name)
  window_size = calculate_window_size(num_timesteps_context)
  valid_timesteps = build_valid_timesteps(intervals, window_size)

  if split:
    total = len(valid_timesteps)
    test_count = int(total * constants.TEST_FRACTION)
    val_count = int(total * constants.VAL_FRACTION)
    train_count = total - test_count - val_count

    if split == 'train':
      valid_timesteps = valid_timesteps[:train_count]
    elif split == 'val':
      val_start = max(0, train_count - num_timesteps_context)
      valid_timesteps = valid_timesteps[val_start:train_count + val_count]
    elif split == 'test':
      test_start = max(0, train_count + val_count - num_timesteps_context)
      valid_timesteps = valid_timesteps[test_start:]
    elif split == 'test_holdout':
      holdout_start = max(0, total - constants.MAX_CONTEXT_LENGTH - constants.PREDICTION_WINDOW_LENGTH)
      valid_timesteps = valid_timesteps[holdout_start:]

  # Use non-slice indexing for non-contiguous case
  is_contiguous = len(valid_timesteps) == (valid_timesteps[-1] - valid_timesteps[0] + 1)
  if is_contiguous:
    return spec[ts.d['t'][slice(valid_timesteps[0], valid_timesteps[-1] + 1)]].translate_to[0]
  else:
    return spec[ts.d['t'][valid_timesteps]].translate_to[0]


def get_rastermap_indices(timeseries: str, dataset_name: str = constants.DEFAULT_DATASET) -> np.ndarray:
  """Gets rastermap indices with dataset-aware fallback."""
  dataset_config = constants.get_dataset_config(dataset_name)
  if timeseries not in dataset_config['rastermap_sortings']:
    raise ValueError(f'{timeseries} not in dataset {dataset_name} rastermap sortings.')
  return json.loads(
      file.Path(
          dataset_config['rastermap_sortings'][timeseries],
      ).read_text('rt')
  )


def get_indices_to_invert_rastermap_sorting(timeseries: str, dataset_name: str = constants.DEFAULT_DATASET) -> np.ndarray:
  """Gets indices that invert rastermap sorting with dataset-aware fallback."""
  return np.argsort(get_rastermap_indices(timeseries, dataset_name=dataset_name))
