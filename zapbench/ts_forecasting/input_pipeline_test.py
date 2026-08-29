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

"""Tests for input pipeline."""

from typing import Any, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from zapbench.ts_forecasting import input_pipeline
from zapbench.ts_forecasting.configs import common
from zapbench.ts_forecasting.configs import tide


def build_placeholder_spec(shape: Sequence[int]) -> dict[str, Any]:
  """Builds a placeholder spec containing all ones."""
  return {
      'driver': 'array',
      'dtype': 'float32',
      'array': np.ones(shape).tolist(),
  }


def get_specs_for_test(
    num_timesteps: int,
    num_features: int,
    num_dynamic_covariates: int,
    num_poco_features: int = 0,
):
  """Returns specs for testing."""
  specs = {
      'covariates': build_placeholder_spec(
          (num_timesteps, num_dynamic_covariates)
      ),
      'timeseries': build_placeholder_spec((num_timesteps, num_features)),
  }
  if num_poco_features:
    specs['poco_embeddings'] = build_placeholder_spec(
        (num_timesteps, num_poco_features)
    )
  return specs


class InputPipelineTest(parameterized.TestCase):

  def test_tide_config_without_poco_keeps_existing_sources_and_shapes(self):
    config = tide.get_config('dataset_name=subject_01,seed=1')

    self.assertFalse(config.use_poco_embeddings)
    self.assertNotIn('poco_embeddings', config.infer_spec)
    for specs in config.train_specs + config.val_specs:
      self.assertNotIn('poco_embeddings', specs)
    self.assertEqual(
        config.covariates,
        ('covariates_static', 'covariates_input', 'covariates_output'),
    )
    self.assertEqual(config.covariates_shapes[1][-1], 16)
    self.assertEqual(config.covariates_shapes[2][-1], 16)
    self.assertTrue(config.tide_config.ablate_past_covariates)

  def test_tide_config_with_poco_adds_aligned_sources_and_shapes(self):
    config = tide.get_config(
        'dataset_name=subject_01,use_poco_embeddings=True,seed=1'
    )

    self.assertTrue(config.use_poco_embeddings)
    self.assertIn('poco_embeddings', config.infer_spec)
    self.assertNotIn(
        'file:////', config.infer_spec['poco_embeddings']['kvstore']
    )
    for specs in config.train_specs + config.val_specs:
      self.assertIn('poco_embeddings', specs)
      self.assertEqual(
          specs['poco_embeddings']['transform']['output'][0],
          specs['covariates']['transform']['output'][0],
      )
    self.assertEqual(
        config.covariates,
        (
            'covariates_static',
            'covariates_input',
            'covariates_output',
            'poco_covariates',
        ),
    )
    self.assertEqual(config.covariates_shapes[1][-1], 16)
    self.assertEqual(config.covariates_shapes[2][-1], 16)
    self.assertEqual(config.covariates_shapes[3], (1, 8))
    self.assertTrue(config.tide_config.ablate_past_covariates)
    self.assertIn('poco_embeddings_input', config.infer_batching_str)
    self.assertIn('poco_embeddings_output', config.infer_batching_str)

  def test_tide_config_with_poco_requires_registered_spec(self):
    with self.assertRaisesRegex(
        ValueError,
        'janelia_pretrain_poco_embeddings not in dataset janelia_pretrain',
    ):
      tide.get_config(
          'dataset_name=janelia_pretrain,use_poco_embeddings=True,seed=1'
      )

  @parameterized.named_parameters(
      dict(testcase_name='wrong_rank', rank=3, shape=(100, 1, 8)),
      dict(testcase_name='wrong_width', rank=2, shape=(100, 7)),
  )
  def test_tide_config_with_poco_requires_rank_two_width_eight(
      self, rank: int, shape: tuple[int, ...]
  ):
    invalid_spec = mock.Mock(rank=rank, shape=shape)
    with (
        mock.patch(
            'zapbench.ts_forecasting.configs.tide.data_utils.get_covariate_spec',
            return_value=invalid_spec,
        ),
        self.assertRaisesRegex(ValueError, r'must have shape \(T, 8\)'),
    ):
      tide.get_config('dataset_name=subject_01,use_poco_embeddings=True,seed=1')

  @parameterized.named_parameters(
      dict(
          testcase_name='short_context',
          timesteps_input=4,
      ),
      dict(
          testcase_name='long_context',
          timesteps_input=256,
      ),
      dict(
          testcase_name='repeated_specs',
          num_spec_repetitions=2,
      ),
      dict(
          testcase_name='poco_embeddings',
          num_poco_features=8,
      ),
  )
  def test_create_datasets(
      self,
      timesteps_input: int = 4,
      timesteps_output: int = 32,
      num_timesteps: int = 1000,
      num_features: int = 16,
      num_dynamic_covariates: int = 2,
      num_poco_features: int = 0,
      num_spec_repetitions: int = 1,
  ):
    specs = get_specs_for_test(
        num_timesteps=num_timesteps,
        num_features=num_features,
        num_dynamic_covariates=num_dynamic_covariates,
        num_poco_features=num_poco_features,
    )

    config = common.get_config(
        timesteps_input=timesteps_input, timesteps_output=timesteps_output
    )
    config.train_specs = [specs] * num_spec_repetitions
    config.val_specs = [specs] * num_spec_repetitions

    config.per_device_batch_size = 8

    train_loader, num_train_records, val_loader, num_val_records = (
        input_pipeline.create_datasets(config, seed=1)
    )

    assert (
        num_train_records
        == (
            num_timesteps - config.timesteps_input - config.timesteps_output + 1
        )
        * num_spec_repetitions
    )
    assert num_val_records == num_train_records

    train_iter = iter(train_loader)
    batch = next(train_iter)
    assert (
        batch['timestep'].sum() > np.arange(config.per_device_batch_size).sum()
    )  # Shuffling
    assert batch['timeseries_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_features,
    )
    assert batch['timeseries_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_features,
    )
    assert batch['covariates_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert batch['covariates_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_dynamic_covariates,
    )
    if num_poco_features:
      self.assertEqual(
          batch['poco_covariates'].shape,
          (config.per_device_batch_size, num_poco_features),
      )
      np.testing.assert_array_equal(batch['poco_covariates'], 1)
      self.assertNotIn('poco_embeddings_input', batch)
      self.assertNotIn('poco_embeddings_output', batch)
    else:
      self.assertNotIn('poco_covariates', batch)

    val_iter = iter(val_loader)
    batch = next(val_iter)
    assert (
        batch['timestep'].sum() == np.arange(config.per_device_batch_size).sum()
    )  # No shuffling
    assert batch['timeseries_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_features,
    )
    assert batch['timeseries_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_features,
    )
    assert batch['covariates_input'].shape == (
        config.per_device_batch_size,
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert batch['covariates_output'].shape == (
        config.per_device_batch_size,
        config.timesteps_output,
        num_dynamic_covariates,
    )
    if num_poco_features:
      self.assertEqual(
          batch['poco_covariates'].shape,
          (config.per_device_batch_size, num_poco_features),
      )
      np.testing.assert_array_equal(batch['poco_covariates'], 1)
      self.assertNotIn('poco_embeddings_input', batch)
      self.assertNotIn('poco_embeddings_output', batch)
    else:
      self.assertNotIn('poco_covariates', batch)

  @parameterized.named_parameters(
      dict(
          testcase_name='short_context',
          timesteps_input=4,
      ),
      dict(
          testcase_name='long_context',
          timesteps_input=256,
      ),
  )
  def test_inference_source(
      self,
      timesteps_input: int = 4,
      timesteps_output: int = 32,
      num_timesteps: int = 1000,
      num_features: int = 16,
      num_dynamic_covariates: int = 2,
  ):
    specs = get_specs_for_test(
        num_timesteps=num_timesteps,
        num_features=num_features,
        num_dynamic_covariates=num_dynamic_covariates,
    )

    config = common.get_config(
        timesteps_input=timesteps_input, timesteps_output=timesteps_output
    )
    config.infer_spec = specs
    config.infer_batching_str = (
        'expand_dims(keys=("timeseries_input","timeseries_output"),axis=0)'
    )

    inference_source = input_pipeline.create_inference_source_with_transforms(
        config
    )

    assert len(inference_source) == (
        num_timesteps - config.timesteps_input - config.timesteps_output + 1
    )

    # Batch dimension of 1 due to infer_batching_str
    assert inference_source[0]['timeseries_input'].shape == (
        1,
        config.timesteps_input,
        num_features,
    )
    assert inference_source[0]['timeseries_output'].shape == (
        1,
        config.timesteps_output,
        num_features,
    )

    # No batch dimension
    assert inference_source[0]['covariates_input'].shape == (
        config.timesteps_input,
        num_dynamic_covariates,
    )
    assert inference_source[0]['covariates_output'].shape == (
        config.timesteps_output,
        num_dynamic_covariates,
    )

  def test_inference_source_with_poco_batches_separate_origin_covariates(self):
    num_timesteps = 100
    num_features = 4
    num_dynamic_covariates = 3
    num_poco_features = 8
    config = tide.get_config(
        'dataset_name=subject_01,timesteps_input=4,'
        'use_poco_embeddings=True,seed=1'
    )
    config.infer_spec = get_specs_for_test(
        num_timesteps=num_timesteps,
        num_features=num_features,
        num_dynamic_covariates=num_dynamic_covariates,
        num_poco_features=num_poco_features,
    )

    inference_source = input_pipeline.create_inference_source_with_transforms(
        config
    )
    batch = inference_source[2]

    self.assertEqual(
        batch['covariates_input'].shape,
        (
            1,
            config.timesteps_input,
            num_dynamic_covariates,
        ),
    )
    self.assertEqual(
        batch['covariates_output'].shape,
        (
            1,
            config.timesteps_output,
            num_dynamic_covariates,
        ),
    )
    self.assertEqual(batch['poco_covariates'].shape, (1, num_poco_features))
    np.testing.assert_array_equal(batch['poco_covariates'], 1)
    self.assertNotIn('poco_embeddings_input', batch)
    self.assertNotIn('poco_embeddings_output', batch)


if __name__ == '__main__':
  absltest.main()
