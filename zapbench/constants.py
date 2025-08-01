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

"""Constants for ZAPBench."""
# pylint: disable=line-too-long

# Dataset-Agnostic Parameters (Global constants)
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2
MAX_CONTEXT_LENGTH = 256
PREDICTION_WINDOW_LENGTH = 32
CONDITION_PADDING = 1

# Backward compatibility
DEFAULT_DATASET = '240930_traces'

# Dataset Registry (Dataset-Specific + Universal Parameters)
DATASET_CONFIGS = {
    '240930_traces': {
        # Dataset-Specific Parameters (MUST change)
        'condition_offsets': (
            0,
            649,
            2422,
            3078,
            3735,
            5047,
            5638,
            6623,
            7279,
            7879,
        ),
        'condition_names': (
            'gain',
            'dots',
            'flash',
            'taxis',
            'turning',
            'position',
            'open loop',
            'rotation',
            'dark',
        ),
        'conditions_train': (0, 1, 2, 4, 5, 6, 7, 8),
        'conditions_holdout': (3,),
        'timeseries_name': '240930_traces',
        'covariate_series_name': '240930_stimuli_features',
        'specs': {
            '240930_traces': {
                'kvstore': 'gs://zapbench-release/volumes/20240930/traces/',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[7879], 71721],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            '240930_stimuli_features': {
                'kvstore': (
                    'gs://zapbench-release/volumes/20240930/stimuli_features/'
                ),
                'driver': 'zarr',
                'rank': 2,
                'metadata': {'shape': [7879, 26]},
                'transform': {
                    'input_inclusive_min': [0, 0],
                    'input_exclusive_max': [[7879], [26]],
                    'input_labels': ['t', 'f'],
                },
            },
            '240930_stimulus_evoked_response': {
                'kvstore': 'gs://zapbench-release/volumes/20240930/stimulus_evoked_response/',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[7879], 71721],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {
            '240930_traces': (-0.25, 1.5),
        },
        # Universal Parameters (MAY change) - present for this dataset
        'position_embedding_specs': {
            '240930_traces': {
                'kvstore': (
                    'gs://zapbench-release/volumes/20240930/position_embedding/'
                ),
                'driver': 'zarr',
                'rank': 2,
                'metadata': {'shape': [71721, 192]},
                'transform': {
                    'input_inclusive_min': [0, 0],
                    'input_exclusive_max': [[71721], [192]],
                    'input_labels': ['f', 'a'],
                },
            }
        },
        'segmentation_dataframes': {
            '240930_traces': 'gs://zapbench-release/volumes/20240930/segmentation/dataframe.json',
        },
        'rastermap_specs': {
            '240930_traces': {
                'kvstore': 'gs://zapbench-release/volumes/20240930/traces_rastermap_sorted/s0/',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[7879], 71721],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            }
        },
        'rastermap_sortings': {
            '240930_traces': 'gs://zapbench-release/volumes/20240930/traces_rastermap_sorted/sorting.json',
        },
    },
}


def get_dataset_config(dataset_name: str = DEFAULT_DATASET) -> dict:
  """Get dataset configuration with fallbacks for default dataset."""
  if dataset_name not in DATASET_CONFIGS:
    raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_CONFIGS")

  config = DATASET_CONFIGS[dataset_name].copy()
  default_config = DATASET_CONFIGS[DEFAULT_DATASET]

  # Apply fallbacks for Universal Parameters (MAY change)
  universal_params = [
      'position_embedding_specs',
      'segmentation_dataframes',
      'rastermap_specs',
      'rastermap_sortings',
  ]
  for param in universal_params:
    if param not in config:
      config[param] = default_config[param]

  return config
