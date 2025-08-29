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

# Dataset registry
DATASET_CONFIGS = {
    '240930_traces': {
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
        'condition_intervals': (
            ((0, 649),),
            ((649, 2422),),
            ((2422, 3078),),
            ((3078, 3735),),
            ((3735, 5047),),
            ((5047, 5638),),
            ((5638, 6623),),
            ((6623, 7279),),
            ((7279, 7879),),
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
    'subject_01': {
        'condition_intervals': ((0, 2879),),
        'condition_names': ('dark-taxis',),
        'conditions_train': (2,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_01',
        'covariate_series_name': 'subject_01_stimuli_features',
        'specs': {
            'subject_01': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_01_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 83205],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_01_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_01_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_01_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_01_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_01': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_02': {
        'condition_intervals': ((0, 3519),),
        'condition_names': ('dark-taxis',),
        'conditions_train': (2,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_02',
        'covariate_series_name': 'subject_02_stimuli_features',
        'specs': {
            'subject_02': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_02_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3520], 82514],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_02_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_02_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3520], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_02_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_02_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3520], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_02': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_03': {
        'condition_intervals': ((0, 2559),),
        'condition_names': ('dark-taxis',),
        'conditions_train': (2,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_03',
        'covariate_series_name': 'subject_03_stimuli_features',
        'specs': {
            'subject_03': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_03_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2560], 83662],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_03_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_03_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2560], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_03_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_03_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2560], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_03': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_04': {
        'condition_intervals': ((0, 2879),),
        'condition_names': ('dark-taxis',),
        'conditions_train': (2,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_04',
        'covariate_series_name': 'subject_04_stimuli_features',
        'specs': {
            'subject_04': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_04_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 90034],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_04_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_04_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_04_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_04_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_04': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_05': {
        'condition_intervals': ((0, 2879),),
        'condition_names': ('dark-taxis',),
        'conditions_train': (2,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_05',
        'covariate_series_name': 'subject_05_stimuli_features',
        'specs': {
            'subject_05': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_05_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 97766],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_05_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_05_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_05_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_05_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[2880], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_05': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_06': {
        'condition_intervals': ((0, 3779),),
        'condition_names': ('taxis',),
        'conditions_train': (1,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_06',
        'covariate_series_name': 'subject_06_stimuli_features',
        'specs': {
            'subject_06': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_06_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3780], 92538],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_06_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_06_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3780], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_06_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_06_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3780], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_06': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_07': {
        'condition_intervals': ((0, 1649),),
        'condition_names': ('taxis',),
        'conditions_train': (1,),
        'conditions_holdout': (),
        'timeseries_name': 'subject_07',
        'covariate_series_name': 'subject_07_stimuli_features',
        'specs': {
            'subject_07': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_07_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1650], 93810],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_07_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_07_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1650], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_07_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_07_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1650], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_07': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_12': {
        'condition_intervals': (((0, 1229), (3425, 4629)), ((2100, 2449), (5500, 6139)), ((1230, 2099), (4630, 5499)), ((3125, 3424),)),
        'condition_names': ('taxis', 'dark', 'opt_response', 'looming'),
        'conditions_train': (1, 3, 4),
        'conditions_holdout': (5,),
        'timeseries_name': 'subject_12',
        'covariate_series_name': 'subject_12_stimuli_features',
        'specs': {
            'subject_12': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_12_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[6140], 58105],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_12_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_12_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[6140], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_12_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_12_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[6140], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_12': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_13': {
        'condition_intervals': (((0, 414), (2280, 2984)), ((800, 1024), (3370, 3594)), ((415, 799), (2985, 3369)), ((1695, 2279), (4265, 4839))),
        'condition_names': ('taxis', 'dark', 'opt_response', 'looming'),
        'conditions_train': (1, 3, 4),
        'conditions_holdout': (5,),
        'timeseries_name': 'subject_13',
        'covariate_series_name': 'subject_13_stimuli_features',
        'specs': {
            'subject_13': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_13_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4840], 73202],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_13_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_13_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4840], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_13_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_13_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4840], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_13': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_14': {
        'condition_intervals': (((0, 834), (2940, 3764)), ((1600, 1949),), ((835, 1599), (3765, 3889)), ((2475, 2939),)),
        'condition_names': ('taxis', 'dark', 'opt_response', 'looming'),
        'conditions_train': (1, 3, 4),
        'conditions_holdout': (5,),
        'timeseries_name': 'subject_14',
        'covariate_series_name': 'subject_14_stimuli_features',
        'specs': {
            'subject_14': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_14_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3890], 72412],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_14_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_14_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3890], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_14_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_14_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[3890], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_14': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_15': {
        'condition_intervals': (((0, 844), (2415, 3314)), ((1495, 1809), (3965, 4279)), ((845, 1494), (3315, 3964)), ((2135, 2414), (4605, 4879))),
        'condition_names': ('taxis', 'dark', 'opt_response', 'looming'),
        'conditions_train': (1, 3, 4),
        'conditions_holdout': (5,),
        'timeseries_name': 'subject_15',
        'covariate_series_name': 'subject_15_stimuli_features',
        'specs': {
            'subject_15': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_15_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4880], 53180],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_15_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_15_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4880], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_15_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_15_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[4880], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_15': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_16': {
        'condition_intervals': (((0, 729),), ((730, 1359),)),
        'condition_names': ('taxis', 'opt_response'),
        'conditions_train': (1,),
        'conditions_holdout': (4,),
        'timeseries_name': 'subject_16',
        'covariate_series_name': 'subject_16_stimuli_features',
        'specs': {
            'subject_16': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_16_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1877], 62737],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_16_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_16_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1877], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_16_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_16_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[1877], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_16': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
    },
    'subject_17': {
        'condition_intervals': (((0, 935), (2750, 3745)), ((1708, 2043), (4518, 4853)), ((936, 1707), (3746, 4517)), ((2469, 2749), (5279, 5553))),
        'condition_names': ('taxis', 'dark', 'opt_response', 'looming'),
        'conditions_train': (1, 3, 4),
        'conditions_holdout': (5,),
        'timeseries_name': 'subject_17',
        'covariate_series_name': 'subject_17_stimuli_features',
        'specs': {
            'subject_17': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_17_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[5554], 63922],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'covariate_specs': {
            'subject_17_stimuli_features': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_17_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[5554], 16],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            'subject_17_behavioral_covariates': {
                'kvstore': 'file:///Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files/subject_17_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[5554], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },
        'min_max_values': {'subject_17': (-0.25, 3.0)},
        'position_embedding_specs': {},
        'segmentation_dataframes': {},
        'rastermap_specs': {},
        'rastermap_sortings': {},
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
