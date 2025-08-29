import numpy as np
import h5py
import scipy.io
import tensorstore as ts
from typing import Tuple, Dict, List, Any
import os


PATH_JANELIA = "/Users/sebastianvoigtlaender/vault/neural_data/janelia"
PATH_STORE = "/Users/sebastianvoigtlaender/vault/neural_data/janelia/ts_files"

SUBJECT_ID_LIST = [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17]
N_STIMULUS_ENCODINGS = 16
CONDITION_NAMES = ['spontaneous', 'taxis', 'dark-taxis', 'dark', 'opt_response', 'looming']
CONDITION_MAP = {
    'Spontaneous': 0,
    'phototaxis': 1,
    'PT': 1,
    '16 permut: B/W/phototaxis*2': 2,
    'DF': 3,
    'OMR': 4,
    'Looming': 5
}

ZAP_AVG = 0.11625818
ZAP_STD = 0.07106597


def fill_gaps(states: np.ndarray, max_interval: int) -> np.ndarray:
    x = states.copy()
    current = 0
    i = 0
    while i < len(states):
        if states[i] != 0:
            current = states[i]
            i += 1
        else:
            start = i
            while i < len(states) and states[i] == 0:
                i += 1
            if i - start <= max_interval and current != 0:
                for j in range(start, i):
                    x[j] = current
    return x


def fill_start(x: np.ndarray) -> np.ndarray:
    i = 0
    pt = 0
    while not pt and i < len(x):
        pt = x[i]
        i += 1
    if i > 1:
        x[:i-1] = x[i-1]
    return x


def get_condition_intervals(x: np.ndarray, condition_ix_list: List[int]) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    interval_bounds_by_condition = []
    for condition_ix in condition_ix_list:
        mask = (x == condition_ix).astype(int)
        mask_diff = np.diff(mask)
        starts = np.where(mask_diff == 1)[0] + 1
        ends = np.where(mask_diff == -1)[0]
        if mask[0] == 1:
            starts = np.insert(starts, 0, 0)
        if mask[-1] == 1:
            ends = np.append(ends, len(mask)-1)
        interval_bounds = tuple((int(s), int(e)) for s, e in zip(starts, ends))
        interval_bounds_by_condition.append(interval_bounds)
    return tuple(interval_bounds_by_condition)


def get_condition_cfg(data_struct: Dict[str, Any]) -> Tuple[Tuple[int, ...], Tuple[Tuple[Tuple[int, int], ...], ...]]:
    stim_full = data_struct['stim_full'].ravel()

    if 'stimset' in data_struct.dtype.fields:
        stim_full_harmonized = np.zeros_like(stim_full)
        n_conditions = data_struct['stimset'].shape[-1]

        for i in range(n_conditions):
            name = data_struct['stimset'][0, i]['name'][0]
            pattern = np.unique(data_struct['stimset'][0, i]['pattern'])
            indices = np.where(np.isin(data_struct['stim_full'].ravel(), pattern[pattern != 3]))[0]
            stim_full_harmonized[indices] = CONDITION_MAP[name]

        x = fill_gaps(stim_full_harmonized, 100)
        x = fill_start(x)
        x = x.astype(int)
        condition_ix_list = np.unique(x)
        condition_intervals = get_condition_intervals(x, condition_ix_list)

    else:
        i_condition = CONDITION_MAP[data_struct['timelists_names'][0, 0][0]]
        condition_ix_list = [i_condition]
        condition_intervals = ((0, len(stim_full)-1),)

    conditions_train = tuple(int(ix) for ix in condition_ix_list)
    return conditions_train, condition_intervals


def save_tensorstore(x: np.ndarray, subject_id: str, data_type: str) -> None:
    spec = {
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': f'{PATH_STORE}/subject_{subject_id}_{data_type}.zarr'},
        'metadata': {
            'shape': list(x.shape),
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [512, min(512, x.shape[-1])]}},
            'chunk_key_encoding': {'name': 'default'},
            'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}}],
            'data_type': 'float32',
            'fill_value': 0.0
        }
    }
    ds = ts.open(spec, create=True).result()
    ds[...] = x


def write_spec_to_constants_file(subject_key: str, spec: Dict[str, Any]) -> None:
    constants_file = "/Users/sebastianvoigtlaender/git/zapbench/constants.txt"

    spec_lines = []
    spec_lines.append(f"    '{subject_key}': {{")

    spec_lines.append(f"        'condition_intervals': {spec['condition_intervals']},")
    spec_lines.append(f"        'condition_names': {spec['condition_names']},")
    spec_lines.append(f"        'conditions_train': {spec['conditions_train']},")
    spec_lines.append(f"        'conditions_holdout': {spec['conditions_holdout']},")
    spec_lines.append(f"        'timeseries_name': '{spec['timeseries_name']}',")
    spec_lines.append(f"        'covariate_series_name': '{spec['covariate_series_name']}',")

    spec_lines.append("        'specs': {")
    for specs_key, specs_val in spec['specs'].items():
        spec_lines.append(f"            '{specs_key}': {{")
        spec_lines.append(f"                'kvstore': '{specs_val['kvstore']}',")
        spec_lines.append(f"                'driver': '{specs_val['driver']}',")
        spec_lines.append("                'transform': {")
        spec_lines.append(f"                    'input_exclusive_max': {specs_val['transform']['input_exclusive_max']},")
        spec_lines.append(f"                    'input_inclusive_min': {specs_val['transform']['input_inclusive_min']},")
        spec_lines.append(f"                    'input_labels': {specs_val['transform']['input_labels']},")
        spec_lines.append("                },")
        spec_lines.append("            },")
    spec_lines.append("        },")

    spec_lines.append("        'covariate_specs': {")
    for cov_key, cov_val in spec['covariate_specs'].items():
        spec_lines.append(f"            '{cov_key}': {{")
        spec_lines.append(f"                'kvstore': '{cov_val['kvstore']}',")
        spec_lines.append(f"                'driver': '{cov_val['driver']}',")
        spec_lines.append("                'transform': {")
        spec_lines.append(f"                    'input_exclusive_max': {cov_val['transform']['input_exclusive_max']},")
        spec_lines.append(f"                    'input_inclusive_min': {cov_val['transform']['input_inclusive_min']},")
        spec_lines.append(f"                    'input_labels': {cov_val['transform']['input_labels']},")
        spec_lines.append("                },")
        spec_lines.append("            },")
    spec_lines.append("        },")

    spec_lines.append(f"        'min_max_values': {spec['min_max_values']},")
    spec_lines.append(f"        'position_embedding_specs': {spec['position_embedding_specs']},")
    spec_lines.append(f"        'segmentation_dataframes': {spec['segmentation_dataframes']},")
    spec_lines.append(f"        'rastermap_specs': {spec['rastermap_specs']},")
    spec_lines.append(f"        'rastermap_sortings': {spec['rastermap_sortings']},")
    spec_lines.append("    },")

    with open(constants_file, 'r') as f:
        content = f.read()

    insertion_point = content.rfind('}\n}')
    new_content = content[:insertion_point] + '\n'.join(spec_lines) + '\n' + content[insertion_point:]

    with open(constants_file, 'w') as f:
        f.write(new_content)


def get_janelia_spec(subject_id: int,
                     condition_intervals: Tuple[Tuple[Tuple[int, int], ...], ...],
                     condition_names: Tuple[str, ...],
                     conditions_train: Tuple[int, ...],
                     conditions_holdout: Tuple[int, ...],
                     n_t: int,
                     n_neurons: int) -> Dict[str, Any]:
    subject_str = f'subject_{subject_id:02d}'

    spec = {
        'condition_intervals': condition_intervals,
        'condition_names': condition_names,
        'conditions_train': conditions_train,
        'conditions_holdout': conditions_holdout,

        'timeseries_name': subject_str,
        'covariate_series_name': f'{subject_str}_stimuli_features',

        'specs': {
            subject_str: {
                'kvstore': f'file://{PATH_STORE}/{subject_str}_traces.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[n_t], n_neurons],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },

        'covariate_specs': {
            f'{subject_str}_stimuli_features': {
                'kvstore': f'file://{PATH_STORE}/{subject_str}_stimuli.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[n_t], N_STIMULUS_ENCODINGS],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
            f'{subject_str}_behavioral_covariates': {
                'kvstore': f'file://{PATH_STORE}/{subject_str}_behavioral_covariates.zarr',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[n_t], 5],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                },
            },
        },

        'min_max_values': {
            subject_str: (-0.25, 3.0),
        },
        'position_embedding_specs': {
        },
        'segmentation_dataframes': {
        },
        'rastermap_specs': {
        },
        'rastermap_sortings': {
        },
    }

    return spec


def process_janelia_subject(subject_id: int) -> Dict[str, Any]:

    h5_path = f"{PATH_JANELIA}/subject_{subject_id:02d}"
    if not os.path.exists(f"{h5_path}/TimeSeries.h5"):
        raise FileNotFoundError(f"TimeSeries.h5 not found for subject {subject_id:02d}")

    with h5py.File(f"{h5_path}/TimeSeries.h5", "r") as f:
        x_traces = np.array(f['CellResp']).astype(np.float32)

    jan_avg, jan_std = np.mean(x_traces), np.std(x_traces)
    x_traces_normalized = ((x_traces - jan_avg) / jan_std) * ZAP_STD + ZAP_AVG
    x_traces_normalized = np.clip(x_traces_normalized, -0.25, 3.0)
    n_t, n_neurons = x_traces_normalized.shape

    save_tensorstore(x_traces_normalized, f"{subject_id:02d}", "traces")

    mat_path = f"{PATH_JANELIA}/subject_{subject_id:02d}/data_full.mat"
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"data_full.mat not found for subject {subject_id:02d}")

    mat_data = scipy.io.loadmat(mat_path)
    data_struct = mat_data['data'][0, 0]

    stimuli = data_struct['stim_full'].ravel().astype(int)
    stimuli_onehot = np.eye(N_STIMULUS_ENCODINGS, dtype=np.float32)[stimuli]
    assert stimuli_onehot.shape == (n_t, N_STIMULUS_ENCODINGS)
    save_tensorstore(stimuli_onehot, f"{subject_id:02d}", "stimuli")

    if 'Behavior_full' in data_struct.dtype.names:
        behavioral_covariates = data_struct['Behavior_full'].T.astype(np.float32)
        assert behavioral_covariates.shape == (n_t, 5), f"Expected (n_t, 5), got {behavioral_covariates.shape}"
        save_tensorstore(behavioral_covariates, f"{subject_id:02d}", "behavioral_covariates")
    else:
        raise ValueError(f"Behavior_full not found for subject {subject_id:02d}")

    conditions, condition_intervals = get_condition_cfg(data_struct)
    condition_names = tuple(CONDITION_NAMES[i] for i in conditions)

    if len(conditions) == 1:
        conditions_train = conditions
        conditions_holdout = ()
    else:
        conditions_train = conditions[:-1]
        conditions_holdout = (conditions[-1],)



    spec = get_janelia_spec(
        subject_id=subject_id,
        condition_intervals=condition_intervals,
        condition_names=condition_names,
        conditions_train=conditions_train,
        conditions_holdout=conditions_holdout,
        n_t=n_t,
        n_neurons=n_neurons
    )

    return spec


def main():
    os.makedirs(PATH_STORE, exist_ok=True)

    for subject_id in SUBJECT_ID_LIST:
        try:
            spec = process_janelia_subject(subject_id)
            subject_key = f'subject_{subject_id:02d}'
            write_spec_to_constants_file(subject_key, spec)
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
