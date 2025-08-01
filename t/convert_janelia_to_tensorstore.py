#!/usr/bin/env python3
"""Convert Janelia dataset to ZAPBench TensorStore format.

This script converts the Janelia .mat and .h5 data to the same TensorStore
format used by ZAPBench, enabling direct use with existing models.
"""

import numpy as np
import h5py
import scipy.io
import tensorstore as ts
import json
from pathlib import Path
import argparse


def load_janelia_data(janelia_path: str, subject_id: int = 14):
    """Load neural activity and metadata from Janelia dataset.
    
    Args:
        janelia_path: Path to Janelia dataset directory
        subject_id: Subject ID to load
        
    Returns:
        tuple: (neural_activity, metadata_dict)
    """
    subject_path = Path(janelia_path) / f"subject_{subject_id}"
    
    # Load neural activity from h5 file
    with h5py.File(subject_path / "TimeSeries.h5", "r") as h5_file:
        # Use CellResp (raw responses) - shape should be (timesteps, neurons)
        neural_activity = np.array(h5_file['CellResp'])
        print(f"Neural activity shape: {neural_activity.shape}")
        
        # Also get the valid neuron indices
        if 'absIX' in h5_file:
            valid_indices = np.array(h5_file['absIX']).flatten()
            print(f"Valid indices shape: {valid_indices.shape}")
        else:
            valid_indices = None
    
    # Load metadata from .mat file
    mat_data = scipy.io.loadmat(subject_path / "data_full.mat")
    data_struct = mat_data['data'][0, 0]
    
    # Extract key metadata
    metadata = {
        'frame_rate': float(data_struct['fpsec'][0, 0]),
        'num_cells_full': int(data_struct['numcell_full'][0, 0]),
        'periods': int(data_struct['periods'][0, 0]),
        'cell_positions': data_struct['CellXYZ'],  # (num_cells, 3)  
        'cell_positions_norm': data_struct['CellXYZ_norm'],
        'stimulus_full': data_struct['stim_full'],  # (1, timesteps)
        'behavior_full': data_struct['Behavior_full'],  # (5, timesteps)
        'eye_full': data_struct['Eye_full'],  # (2, timesteps)
        'valid_indices': valid_indices
    }
    
    return neural_activity, metadata


def create_covariates(metadata: dict, timesteps: int):
    """Create covariate features from metadata.
    
    Args:
        metadata: Metadata dictionary from load_janelia_data
        timesteps: Number of timesteps in neural data
        
    Returns:
        np.ndarray: Covariate array of shape (timesteps, n_features)
    """
    covariates = []
    
    # Stimulus features (4 unique values -> one-hot encode)
    stimulus = metadata['stimulus_full'].flatten()[:timesteps]
    stimulus_onehot = np.eye(4)[stimulus.astype(int)]  # (timesteps, 4)
    covariates.append(stimulus_onehot)
    
    # Behavioral features (5 channels)
    behavior = metadata['behavior_full'][:, :timesteps].T  # (timesteps, 5)
    covariates.append(behavior)
    
    # Eye movement features (2 channels)  
    eye = metadata['eye_full'][:, :timesteps].T  # (timesteps, 2)
    covariates.append(eye)
    
    # Combine all features
    all_covariates = np.concatenate(covariates, axis=1)  # (timesteps, 11)
    
    return all_covariates.astype(np.float32)


def save_to_tensorstore(data: np.ndarray, output_path: str, name: str):
    """Save data array to TensorStore in Zarr3 format.
    
    Args:
        data: Data array to save
        output_path: Base output directory
        name: Name for this dataset (e.g., 'traces', 'covariates')
    """
    output_dir = Path(output_path) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorStore spec similar to ZAPBench format
    spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': str(output_dir),
        },
        'metadata': {
            'shape': list(data.shape),
            'data_type': str(data.dtype),
        },
        'create': True,
        'delete_existing': True,
    }
    
    # Open and write data
    dataset = ts.open(spec).result()
    dataset[:].write(data).result()
    
    print(f"Saved {name} with shape {data.shape} to {output_dir}")
    
    return spec


def create_zapbench_constants(metadata: dict, neural_shape: tuple, 
                            covariates_shape: tuple, output_path: str):
    """Create constants file similar to ZAPBench format.
    
    Args:
        metadata: Metadata dictionary
        neural_shape: Shape of neural activity data (timesteps, neurons)
        covariates_shape: Shape of covariates data (timesteps, features)
        output_path: Output directory path
    """
    # For Janelia data, we don't have multiple experimental conditions
    # So we'll treat the entire timeseries as one condition
    timesteps, neurons = neural_shape
    
    constants = {
        # Single condition covering entire timeseries
        'CONDITION_OFFSETS': (0, timesteps),
        'CONDITION_PADDING': 1,
        'CONDITIONS_TRAIN': (0,),
        'CONDITIONS_HOLDOUT': (),  # No holdout condition for now
        'CONDITIONS': (0,),
        'CONDITION_NAMES': ('janelia_full',),
        'VAL_FRACTION': 0.1,
        'TEST_FRACTION': 0.2,
        'MAX_CONTEXT_LENGTH': 256,
        'PREDICTION_WINDOW_LENGTH': 32,
        'TIMESERIES_NAME': 'janelia_traces',
        'COVARIATE_SERIES_NAME': 'janelia_covariates',
        
        # TensorStore specs
        'SPECS': {
            'janelia_traces': {
                'kvstore': f'file://{Path(output_path).absolute()}/traces/',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[timesteps], neurons],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                }
            }
        },
        
        'COVARIATE_SPECS': {
            'janelia_covariates': {
                'kvstore': f'file://{Path(output_path).absolute()}/covariates/',
                'driver': 'zarr3',
                'transform': {
                    'input_exclusive_max': [[timesteps], covariates_shape[1]],
                    'input_inclusive_min': [0, 0],
                    'input_labels': ['t', 'f'],
                }
            }
        },
        
        # Data characteristics
        'MIN_MAX_VALUES': {
            'janelia_traces': (float(np.min(neural_shape)), float(np.max(neural_shape))),
        },
        
        # Metadata
        'FRAME_RATE': metadata['frame_rate'],
        'NUM_CELLS': neurons,
        'EXPERIMENT_INFO': {
            'subject_id': 14,  # Default subject
            'num_cells_full': metadata['num_cells_full'],
            'periods': metadata['periods'],
        }
    }
    
    # Save constants as JSON
    constants_path = Path(output_path) / 'janelia_constants.json'
    with open(constants_path, 'w') as f:
        json.dump(constants, f, indent=2)
    
    print(f"Saved constants to {constants_path}")
    return constants


def main():
    parser = argparse.ArgumentParser(description='Convert Janelia data to TensorStore format')
    parser.add_argument('--janelia_path', type=str, 
                       default='/Users/s/vault/neural_data/janelia/',
                       help='Path to Janelia dataset directory')
    parser.add_argument('--subject_id', type=int, default=14,
                       help='Subject ID to convert')
    parser.add_argument('--output_path', type=str,
                       default='/Users/s/vault/zapbench_format/janelia/',
                       help='Output directory for TensorStore data')
    
    args = parser.parse_args()
    
    print("Loading Janelia data...")
    neural_activity, metadata = load_janelia_data(args.janelia_path, args.subject_id)
    
    # Convert to float32 for consistency with ZAPBench
    neural_activity = neural_activity.astype(np.float32)
    
    print("Creating covariates...")
    covariates = create_covariates(metadata, neural_activity.shape[0])
    
    print("Saving to TensorStore format...")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save neural activity
    save_to_tensorstore(neural_activity, args.output_path, 'traces')
    
    # Save covariates
    save_to_tensorstore(covariates, args.output_path, 'covariates')
    
    # Create constants file
    constants = create_zapbench_constants(
        metadata, neural_activity.shape, covariates.shape, args.output_path
    )
    
    print("\nConversion completed!")
    print(f"Neural activity: {neural_activity.shape}")
    print(f"Covariates: {covariates.shape}")
    print(f"Output directory: {output_path}")
    print(f"Frame rate: {metadata['frame_rate']:.2f} fps")
    
    # Print usage instructions
    print("\nTo use with ZAPBench models:")
    print("1. Update config files to use the new constants")
    print("2. Set TIMESERIES_NAME = 'janelia_traces'")
    print("3. Set COVARIATE_SERIES_NAME = 'janelia_covariates'")


if __name__ == '__main__':
    main()