#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from zapbench.ts_forecasting.configs import linear
from zapbench.ts_forecasting import input_pipeline
from zapbench.ts_forecasting import data_source
from connectomics.jax import grain_util
import grain.python as grain
import jax

print("🔍 Verifying new unified record mapping implementation...")
print("=" * 60)

config = linear.get_config("timesteps_input=4")
drop_remainder = True
shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)
all_ops = input_pipeline.get_all_ops()
transformations = list(grain_util.parse(config.pre_process_str, all_ops))
process_batch_size = jax.local_device_count() * config.per_device_batch_size
batch_op = grain.Batch(batch_size=process_batch_size, drop_remainder=drop_remainder)
transformations.append(batch_op)
transformations += list(grain_util.parse(config.batch_process_str, all_ops))

print(f"📋 Config: timesteps_input={config.timesteps_input}, timesteps_output={config.timesteps_output}")
print(f"📋 Training conditions: {config.train_conditions}")
print(f"📋 Sequential data source: {config.sequential_data_source}")

train_source = data_source.ConcatenatedTensorStoreTimeSeries(*[
    data_source.MergedTensorStoreTimeSeries(*[
        data_source.TensorStoreTimeSeries(
            config=data_source.TensorStoreTimeSeriesConfig(
                input_spec=input_spec.to_dict() if hasattr(input_spec, 'to_dict') else input_spec,
                timesteps_input=config.timesteps_input,
                timesteps_output=config.timesteps_output,
            ),
            prefetch=config.prefetch,
            prefix=name,
            sequential=config.sequential_data_source,
        )
        for name, input_spec in series.items()
    ])
    for series in config.train_specs
])

print(f"\n🔢 Total train_source length: {len(train_source)}")

# Verify each condition's data sources
print("\n🔍 Inspecting individual condition data sources:")
print(f"Number of conditions in train_source: {len(train_source.srcs)}")

for condition_idx, merged_source in enumerate(train_source.srcs):
    print(f"\n📊 Condition {config.train_conditions[condition_idx]}:")
    print(f"  Merged source length: {len(merged_source)}")
    
    # Check each series in the merged source  
    for series_idx, ts_source in enumerate(merged_source.srcs):
        series_name = ts_source.prefix if hasattr(ts_source, 'prefix') else f"series_{series_idx}"
        print(f"  📈 Series '{series_name}':")
        print(f"    Length: {len(ts_source)}")
        print(f"    Has record_key_to_index: {hasattr(ts_source, 'record_key_to_index')}")
        
        if hasattr(ts_source, 'record_key_to_index'):
            mapping = ts_source.record_key_to_index
            print(f"    Record mapping size: {len(mapping)}")
            
            # Check if it's contiguous (old behavior) or non-contiguous (new gaps)
            if len(mapping) > 0:
                min_idx = min(mapping.values())
                max_idx = max(mapping.values())
                expected_contiguous = max_idx - min_idx + 1
                is_contiguous = len(mapping) == expected_contiguous
                print(f"    Index range: {min_idx} → {max_idx} (span: {expected_contiguous})")
                print(f"    Is contiguous: {is_contiguous}")
                
                # Show first few and last few mappings
                sorted_keys = sorted(mapping.keys())
                if len(sorted_keys) >= 6:
                    first_mappings = {k: mapping[k] for k in sorted_keys[:3]}
                    last_mappings = {k: mapping[k] for k in sorted_keys[-3:]}
                    print(f"    First mappings: {first_mappings}")
                    print(f"    Last mappings: {last_mappings}")
                else:
                    print(f"    All mappings: {dict(sorted(mapping.items()))}")
                
                # Verify consecutive property for contiguous case
                if is_contiguous:
                    consecutive = all(mapping[k] == mapping[k-1] + 1 for k in sorted_keys[1:])
                    print(f"    Consecutive mapping: {consecutive}")
        
        # Check input spec structure
        if hasattr(ts_source, 'config') and hasattr(ts_source.config, 'input_spec'):
            spec = ts_source.config.input_spec
            has_advanced_indexing = (
                'transform' in spec and 
                'output' in spec['transform'] and 
                any('index_array' in output for output in spec['transform']['output'])
            )
            print(f"    Has advanced indexing: {has_advanced_indexing}")
            
            if has_advanced_indexing:
                for output in spec['transform']['output']:
                    if 'index_array' in output:
                        index_array = output['index_array'][0]
                        print(f"    Index array length: {len(index_array)}")
                        print(f"    Index array range: {index_array[0]} → {index_array[-1]}")
                        break

print(f"\n🎯 Testing data loading consistency...")

train_sampler = grain.IndexSampler(
    num_records=len(train_source),
    shuffle=False,  # Use deterministic order for verification
    seed=42,
    num_epochs=1,
    shard_options=shard_options,
)

train_loader = grain.DataLoader(
    data_source=train_source,
    sampler=train_sampler,
    operations=transformations,
    worker_count=config.grain_num_workers,
)

train_iter = iter(train_loader)

# Load first batch and verify structure
batch = next(train_iter)
print(f"✅ Successfully loaded first batch!")
print(f"Batch keys: {list(batch.keys())}")
for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

# Test a few individual records directly
print(f"\n🔍 Testing direct record access:")
for i in [0, len(train_source)//2, len(train_source)-1]:
    try:
        record = train_source[i]
        print(f"  Record {i}: ✅ Success")
        for key, value in record.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: shape {value.shape}")
    except Exception as e:
        print(f"  Record {i}: ❌ Failed - {e}")

print(f"\n✅ Verification complete!")
print("=" * 60)