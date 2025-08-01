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

config = linear.get_config("timesteps_input=4")
drop_remainder = True
shard_options = grain.ShardByJaxProcess(drop_remainder=drop_remainder)
all_ops = input_pipeline.get_all_ops()
transformations = list(grain_util.parse(config.pre_process_str, all_ops))
process_batch_size = jax.local_device_count() * config.per_device_batch_size
batch_op = grain.Batch(batch_size=process_batch_size, drop_remainder=drop_remainder)
transformations.append(batch_op)
transformations += list(grain_util.parse(config.batch_process_str, all_ops))

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

train_sampler = grain.IndexSampler(
    num_records=len(train_source),
    shuffle=True,
    seed=42,
    num_epochs=10,
    shard_options=shard_options,
)

train_loader = grain.DataLoader(
    data_source=train_source,
    sampler=train_sampler,
    operations=transformations,
    worker_count=config.grain_num_workers,
)

train_iter = iter(train_loader)
batch = next(train_iter)