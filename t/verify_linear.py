#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp

import zapbench.models.util as model_util
from zapbench.ts_forecasting.configs import linear

config = linear.get_config()
model = model_util.model_from_config(config)
init_rng, dropout_rng = jax.random.split(jax.random.PRNGKey(42), num=2)
variables = model.init(init_rng, jnp.ones(config.series_shape), train=False)
params = variables["params"]
batch_stats = variables.get("batch_stats", None)
n_neurons = 64

input_batch = jnp.ones((8, 4, n_neurons))
output = model.apply(variables, input_batch, train=False)

print("=== Model Structure ===")
print(model)

print("\n=== Parameter Shapes ===")
print(jax.tree_util.tree_map(lambda x: x.shape, params))

print("\n=== Detailed Parameters ===")
for path, param in jax.tree_util.tree_flatten_with_path(params)[0]:
  path_str = ".".join(str(k.key) for k in path)
  print(f"{path_str:20} {param.shape} ({param.size:,} params)")

print("\n=== Model Tabulate ===")
print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 4, n_neurons))))

print(f"\nOutput shape: {output.shape}")
