#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import importlib

import jax
import jax.numpy as jnp

import zapbench.models.util as model_util

# List of config modules to test (excluding infer.py and common.py)
CONFIG_MODULES = ["linear", "mean", "stimulus", "tide", "timemix", "tsmixer"]


def verify_model(config_name):
  """Verify input/output shapes for a given config."""
  print(f"\n{'='*60}")
  print(f"Testing config: {config_name}")
  print(f"{'='*60}")

  try:
    # Import the config module
    config_module = importlib.import_module(
        f"zapbench.ts_forecasting.configs.{config_name}"
    )
    config = config_module.get_config()

    print(f"Model class: {config.model_class}")
    print(f"Series shape: {config.series_shape}")
    print(f"Timesteps input: {getattr(config, 'timesteps_input', 'N/A')}")
    print(f"Timesteps output: {getattr(config, 'timesteps_output', 'N/A')}")

    # Create model
    model = model_util.model_from_config(config)

    # Initialize model
    init_rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones(config.series_shape)
    variables = model.init(init_rng, dummy_input, train=False)

    # Test with a batch
    batch_size = 8
    input_shape = (batch_size,) + config.series_shape[
        1:
    ]  # Remove leading 1 from series_shape
    test_input = jnp.ones(input_shape)

    print(f"Test input shape: {test_input.shape}")

    # Get model output
    output = model.apply(variables, test_input, train=False)
    print(f"Model output shape: {output.shape}")

    # Count parameters
    params = variables["params"]
    total_params = sum(
        param.size for param in jax.tree_util.tree_leaves(params)
    )
    print(f"Total parameters: {total_params:,}")

    return True

  except Exception as e:
    print(f"ERROR testing {config_name}: {str(e)}")
    return False


def main():
  print("ZAPBench TS Forecasting Models Verification")
  print(f"JAX devices: {jax.devices()}")

  success_count = 0
  total_count = len(CONFIG_MODULES)

  for config_name in CONFIG_MODULES:
    success = verify_model(config_name)
    if success:
      success_count += 1

  print(f"\n{'='*60}")
  print(f"SUMMARY: {success_count}/{total_count} configs verified successfully")
  print(f"{'='*60}")


if __name__ == "__main__":
  main()
  main()
  main()
  main()
  main()
  main()
  main()
  main()
