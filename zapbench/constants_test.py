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

"""Tests for dataset constants."""

import os
import runpy
from unittest import mock

from absl.testing import absltest
from zapbench import constants


class ConstantsTest(absltest.TestCase):

  def test_local_file_uris_use_absolute_root(self):
    root_path = '/space/vault/zapbench/data/example'
    with mock.patch.dict(os.environ, {'ROOT_PATH': root_path}):
      registry = runpy.run_path(constants.__file__)['DATASET_CONFIGS']

    file_uris = []

    def collect_file_uris(value):
      if isinstance(value, str) and value.startswith('file://'):
        file_uris.append(value)
      elif isinstance(value, dict):
        for child in value.values():
          collect_file_uris(child)
      elif isinstance(value, (list, tuple)):
        for child in value:
          collect_file_uris(child)

    collect_file_uris(registry)
    self.assertTrue(file_uris)

    expected_prefix = f'file://{root_path}/ts_files/'
    for uri in file_uris:
      self.assertTrue(uri.startswith(expected_prefix), uri)
      self.assertNotIn('file:////space/', uri)


if __name__ == '__main__':
  absltest.main()
