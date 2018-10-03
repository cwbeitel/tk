# coding=utf-8
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

"""Definition of GithubFunctionDocstring problem."""

import csv

import tensorflow as tf

from tensor2tensor.utils import registry

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

from tk.data_generators import function_docstring


class TestProblem(tf.test.TestCase):
    
  def test_instantiate(self):
    
    _ = registry.problem("github_function_docstring")
    
  def test_generates(self):
    
    skip = False
    
    if not skip:
        
      problem_object = registry.problem("github_function_docstring")
        
      data_dir = "/mnt/nfs-east1-d/data"
      tmp_dir = "/mnt/nfs-east1-d/tmp"
        
      problem_object = registry.problem("github_function_docstring")
      problem_object.generate_data(data_dir, tmp_dir)
          
      example = tfe.Iterator(problem_object.dataset(Modes.TRAIN, data_dir)).next()

      # TODO: Test something about the example

    
if __name__ == "__main__":
  tf.test.main()