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

"""Tests of modified similarity transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import atexit
import subprocess
import socket
import shlex

import grpc
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.serving import serving_utils
from tensor2tensor.serving import export
from tensor2tensor.utils import decoding
from tensor2tensor.utils import usr_dir
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib

from tk.models import similarity_transformer

FLAGS = tf.flags.FLAGS


def encode(input_str, output_str=None, encoders=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}


def _get_t2t_usr_dir():
  """Get the path to the t2t usr dir."""
  return os.path.join(os.path.realpath(__file__), "../")


class TestSimilarityTransformerExport(tf.test.TestCase):

  def test_trains(self):
    """Test that we can export and query the model via tf.serving."""

    FLAGS.t2t_usr_dir = _get_t2t_usr_dir()
    FLAGS.problem = "github_function_docstring"
    FLAGS.data_dir = "/mnt/nfs-east1-d/data"
    FLAGS.tmp_dir = "/mnt/nfs-east1-d/tmp"
    FLAGS.output_dir = tempfile.mkdtemp()
    FLAGS.model = "similarity_transformer_dev"
    FLAGS.hparams_set = "similarity_transformer_tiny"
    FLAGS.train_steps = 1000
    FLAGS.schedule = "train"
    FLAGS.hparams = "'loss_variant=slicenet'"

    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    
    t2t_trainer.main(None)


if __name__ == "__main__":
  tf.test.main()