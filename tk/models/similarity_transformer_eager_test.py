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

"""Tests of modified similarity transformer model.

TODO: Currently this test assumes data_dir is /mnt/nfs-east1-d/data

"""

import tensorflow as tf

from tensor2tensor.utils import registry

import numpy as np

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

from tk.models import similarity_transformer


"""
class TestLossFunctions(tf.test.TestCase):
    
  def setUp(self):
    
    # Create two tensors, each with a 3-dimensional (columns) embedding vector for
    # each of three examples (rows).

    self.t1 = tf.convert_to_tensor(
        np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0], [5.0, 6.0, 3.0]], dtype=np.float32),
        dtype=tf.float32)

    self.t2 = tf.convert_to_tensor(
        np.array([[4.0, 3.0, 3.0], [2.0, 1.0, 3.0], [7.0, 8.0, 3.0]], dtype=np.float32),
        dtype=tf.float32)

  def test_losses(self):
    

    for loss_variant in similarity_transformer._LOSS_VARIANTS:
      similarity_transformer.similarity_cost(self.t1, self.t2, loss_variant)

"""

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None, encoders=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, 1, -1, 1])  # Make it 3D.
  return batch_inputs


def training_test(hparams, p_hparams, problem_object, data_dir):

    model = registry.model("similarity_transformer_dev")(
        hparams, tf.estimator.ModeKeys.TRAIN, p_hparams
    )

    batch_size = 1
    train_dataset = problem_object.dataset(Modes.TRAIN, data_dir)
    train_dataset = train_dataset.repeat(None).batch(batch_size)

    optimizer = tf.train.AdamOptimizer()

    @tfe.implicit_value_and_gradients
    def loss_fn(features):
      _, losses = model(features)
      return losses["training"]

    NUM_STEPS = 10

    for count, example in enumerate(tfe.Iterator(train_dataset)):
      loss, gv = loss_fn(example)
      optimizer.apply_gradients(gv)


class TestSimilarityTransformerDevModel(tf.test.TestCase):

  def setUp(self):
    self.problem_object = registry.problem("github_constrained_embedding")
    self.data_dir = "/mnt/nfs-east1-d/data"
    self.hparams = registry.hparams("similarity_transformer_tiny")
    self.hparams.data_dir = self.data_dir
    self.p_hparams = self.problem_object.get_hparams(self.hparams)
    self.encoders = self.problem_object.feature_encoders(self.data_dir)

  def test_trains(self):
    """Test that we can run one 10 iterations of training with batch size of 1."""
    
    model = registry.model("constrained_embedding_transformer_v2")(
        self.hparams, tf.estimator.ModeKeys.TRAIN, self.p_hparams
    )

    batch_size = 1
    train_dataset = self.problem_object.dataset(Modes.TRAIN, self.data_dir)
    train_dataset = train_dataset.repeat(None).batch(batch_size)

    optimizer = tf.train.AdamOptimizer()

    @tfe.implicit_value_and_gradients
    def loss_fn(features):
      _, losses = model(features)
      return losses["training"]

    NUM_STEPS = 10

    for count, example in enumerate(tfe.Iterator(train_dataset)):
      loss, gv = loss_fn(example)
      optimizer.apply_gradients(gv)

      if count % 1 == 0:
       print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
      if count >= NUM_STEPS:
       break
    
    # TODO: Test something about the resulting model

  """
  Not clear why these tests are failing.
  def test_infer(self):

    model = registry.model("similarity_transformer_dev")(
        self.hparams, tf.estimator.ModeKeys.TRAIN, self.p_hparams
    )
    
    query = "hello world"
    
    _ = model.encode_string(encode(query, encoders=self.encoders))
    
    # TODO: Test something about the result

  def test_pair_distance(self):

    model = registry.model("similarity_transformer_dev")(
        self.hparams, tf.estimator.ModeKeys.TRAIN, self.p_hparams
    )
    
    # TODO: Set loss_variant hparam
    
    code1 = "def my_function(query):  print(query)"
    code2 = "nronfg vmo5i n6565-23 wrdnds vdmam65 3ehn bdp"

    code1_emb = model.encode_code(encode(code1, encoders=self.encoders))
    code2_emb = model.encode_code(encode(code2, encoders=self.encoders))

    model.similarity_cost(code1_emb, code2_emb)
    
    # TODO: Test something about the computed loss
    """

if __name__ == "__main__":
  tf.test.main()