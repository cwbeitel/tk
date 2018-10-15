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

"""Modified similarity transformer model."""

import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models.transformer import Transformer

from tensor2tensor.models.transformer import transformer_base


_LOSS_VARIANTS = [
    "slicenet",
    "kfnet",
    "simple_cd"
]


def _cosine_similarity(a, b):
  """Computes the cosine similarity between two tensors."""

  with tf.name_scope("cosine_similarity"):

    a = tf.nn.l2_normalize(a, 1)
    b = tf.nn.l2_normalize(b, 1)
    return tf.matmul(a, tf.transpose(b)) 


def slicenet_similarity_cost(a, b, margin=0.2):
  """Hinge cosine similarity poached from slicenet.
  
  TODO: Not clear on cost_im or why we're clearing the diagonals.

  """

  with tf.name_scope("slicenet_loss"):

    a, b = common_layers.pad_to_same_length(a, b)

    cosine_similarity = _cosine_similarity(a, b)

    diagonal = tf.diag_part(cosine_similarity)
    cost_s = tf.maximum(0.0, margin - diagonal + cosine_similarity)
    cost_im = tf.maximum(
        0.0, margin - tf.reshape(diagonal, [-1, 1]) + cosine_similarity)

    # Clear diagonals.
    batch_size = tf.shape(a)[0]
    empty_diagonal_mat = tf.ones_like(cost_s) - tf.eye(batch_size)
    cost_s *= empty_diagonal_mat
    cost_im *= empty_diagonal_mat

    return tf.reduce_mean(cost_s) + tf.reduce_mean(cost_im)


def simple_similarity_cost(a, b, k=2):
  """Experimental simplified cosine distance loss.
  
  TODO: Consider making k a function of batch size to control weighting of
  importance of self- vs non-self distances.
  
  """

  with tf.name_scope("simple_cd_loss"):

    cosine_similarity = _cosine_similarity(a, b)

    # get scores that refer to two embeddings that should correspond
    self_cosine_similarity = tf.diag_part(cosine_similarity)
    
    # Sum the values off the diagonal with 1 - values on diagonal, k >=0, maybe 2
    # Will have values on range [-B^2, B^2] for batch size B.
    return tf.reduce_mean(cosine_similarity) - k*tf.reduce_mean(self_cosine_similarity)


def kubeflow_similarity_cost(a, b, scale_factor=20, target=0.2):
  """Modification to original kubeflow code_search example loss.

  Notes:
  - Use cosine similarity instead of cosine distance.
  - Allow specification of stretch factor.

  TODO: Consider adding a shift factor that is a function of the query distance
  and the stretch factor.

  Args:
    a (tensor): An embedding vector.
    b (tensor): An embedding vector.
    scale_factor (float): Scale c.d. from [-1,1] to [-sf,sf].

  """

  with tf.name_scope("kf_loss"):
        
    shift = scale_factor*(1 - target)

    cosine_similarity = _cosine_similarity(a, b)
    cosine_similarity_flat = tf.reshape(cosine_similarity, [-1, 1])
    cosine_similarity_flat = scale_factor * cosine_similarity_flat - shift

    # Positive samples on the diagonal, reshaped as row-major.
    label_matrix = tf.eye(tf.shape(cosine_similarity)[0], dtype=tf.float32)
    label_matrix_flat = tf.reshape(label_matrix, [-1, 1])

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=label_matrix_flat,
                                                logits=cosine_similarity_flat)


def similarity_cost(a, b, loss_variant):
  """Compute either the kfnet or slicenet cosine similarity loss."""

  if loss_variant == "slicenet":
    return slicenet_similarity_cost(a, b)
  elif loss_variant == "kfnet":
    return kubeflow_similarity_cost(a, b)
  elif loss_variant == "simple_cd":
    return kubeflow_similarity_cost(a, b)
  else:
    raise ValueError("Unrecognize loss variant: %s" % loss_variant)


def encode(tensor, hparams):
  """Encoder."""

  tensor = common_layers.flatten4d3d(tensor)

  (encoder_input, encoder_self_attention_bias, _) = (
      transformer.transformer_prepare_encoder(tensor,
                                              problem.SpaceID.EN_TOK,
                                              hparams))

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  encoder_output = transformer.transformer_encoder(
      encoder_input, encoder_self_attention_bias, hparams)

  encoder_output = tf.reduce_mean(encoder_output, axis=1)

  return encoder_output


@registry.register_model
class SimilarityTransformerPretrainStringEncoding(Transformer):

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    with tf.variable_scope("string_encoder"):
      return super(SimilarityTransformerPretrainStringEncoding, self).encode(
          inputs, target_space, hparams, features, losses)

  def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias,
             decoder_self_attention_bias, hparams, cache=None, nonpadding=None,
             losses=None):
    with tf.variable_scope("string_decoder"):
      return super(SimilarityTransformerPretrainStringEncoding, self).decode(
          decoder_input, encoder_output, encoder_decoder_attention_bias,
          decoder_self_attention_bias, hparams, cache, nonpadding,
          losses)


@registry.register_model
class SimilarityTransformerPretrainCodeEncoding(Transformer):
    
  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    with tf.variable_scope("code_encoder"):
      return super(SimilarityTransformerPretrainCodeEncoding, self).encode(
          inputs, target_space, hparams, features, losses)

  def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias,
             decoder_self_attention_bias, hparams, cache=None, nonpadding=None,
             losses=None):
    with tf.variable_scope("code_decoder"):
      return super(SimilarityTransformerPretrainCodeEncoding, self).decode(
          decoder_input, encoder_output, encoder_decoder_attention_bias,
          decoder_self_attention_bias, hparams, cache, nonpadding,
          losses)


@registry.register_model
class SimilarityTransformerDev(Transformer):

  def encode_string(self, inputs, features=None, losses=None):
    hparams = self._hparams
    target_space = problem.SpaceID.EN_TOK
    with tf.variable_scope("string_encoder"):
      encoder_output, _ = super(SimilarityTransformerDev, self).encode(
          inputs, target_space, hparams, features, losses)
      return tf.reduce_mean(encoder_output, axis=1)

  def encode_code(self, inputs, features=None, losses=None):
    hparams = self._hparams
    target_space = problem.SpaceID.EN_TOK
    with tf.variable_scope("code_encoder"):
      encoder_output, _ = super(SimilarityTransformerDev, self).encode(
          inputs, target_space, hparams, features, losses)
      return tf.reduce_mean(encoder_output, axis=1)

  def body(self, features):
    string_embedding = self.encode_string(features["inputs"])
    code_embedding = self.encode_code(features["targets"])
    loss = self.similarity_cost(string_embedding, code_embedding)
    return string_embedding, {"training": loss}

  def similarity_cost(self, a, b):
    loss_variant = self.hparams.loss_variant
    return similarity_cost(a, b, loss_variant)

  def infer(self, features=None, **kwargs):
    del kwargs
    if "targets" not in features:
      features["targets"] = tf.zeros_like(features["inputs"])
    predictions, _ = self(features)
    return predictions


@registry.register_model
class SimilarityTransformerStringEncoder(SimilarityTransformerDev):
  """String encoder model restricted to string encoding branch."""

  def body(self, features):
    return self.encode_string(features["inputs"]), {"training": 0.0}


@registry.register_model
class SimilarityTransformerCodeEncoder(SimilarityTransformerDev):
  """Code encoder model restricted to code encoding branch."""

  def body(self, features):
    return self.encode_code(features["inputs"]), {"training": 0.0}


"""
@registry.register_model
class SimilarityTransformerDev(t2t_model.T2TModel):

  def top(self, body_output, _):  # pylint: disable=no-self-use
    return body_output

  def body(self, features):

    loss_variant = self.hparams.loss_variant

    string_embedding = self.embed_string(features["inputs"])

    if 'targets' in features:
      code_embedding = self.embed_code(features["targets"])
      loss = self.similarity_cost(string_embedding, code_embedding, loss_variant)
      return string_embedding, {"training": loss}

    return string_embedding, {"training": 0.0}

  def embed_string(self, string_tensor):
    with tf.variable_scope("string_embedding"):
      return encode(string_tensor, self._hparams)

  def embed_code(self, code_tensor):
    with tf.variable_scope("code_embedding"):
      return encode(code_tensor, self._hparams)

  def similarity_cost(self, a, b, loss_variant):
    return similarity_cost(a, b, loss_variant)

  def infer(self, features=None, **kwargs):
    del kwargs

    predictions, _ = self(features)
    return predictions
"""

@registry.register_hparams
def similarity_transformer_tiny():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  hparams.docs_encoder_trainable = True
  hparams.code_encoder_trainable = True
  hparams.initializer = None
  hparams.add_hparam("loss_variant", "kfnet")
  return hparams
