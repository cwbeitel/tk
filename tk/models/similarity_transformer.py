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
    return tf.matmul(a, b, transpose_b=True)


def slicenet_similarity_cost(a, b, hparams=None):
  """Hinge cosine similarity poached from slicenet.
  
  TODO: Not clear on cost_im or why we're clearing the diagonals.

  """

  margin = 0.2

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


def simple_similarity_cost(a, b, hparams=None):
  """Experimental simplified cosine distance loss.
  
  TODO: Consider making k a function of batch size to control weighting of
  importance of self- vs non-self distances.
  
  """

  if hasattr(hparams, "cs_balanced_k"):
    k = hparams.cs_balanced_k

  with tf.name_scope("simple_cd_loss"):

    cosine_similarity = _cosine_similarity(a, b)

    #cosine_similarity = tf.maximum(0.0, cosine_similarity)
    
    # get scores that refer to two embeddings that should correspond
    self_cosine_similarity = tf.diag_part(cosine_similarity)
    
    # Sum the values off the diagonal with 1 - values on diagonal, k >=0, maybe 2
    # Will have values on range [-B^2, B^2] for batch size B.
    return tf.reduce_mean(cosine_similarity) - k*tf.reduce_mean(self_cosine_similarity) + k


def kubeflow_similarity_cost(a, b, hparams=None):
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
 
  if hasattr(hparams, "cs_unbalanced_scale_factor"):
    scale_factor = hparams.cs_unbalanced_scale_factor

  if hasattr(hparams, "cs_unbalanced_target"):
    target = hparams.cs_unbalanced_target

  with tf.name_scope("kf_loss"):
        
    shift = scale_factor*(1 - target)

    cosine_similarity = _cosine_similarity(a, b)
    cosine_similarity_flat = tf.reshape(cosine_similarity, [-1, 1])
    cosine_similarity_flat = scale_factor * cosine_similarity_flat #- shift

    # Positive samples on the diagonal, reshaped as row-major.
    label_matrix = tf.eye(tf.shape(cosine_similarity)[0], dtype=tf.float32)
    label_matrix_flat = tf.reshape(label_matrix, [-1, 1])

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=label_matrix_flat,
                                                   logits=cosine_similarity_flat)

"""
Consider this:

def loss(x1, x2, y):
    # Euclidean distance between x1,x2
    l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(x1, x2)),
                                    reduction_indices=1))

    # you can try margin parameters
    margin = tf.constant(1.)     

    labels = tf.to_float(y)

    match_loss = tf.square(l2diff, 'match_term')
    mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)), 'mismatch_term')

    # if label is 1, only match_loss will count, otherwise mismatch_loss
    loss = tf.add(tf.mul(labels, match_loss), \
                  tf.mul((1 - labels), mismatch_loss), 'loss_add')

    loss_mean = tf.reduce_mean(loss)
    return loss_mean

loss_ = loss(x1_, x2_, y_)

"""

def similarity_cost(a, b, hparams=None):
  """Compute either the kfnet or slicenet cosine similarity loss."""

  costs = {}
  
  assert hasattr(hparams, "use_transpose_similarity")
  use_transpose_similarity = hparams.use_transpose_similarity

  assert hasattr(hparams, "loss_variant")
  loss_variant = hparams.loss_variant

  if hasattr(hparams, "use_transpose_cosine_similarity"):
    use_transpose_cosine_similarity = hparams.use_transpose_cosine_similarity
  else:
    use_transpose_cosine_similarity = 0

  ts_multiplier = 1
  if hasattr(hparams, "transpose_similarity_multiplier"):
    ts_multiplier = hparams.transpose_similarity_multiplier

  if (use_transpose_similarity == 1):
    cs = _cosine_similarity(a, b)

    # Compute the vector difference between the positions of a code and doc
    # string pair in similarity space (i.e. the vector space constructed by
    # taking the cosine similarity between all embeddings in a batch).
    ts = tf.reduce_mean(tf.abs(cs - tf.transpose(cs)))
    ts = ts_multiplier*ts

    """
    ts = tf.log(tf.reduce_sum(tf.abs(cs - tf.transpose(cs))) + 1)
    ts = ts_multiplier*ts
    """
    
    """
    # Euclidean distance
    ts = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(cs, tf.transpose(cs))),
                               reduction_indices=1))
    ts = ts_multiplier*ts
    """

    costs.update({"transpose_similarity": ts})

  if (use_transpose_cosine_similarity == 1):

    cs = _cosine_similarity(a, b)

    # Compute the similarity of doc string and code similarity vectors.
    # I.e. the similarity of a pair in terms of the vector comprised of
    # their similarity to every other point in the set (i.e. in similarity
    # space).
    cs2 = 0.5*(1 - tf.reduce_mean(_cosine_similarity(cs, tf.transpose(cs))))
    
    cs2 = cs2 * ts_multiplier
    
    costs.update({"transpose_cosine_similarity": cs2})

  non_ts_multiplier = 1
  if hasattr(hparams, "non_ts_similarity_multiplier"):
    non_ts_multiplier = hparams.non_ts_similarity_multiplier
    
  # TODO: Update to new loss variant nicknames
  if loss_variant == "slicenet":
    cost = non_ts_multiplier * slicenet_similarity_cost(a, b, hparams)
    costs.update({"cs_hinge": cost})
  elif loss_variant == "kfnet":
    cost = non_ts_multiplier * kubeflow_similarity_cost(a, b, hparams)
    costs.update({"cs_unbalanced": cost})
  elif loss_variant == "simple_cd":
    cost = non_ts_multiplier * simple_similarity_cost(a, b, hparams)
    costs.update({"cs_balanced": cost})
  elif loss_variant == "none":
    next
  else:
    raise ValueError("Unrecognize loss variant: %s" % loss_variant)

  return costs


@registry.register_model
class ConstrainedEmbeddingTransformer(Transformer):

  def body(self, features):
    hparams = self._hparams
    target_space = problem.SpaceID.EN_TOK

    if self._hparams.mode == tf.estimator.ModeKeys.PREDICT:
      encoded, _ = self.encode(features["inputs"], target_space, hparams,
                               features=features)
      return tf.reduce_mean(encoded, axis=1), {"training": 0.0}
  
    features_alt = {"inputs":features["docstring"], "targets":features["code"]}
    features_alt = self.bottom(features_alt)

    string_embedding, _ = self.encode(features_alt["inputs"], target_space, hparams,
                                      features=features_alt)
    string_embedding = tf.reduce_mean(string_embedding, axis=1)
    code_embedding, _ = self.encode(features_alt["targets"], target_space, hparams,
                                    features=features_alt)
    code_embedding = tf.reduce_mean(code_embedding, axis=1)
    
    sc = similarity_cost(string_embedding,
                         code_embedding,
                         hparams)

    if hasattr(hparams, "compute_mapping_loss") and hparams.compute_mapping_loss == 0:
      foo_training_loss = {"training": sum([val for key, val in sc.items()])}
      sc.update(foo_training_loss)
      return code_embedding, sc

    ret = super(ConstrainedEmbeddingTransformer, self).body(features)

    if isinstance(ret, tf.Tensor) or isinstance(ret, tf.EagerTensor):
      return ret, sc

    elif isinstance(ret, tuple):
      if not isinstance(ret[1], dict):
        raise ValueError("Unexpected second type in superclass body return.")
      ret[1].update(sc)
      return ret

    
MODEL_NAME = 'cs_similarity_transformer'

# We don't use the default name because there is already an older version
# included as part of the T2T library with the default name.
@registry.register_model(MODEL_NAME)
class SimilarityTransformer(t2t_model.T2TModel):
  """Transformer Model for Similarity between two strings.
  This model defines the architecture using two transformer
  networks, each of which embed a string and the loss is
  calculated as a Binary Cross-Entropy loss. Normalized
  Dot Product is used as the distance measure between two
  string embeddings.
  """
  def top(self, body_output, _):  # pylint: disable=no-self-use
    return body_output

  def body(self, features):

    if self.hparams.mode != tf.estimator.ModeKeys.PREDICT:
      # In training mode we need to embed both the queries and the code
      # using the inputs and targets respectively.
      with tf.variable_scope('string_embedding'):
        string_embedding = self.encode(features, 'inputs')

      with tf.variable_scope('code_embedding'):
        code_embedding = self.encode(features, 'targets')

      string_embedding_norm = tf.nn.l2_normalize(string_embedding, axis=1)
      code_embedding_norm = tf.nn.l2_normalize(code_embedding, axis=1)

      # All-vs-All cosine distance matrix, reshaped as row-major.
      cosine_dist = 1.0 - tf.matmul(string_embedding_norm, code_embedding_norm,
                                      transpose_b=True)
      cosine_dist_flat = tf.reshape(cosine_dist, [-1, 1])

      # Positive samples on the diagonal, reshaped as row-major.
      label_matrix = tf.eye(tf.shape(cosine_dist)[0], dtype=tf.int32)
      label_matrix_flat = tf.reshape(label_matrix, [-1])

      logits = tf.concat([1.0 - cosine_dist_flat, cosine_dist_flat], axis=1)
      labels = tf.one_hot(label_matrix_flat, 2)

      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
      result = string_embedding
      return result, {'training': loss}


    # In predict mode we conditionally embed either the string query
    # or the code based on the embed_code feature. In both cases the
    # input will be in the inputs feature but the variable scope will
    # be different
    # Define predicates to be used with tf.cond
    def embed_string():
      with tf.variable_scope('string_embedding'):
        string_embedding = self.encode(features, 'inputs')
      return string_embedding

    def embed_code():
      with tf.variable_scope('code_embedding'):
        code_embedding = self.encode(features, 'inputs')
      return code_embedding

    embed_code_feature = features.get('embed_code')

    # embed_code_feature will be a tensor because inputs will be a batch
    # of inputs. We need to reduce that down to a single value for use
    # with tf.cond; so we simply take the max of all the elements.
    # This implicitly assume all inputs have the same value.
    is_embed_code = tf.reduce_max(embed_code_feature)
    result = tf.cond(is_embed_code > 0, embed_code, embed_string)

    result = tf.nn.l2_normalize(result)
    return result

  def encode(self, features, input_key):
    hparams = self._hparams
    inputs = common_layers.flatten4d3d(features[input_key])

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer.transformer_prepare_encoder(inputs, problem.SpaceID.EN_TOK,
                                                hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer.transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, input_key))

    encoder_output = tf.reduce_mean(encoder_output, axis=1)

    return encoder_output

  def infer(self, features=None, **kwargs):
    del kwargs
    predictions, _ = self(features)
    return predictions


@registry.register_hparams
def similarity_transformer_base():
  hparams = transformer_base()
  hparams.add_hparam("loss_variant", "kfnet")
  hparams.add_hparam("use_transpose_similarity", 1)
  hparams.add_hparam("cs_balanced_k", 1.0)
  hparams.add_hparam("cs_unbalanced_scale_factor", 20.0)
  hparams.add_hparam("cs_unbalanced_target", 0.0)
  hparams.add_hparam("transpose_similarity_multiplier", 5.0)
  hparams.add_hparam("compute_mapping_loss", 1)
  hparams.add_hparam("use_transpose_cosine_similarity", 1)
  hparams.add_hparam("non_ts_similarity_multiplier", 1)
  hparams.add_hparam("predict_input_modality", "code")
  return hparams


@registry.register_hparams
def similarity_transformer_tiny():
  hparams = similarity_transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams
