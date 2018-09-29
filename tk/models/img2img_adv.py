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

"""Adversarial transformer model."""

import tensorflow as tf

import copy

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from tensor2tensor.models.image_transformer_2d import img2img_transformer2d_tiny

from tensor2tensor.models.research import transformer_vae


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def reverse_gradient(x):
  
  return -x + tf.stop_gradient(2 * x)


# ------------------------------------------------------------
# Copied from t2t master when needing to revert back to prior version where
# deep_discriminator was not yet in common_layers.

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def deep_discriminator(x,
                       batch_norm,
                       is_training,
                       filters=512,
                       filter_size=4,
                       stride=2,
                       output_size=1024,
                       tanh_output=False):
  """Discriminator architecture based on InfoGAN."""
  with tf.variable_scope(
      "discriminator", initializer=tf.random_normal_initializer(stddev=0.02)):
    
    batch_size, height, width = shape_list(x)[:3]
    
     # Mapping x from [bs, h, w, c] to [bs, 1]
    net = tf.layers.conv2d(
        x, filters, filter_size, strides=stride, padding="SAME", name="conv1")
    
    net = lrelu(net)

    net = tf.layers.conv2d(net, 2 * filters, filter_size, strides=stride,
        padding="SAME", name="conv2")

    if batch_norm:
      tf.logging.info("Performing discriminator batch norm.")
      net = tf.layers.batch_normalization(
          net, training=is_training, momentum=0.999, name="d_bn2")

    net = lrelu(net)

    size = height * width

    x_shape = x.get_shape().as_list()
    
    #if not isinstance(size * filters / 8, int):
    #    raise ValueError("size * filters should be divisible by 8")

    if x_shape[1] is None or x_shape[2] is None:
      net = tf.reduce_mean(net, axis=[1, 2])
    else:
      s = shape_list(net)
      # Or alternatively size * filters / 8
      net = tf.reshape(net, [batch_size, s[1] * s[2] * s[3]])

    net = tf.layers.dense(net, output_size, name="d_fc3")

    if batch_norm:
      net = tf.layers.batch_normalization(
          net, training=is_training, momentum=0.999, name="d_bn3")

    if tanh_output:
      tf.logging.info("Discriminator final nonlinearity: tanh.")
      net = tf.tanh(net)
    else:
      tf.logging.info("Discriminator final nonlinearity: relu.")
      net = lrelu(net)

    return net

# ------------------------------------------------------------

@registry.register_model
class Img2imgTransformerAdversarial(t2t_model.T2TModel):
  """Image 2 Image transformer net."""

  def vanilla_gan_loss(self, real_input, fake_input, is_training):
    
    with tf.variable_scope("gan_loss"):

      # Using common_layers.deep_discriminator
      d_r = deep_discriminator(
          real_input,
          batch_norm=self.hparams.discriminator_batchnorm,
          is_training=is_training,
          filters=self.hparams.discriminator_num_filters,
          tanh_output=self.hparams.discriminator_tanh_output)

      d_f = deep_discriminator(
          reverse_gradient(fake_input),
          batch_norm=self.hparams.discriminator_batchnorm,
          is_training=is_training,
          filters=self.hparams.discriminator_num_filters,
          tanh_output=self.hparams.discriminator_tanh_output)
        
      smoothing = self.hparams.label_smoothing_factor
      tf.logging.info("Smoothing GAN loss labels by factor %s" % (1 - smoothing))

      d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_r,
          labels=tf.ones_like(d_r) * (1 - smoothing)))
        
      d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_f,
          labels=tf.zeros_like(d_f)))
    
      g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=d_f,
          labels=tf.ones_like(d_f) * (1 - smoothing)))

      return {"discriminator_loss": d_loss, "generator_loss": g_loss}

  def generator(self, inputs, targets):
    """From tensor2tensor.models.img2img_transformer_2d."""

    hparams = copy.copy(self._hparams)

    encoder_input = cia.prepare_encoder(inputs, hparams)

    encoder_output = cia.transformer_encoder_layers(
        encoder_input,
        hparams.num_encoder_layers,
        hparams,
        attention_type=hparams.enc_attention_type,
        name="encoder")

    decoder_input, rows, cols = cia.prepare_decoder(
        targets, hparams)

    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_decoder_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        name="decoder")

    output = cia.create_output(decoder_output, rows, cols, targets, hparams)

    return output

  def body(self, features):
    """Map features through generator & optionally discriminator, GAN loss."""

    # Examples might come in with other types but we'll need these to be floats.
    targets = tf.cast(features["targets"], tf.float32)
    inputs = tf.cast(features["inputs"], tf.float32)
    
    hparams = copy.copy(self._hparams)
    
    import pprint
    pprint.pprint(hparams.__dict__)

    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

    allowed_loss_variants = ["non_gan", "vanilla_gan",
                             "hybrid_vanilla_gan"]

    if hparams.loss_variant not in allowed_loss_variants:
      raise ValueError("Unrecognized loss variant: %s" % hparams.loss_variant)
    
    # Run the generator and obtain a tensor of logits
    generator_output = self.generator(inputs, targets)
    
    # NEW, emulating transformer aux.
    output, losses = self._normalize_body_output(generator_output)

    if "vanilla_gan" in hparams.loss_variant:
        
      tf.logging.info("Vanilla GAN loss will be computed.")
    
      batch_size, height, width, num_channels = common_layers.shape_list(generator_output)[:4]

      #with tf.variable_scope(
      #    "glue", initializer=tf.random_normal_initializer(stddev=0.02)):

      discriminator_input = tf.layers.dense(generator_output, 1, name="fc0")

      discriminator_input = tf.reshape(discriminator_input,
                                       [batch_size,
                                        height,
                                        width,
                                        num_channels])

      gan_loss = self.vanilla_gan_loss(targets,
                                       discriminator_input,
                                       is_training=is_training)
      losses.update(gan_loss)
    
    # Are we computing a hybrid loss?
    if "hybrid" not in hparams.loss_variant and hparams.loss_variant != "non_gan":
      tf.logging.info("Hybrid loss will not be computed (GAN only).")
      # We don't want to compute the default t2t (non-gan) loss for this
      # modality. By returning "training" in losses, t2t will skip computing
      # this.
      losses["training"] = tf.constant(0., dtype=tf.float32)
        
      return tf.zeros_like(targets), losses 
      
    tf.logging.info("Hybrid loss will be computed.")
    return generator_output, losses

  def infer(self, features):
    inputs = tf.cast(features["inputs"], tf.float32)
    targets = tf.cast(features["targets"], tf.float32)
    g_logits = self.generator(inputs, targets)
    g = tf.cast(tf.argmax(g_logits, axis=-1), tf.uint8)
    return g

  
@registry.register_hparams
def img2img_transformer_2d_adversarial():
  """Basic parameters for an adversarial_transformer."""
  hparams = img2img_transformer2d_tiny()
  hparams.label_smoothing = 0.0
  hparams.hidden_size = 128
  hparams.batch_size = 64
  hparams.add_hparam("z_size", 64)
  hparams.add_hparam("c_dim", 1)
  hparams.add_hparam("height", 28)
  hparams.add_hparam("width", 28)
  hparams.add_hparam("discriminator_batchnorm", int(True))
  hparams.add_hparam("l2_multiplier", 0.1)
  hparams.add_hparam("loss_variant", "vanilla_gan")
  hparams.add_hparam("gan_loss_multiplier", 2)
  hparams.add_hparam("num_compress_steps", 3)
  hparams.add_hparam("num_sliced_vecs", 4096)
  hparams.add_hparam("discriminator_num_filters", 64)
  hparams.add_hparam("discriminator_tanh_output", False)
  hparams.add_hparam("label_smoothing_factor", 0.0)
  hparams.learning_rate_warmup_steps = 2000
  return hparams

def img2img_transformer_2d_adversarial_fs256_hs256_dnf128():
  hparams = img2img_transformer_2d_adversarial()
  hparams.loss_variant = "hybrid_vanilla_gan"
  hparams.label_smoothing_factor = 0.1
  hparams.batch_size = 16
  hparams.discriminator_tanh_output = True
  hparams.filter_size = 256
  hparams.hidden_size = 256
  hparams.discriminator_num_filters = 128
  return hparams

def img2img_transformer_2d_adversarial_fs512_hs512_dnf256():
  hparams = img2img_transformer_2d_adversarial()
  hparams.loss_variant = "hybrid_vanilla_gan"
  hparams.label_smoothing_factor = 0.1
  hparams.batch_size = 16
  hparams.discriminator_tanh_output = True
  hparams.filter_size = 512
  hparams.hidden_size = 512
  hparams.discriminator_num_filters = 256
  return hparams

def img2img_transformer_2d_adversarial_fs1024_hs1024_dnf512():
  hparams = img2img_transformer_2d_adversarial()
  hparams.loss_variant = "hybrid_vanilla_gan"
  hparams.label_smoothing_factor = 0.1
  hparams.batch_size = 16
  hparams.discriminator_tanh_output = True
  hparams.filter_size = 1024
  hparams.hidden_size = 1024
  hparams.discriminator_num_filters = 512
  return hparams
