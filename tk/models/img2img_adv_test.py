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

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from tensor2tensor.data_generators import allen_brain
from tensor2tensor.models import image_transformer_2d

from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.utils import registry

import tensor2tensor

from tk.models import img2img_adv as models

tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

import os
import shutil
import tempfile
import numpy as np


def try_importing_pil_image():
  """Import a PIL Image object if the function is called."""
  try:
    from PIL import Image
  except ImportError:
    tf.logging.error("Can't import Image from PIL (Pillow). Please install it, "
                     "such as by running `pip install Pillow`.")
    exit(1)

  return Image


def mock_raw_image(x_dim=1024, y_dim=1024, num_channels=3,
                   output_path=None, write_image=True):
  """Generate random `x_dim` by `y_dim`, optionally to `output_path`.

  Args:
    output_path: str, path to which to write image.
    x_dim: int, the x dimension of generated raw image.
    y_dim: int, the x dimension of generated raw image.
    return_raw_image: bool, whether to return the generated image (as a
      numpy array).

  Returns:
    numpy.array: The random `x_dim` by `y_dim` image (i.e. array).

  """

  rand_shape = (x_dim, y_dim, num_channels)
  tf.logging.debug(rand_shape)

  if num_channels != 3:
    raise NotImplementedError("mock_raw_image for channels != 3 not yet "
                              "implemented.")

  img = np.random.random(rand_shape)
  img = np.uint8(img*255)

  if write_image:
    if not isinstance(output_path, str):
      raise ValueError("Output path must be of type str if write_image=True, "
                       "saw %s." % output_path)

    image_obj = try_importing_pil_image()
    pil_img = image_obj.fromarray(img, mode="RGB")
    with tf.gfile.Open(output_path, "w") as f:
      pil_img.save(f, "jpeg")

  return img


def mock_raw_data(tmp_dir, raw_dim=1024, num_channels=3, num_images=1):
  """Mock a raw data download directory with meta and raw subdirs.

  Notes:

    * This utility is shared by tests in both allen_brain_utils and
      allen_brain so kept here instead of in one of *_test.

  Args:
    tmp_dir: str, temporary dir in which to mock data.
    raw_dim: int, the x and y dimension of generated raw imgs.

  """

  tf.gfile.MakeDirs(tmp_dir)

  for image_id in range(0, num_images):

    raw_image_path = os.path.join(tmp_dir, "%s.jpg" % image_id)

    mock_raw_image(x_dim=raw_dim, y_dim=raw_dim,
                   num_channels=num_channels,
                   output_path=raw_image_path)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)
    

class TestAdversarialTransformer(tf.test.TestCase):
 
  def test_runs(self):

    problem_object = allen_brain.Img2imgAllenBrainDim8to32()

    with TemporaryDirectory() as tmp_dir:

      mock_raw_data(tmp_dir, raw_dim=256, num_images=730)

      with TemporaryDirectory() as data_dir:

        problem_object.generate_data(data_dir, tmp_dir)

        input_xy_dim = problem_object.input_dim
        target_xy_dim = problem_object.output_dim
        num_channels = problem_object.num_channels

        hparams = models.img2img_transformer_2d_adversarial()
        hparams.data_dir = data_dir
        hparams.discriminator_tanh_output = True
        hparams.label_smoothing_factor = 0.1
        hparams.discriminator_num_filters = 64
        hparams.hidden_size = 64
        hparams.loss_variant = "hybrid_vanilla_gan"
        hparams.filter_size = 64
        hparams.batch_size = 1

        p_hparams = problem_object.get_hparams(hparams)

        model = models.Img2imgTransformerAdversarial(
            hparams, tf.estimator.ModeKeys.TRAIN, p_hparams
        )

        @tfe.implicit_value_and_gradients
        def loss_fn(features):
          _, losses = model(features)

          if hparams.loss_variant == "non_gan":
            return losses["training"]
          elif hparams.loss_variant == "vanilla_gan":
            return losses["discriminator_loss"] + losses["generator_loss"]
          elif hparams.loss_variant == "hybrid_vanilla_gan":
            return losses["training"] + losses["discriminator_loss"] + losses["generator_loss"]

        batch_size = 1
        train_dataset = problem_object.dataset(Modes.TRAIN, data_dir)
        train_dataset = train_dataset.repeat(None).batch(batch_size)

        optimizer = tf.train.AdamOptimizer()

        example = tfe.Iterator(train_dataset).next()
        example["targets"] = tf.reshape(example["targets"],
                                        [batch_size,
                                         target_xy_dim,
                                         target_xy_dim,
                                         num_channels])
        a, gv = loss_fn(example)

        if hparams.loss_variant == "hybrid_vanilla_gan":
            # Check that the loss is near 7.66
            self.assertTrue(abs(a.numpy() - 7.66) < 1)
        elif hparams.loss_variant == "non_gan":
            # Check that the loss is near 5.56
            self.assertTrue(abs(a.numpy() - 5.56) < 1)

        optimizer.apply_gradients(gv)
    
        
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
