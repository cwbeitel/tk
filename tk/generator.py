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

"""Data generators for histology image analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import math

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

import tensorflow as tf

_MEDIUM_IMAGE_SIZE = 1024

_MEDIUM_IMAGE_PREFIX = "128x128"

# For development, in practice use "all"
_MAX_RAW_EXAMPLES = 10


def get_subimage_paths(data_root, size):
       
    meta_root = os.path.join(data_root, "meta")
    manifest_prefix = "%sx%s_path_manifest" % (size, size)
    paths = []
    
    for filename in os.listdir(meta_root):
        if filename.startswith(manifest_prefix):
            path = os.path.join(meta_root, filename)
            with open(path, "r") as f:
                for line in f:
                    paths.append(line.strip())

    return paths


def _get_case_file_paths(raw_data_root, case, size, training_fraction=0.95):
    
    paths = get_subimage_paths(raw_data_root, size)
    
    split_index = math.floor(len(paths)*training_fraction)
    
    if case == 1:
        return paths[:split_index]
    else:
        return paths[split_index:]

    
def _generator(tmp_dir, training, size=_MEDIUM_IMAGE_SIZE,
                                image_prefix=_MEDIUM_IMAGE_PREFIX,
                                max_raw_examples=_MAX_RAW_EXAMPLES):
  image_files = _get_case_file_paths(tmp_dir, training, size=size)
  tf.logging.info("Loaded case file paths (n=%s)" % len(image_files))
  height = size
  width = size
  const_label = 0
  for filename in image_files:
    with open(filename, "rb") as f:
      encoded_image = f.read()
    
      yield {
          "image/encoded": [encoded_image],
          "image/format": ["jpeg"],
          "image/height": [height],
          "image/width": [width]
      }


# V2
@registry.register_problem
class AllenBrainImage2image(problem.Problem):

  @property
  def train_shards(self):
    return 20

  @property
  def dev_shards(self):
    return 10

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  @property
  def input_dim(self):
    """The x and y dim of input image."""
    return 32

  @property
  def output_dim(self):
    """The x and y dim of input image."""
    return 32

  def example_reading_spec(self, label_repr=None):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_channels),
    }

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):

    #example["inputs"] = tf.image.per_image_standardization(example["inputs"])
    
    inputs = example["inputs"]
    # For Img2Img resize input and output images as desired.
    example["inputs"] = image_utils.resize_by_area(inputs, self.input_dim)
    example["targets"] = image_utils.resize_by_area(inputs, self.output_dim)

    return example

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]
    if self._was_reversed:
      eval_metrics += [metrics.Metrics.IMAGE_SUMMARY]
    return eval_metrics

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(channels=self.num_channels),
        "targets": text_encoder.ImageEncoder(channels=self.num_channels)
    }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=True),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256)}
    p.target_modality = ("image:identity", 256)
    p.batch_size_multiplier = 256
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return _generator(
          tmp_dir, int(True), size=_MEDIUM_IMAGE_SIZE)
    else:
      return _generator(
          tmp_dir, int(False), size=_MEDIUM_IMAGE_SIZE)


@registry.register_problem
class AllenBrainImage2imageIdentityTiny(AllenBrainImage2image):
    
  @property
  def input_dim(self):
    """The x and y dim of input image."""
    return 8

  @property
  def output_dim(self):
    """The x and y dim of input image."""
    return 8


@registry.register_problem
class AllenBrainImage2imageUpscale(AllenBrainImage2image):
    
  @property
  def input_dim(self):
    """The x and y dim of input image."""
    return 32

  @property
  def output_dim(self):
    """The x and y dim of input image."""
    return 64
