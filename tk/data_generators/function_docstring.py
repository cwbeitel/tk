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
from six import StringIO
import tempfile

import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.layers import common_layers
import numpy as np



class GithubFunctionDocstring(text_problems.Text2TextProblem):
  """Function and Docstring similarity Problem.
  This problem contains the data consisting of function
  and docstring pairs as CSV files. The files are structured
  such that they contain two columns without headers containing
  the docstring tokens and function tokens. The delimiter is
  ",".
  """

  DATA_PATH_PREFIX = "gs://kubeflow-examples/t2t-code-search/raw_data"

  @property
  def pair_files_list(self):
    """Return URL and file names.
    This format is a convention across the Tensor2Tensor (T2T)
    codebase. It should be noted that the file names are currently
    hardcoded. This is to preserve the semantics of a T2T problem.
    In case a change of these values is desired, one must subclass
    and override this property.
    # TODO(sanyamkapoor): Manually separate train/eval data set.
    Returns:
      A list of the format,
        [
          [
            "STRING",
            ("STRING", "STRING", ...)
          ],
          ...
        ]
      Each element is a list of size 2 where the first represents
      the source URL and the next is an n-tuple of file names.
      In this case, the tuple is of size 1 because the URL points
      to a file itself.
    """
    return [
        [
            "{}/func-doc-pairs-000{:02}-of-00100.csv".format(
                self.DATA_PATH_PREFIX, i),
            ("func-doc-pairs-000{:02}-of-00100.csv".format(i),)
        ]
        for i in range(100)
    ]

  @property
  def is_generate_per_split(self):
    return False

  @property
  def approx_vocab_size(self):
    return 2**13

  @property
  def max_samples_for_vocab(self):
    # FIXME(sanyamkapoor): This exists to handle memory explosion.
    return int(2e5)

  def get_csv_files(self, _data_dir, tmp_dir, _dataset_split):
    return [
        generator_utils.maybe_download(tmp_dir, file_list[0], uri)
        for uri, file_list in self.pair_files_list
    ]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """A generator to return data samples.Returns the data generator to return.
    Args:
      data_dir: A string representing the data directory.
      tmp_dir: A string representing the temporary directory and is
              used to download files if not already available.
      dataset_split: Train, Test or Eval.
    Yields:
      Each element yielded is of a Python dict of the form
        {"inputs": "STRING", "targets": "STRING", "embed_code": [0]}
    """
    csv_files = self.get_csv_files(data_dir, tmp_dir, dataset_split)

    for pairs_file in csv_files:
      tf.logging.debug("Reading {}".format(pairs_file))
      with tf.gfile.Open(pairs_file) as csv_file:
        for line in csv_file:
          reader = csv.reader(StringIO(line))
          for docstring_tokens, function_tokens in reader:
            yield {
                "inputs": docstring_tokens,
                "targets": function_tokens,
                "embed_code": [0]
            }

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = super(GithubFunctionDocstring,
                                                self).example_reading_spec()
    data_fields["embed_code"] = tf.FixedLenFeature([1], dtype=tf.int64)

    data_items_to_decoders = {
      "inputs": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="inputs"),
      "targets": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="targets"),
      "embed_code": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="embed_code")
    }
    return data_fields, data_items_to_decoders

  def eval_metrics(self):  # pylint: disable=no-self-use
    return [
        metrics.Metrics.ACC
    ]


def _random_mask_sequence(sequence):
  """'quick brown fox' -> 'quick ##### fox'"""
  arr = sequence.split()
  ind = np.random.randint(len(arr))
  arr[ind] = ''.join(["#" for _ in range(0, len(arr[ind]))])
  return ' '.join(arr)


@registry.register_problem
class GithubStringInpaint(GithubFunctionDocstring):
  """Predict full string from truncated string.
  
  TODO: Do this with a preprocess_example method instead of generating
        different examples.
  """

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    csv_files = self.get_csv_files(data_dir, tmp_dir, dataset_split)

    for pairs_file in csv_files:
      tf.logging.debug("Reading {}".format(pairs_file))
      with tf.gfile.Open(pairs_file) as csv_file:
        for line in csv_file:
          reader = csv.reader(StringIO(line))
          for docstring_tokens, _ in reader:

            example = {
                "inputs": _random_mask_sequence(docstring_tokens),
                "targets": docstring_tokens,
                "embed_code": [0]
            }

            yield example


@registry.register_problem
class GithubCodeInpaint(GithubFunctionDocstring):
  """Code in-painting given randomly ablated regions."""

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    csv_files = self.get_csv_files(data_dir, tmp_dir, dataset_split)

    for pairs_file in csv_files:
      tf.logging.debug("Reading {}".format(pairs_file))
      with tf.gfile.Open(pairs_file) as csv_file:
        for line in csv_file:
          reader = csv.reader(StringIO(line))
          for _, function_tokens in reader:

            example = {
                "inputs": _random_mask_sequence(function_tokens),
                "targets": function_tokens,
                "embed_code": [0]
            }

            yield example


@registry.register_problem
class GithubMultiProblemBase(GithubFunctionDocstring):

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    csv_files = self.get_csv_files(data_dir, tmp_dir, dataset_split)

    for pairs_file in csv_files:
      tf.logging.debug("Reading {}".format(pairs_file))
      with tf.gfile.Open(pairs_file) as csv_file:
        for line in csv_file:
          reader = csv.reader(StringIO(line))
          for docstring_tokens, function_tokens in reader:

            yield {
                "inputs": _random_mask_sequence(function_tokens),
                "targets": function_tokens,
                "embed_code": [0]
            }

            yield {
                "inputs": _random_mask_sequence(docstring_tokens),
                "targets": docstring_tokens,
                "embed_code": [0]
            }


def random_mask(tensor):
  mask = tf.random_uniform(common_layers.shape_list(tensor),
                           0, 10, dtype=tf.float32)
  mask = tf.cast(tf.greater(mask, 1), tf.int64)
  return tf.multiply(tensor, mask)


# yes hackily repetitive

@registry.register_problem
class GithubConstrainedEmbedding(GithubFunctionDocstring):

  @property
  def approx_vocab_size(self):
    return 2**13

  def preprocess_example(self, example, mode, hparams):

    example["docstring"] = example["inputs"]
    example["code"] = example["targets"]
    if np.random.randint(2) == 0:
      # docstring un-masking
      example["targets"] = example["inputs"]
      example["inputs"] = random_mask(example["inputs"])
    else:
      # code un-masking
      example["inputs"] = random_mask(example["targets"])

    return example


@registry.register_problem
class GithubConstrainedEmbedding16k(GithubFunctionDocstring):

  @property
  def approx_vocab_size(self):
    return 2**14

  def preprocess_example(self, example, mode, hparams):

    example["docstring"] = example["inputs"]
    example["code"] = example["targets"]
    if np.random.randint(2) == 0:
      # docstring un-masking
      example["targets"] = example["inputs"]
      example["inputs"] = random_mask(example["inputs"])
    else:
      # code un-masking
      example["inputs"] = random_mask(example["targets"])

    return example


@registry.register_problem
class GithubConstrainedEmbedding32k(GithubFunctionDocstring):

  @property
  def approx_vocab_size(self):
    return 2**15

  def preprocess_example(self, example, mode, hparams):

    example["docstring"] = example["inputs"]
    example["code"] = example["targets"]
    if np.random.randint(2) == 0:
      # docstring un-masking
      example["targets"] = example["inputs"]
      example["inputs"] = random_mask(example["inputs"])
    else:
      # code un-masking
      example["inputs"] = random_mask(example["targets"])

    return example


@registry.register_problem
class GithubConstrainedEmbedding64k(GithubFunctionDocstring):

  @property
  def approx_vocab_size(self):
    return 2**16

  def preprocess_example(self, example, mode, hparams):

    example["docstring"] = example["inputs"]
    example["code"] = example["targets"]
    if np.random.randint(2) == 0:
      # docstring un-masking
      example["targets"] = example["inputs"]
      example["inputs"] = random_mask(example["inputs"])
    else:
      # code un-masking
      example["inputs"] = random_mask(example["targets"])

    return example
