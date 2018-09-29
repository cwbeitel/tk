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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import math
import os
import json

import numpy as np
import requests

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

_BASE_EXAMPLE_IMAGE_SIZE = 64


# A 100 image random subset of non-failed acquisitions of Mouse imaging
# products from Allen Brain Institute (api.brain-map.org) dataset. The
# full set (or a desired subset) of image IDs can be obtained following
# the steps described here: http://help.brain-map.org/display/api,
# e.g. https://gist.github.com/cwbeitel/5dffe90eb561637e35cdf6aa4ee3e704
_IMAGE_IDS = [
    "100883774", "100883774", "100883784", "100883784", "100883779", "100883779",
    "100883780", "100883780", "100883783", "100883783", "100883786", "100883786",
    "100883826", "100883826", "100883819", "100883819", "100883816", "100883816",
    "100883825", "100883825", "100883798", "100883798", "100883785", "100883785",
    "100883815", "100883815", "100883789", "100883789", "100883822", "100883822",
    "100883811", "100883811", "100883799", "100883799", "100883790", "100883790",
    "100883787", "100883787", "100883804", "100883804", "100883813", "100883813",
    "100883818", "100883818", "100883844", "100883844", "100883849", "100883849",
    "100883835", "100883835", "100883863", "100883863", "100883838", "100883838",
    "100883864", "100883864", "100883862", "100883862", "100883852", "100883852",
    "100883866", "100883866", "100883839", "100883839", "100883841", "100883841",
    "100883861", "100883861", "100883848", "100883848", "100883855", "100883855",
    "100883858", "100883858", "100883869", "100883869", "100883882", "100883882",
    "100883898", "100883898", "100883905", "100883905", "100883880", "100883880",
    "100883902", "100883902", "100883910", "100883910", "100883893", "100883893",
    "100883912", "100883912", "100883909", "100883909", "100883895", "100883895",
    "100883886", "100883886", "100883879", "100883879", "100883873", "100883873",
    "100883906", "100883906", "100883899", "100883899", "100883889", "100883889",
    "100883887", "100883887", "100883876", "100883876", "100883883", "100883883",
    "100883896", "100883896", "100883901", "100883901", "100883884", "100883884",
    "100883907", "100883907", "100883888", "100883888", "100883922", "100883922",
    "100883929", "100883929", "100883934", "100883934", "100883957", "100883957",
    "100883932", "100883932", "100883955", "100883955", "100883949", "100883949",
    "100883931", "100883931", "100883937", "100883937", "100883925", "100883925",
    "100883952", "100883952", "100883916", "100883916", "100883919", "100883919",
    "100883933", "100883933", "100883988", "100883988", "100883965", "100883965",
    "100883997", "100883997", "100883972", "100883972", "100883977", "100883977",
    "100883999", "100883999", "100883962", "100883962", "100884001", "100884001",
    "100883976", "100883976", "100883993", "100883993", "100883979", "100883979",
    "100883995", "100883995", "100883991", "100883991", "100883978", "100883978",
    "100883996", "100883996", "100883982", "100883982", "100883971", "100883971",
    "100883983", "100883983", "100883984", "100883984", "100883964", "100883964",
    "100883969", "100883969", "100884045", "100884045", "100884048", "100884048",
    "100884018", "100884018", "100884015", "100884015", "100884046", "100884046",
    "100884016", "100884016", "100884038", "100884038", "100884026", "100884026",
    "100884020", "100884020", "100884014", "100884014", "100884012", "100884012",
    "100884025", "100884025", "100884040", "100884040", "100884033", "100884033",
    "100884034", "100884034", "100884004", "100884004", "100884036", "100884036",
    "100884007", "100884007", "100884069", "100884069", "100884088", "100884088",
    "100884080", "100884080", "100884067", "100884067", "100884078", "100884078",
    "100884064", "100884064", "100884071", "100884071", "100884092", "100884092",
    "100884056", "100884056", "100884087", "100884087", "100884089", "100884089",
    "100884052", "100884052", "100884135", "100884135", "100884124", "100884124",
    "100884094", "100884094", "100884114", "100884114", "100884097", "100884097",
    "100884108", "100884108", "100884116", "100884116", "100884110", "100884110",
    "100884119", "100884119", "100884095", "100884095", "100884128", "100884128",
    "100884127", "100884127", "100884109", "100884109", "100884096", "100884096",
    "100884093", "100884093", "100884104", "100884104", "100884123", "100884123",
    "100884107", "100884107", "100884106", "100884106", "100887188", "100887188",
    "100887186", "100887186", "100883814", "100883814", "100883961", "100883961",
    "100883908", "100883908", "100884133", "100884133", "100884013", "100884013",
    "100884022", "100884022", "100884072", "100884072", "100883860", "100883860",
    "100883782", "100883782", "100884000", "100884000", "100887185", "100887185",
    "100883953", "100883953", "100883948", "100883948", "100883807", "100883807",
    "100884027", "100884027", "100883830", "100883830", "100884057", "100884057",
    "100883951", "100883951", "100883992", "100883992", "100883808", "100883808",
    "100884077", "100884077", "100884101", "100884101", "100883877", "100883877",
    "100883914", "100883914", "100883928", "100883928", "100883778", "100883778",
    "100883833", "100883833", "100883990", "100883990", "100884066", "100884066",
    "100884081", "100884081", "100883775", "100883775", "100883942", "100883942",
    "100884039", "100884039", "100883840", "100883840", "100883940", "100883940",
    "100883836", "100883836", "100883856", "100883856", "100883913", "100883913",
    "100883823", "100883823", "100884120", "100884120", "100884019", "100884019",
    "100883975", "100883975", "100884063", "100884063", "100884105", "100884105",
    "100883939", "100883939", "100883881", "100883881", "100883950", "100883950",
    "100883987", "100883987", "100883797", "100883797", "100884073", "100884073",
    "100883958", "100883958", "100883834", "100883834", "100883843", "100883843",
    "100884035", "100884035", "100883960", "100883960", "100883859", "100883859",
    "100883930", "100883930", "100883994", "100883994", "100884003", "100884003",
    "100883897", "100883897", "100883927", "100883927", "100884037", "100884037",
    "100884131", "100884131", "100884024", "100884024", "100884011", "100884011",
    "100883788", "100883788", "100883998", "100883998", "100883772", "100883772",
    "100884050", "100884050", "100883936", "100883936", "100884028", "100884028",
    "100884060", "100884060", "100883769", "100883769", "100883792", "100883792",
    "100883802", "100883802", "100883832", "100883832", "100884008", "100884008",
    "100883891", "100883891", "100884115", "100884115", "100884111", "100884111",
    "100884103", "100884103", "100884084", "100884084", "100883796", "100883796",
    "100883900", "100883900", "100883973", "100883973", "100884085", "100884085",
    "100883829", "100883829", "100883831", "100883831", "100884005", "100884005",
    "100883980", "100883980", "100884054", "100884054", "100884021", "100884021",
    "100883935", "100883935", "100884102", "100884102", "100884099", "100884099",
    "100884062", "100884062", "100884055", "100884055", "100887187", "100887187",
    "100884112", "100884112", "100883920", "100883920", "100884122", "100884122",
    "100883943", "100883943", "100884083", "100884083", "100883911", "100883911",
    "100884044", "100884044", "100883947", "100883947", "100883810", "100883810",
    "100883806", "100883806", "100884043", "100884043", "100883894", "100883894",
    "100883963", "100883963", "100884059", "100884059", "100883773", "100883773",
    "100883885", "100883885", "100883956", "100883956", "100883974", "100883974",
    "100883924", "100883924", "100884076", "100884076", "100883857", "100883857",
    "100884006", "100884006", "100884075", "100884075", "100883915", "100883915",
    "100884051", "100884051", "100883868", "100883868", "100884017", "100884017",
    "100883851", "100883851", "100883892", "100883892", "100883781", "100883781",
    "100883828", "100883828", "100883821", "100883821", "100884130", "100884130",
    "100883959", "100883959", "100883875", "100883875", "100883824", "100883824",
    "100883795", "100883795", "100883967", "100883967", "100883793", "100883793",
    "100884121", "100884121", "100884125", "100884125", "100884082", "100884082",
    "100884129", "100884129", "100884010", "100884010", "100884068", "100884068",
    "100883970", "100883970", "100883809", "100883809", "100884098", "100884098",
    "100883903", "100883903", "100884091", "100884091", "100883812", "100883812",
    "100884058", "100884058", "100883917", "100883917", "100883854", "100883854",
    "100884053", "100884053", "100883870", "100883870", "100884002", "100884002",
    "100883776", "100883776", "100883944", "100883944", "100883837", "100883837",
    "100883794", "100883794", "100883945", "100883945", "100884009", "100884009",
    "100883817", "100883817", "100883803", "100883803", "100884042", "100884042",
    "100883777", "100883777", "100884079", "100884079", "100883820", "100883820",
    "100884117", "100884117", "100884113", "100884113", "100883872", "100883872",
    "100883941", "100883941", "100883850", "100883850", "100883923", "100883923",
    "100883874", "100883874", "100884126", "100884126", "100884047", "100884047",
    "100884090", "100884090", "100883865", "100883865", "100883827", "100883827",
    "100883985", "100883985", "100884065", "100884065", "100884132", "100884132",
    "100884041", "100884041", "100883800", "100883800", "100883966", "100883966",
    "100883878", "100883878", "100884100", "100884100", "100884086", "100884086",
    "100883946", "100883946", "100883904", "100883904", "100883890", "100883890",
    "100883771", "100883771", "100883801", "100883801", "100883847", "100883847",
    "100883981", "100883981", "100884049", "100884049", "100884070", "100884070",
    "100883842", "100883842", "100883871", "100883871", "100884074", "100884074",
    "100883845", "100883845", "100884118", "100884118", "100883918", "100883918",
    "100884134", "100884134", "100883968", "100883968", "100884023", "100884023",
    "100883986", "100883986", "100884136", "100884136", "100883938", "100883938",
    "100883770", "100883770", "100883791", "100883791", "100883805", "100883805",
    "100883846", "100883846", "100883867", "100883867", "100883921", "100883921",
    "100883926", "100883926", "100883954", "100883954", "999"
]

# HACK: There are duplicates in the above.
_IMAGE_IDS = list(set(_IMAGE_IDS))

def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


def _get_case_file_paths(tmp_dir, case, training_fraction=0.95):
  """Obtain a list of image paths corresponding to training or eval case.

  Args:
    tmp_dir: str, the root path to which raw images were written, at the
      top level having meta/ and raw/ subdirs.
    case: bool, whether obtaining file paths for training (true) or eval
      (false).
    training_fraction: float, the fraction of the sub-image path list to
      consider as the basis for training examples.

  Returns:
    list: A list of file paths.

  Raises:
    ValueError: if images not found in tmp_dir, or if training_fraction would
      leave no examples for eval.
  """

  paths = tf.gfile.Glob("%s/*.jpg" % tmp_dir)

  if not paths:
    raise ValueError("Search of tmp_dir (%s) " % tmp_dir,
                     "for subimage paths yielded an empty list, ",
                     "can't proceed with returning training/eval split.")

  split_index = int(math.floor(len(paths)*training_fraction))

  if split_index >= len(paths):
    raise ValueError("For a path list of size %s "
                     "and a training_fraction of %s "
                     "the resulting split_index of the paths list, "
                     "%s, would leave no elements for the eval "
                     "condition." % (len(paths),
                                     training_fraction,
                                     split_index))

  if case:
    return paths[:split_index]
  else:
    return paths[split_index:]


def maybe_download_image_dataset(image_ids, target_dir):
  """Download a set of images from api.brain-map.org to `target_dir`.

  Args:
    image_ids: list, a list of image ids.
    target_dir: str, a directory to which to download the images.
  """

  tf.gfile.MakeDirs(target_dir)

  num_images = len(image_ids)

  for i, image_id in enumerate(image_ids):

    destination = os.path.join(target_dir, "%s.jpg" % i)
    tmp_destination = "%s.temp" % destination

    source_url = ("http://api.brain-map.org/api/v2/"
                  "section_image_download/%s" % image_id)

    if tf.gfile.Exists(destination):
      tf.logging.info("Image with ID already present, "
                      "skipping download (%s of %s)." % (
                          i+1, num_images
                      ))
      continue

    tf.logging.info("Downloading image with id %s (%s of %s)" % (
        image_id, i+1, num_images
    ))

    response = requests.get(source_url, stream=True)

    response.raise_for_status()

    with tf.gfile.Open(tmp_destination, "w") as f:
      for block in response.iter_content(1024):
        f.write(block)

    tf.gfile.Rename(tmp_destination, destination)


def maybe_compute_image_statistics(image_files):
    """

    Notes:
        As a quick hack we'll assume the mean of image means and mean of standard
        deviations both reflect the mean and standard deviation of the set as a
        whole. We can't get the exact sample standard deviation this way because
        each image has a different mean and the SD's are computed relative to that.
        The mean of the dataset as a whole can be computed in this way since each
        image should be the same size.
    
    """

    print("Computing image statistics...")
    
    image_obj = PIL_Image()

    means = []
    stdevs = []

    for i, image_file_name in enumerate(image_files):

        print("Processing image %s of %s..." % (i + 1, len(image_files)))

        fn_split = image_file_name.split(".")
        assert len(fn_split) == 2
        
        meta_fname = fn_split[0] + ".meta"
        
        if tf.gfile.Exists(meta_fname):

            print("Found an existing metadata file...")
            
            with open(meta_fname, "r") as f:
                meta = json.loads(f.read())
            mn = float(meta["mean"])
            stdev = float(meta["stdev"])
                
        else:
            
            print("Did not find an existing metadata file, computing...")
            
            meta = {}

            img = image_obj.open(image_file_name)
            img = np.float32(img)

            mn = np.mean(img)
            stdev = np.std(img)
            
            with open(meta_fname, "w") as f:
                print("Writing metadata to file: %s" % meta_fname)
                f.write(json.dumps({"stdev": str(stdev), "mean": str(mn)}))

        means.append(mn)
        stdevs.append(stdev)
        
        print("Saw mean and stdev: %s, %s" % (mn, stdev))

    return np.mean(means), np.mean(stdevs)


def random_square_mask(shape, fraction):
  """Create a numpy array with specified shape and masked fraction.

  Args:
    shape: tuple, shape of the mask to create.
    fraction: float, fraction of the mask area to populate with `mask_scalar`.

  Returns:
    numpy.array: A numpy array storing the mask.
  """

  mask = np.ones(shape)

  patch_area = shape[0]*shape[1]*fraction
  patch_dim = np.int(math.floor(math.sqrt(patch_area)))
  if patch_area == 0 or patch_dim == 0:
    return mask

  x = np.random.randint(shape[0] - patch_dim)
  y = np.random.randint(shape[1] - patch_dim)

  mask[x:(x + patch_dim), y:(y + patch_dim), :] = 0

  return mask


def _generator(tmp_dir, training, size=_BASE_EXAMPLE_IMAGE_SIZE,
               training_fraction=0.95):
  """Base problem example generator for Allen Brain Atlas problems.

  Args:

    tmp_dir: str, a directory where raw example input data has been stored.
    training: bool, whether the mode of operation is training (or,
      alternatively, evaluation), determining whether examples in tmp_dir
      prefixed with train or dev will be used.
    size: int, the image size to add to the example annotation.
    training_fraction: float, the fraction of the sub-image path list to
      consider as the basis for training examples.

  Yields:
    A dictionary representing the images with the following fields:
      * image/encoded: The string encoding the image as JPEG.
      * image/format: The string "jpeg" indicating the image format.
      * image/height: The integer indicating the image height.
      * image/width: The integer indicating the image height.

  """

  maybe_download_image_dataset(_IMAGE_IDS, tmp_dir)

  image_files = _get_case_file_paths(tmp_dir=tmp_dir,
                                     case=training,
                                     training_fraction=training_fraction)

  image_obj = PIL_Image()
    
  mn, sd = maybe_compute_image_statistics(image_files)

  tf.logging.info("Loaded case file paths (n=%s)" % len(image_files))
  height = size
  width = size

  for input_path in image_files:

    img = image_obj.open(input_path)
    img = np.float32(img)
    shape = np.shape(img)

    for h_index in range(0, int(math.floor(shape[0]/size))):

      h_offset = h_index * size
      h_end = h_offset + size - 1

      for v_index in range(0, int(math.floor(shape[1]/size))):

        v_offset = v_index * size
        v_end = v_offset + size - 1

        # Extract a sub-image tile.
        subimage = np.uint8(img[h_offset:h_end, v_offset:v_end])  # pylint: disable=invalid-sequence-index

        # Filter images that are likely background (not tissue).
        if np.amax(subimage) < 230:
          continue

        subimage = image_obj.fromarray(subimage)
        buff = BytesIO()
        subimage.save(buff, format="JPEG")
        subimage_encoded = buff.getvalue()

        yield {
            "image/encoded": [subimage_encoded],
            "image/format": ["jpeg"],
            "image/height": [height],
            "image/width": [width]
        }


@registry.register_problem
class Img2imgAllenBrain2(problem.Problem):
  """Allen Brain Atlas histology dataset.

  See also: http://help.brain-map.org/

  Notes:

    * 64px to 64px identity mapping problem, no in-painting.

  """

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  @property
  def training_fraction(self):
    return 0.95

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  @property
  def input_dim(self):
    """The x and y dimension of the input image."""
    # By default, there is no input image, only a target.
    return 64

  @property
  def output_dim(self):
    """The x and y dimension of the target image."""
    return 64

  @property
  def inpaint_fraction(self):
    """The fraction of the input image to be in-painted."""
    # By default, no in-painting is performed.
    return None

  @property
  def should_standardize(self):
    return False

  def preprocess_example(self, example, mode, hparams):

    # Crop to target shape instead of down-sampling target, leaving target
    # of maximum available resolution.
    target_shape = (self.output_dim, self.output_dim, self.num_channels)
    target = tf.random_crop(example["targets"], target_shape)
    
    example["targets"] = target
    
    if self.should_standardize:
      example["targets"] = tf.image.per_image_standardization(target)

    inputs = image_utils.resize_by_area(target, self.input_dim)
    
    example["inputs"] = inputs
    
    if self.should_standardize:
      example["inputs"] = tf.image.per_image_standardization(inputs)

    if self.inpaint_fraction is not None and self.inpaint_fraction > 0:

      mask = random_square_mask((self.input_dim,
                                 self.input_dim,
                                 self.num_channels),
                                self.inpaint_fraction)

      example["inputs"] = tf.multiply(
          tf.convert_to_tensor(mask, dtype=tf.int64),
          example["inputs"])

      if self.input_dim is None:
        raise ValueError("Cannot train in-painting for examples with "
                         "only targets (i.e. input_dim is None, "
                         "implying there are only targets to be "
                         "generated).")

    return example

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(channels=self.num_channels),
        "targets": text_encoder.ImageEncoder(channels=self.num_channels)
    }

  def example_reading_spec(self):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "targets":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_channels),
    }

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC,
        metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY
    ]
    return eval_metrics

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    
    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=True),
        self.generator(tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256)}
    p.target_modality = ("image:identity", 256)
    p.batch_size_multiplier = 256
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generator(self, tmp_dir, is_training):
    if is_training:
      return _generator(tmp_dir, True, size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)
    else:
      return _generator(tmp_dir, False, size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)


@registry.register_problem
class Img2imgAllenBrain2Dim48to64(Img2imgAllenBrain2):
  """48px to 64px resolution up-sampling problem."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 48

  @property
  def output_dim(self):
    return 64


@registry.register_problem
class Img2imgAllenBrain2Dim8to32(Img2imgAllenBrain2):
  """8px to 32px resolution up-sampling problem."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 8

  @property
  def output_dim(self):
    return 32


@registry.register_problem
class Img2imgAllenBrain2Dim16to16Paint1(Img2imgAllenBrain2):
  """In-painting problem (1%) with no resolution upsampling."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 16

  @property
  def output_dim(self):
    return 16

  @property
  def inpaint_fraction(self):
    return 0.01


@registry.register_problem
class Img2imgAllenBrain2Dim8to32Stand(Img2imgAllenBrain2):
  """In-painting problem (1%) with no resolution upsampling."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 8

  @property
  def output_dim(self):
    return 32

  @property
  def should_standardize(self):
    return True
