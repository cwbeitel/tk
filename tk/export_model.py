
import os

import tensorflow as tf
import tk

from tensor2tensor.serving import export
from tensor2tensor.utils import usr_dir

def _get_t2t_usr_dir():
  """Get the path to the t2t usr dir."""
  return "/home/jovyan/nfs-work/tk/tk"
  return os.path.join(os.path.realpath(__file__), "../")

FLAGS = tf.flags.FLAGS

FLAGS.output_dir = "/mnt/nfs-east1-d/ckpts/cs-dist-tr-morex-j1017-1758-a24a/output"
FLAGS.problem = "github_constrained_embedding"
FLAGS.model = "constrained_embedding_transformer"
FLAGS.hparams_set = "similarity_transformer_tiny"

FLAGS.t2t_usr_dir = _get_t2t_usr_dir()
FLAGS.data_dir = "/mnt/nfs-east1-d/data"
FLAGS.tmp_dir = "/mnt/nfs-east1-d/tmp"
#FLAGS.export_dir = os.path.join(FLAGS.output_dir, "export")

#usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

export.main(None)

