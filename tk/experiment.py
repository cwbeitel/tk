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

"""Extend t2t-trainer with startup tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging

import tensorflow as tf

from tensor2tensor.bin.t2t_trainer import FLAGS
from tensor2tensor.bin.t2t_trainer import main as trainer_main

from tk.kube import AttachedVolume
from tk.kube import LocalSSD
from tk.kube import Resources
from tk.kube import TFJob
from tk.kube import TFJobReplica
from tk.util import hack_dict_to_cli_args


class T2TExperiment(TFJob):
  """A Job that runs a training experiment."""

  def __init__(self,
               app_root,
               image,
               num_worker_replicas=0,
               num_ps_replicas=0,
               cpu=1,
               memory="1Gi",
               num_gpu=0,
               volume_claim_id=None,
               selector_labels={},
               num_local_ssd=0,
               *args, **kwargs):
    """Configure a T2TExperiment object.

    Vars:
        app_root (str):
        image (str): 
        num_worker_replicas (int): 
        num_ps_replicas (int):
        cpu (int): 
        memory (str):
        pvc_id (str): ID of a PVC to mount to the training volume.
        stage_data_dir_to_local_ssd (bool): HACK: Whether to mount local
            SSD and at the start of training stage contents 
        selector_labels (dict):

    """

    # HACK: Should check type at arg parser level not here, this change
    # is part of a larger change in way additional args are loaded i.e.
    # parsing FLAGS for called apps instead of providing a dict template.
    if isinstance(num_worker_replicas, str):
      num_worker_replicas = int(num_worker_replicas)

    if (not isinstance(num_worker_replicas, int) or num_worker_replicas < 0):
      raise ValueError("The number of worker replicas must be an "
                       "integer greater than or equal to zero.")

    if (not isinstance(num_ps_replicas, int) or 
      num_ps_replicas < 0):
      raise ValueError("The number of ps replicas must be an "
                       "integer greater than or equal to zero.")

    command = ["sh", os.path.join(app_root, "job.sh")]

    # TODO: For now, just run all components with the same resources.
    master_resources = Resources(limits={
      "cpu": cpu,
      "memory": memory,
      "nvidia.com/gpu": num_gpu
    })

    worker_resources = Resources(limits={
      "cpu": cpu,
      "memory": memory,
      "nvidia.com/gpu": num_gpu
    })

    ps_resources = Resources(limits={
      "cpu": "100m",
      "memory": "1Gi",
      "nvidia.com/gpu": num_gpu
    })

    volumes = []

    if volume_claim_id is not None:
      volumes.append(AttachedVolume(volume_claim_id))

    if num_local_ssd > 0:
      for i in range(num_local_ssd):
        volumes.append(LocalSSD(disk_id=i))

    if len(volumes) == 0:
      volumes = None

    replicas = [
      TFJobReplica(replica_type="MASTER",
                   num_replicas=1,
                   args=command,
                   image=image,
                   resources=master_resources,
                   attached_volumes=volumes,
                   node_selector=selector_labels)
    ]

    if num_ps_replicas > 0:
      replicas.append(
        TFJobReplica(replica_type="PS",
                     num_replicas=num_ps_replicas,
                     args=command,
                     image=image,
                     resources=ps_resources,
                     attached_volumes=volumes,
                     node_selector=selector_labels
                    )
      )

    if num_worker_replicas > 0:
      replicas.append(
        TFJobReplica(replica_type="WORKER",
                     num_replicas=num_worker_replicas,
                     args=command,
                     image=image,
                     resources=worker_resources,
                     attached_volumes=volumes,
                     node_selector=selector_labels)
      )

    super(T2TExperiment, self).__init__(command=command,
                                        replicas=replicas,
                                        *args, **kwargs)


def tf_config_to_additional_flags():
  """Read TF_CONFIG and set relevant t2t FLAGS."""

  tf_config = os.environ["TF_CONFIG"]

  tf_config = json.loads(tf_config)

  tf.logging.info("Loaded TF_CONFIG: %s" % tf_config)

  if "cluster" not in tf_config:
    raise ValueError("TF_CONFIG environment variable should always "
                     "have a 'cluster' field, saw %s" % tf_config)

  cluster_spec = tf_config["cluster"]

  if "master" not in cluster_spec or len(cluster_spec["master"]) == 0:
    raise ValueError("Expected at least one master defined in "
                     "master field of cluster_spec.")

  masters = cluster_spec["master"]
  num_masters = len(masters)
  tf.logging.info("num_masters: %s" % num_masters)

  ps_tasks = [] if "ps" not in cluster_spec else cluster_spec["ps"]
  num_ps = len(ps_tasks)
  tf.logging.info("num_ps: %s" % num_ps)

  worker_tasks = [] if "worker" not in cluster_spec else cluster_spec["worker"]
  num_workers = len(worker_tasks)
  tf.logging.info("worker_tasks: %s" % num_workers)

  master_address = "grpc://%s" % masters[0]
  tf.logging.info("master address: %s" % master_address)
  
  tid = tf_config["task"]["index"]
  task_type = tf_config["task"]["type"]

  #FLAGS.master = master_address
  #FLAGS.ps_replicas = num_ps
  #FLAGS.worker_replicas = num_workers + num_masters
  #FLAGS.worker_id = tid
  #FLAGS.worker_job = '/job:%s' % task_type
  FLAGS.ps_gpu = 0
  #FLAGS.schedule = 'train'
  #FLAGS.sync = True if (FLAGS.worker_replicas > 1) else False

  #if task_type == "ps":
  #  FLAGS.schedule = "run_std_server"

  return task_type, tid


def configure_logging(worker_name):
  """Configure logging to file named after worker."""

  logs_filename = "%s-logs.txt" % worker_name

  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)

  tf.gfile.MakeDirs(FLAGS.output_dir)
  fh = logging.FileHandler(os.path.join(FLAGS.output_dir,
                                        logs_filename))
  fh.setLevel(logging.DEBUG)
  log.addHandler(fh)


def start_bg_resource_logger(base_path):
  cmd = ('python -m tk.resource_logger --base_path %s &' % base_path)
  os.system(cmd)


def main(argv):
  """Configure, setup logging, and train."""

  #os.system("started > %s" % os.path.join(FLAGS.output_dir,
  #                                        "status.txt"))

  task_type, task_id = tf_config_to_additional_flags()

  worker_name = "%s-%s" % (task_type, task_id)
  FLAGS.output_dir = os.path.join(FLAGS.output_dir,
                                  worker_name)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  configure_logging(worker_name)

  start_bg_resource_logger(FLAGS.output_dir)

  trainer_main(None)

  #os.system("done > %s" % os.path.join(FLAGS.output_dir,
  #                                     "status.txt"))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()