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
               master_gpu=0,
               worker_gpu=0,
               ps_gpu=0,
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

    master_command = ["sh", os.path.join(app_root, "master-job.sh")]
    ps_command = ["sh", os.path.join(app_root, "ps-job.sh")]
    worker_command = ["sh", os.path.join(app_root, "worker-job.sh")]

    # TODO: For now, just run all components with the same resources.
    master_resources = Resources(limits={
      "cpu": cpu,
      "memory": memory,
      "nvidia.com/gpu": master_gpu
    })

    worker_resources = Resources(limits={
      "cpu": cpu,
      "memory": memory,
      "nvidia.com/gpu": worker_gpu
    })

    ps_resources = Resources(limits={
      "cpu": cpu,
      "memory": memory,
      "nvidia.com/gpu": ps_gpu
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
                   args=master_command,
                   image=image,
                   resources=master_resources,
                   attached_volumes=volumes,
                   node_selector=selector_labels)
    ]

    if num_ps_replicas > 0:
      replicas.append(
        TFJobReplica(replica_type="PS",
                     num_replicas=num_ps_replicas,
                     args=ps_command,
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
                     args=worker_command,
                     image=image,
                     resources=worker_resources,
                     attached_volumes=volumes,
                     node_selector=selector_labels)
      )

    super(T2TExperiment, self).__init__(command="",
                                        replicas=replicas,
                                        *args, **kwargs)


def tf_config_to_additional_flags():
  """Read TF_CONFIG and set relevant t2t FLAGS."""

  if "TF_CONFIG" not in os.environ:
    tf.logging.info("No TF_CONFIG present, returning dummy.")
    task_type = "master"
    tid = 0
    #FLAGS.master = None
    #FLAGS.ps_replicas = 0
    #FLAGS.worker_id = tid
    #FLAGS.worker_job = '/job:%s' % task_type
    #FLAGS.worker_gpu = 0
    #FLAGS.worker_replicas = 1
    #FLAGS.schedule = 'train'
    return task_type, 0

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

  FLAGS.master = master_address
  FLAGS.ps_replicas = num_ps

  if task_type == "ps":
    FLAGS.schedule = "run_std_server"
    return task_type, tid

  FLAGS.worker_id = tid
  FLAGS.worker_job = '/job:%s' % task_type
  FLAGS.worker_gpu = 0
  FLAGS.worker_replicas = 1

  FLAGS.sync = True
  FLAGS.schedule = 'train'

  return task_type, tid


def main(argv):
  """Configure, setup logging, and train."""

  task_type, task_id = tf_config_to_additional_flags()

  tf.gfile.MakeDirs(FLAGS.output_dir)

  worker_name = "%s-%s" % (task_type, task_id)
    
  trainer_main(None)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()