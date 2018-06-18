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

"""Extend t2t-trainer from t2t with startup tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging

import tensorflow as tf

from tensor2tensor.bin.t2t_trainer import main as trainer_main
from tensor2tensor.bin.t2t_trainer import FLAGS
# Make the arguments for this script a superset of those of
# t2t-trainer.

from tk.kube import TFJob, TFJobReplica, LocalSSD, AttachedVolume, Resources
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
        master_resources = Resources(requests={
          "cpu": cpu,
          "memory": memory,
          "nvidia.com/gpu": num_gpu
        })
        
        worker_resources = Resources(requests={
           "cpu": cpu,
           "memory": memory,
           "nvidia.com/gpu": num_gpu
        })

        ps_resources = Resources(requests={
           "cpu": "100m",
           "memory": "1Gi",
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
                             node_selector=selector_labels)
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
    
    # Most of this below is duplicated from 
    # https://github.com/tensorflow/tensor2tensor/blob/master/
    # tensor2tensor/bin/make_tf_configs.py
    # but necessary because unlike there where cli args and TF_CONFIG are
    # being constructed from lists of ps and worker addresses here we already
    # have a TF_CONFIG and we want to construct cmd_line_flags from those.
    for task_type, jobs in (("master", masters), ("ps", ps_tasks)):
        for idx, job in enumerate(jobs):
            if task_type == "master":
                FLAGS.master = "grpc://%s" % job
                FLAGS.ps_replicas = num_ps
                FLAGS.worker_replicas = num_masters
                FLAGS.worker_gpu = 1
                FLAGS.worker_id = idx
                FLAGS.worker_job = '/job:master'
                FLAGS.ps_gpu = 1
                FLAGS.schedule = 'train'
                FLAGS.sync = True if num_masters == 1 else False

            else:
                FLAGS.master = "grpc://%s" % job
                FLAGS.schedule = "run_std_server"

        
def stage_data_dir_to_ssd_mount(data_dir, ssd_mount_path):

    import util
  
    tf.logging.info(util.run_and_output([
        "mkdir", "-p", ssd_mount_path
    ]))

    if FLAGS.data_dir.startswith("gs://"):
        tf.logging.info(util.run_and_output([
            "gsutil", "-m", "cp",
            "-r",
            data_dir + "/*",
            ssd_mount_path
        ]))
    else:
        tf.logging.info(util.run_and_output([
            "cp",
            "-r",
            FLAGS.data_dir + "/*",
            ssd_mount_path
        ]))  

    tf.logging.info(util.run_and_output([
        "ls", ssd_mount_path
    ]))


def main(argv):

    if FLAGS.ssd_mount_path:
        stage_data_dir_to_ssd_mount(FLAGS.data_dir,
                                    FLAGS.ssd_mount_path)
        FLAGS.data_dir = FLAGS.ssd_mount_path

    tf_config_to_additional_flags()

    trainer_main(None)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()