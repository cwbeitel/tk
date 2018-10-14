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
import shutil
import pprint

import tensorflow as tf

from tensor2tensor.bin.t2t_trainer import FLAGS
from tensor2tensor.bin.t2t_trainer import main as trainer_main

from tk.kube import AttachedVolume
from tk.kube import LocalSSD
from tk.kube import Resources
from tk.kube import TFJob
from tk.kube import TFJobReplica
from tk.util import hack_dict_to_cli_args
from tk.util import generate_job_name


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


def _stage(local_app_root, remote_app_root):
    """Stage data from `local_app_root` to `remote_app_root`.
    
    Args:
        local_app_root (str): Directory path on local FS.
        remote_app_root (str): Directory path on remote FS.
    """
    
    if not os.path.exists(local_app_root):
        raise ValueError("Can't stage from a non-existent source, "
                         "saw %s" % local_app_root)

    shutil.copytree(local_app_root, remote_app_root)


def configure_experiment(base_name, num_gpu_per_worker=1,
                         problem="img2img_allen_brain_dim8to32",
                         model="img2img_transformer",
                         hparams_set="img2img_transformer2d_tiny",
                         num_steps=100000,
                         num_workers=0,
                         num_ps=0,
                         ps_gpu=1,
                         log_device_placement=False,
                         profile=False,
                         dbgprofile=False,
                         extra_hparams={
                           "batch_size": 4
                         },
                         app_root="/mnt/nfs-east1-d/work/tk",
                         base_image="tensorflow/tensorflow:latest-gpu",
                         reuse_output_dir=None):
    """Wrapper to construct args object and produce job scripts.

    Args:
        base_name (str): The base name to be used to identify the experiment.
    """
    
    output_dir = os.path.join(app_root, "output")

    job_name = generate_job_name(base_name)

    train_args = {
        "problem": problem,
        "model": model,
        "hparams_set": hparams_set,
        "data_dir": "/mnt/nfs-east1-d/data",
        "output_dir": output_dir,
        "train_steps": num_steps,
        "schedule": "train",
        "profile": profile,
        "log_device_placement": log_device_placement,
        "worker_gpu": num_gpu_per_worker,
        "ps_gpu": ps_gpu,
        "save_checkpoints_secs": 1800,
        "dbgprofile": dbgprofile, # Saves profiling timelines, viewable in chrome://tracing
        "ssd_mount_path": "/mnt/disks/ssd0",
        "worker_gpu_memory_fraction": 0.95,
        #"hparams": "'batch_size=%s'" % batch_size
    }
    
    #if isinstance(loss_variant, str):
    #    train_args["hparams"] = "'batch_size=%s,loss_variant=%s'" % (batch_size,
    #                                                                 loss_variant)
    
    hparams = ""
    for k, v in extra_hparams.items():
        if len(hparams) != 0:
            hparams += ","
        hparams += "%s=%s" % (k, v)
        
    train_args["hparams"] = "'%s'" % hparams

    args = {
        "job_name": job_name,
        "volume_claim_id": "nfs-east1-d",
        "app_root": app_root,
        "gcp_project": "foo",
        "namespace": "kubeflow",
        "image": base_image,
        "smoke": True,
        "batch": False,
        "train_args": train_args,
        "cpu": 7,
        "memory": "40Gi",
        "num_gpu": num_gpu_per_worker,
        
        # DEV
        "master_gpu": num_gpu_per_worker,
        "ps_gpu": ps_gpu,
        "worker_gpu": num_gpu_per_worker,
        # --

        "num_local_ssd": 1,
        "no_wait": True,
        "num_worker_replicas": num_workers,
        "num_ps_replicas": num_ps,
        "selector_labels": {
          "cloud.google.com/gke-nodepool": "train-gpu-preemptible-%sx-hm" % num_gpu_per_worker,
          "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
        }
    }

    local_app_root = args["app_root"]

    testing_storage_base = "/mnt/nfs-east1-d/comparisons/%s" % base_name
    
    remote_app_root = "%s/%s" % (testing_storage_base,
                                 args["job_name"])

    output_dir_root = "gs://kubeflow-rl-checkpoints/comparisons/%s" % base_name

    # Put training checkpoints in a folder like so:
    # gs://kubeflow-rl-checkpoints/comparisons/[exp base name]/[job id]/output/
    args["train_args"]["output_dir"] = os.path.join(output_dir_root,
                                                      args["job_name"],
                                                      "output")

    if reuse_output_dir is not None:
      args["train_args"]["output_dir"] = reuse_output_dir

    args["train_args"]["t2t_usr_dir"] = os.path.join(output_dir_root,
                                                     args["job_name"],
                                                     "tk")

    print("train_args:")
    pprint.pprint(args["train_args"])

    for job_type in ["master", "ps"]:
        
        with open(os.path.join(local_app_root, "%s-job.sh" % job_type), "w") as f:
          f.write("ls /mnt\n")
          #f.write("mkdir /tmp/deps")
          #f.write("tar -xzf %s/deps.tgz /tmp/deps\n" % remote_app_root")
          f.write("cp -r /mnt/nfs-east1-d/data/* /mnt/ssd0/\n")
          f.write("pip install -e %s/vendor/tensor2tensor\n" % remote_app_root)
          f.write("pip install -e %s\n" % remote_app_root)
          f.write("nvidia-smi\n")
          f.write("python -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'\n")
          f.write("python -c 'import tensorflow as tf; print(tf.__version__)'\n")
          f.write("echo ${TF_CONFIG}\n")
          f.write("cd %s\n" % remote_app_root)
          cmd = ["python", "-m", "tk.experiment"]

          cmd.extend(hack_dict_to_cli_args(args["train_args"]))
          f.write(" ".join(cmd) + "\n")
          f.write("nvidia-smi\n")
          logging.info(local_app_root)
    
    _stage(local_app_root, remote_app_root)
    args["app_root"] = remote_app_root
    args["batch"] = True

    return args


def main(argv):
  """Configure, setup logging, and train."""

  task_type, task_id = tf_config_to_additional_flags()

  tf.gfile.MakeDirs(FLAGS.output_dir)

  worker_name = "%s-%s" % (task_type, task_id)

  trainer_main(None)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()