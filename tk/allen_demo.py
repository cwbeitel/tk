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

import math
import os
import logging
import shutil

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pprint 

import subprocess
import requests

import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

from tk.util import hack_dict_to_cli_args
from tk import experiment
from tk import util

from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import allen_brain

from tensorboard.backend.event_processing import event_file_loader
from protobuf_to_dict import protobuf_to_dict

logging.getLogger().setLevel(logging.INFO)

from tk.util import hack_dict_to_cli_args
from tk import experiment
from tk import util

"""Code for blog post."""


def _event_dict_list_from_events_file(event_file_path):
    """Given an event file path, load the event data.
    
    Args:
        event_file_path (str): A path to a TensorFlow events.out* file.
    """
    loader = event_file_loader.EventFileLoader(event_file_path)
    events = []
    for event in loader.Load():
      events.append(event)

    output_events = []

    for thing in events:
        d = protobuf_to_dict(thing)

        output_event = {}
        if "wall_time" in d:
            output_event["wall_time"] = d["wall_time"]
        if "step" in d:
            output_event["step"] = d["step"]
        if "summary" in d.keys():
            values = {}
            for value in d["summary"]["value"]:
                if "simple_value" in value.keys():
                    output_event[value["tag"]] = value["simple_value"]

        if "loss" in output_event:
            output_events.append(output_event)
    
    return output_events


def event_data_for_comparison(comparison_root, events_subdir_query="*/output/events.out*"):
    """Given root dir path for comparison, load all event data beneath.
    
    Args:
        comparison_root (str): The root path beneath which comparison logs and
            events are being stored.
        events_subdir_query (str): The file path pattern to use to locate event
            files beneath `comparison_root`.
    """
    event_data = []
    tf.logging.info("Identifying event files in experiment subdirectories...")
    paths = tf.gfile.Glob(os.path.join(comparison_root, events_subdir_query))

    for i, experiment in enumerate(paths):
        tf.logging.info("Processing experiment events (%s of %s)" % (i + 1, len(paths)))
        event_list = _event_dict_list_from_events_file(experiment)
        event_dict = {"source_path": experiment,
                      "events": event_list}

        event_data.append(event_dict)
    tf.logging.info("Finished loading event data for comparison.")
    return event_data


def show_experiment_loss(experiments_data):
    """Given a collection of TensorFlow events data, display labeled loss plots.
    
    Args:
        experiments_data (list): A list of event data dictionaries.
    """
    
    plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
    for i, experiment_data in enumerate(experiments_data):
        wall_times = [event["wall_time"] for event in experiment_data["events"]]
        minwt = min(wall_times)
        relative_times = [wt - minwt for wt in wall_times]
        losses = [event["loss"] for event in experiment_data["events"]]
        autocorr = math.floor(pandas.Series(losses).autocorr()*1000)/1000
        tag = experiment_data["source_path"].split("/")[-3]
        plt.plot(relative_times, losses, label="%s, ac=%s" % (tag, autocorr))

    plt.legend()
    plt.show()


def infer(predictions):
    """Produce a predicted image using argmax of model-generated logits.
    
    Args:
        predictions (tf.Tensor): A 3-D tensor of logits.
    """
    p = predictions.numpy()
    x_max = p.shape[1] # X-position in image
    y_max = p.shape[2] # Y-position in image
    c_max = p.shape[3] # Color channel (r,g,b)

    # The image we will populate
    image = np.zeros((1, x_max, y_max, c_max), dtype=np.uint8)

    batch_idx = 0

    for x in range(0, x_max):
        for y in range(0, y_max):
            for c in range(0, c_max):
                # Get the index of the greatest value in p[0][x][y][c]
                val = np.argmax(p[0][x][y][c])
                image[batch_idx][x][y][c] = np.uint8(np.argmax(p[0][x][y][c]))
    return image[0]


def predict_ith(ckpt_path, dataset, model, offset=1234, input_dim=8, output_dim=32):
    """Apply model to input obtained from dataset modulo offset.
    
    Args:
        offset (int): The offset within `dataset` where we will extract an example.
        ckpt_path (str): The path from which checkpoints will be loaded.
        dataset (`obj`): A tf.data dataset object.
        model (`obj`): A tensor2tensor model object loaded from the registry.
    """

    with tfe.restore_variables_on_create(ckpt_path):
      for count, example in enumerate(tfe.Iterator(dataset)):
          if count > offset:
            break
      fig=plt.figure(figsize=(8, 8))
      example["inputs"] = tf.reshape(example["inputs"], [1, input_dim, input_dim, 3])
      fig.add_subplot(1, 3, 1)
      plt.imshow(example["inputs"].numpy()[0])
      fig.add_subplot(1, 3, 2)
      example["targets"] = tf.reshape(example["targets"], [1, output_dim, output_dim, 3])
      plt.imshow(example["targets"].numpy()[0])
      example["targets"] = tf.reshape(tf.zeros((1, output_dim, output_dim, 3), dtype=np.uint8),
                                      [1, output_dim, output_dim, 3])
      predictions, _ = model(example)
      fig.add_subplot(1,3,3)
      inferred = infer(predictions)
      plt.imshow(inferred)
      plt.show()
        
    return example, predictions, inferred


def patchwise_infer(input_image, ckpt_path, target_shape, input_stride=8,
                    target_stride=32, offset=0):
    """Given a small image, infer a large image using non-overlapping patches.
    
    Args:
        input_image (np.Array): An array of image data.
        ckpt_path (str): The path where checkpoint data is stored.
        target_shape (tuple): The shape of the target to infer.
    """
    
    target_image = np.zeros(target_shape, dtype=np.uint8)

    input_shape = np.shape(input_image)

    upscale_factor = target_stride / input_stride

    with tfe.restore_variables_on_create(ckpt_path):

        x_index_max = input_shape[0] // input_stride
        for x_index in range(0, x_index_max):
            
            tf.logging.info("processing row %s of %s" % (
                x_index, x_index_max))

            x_offset_toggle = 0
            if x_index == 1:
                x_offset_toggle = 1

            for y_index in range(0, input_shape[1] // input_stride):

                example = {}
                
                y_offset_toggle = 0
                if y_index == 1:
                    y_offset_toggle = 1

                input_x_start = x_index * input_stride - offset * x_offset_toggle
                input_x_end = input_x_start + input_stride
                input_y_start = y_index * input_stride - offset * y_offset_toggle
                input_y_end = input_y_start + input_stride

                input_patch = input_image[input_x_start:input_x_end,
                                          input_y_start:input_y_end]

                source_reshape = [1, input_stride, input_stride, 3]
                example["inputs"] = tf.reshape(input_patch, source_reshape)

                upscale_offset = offset * upscale_factor
                target_x_start = int(math.floor(x_index * target_stride - upscale_offset * x_offset_toggle))
                target_y_start = int(math.floor(y_index * target_stride - upscale_offset * y_offset_toggle))
                target_x_end = target_x_start + target_stride
                target_y_end =  target_y_start + target_stride

                zero_target = tf.zeros((1, target_stride, target_stride, 3),
                                       dtype=np.uint8)
                #zero_target = target_image[target_x_start:target_x_end,
                #                           target_y_start:target_y_end]
                target_reshape = [1, target_stride, target_stride, 3]
                example["targets"] = tf.reshape(zero_target, target_reshape)

                predictions, _ = model(example)

                prediction = infer(predictions)

                try:
                  target_image[target_x_start:target_x_end,
                               target_y_start:target_y_end] = prediction

                except Exception as e:
                    tf.logging.info("Ran out of space in target, skipping...")

    return target_image


def example_apply_model(ckpt_path,
                        hparams_set="img2img_transformer2d_tiny",
                        problem_name="img2img_allen_brain_dim8to32",
                        model_name="img2img_transformer",
                        data_dir="/mnt/nfs-east1-d/data",
                        input_dim=8,
                        output_dim=32):

    # HACK: Avoid re-instantiating the model which causes problems...
    # TODO: Better way to handle this, e.g. delete from globals.
    if 'model' not in globals():

        hp = trainer_lib.create_hparams(
            hparams_set,
            data_dir=data_dir,
            problem_name=problem_name)

        model = registry.model(model_name)(hp, Modes.TRAIN)
    
    problem_object = problems.problem(problem_name)
    
    dataset = problem_object.dataset(Modes.TRAIN, data_dir)
    
    with tfe.restore_variables_on_create(ckpt_path):
      for count, example in enumerate(tfe.Iterator(dataset)):
          if count > 1234:
            break

      # Example input
      fig=plt.figure(figsize=(8, 8))
      example["inputs"] = tf.reshape(example["inputs"], [1, input_dim, input_dim, 3])
      fig.add_subplot(1, 3, 1)
      plt.imshow(example["inputs"].numpy()[0])

      # Example target
      fig.add_subplot(1, 3, 2)
      example["targets"] = tf.reshape(example["targets"], [1, output_dim, output_dim, 3])
      plt.imshow(example["targets"].numpy()[0])

      # Dummy target (expected by model)
      example["targets"] = tf.reshape(tf.zeros((1, output_dim, output_dim, 3), dtype=np.uint8),
                                      [1, output_dim, output_dim, 3])
      
      # Produce and display prediction
      predictions, _ = model(example)
      fig.add_subplot(1,3,3)
      inferred = demo.infer(predictions)
      plt.imshow(inferred)
      plt.show()
        
    return example, predictions, inferred


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
                         }):
    """Wrapper to construct args object and produce job scripts.

    Args:
        base_name (str): The base name to be used to identify the experiment.
    """

    app_root = "/mnt/nfs-east1-d/work/tk"
    
    output_dir = os.path.join(app_root, "output")

    job_name = util.generate_job_name(base_name)

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
    
    print("train_args:")
    pprint.pprint(train_args)

    args = {
        "job_name": job_name,
        "volume_claim_id": "nfs-east1-d",
        "app_root": app_root,
        "gcp_project": "foo",
        "namespace": "kubeflow",
        "image": "tensorflow/tensorflow:latest-gpu",
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

    args["train_args"]["t2t_usr_dir"] = os.path.join(output_dir_root,
                                                     args["job_name"],
                                                     "tk")

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


def maybe_download_file(url):
    local_filename = "/tmp/allen-brain-example.jpg"
    if os.path.exists(local_filename):
        print("Local example image already exists, skipping download.")
        return local_filename
    print("Downloading example image...")
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: 
                f.write(chunk)
    print("Done.")
    return local_filename


def load_example_image():
    local_filename = maybe_download_file("http://api.brain-map.org/api/v2/image_download/100883805")
    img = Image.open(local_filename)
    img = np.float32(img)
    return img


def show_random_examples(problem_object, data_dir, num=1):

    dataset = problem_object.dataset(Modes.TRAIN, data_dir)

    input_dim = 8
    output_dim = 32

    for j in range(0, num):
    
        offset = 1234

        for i, example in enumerate(tfe.Iterator(dataset)):
            if i > offset:
                break

        fig=plt.figure(figsize=(6, 6))
        example["inputs"] = tf.reshape(example["inputs"], [1, input_dim, input_dim, 3])

        fig.add_subplot(1, 2, 1)
        plt.imshow(example["inputs"].numpy()[0])

        fig.add_subplot(1, 2, 2)
        example["targets"] = tf.reshape(example["targets"], [1, output_dim, output_dim, 3])

        plt.imshow(example["targets"].numpy()[0])
