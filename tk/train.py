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

import os
import json
import logging

import tensorflow as tf

from tensor2tensor.bin.t2t_trainer import main as trainer_main
from tensor2tensor.bin.t2t_trainer import FLAGS
# Make this script equivalent to running t2t-trainer so that we can
# add some steps when running as __main__.


def tf_config_to_cmd_line_flags():

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
                cmd_line_flags = [
                        "--master=grpc://%s" % job,
                        "--ps_replicas=%d" % num_ps,
                        "--worker_replicas=%d" % num_masters,
                        "--worker_gpu=1",
                        "--worker_id=%d" % idx,
                        "--worker_job='/job:master'",
                        "--ps_gpu=1",
                        "--schedule=train",
                        "--sync" if num_masters == 1 else "",
                    ]
            else:
                cmd_line_flags = [
                    "--master=grpc://%s" % job,
                    "--schedule=run_std_server",
                ]

    logging.info("constructed extra cmd line flags: %s" % cmd_line_flags)
    return cmd_line_flags


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


def main(argv):

    #cmd_line_flags = tf_config_to_cmd_line_flags()
    #argv.extend(cmd_line_flags)
    #trainer_main(argv)

    tf_config_to_additional_flags()
    trainer_main(None)


if __name__ == "__main__":
    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()