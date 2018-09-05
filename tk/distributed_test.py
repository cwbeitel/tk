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

import threading
import time
import unittest
import multiprocessing

import os
import json
import sys

import tensorflow as tf

from tk.util import generate_job_name
from tk.util import hack_dict_to_cli_args
from tk.util import run_and_output

from tk.experiment import main

"""

*** Question: Do the sys.argv.extend calls affect eachother?

"""

class ThreadContext(object):
  """Start a pool of threads and cleanup on exit.
  
  For use in a test we want to be able to fail with an
  error but still clean up the thread context.

  """
  
  def __init__(self, threads):
    self.threads = threads

  def __enter__(self):
    for t in self.threads:
      t.start()

  def __exit__(self, exc_type, exc_value, traceback):
    for t in self.threads:
      t.terminate()


class TestDistributed(unittest.TestCase):

  def test_local_distributed_setup(self):
    
    def _get_cli_args(tag):
        job_name = generate_job_name("test")
        args = {
            "problem": "img2img_allen_brain_dim8to32",
            "model": "img2img_transformer",
            "hparams_set": "img2img_transformer2d_tiny",
            "data_dir": "/mnt/nfs-east1-d/data",
            "output_dir": "/tmp/%s-%s" % (job_name, tag),
            "train_steps": 1000,
            "schedule": "train",
            "hparams": "batch_size=1"
        }
        return hack_dict_to_cli_args(args)

    def get_tfc(task_type):
        return json.dumps(
          {u'environment': u'cloud',
           u'cluster': {
             u'master': [u'localhost:2222'],
             u'ps': [u'localhost:2221'],
             u'worker': [u'localhost:2220'],
           },
           u'task': {u'index': 0, u'type': task_type.encode("utf-8")}})

    def start_ps(_):
      tf.logging.info("Running parameter server...")
      os.environ["TF_CONFIG"] = get_tfc("ps")
      print(os.environ["TF_CONFIG"])
      sys.argv.extend(_get_cli_args("ps"))
      sys.argv.extend("--schedule=run_std_server")
      tf.app.run()
      tf.logging.info("Finished running parameter server.")

    def start_worker(_):
      tf.logging.info("Running worker...")
      os.environ["TF_CONFIG"] = get_tfc("worker")
      sys.argv.extend(_get_cli_args("worker"))
      tf.app.run()
      tf.logging.info("Finished running worker.")


    threads = [
        multiprocessing.Process(
            target=start_ps, args=(None,)),
        multiprocessing.Process(
            target=start_worker, args=(None,))
    ]
    
    with ThreadContext(threads):

      os.environ["TF_CONFIG"] = get_tfc("master")
      sys.argv.extend(_get_cli_args("master"))
      tf.app.run()

      time.sleep(100)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()