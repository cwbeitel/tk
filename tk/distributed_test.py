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

import tensorflow as tf


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

    def start_ps(_):
      tf.logging.info("Running parameter server...")
      time.sleep(100)

    def start_worker(_):
      tf.logging.info("Running worker...")
      time.sleep(100)

    threads = [
        multiprocessing.Process(
            target=start_ps, args=(None,)),
        multiprocessing.Process(
            target=start_worker, args=(None,))
    ]
    
    with ThreadContext(threads):

      # Run a master referencing the above PS and worker
      # TODO
    

      time.sleep(5)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()