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

"""Periodically log resource information to a file on disk."""

import os
import datetime
import time
import psutil
import json
import subprocess

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('base_path', None, 'Base path to which to write a resource log.')
flags.DEFINE_integer('inter_log_duration', 5, 
                     'Number of seconds between logging a packet.')

def main(_):
    
    base_path = FLAGS.base_path
    if base_path is None:
        raise ValueError("Please provide a base path via --base_path.")

    inter_log_duration = FLAGS.inter_log_duration
    
    def _log_one(destination):
        packet = {
            "memory": psutil.virtual_memory().__dict__,
            "timestamp": str(datetime.datetime.now())
        }
        
        cmd = "echo %s >> %s" % (json.dumps(packet),
                                 destination)

        os.system(cmd)
        
        os.system("nvidia-smi >> %s" % destination)

    destination = os.path.join(base_path, "resources_log.txt")

    while True:
        _log_one(destination)
        time.sleep(inter_log_duration)

            
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()