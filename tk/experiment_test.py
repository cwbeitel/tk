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

import tensorflow as tf

import os
import json

from tk import experiment
from tk import util
import shutil
import json
import logging

from tk.util import hack_dict_to_cli_args


def gen_local_smoke_args(base_name):
    
    app_root = os.path.realpath(os.path.join(
            os.path.split(__file__)[0], "../")
        )
    
    output_dir = os.path.join(app_root, "output")
    
    job_name = util.generate_job_name(base_name)
    args = {
        "job_name": job_name,
        "volume_claim_id": "nfs-east1-d",
        "app_root": app_root,
        "gcp_project": "foo",
        "namespace": "kubeflow",
        "image": "gcr.io/kubeflow-rl/base:0.0.9",
        "problem": "img2img_allen_brain",
        "model": "img2img_transformer",
        "hparams_set": "img2img_transformer2d_tiny",
        "data_dir": "/mnt/nfs-east1-d/data",
        "output_dir": output_dir,
        "smoke": True,
        "batch": False,
        "train_steps": 2,
        "schedule": "train"
    }

    return args


def _stage(local_app_root, remote_app_root):
    if not os.path.exists(local_app_root):
        raise ValueError("Can't stage from a non-existent source, "
                         "saw %s" % local_app_root)
    shutil.copytree(local_app_root, remote_app_root)


def gen_remote_smoke_args(base_name):

    args = gen_local_smoke_args(base_name)
    local_app_root = args["app_root"]

    testing_storage_base = "/mnt/nfs-east1-d/testing"
    remote_app_root = "%s/%s" % (testing_storage_base,
                                 args["job_name"])

    with open(os.path.join(local_app_root, "job.sh"), "w") as f:
      f.write("pip install -e %s" % remote_app_root)
      cmd = ["python", "-m", "tk.experiment"]
      cmd.extend(hack_dict_to_cli_args(args))
      f.write(" ".join(cmd))
      logging.info(local_app_root)
      
    _stage(local_app_root, remote_app_root)
    args["app_root"] = remote_app_root
    args["batch"] = True
    args["output_dir"] = os.path.join(remote_app_root, "output")

    return args

  
class TestParseTFConfig(tf.test.TestCase):

  def test_simple(self):

    os.environ["TF_CONFIG"] = json.dumps(
      {u'environment': u'cloud',
       u'cluster': {
         u'master': [u'enhance-0401-0010-882a-master-5sq4-0:2222']
       },
       u'task': {u'index': 0, u'type': u'master'}})

    flags = experiment.tf_config_to_cmd_line_flags()
    tf.logging.info(flags)


class TestT2TExperiment(tf.test.TestCase):

  def test_generates_expected_config(self):
    """Test that a correctly structured job config is constructed."""
    pass

  def test_e2e_smoke_local(self):

    skip = True

    args = gen_local_smoke_args("test-smoke-experiment")
    job = experiment.T2TExperiment(**args)

    if not skip:
      job.run()

  def test_e2e_small_remote(self):

    skip = False

    args = gen_remote_smoke_args("test-small-experiment")
    job = experiment.T2TExperiment(**args)

    if not skip:
      job.run()
    

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.test.main()
