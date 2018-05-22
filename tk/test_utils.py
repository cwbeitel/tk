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

"""Launcher tests."""

import unittest
import os
import logging
import shutil

from util import generate_job_name

"""

TODO: Get base names for jobs from the method names of the tests launching
    them.

"""

def gen_local_smoke_args(base_name):
    
    app_root = os.path.realpath(os.path.join(
            os.path.split(__file__)[0], "../")
        )
    
    output_dir = os.path.join(app_root, "output")
    
    job_name = generate_job_name(base_name)
    args = {
        "job_name": job_name,
        "volume_claim_id": "nfs-1",
        "app_root": app_root,
        "gcp_project": "foo",
        "namespace": "kubeflow",
        "image": "gcr.io/kubeflow-rl/enhance:0411-0440-41e6",
        #"image": "gcr.io/kubeflow-rl/enhance-runtime:0.0.3",
        "problems": "allen_brain_image2image_upscale",
        "problem": "allen_brain_image2image_upscale",
        "model": "img2img_transformer",
        "hparams_set": "img2img_transformer2d_tiny",
        "data_dir": "/mnt/nfs-1/testing/decode/data_dir/",
        "tmp_dir": "/mnt/nfs-1/datasets/alleninst/mouse-testing",
        "output_dir": output_dir,
        "smoke": True,
        "batch": False,
        "train_steps": 2,
        "eval_steps": 1, # ?
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
    
    testing_storage_base = "/mnt/nfs-1/testing"
    remote_app_root = "%s/%s" % (testing_storage_base,
                                 args["job_name"])
    _stage(local_app_root, remote_app_root)
    args["app_root"] = remote_app_root
    args["batch"] = True
    args["output_dir"] = os.path.join(remote_app_root, "output")

    return args


