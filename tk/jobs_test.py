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

import unittest
import logging

from test_utils import gen_local_smoke_args, gen_remote_smoke_args
from jobs import DownloadJob, T2TDatagenJob, T2TExperiment
from jobs import InferenceJob

"""
TODO:
- Tests not general, tied to specific files and system config (NFS).

"""

class TestDownloadJob(unittest.TestCase):

    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = True

    def test_smoke_local(self):
        args = gen_local_smoke_args("test-smoke-download")
        logging.info("Triggering smoke download job with args: %s" % args)
        job = DownloadJob(**args)
        job.run()
    
    def test_mini_remote(self):
        # Note that this does not yet enforce that a remote job was successful!
        args = gen_remote_smoke_args("test-small-download")
        # For now effectively skip the test by providing a tmp_dir that already
        # contains data
        args["tmp_dir"] = "/mnt/nfs-1/testing/artifacts/tmp_dir"
        job = DownloadJob(**args)
        
        if not self.skip:
            job.run()


class TestInferenceJob(unittest.TestCase):
    
    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = True
       
    def test_smoke_local(self):
        args = gen_local_smoke_args("smoke-inference")
        job = InferenceJob(**args)
        job.run()
    
    def test_smoke_remote(self):
        args = gen_remote_smoke_args("mini-inference")
        args["tmp_dir"] = "/mnt/nfs-1/testing/decode/tmp_dir"
        args["output_dir"] = "/mnt/nfs-1/testing/artifacts/output_dir"
        args["decode_hparams"] = "save_images=True,num_samples=3,beam_size=1"
        args["data_dir"] = "/mnt/nfs-1/testing/artifacts/decode_data_dir"
        #args["image"] = "gcr.io/kubeflow-rl/enhance:0411-0440-41e6"
        job = InferenceJob(**args)
        
        if not self.skip:
            job.run()


class TestT2TDatagenJob(unittest.TestCase):

    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = True

    def test_smoke_local(self):
        args = gen_local_smoke_args("test-smoke-datagen")
        job = T2TDatagenJob(**args)
        job.run()
    
    def test_mini_remote(self):
        args = gen_remote_smoke_args("test-small-datagen")
        
        # Use a directory that contains the result of a previously-run
        # download step.
        # TODO: Figure out how to decouple this from local setup, e.g.
        # maybe-stage down test artifacts from GCS...?
        args["tmp_dir"] = "/mnt/nfs-1/testing/artifacts/tmp_dir"
        
        logging.info("test smoke remote args: %s" % args)
        job = T2TDatagenJob(**args)
        
        if not self.skip:
            job.run()
        

class TestT2TExperiment(unittest.TestCase):

    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = False

    def test_smoke_local(self):
        args = gen_local_smoke_args("test-smoke-experiment")
        job = T2TExperiment(**args)
        job.run()

    def test_small_remote(self):
        args = gen_remote_smoke_args("test-small-experiment")
        job = T2TExperiment(**args)
        
        if not self.skip:
            job.run()
        

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()