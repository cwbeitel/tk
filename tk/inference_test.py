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

import logging
import unittest

from tk.test_utils import gen_local_smoke_args
from tk.test_utils import gen_remote_smoke_args

from tk.jobs import InferenceJob


class TestInferenceJob(unittest.TestCase):
    
    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = True
       
    def test_smoke_local(self):
        args = gen_local_smoke_args("smoke-inference")
        job = InferenceJob(**args)
        
        if not self.skip:
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()