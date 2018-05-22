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

"""Tests of general utilities."""

import unittest
import logging
import os
import shutil

from util import generate_job_name

from test_utils import gen_local_smoke_args, gen_remote_smoke_args
from e2e import main, E2EJob


class TestE2E(unittest.TestCase):
    
    #def test_smoke_local(self):
    #    job_config = _gen_local_smoke_args("smoke-e2e-local")
    #    main(job_config)
    
    def test_e2e_tiny_iden(self):
        """Minimal run in batch."""
        skip_run = False
        args = gen_remote_smoke_args("test-e2e-tiny")
        # For now presume we're operating on previously downloaded data
        args["tmp_dir"] = "/mnt/nfs-1/testing/artifacts/tmp_dir"
        args["decode_hparams"] = "save_images=True,num_samples=3,beam_size=1"
        args["train_steps"] = 2
        args["decode_data_dir"] = "/mnt/nfs-1/testing/artifacts/decode_data_dir"
        args["problem"] = "allen_brain_image2image_upscale"
        job = E2EJob(**args)
        if not skip_run:
            job.run()


# TODO: Decide on the organization of tests - local vs. remote, single
# job vs. e2e, included in modules they call or separated. 

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
