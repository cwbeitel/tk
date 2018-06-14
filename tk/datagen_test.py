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

from tk.jobs import T2TDatagenJob


class TestT2TDatagenJob(unittest.TestCase):

    def setUp(self):
        # Hack: Until have better means of incremental testing
        self.skip = True

    def test_smoke_local(self):
        args = gen_local_smoke_args("test-smoke-datagen")
        job = T2TDatagenJob(**args)
        
        if not self.skip:
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()