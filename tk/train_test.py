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

import os
import json

from train import tf_config_to_cmd_line_flags


# For now tests of trainer are in launcher_test which might change

class TestParseTFConfig(unittest.TestCase):
    
    def test_simple(self):
        
        os.environ["TF_CONFIG"] = json.dumps({u'environment': u'cloud',
                                              u'cluster': {
                                                  u'master': [u'enhance-0401-0010-882a-master-5sq4-0:2222']
                                              },
                                              u'task': {u'index': 0, u'type': u'master'}})
        
        flags = tf_config_to_cmd_line_flags()
        logging.info(flags)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()