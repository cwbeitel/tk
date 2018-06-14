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

import uuid
import datetime
import logging

from tk.kube_tensorboard import TensorBoard

from tk import util


class TestTensorBoardComponent(unittest.TestCase):

    def setUp(self):
        self.log_dir = "foo_log_dir"
        self.namespace = "kubeflow"
        now = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        salt = "%s-%s" % (now, str(uuid.uuid4())[0:8])
        self.job_name_base = "%s-%s" % ("tb-test", salt)
        self.maxDiff = 2000
    
    def test_builds_component(self):
        
        depl_name = self.job_name_base + "-depl"
        srv_name = self.job_name_base + "-svc"
        
        expected_service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {'name': srv_name, 'namespace': self.namespace},
            'spec': {
                'ports': [{'name': 'http', 'port': 80, 'targetPort': 80}],
                'selector': {'app': 'tensorboard', 'tb-job': depl_name}
            }
        }
        
        cmd = ['/usr/local/bin/tensorboard',
               '--logdir=%s' % self.log_dir,
               '--port=80']
        
        expected_deployment = {
            'apiVersion': 'apps/v1beta1',
            'kind': 'Deployment',
            'metadata': {
                'name': depl_name,
                'namespace': self.namespace
            },
            'spec': {
                'replicas': 1,
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'tensorboard',
                            'tb-job': depl_name
                        },
                        'name': depl_name,
                        'namespace': self.namespace
                    },
                    'spec': {
                        'containers': [
                            {
                                'command': cmd,
                                'image': 'gcr.io/tensorflow/tensorflow:latest',
                                'name': 'tensorboard',
                                'ports': [
                                    {'containerPort': 80}
                                ]
                            }
                        ]
                    }
                }
            }
        }
        
        tb = TensorBoard(log_dir=self.log_dir,
                         namespace=self.namespace,
                         job_name_base=self.job_name_base)
        
        self.assertEqual(util.object_as_dict(tb.components["service"]),
                         expected_service)

        self.assertEqual(util.object_as_dict(tb.components["deployment"]),
                         expected_deployment)
        
    
    def test_runs(self):
        """Verify that when .create() both deployment and service are created."""
        
        tb = TensorBoard(self.log_dir,
                         self.namespace,
                         self.job_name_base)
        
        responses = tb.create(poll_and_check=True)
        
        tb.delete()
 

    def test_can_instantiate_without_job_name_base(self):
      tb = TensorBoard(self.log_dir,
                       self.namespace)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()