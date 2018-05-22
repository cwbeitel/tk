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

"""Kubernetes models and utils supporting templating Job's and TFJob's"""

import unittest
import logging
import pprint

from kube import TFJob, TFJobReplica, Resources, AttachedVolume, Job, build_command, Container
from util import gen_timestamped_uid, object_as_dict


class TestContainer(unittest.TestCase):
    
    def test_instantiate(self):
        
        config = {
            "model": "something"
        }
        
        container_args = build_command("t2t-trainer", **config)

        av = AttachedVolume("nfs-1")
        resources = Resources(requests={"cpu": 30})
        image_tag = "gcr.io/kubeflow-rl/enhance:0321-2116-e45a"
        
        cases = [
            {
                "kwargs": {
                    "args": container_args,
                    "image": image_tag,
                    "name": "tensorflow",
                    "resources": resources,
                    "attached_volume": av,
                },
                "expected": {
                    'args': ['t2t-trainer', '--model=something'],
                    'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
                    'name': 'tensorflow',
                    'resources': {'requests': {'cpu': 30}},
                    'volumeMounts': [{'mountPath': '/mnt/nfs-1', 'name': 'nfs-1'}]
                }
            }
        ]
        
        for case in cases:
            self.assertEqual(object_as_dict(Container(**case["kwargs"])),
                             case["expected"])


    def test_expose_ports(self):

        kwargs = {
                "image": "gcr.io/kubeflow-rl/enhance:0321-2116-e45a",
                "args": ['t2t-trainer', '--model=something'],
                "name": "tensorflow",
                "ports": [{"containerPort": 80}]
            }
        expected = {
                'args': ['t2t-trainer', '--model=something'],
                'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
                'name': 'tensorflow',
                'ports': [{"containerPort": 80}]
            }
     
        self.assertEqual(object_as_dict(Container(**kwargs)), expected)
        
    
class TestAttachedVolume(unittest.TestCase):
    
    def test_instantiate(self):
        
        cases = [
            {
                "args": ["nfs-1"],
                "expected": {
                    'volume': {
                        'name': 'nfs-1',
                        'persistentVolumeClaim': {'claimName': 'nfs-1'}
                    },
                    'volume_mount': {'mountPath': '/mnt/nfs-1', 'name': 'nfs-1'}
                }
            }
        ]
        
        for case in cases:
            av = AttachedVolume(*case["args"])
            self.assertEqual(object_as_dict(av),
                             case["expected"])

    
class TestJob(unittest.TestCase):
    
    def test_instantiates_job(self):
        """Test our ability to instantiate a job"""
        
        cases = [
            {
                # Without NFS
                "job_object_args": {
                    "job_name": "kittens",
                    "command": ["ls"],
                    "image": "ubuntu"
                },
                "expected_dict": {
                    'apiVersion': 'batch/v1',
                    'kind': 'Job',
                    'metadata': {
                        'name': 'kittens',
                        'namespace': 'default'
                    },
                    'spec': {
                        'backoffLimit': 4,
                        'template': {
                            'spec': {
                                'containers': [
                                    {
                                        'args': ['ls'],         
                                        'image': 'ubuntu',
                                        'name': 'container'
                                    }
                                ],
                                'restartPolicy': 'Never'
                            }
                        }
                    }
                }
            },
            {
                # With NFS
                "job_object_args": {
                    "job_name": "puppies",
                    "image": "ubuntu",
                    "command": ["ls"],
                    "namespace": "kubeflow",
                    "volume_claim_id": "nfs-1"
                },
                "expected_dict": {
                    'apiVersion': 'batch/v1',
                    'kind': 'Job',
                    'metadata': {
                        'name': 'puppies',
                        'namespace': 'kubeflow'
                    },
                    'spec': {
                        'backoffLimit': 4,
                        'template': {
                            'spec': {
                                'containers': [
                                    {
                                        'args': ['ls'],         
                                        'image': 'ubuntu',
                                        'name': 'container',
                                        'volumeMounts': [{'mountPath': '/mnt/nfs-1',
                                                          'name': 'nfs-1'}]
                                    }
                                ],
                                'restartPolicy': 'Never',
                                'volumes': [{
                                  'name': 'nfs-1',
                                  'persistentVolumeClaim': {
                                      'claimName': 'nfs-1'
                                  }
                                }]
                            }
                        }
                    }
                }
            }
        ]
        
        self.maxDiff = 1000
        
        for case in cases:
            
            job = Job(**case["job_object_args"])
            
            pprint.pprint(object_as_dict(job))
            pprint.pprint(case["expected_dict"])
            
            self.assertEqual(job.as_dict(),
                             case["expected_dict"])


    def test_subclass_smoke_local(self):
        
        args = {
            "job_name": "foo",
            "image": "foo",
        }
        
        class SmokeableJob(Job):
            
            def __init__(self, *args, **kwargs):

                command = ["cat"]

                super(SmokeableJob, self).__init__(command=command,
                                                   *args, **kwargs)

        args["smoke"] = True
        args["batch"] = False
        job = SmokeableJob(**args)
        job.run()


class TestTFJob(unittest.TestCase):
    
    def test_instantiate_tfjob(self):
        """Test that a local TFJob model can be instantiated."""
        
        config = {
            "model": "something"
        }
        
        image = "gcr.io/kubeflow-rl/enhance:0321-2116-e45a"
        resources = Resources(requests={
                             "cpu": 30,
                             "memory": "119Gi"
                         })

        command = [
                "python", "%s/py/train.py" % "/foo",
                "--t2t_usr_dir", "%s/py" % "/foo",
            ]
        
        replicas = [
                TFJobReplica(replica_type="MASTER",
                             num_replicas=1,
                             args=command,
                             image=image,
                             resources=Resources(requests={
                                 "cpu": 30
                             }),
                             attached_volume=AttachedVolume("nfs-1"))
        ]
        
        tfjob = TFJob(command=command,
                      job_name=gen_timestamped_uid(),
                      namespace="kubeflow",
                      replicas = replicas)

    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()