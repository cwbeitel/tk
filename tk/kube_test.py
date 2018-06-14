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

from tk.kube import TFJob, TFJobReplica, Resources, AttachedVolume, Job, build_command, Container, LocalSSD
from tk.util import gen_timestamped_uid, object_as_dict


class TestContainer(unittest.TestCase):
    
    def test_instantiate(self):
        
        config = {
            "model": "something"
        }

        container_args = build_command("t2t-trainer", **config)

        av = AttachedVolume("nfs-1")
        resources = Resources(requests={"cpu": 1,
                                        "memory": "1Gi",
                                        "nvidia.com/gpu": 1})
        image_tag = "gcr.io/kubeflow-rl/enhance:0321-2116-e45a"

        cases = [
            {
                "kwargs": {
                    "args": container_args,
                    "image": image_tag,
                    "name": "tensorflow",
                    "resources": resources,
                    "volume_mounts": [av.volume_mount],
                },
                "expected": {
                    'args': ['t2t-trainer', '--model=something'],
                    'image': 'gcr.io/kubeflow-rl/enhance:0321-2116-e45a',
                    'name': 'tensorflow',
                    'resources': {'requests': {'cpu': 1,
                                               'memory': '1Gi',
                                               'nvidia.com/gpu': 1}},
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
                "kwargs": {"claim_name": "nfs-1"},
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
            av = AttachedVolume(**case["kwargs"])
            self.assertEqual(object_as_dict(av),
                             case["expected"])


class TestLocalSSD(unittest.TestCase):
  
  def test_instantiate(self):

        cases = [
            {
                "kwargs": {},
                "expected": {
                    'volume': {
                        'name': 'ssd0',
                        'hostPath': {
                          'path': '/mnt/disks/ssd0'
                        }
                    },
                    'volume_mount': {
                      'mountPath': '/mnt/ssd0',
                      'name': 'ssd0'
                    }
                }
            }
        ]
        
        for case in cases:
            v = LocalSSD(**case["kwargs"])
            self.assertEqual(object_as_dict(v),
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

    def test_schedules_on_gpu_node_pool(self):
      """Test that we can schedule a job into a GPU node pool."""

      skip = True

      class ReportGPU(Job):

        def __init__(self, *args, **kwargs):

          # Check that a GPU device is visible.
          command = ["nvidia-smi"]

          super(ReportGPU, self).__init__(
            command=command,
            *args, **kwargs)

      jid = gen_timestamped_uid()
      job = ReportGPU(**{
        "job_name": jid,
        "image": "tensorflow/tensorflow:nightly-gpu",
        "node_selector": {
          "gpuspernode": 1,
          "highmem": "true",
          "preemptible": "true",
          "type": "train",
          "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
        },
        "resources": Resources(limits={"nvidia.com/gpu": 1})
      })
      
      # Note: It takes ~5M for the NVIDIA drivers to auto-install after scale-up
      # so if the time-out is longer than that this will fail under that condition.

      if not skip:
        job.run()
        # TODO: Pull logs for job and verify they contain "Tesla K80"
        # TODO: Check that it scheduled on the correct node pool

    def test_local_ssd(self):
      """Test that num_local_ssd leads to local SSD mount."""
      
      skip = True
  
      command = ["ls", "/mnt"]

      jid = gen_timestamped_uid()
      job = Job(**{
        "job_name": jid,
        "image": "ubuntu",
        "num_local_ssd": 1,
        "command": command
      })

      if not skip:
        job.run()
      # TODO: Collect output and check, should be "ssd0"

    def test_complex_command(self):

      skip = False
  
      command = [
            "echo", "'", "pip", "install", "-e", "foo", "'"
            ">", "job.sh",
            "&&",
            "echo", "'", "python", "%s/tk/experiment.py" % "foo",
            "--t2t_usr_dir", "%s/tk" % "foo", "'"
            ">>", "job.sh",
            "&&",
            "cat", "job.sh"
      ]

      ["sh", "/app/job.sh"]

      jid = gen_timestamped_uid()
      job = Job(**{
        "job_name": jid,
        "image": "ubuntu",
        "command": command
      })

      if not skip:
        job.run()

    def test_node_grouping(self):
      """Test that we can create a few jobs and co-locate them.
      
      We will designate one job the "master", with that designated
      in its meta field, with other jobs given pod affinity for this
      attribute. All nodes will have node affinity for some other
      attribute.
      
      Notes:
      
      * One issue with this is that we would need to avoid jobs
        being scheduled onto the same node for some reason other
        than our configuration.
      * One component of the strategy is to collect hostnames
        when jobs run and check that all of these are the same.

      """
      
      skip = True

      class ReportHostname(Job):

        def __init__(self, *args, **kwargs):

            command = ["hostname"]

            super(ReportHostname, self).__init__(
              command=command,
              *args, **kwargs)

      hostnames = []
      
      master_id = gen_timestamped_uid()
      
      master_job = ReportHostname(**{
        "job_name": "foo-%s" % master_id,
        "image": "ubuntu",
        "additional_metadata": {
          "master_id": [master_id]
        }
      })

      if not skip:
        mater_job.run()

      for i in range(0, 10):
        worker_job = ReportHostname(**{
          "job_name": "foo-%i" % i,
          "image": "ubuntu",
          "pod_affinity": {
            "master_id": [master_id]
          }
        })

        if not skip:
          worker_job.run()


class TestTFJob(unittest.TestCase):

    def test_instantiate_tfjob_replica(self):
      """Test that a TFJobReplica model can be instantiated."""
      
      job_name = gen_timestamped_uid()

      replica = TFJobReplica(
        replica_type="MASTER",
        num_replicas=1,
        args="pwd",
        image="tensorflow/tensorflow:nightly-gpu",
        resources=Resources(
          requests={
            "cpu": 7,
            "memory": "16Gi",
            "nvidia.com/gpu": 1
          }),
        attached_volumes=[AttachedVolume("nfs-1"),
                          LocalSSD()],
        additional_metadata={
          "master_name": job_name
        },
        node_selector={
          "gpuspernode": 8,
          "highmem": "true",
          "preemptible": "true"
        },
        pod_affinity={"master_name": [job_name]}
      )

    def test_instantiate_tfjob(self):
        """Test that a local TFJob model can be instantiated."""

        """
        Issues:
        - Isn't running on 8-GPU node, probably because node_selector
          handling is not yet implemented.
        - That may explain why the nvidia setup driver does not seem to have
          run?
        - Hmm... could just go ahead with trying some training which would
          involve modifying the command and re-building the container.
        """
        
        skip = True
        
        # Should be GPU image
        #image = "tensorflow/tensorflow:nightly-gpu"
        image = "gcr.io/kubeflow-rl/base:0.0.9"
        train_resources = Resources(requests={
                             "cpu": 1,
                             "memory": "4Gi",
                             "nvidia.com/gpu": 1
                         })

        command = [
          "nvidia-smi"
        ]

        job_name = gen_timestamped_uid()

        # Pods of the TFJob will run in a node pool that matches these
        # labels.
        job_node_selector = {
          "gpuspernode": 2,
          "highmem": "true",
          "preemptible": "true",
          "type": "train",
          "cloud.google.com/gke-accelerator": "nvidia-tesla-k80"
        }
        
        # A PVC with this ID must exist within the namespace of the job
        pvc_id = "nfs-east1-d"

        replicas = [
          TFJobReplica(replica_type="MASTER",
                       num_replicas=1,
                       args=command,
                       image=image,
                       resources=train_resources,
                       attached_volumes=[AttachedVolume(pvc_id),
                                         LocalSSD()],
                       additional_metadata={
                         "master_name": job_name
                       },
                       node_selector=job_node_selector
                      ),
          TFJobReplica(replica_type="WORKER",
                       num_replicas=1,
                       args=command,
                       image=image,
                       resources=train_resources,
                       attached_volumes=[AttachedVolume(pvc_id),
                                         LocalSSD()],
                       # Worker pods have affinity to be co-located on nodes
                       # with the master pod of this TFJob
                       #pod_affinity={"master_name": [job_name]},
                       node_selector=job_node_selector
                      )
        ]

        """
        TFJobReplica(replica_type="PS",
                     num_replicas=1,
                     args=command,
                     image=image,
                     #resources=Resources(requests={
                     #    "cpu": 8,
                     #    "memory": "8Gi"
                     #}),
                     resources=train_resources,
                     attached_volumes=[AttachedVolume(pvc_id),
                                       LocalSSD()],
                     # PS pods have affinity to be co-located on nodes
                     # with the master pod of this TFJob
                     #pod_affinity={"master_name": [job_name]},
                     node_selector=job_node_selector
                    )
        """

        tfjob = TFJob(command=command,
                      job_name=gen_timestamped_uid(),
                      namespace="kubeflow",
                      replicas=replicas)

        if not skip:
          results = tfjob.batch_run()
          self.assertEqual(results.get("status", {}).get("state", {}), "Succeeded")

          expected = {

          }

          # TODO: Once have one of these that works, put that here.
          #self.assertEqual(tfjob.as_dict(),
          #                 expected)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()