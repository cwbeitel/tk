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
import pprint
import time
import datetime

import kubernetes

from tk.kube import AttachedVolume, Container
import util


def get_client(which_api):

    kubernetes.config.load_kube_config()
    
    allowed_apis = [
        "BatchV1Api",
        "AppsV1beta1Api",
        "CoreV1Api"
    ]
    if which_api not in allowed_apis:
        raise ValueError("Specified api %s not in allowed apis %s" % (
            which_api, allowed_apis
        ))

    client = getattr(kubernetes.client, which_api)()
    
    return client


class TensorBoardService(object):
    """Python model of a TensorBoard Service object."""
    
    def __init__(self,
                 service_name,
                 deployment_name,
                 namespace="default",
                 api_version=None,
                 kind=None,
                 spec=None,
                 *args, **kwargs):
            
        self.apiVersion = api_version if api_version is not None else "v1"
        self.kind = kind if kind is not None else "Service"
        self.metadata = {
            "name": service_name,
            "namespace": namespace
        }
        
        self.spec = spec if spec is not None else {
            "ports": [{
                "name": "http",
                "port": 80,
                "targetPort": 80,
            }],
            "selector": {
                "app": "tensorboard",
                "tb-job": deployment_name
            }
        }
    
    def create(self):
        
        client = get_client("CoreV1Api")
        
        cfg = util.object_as_dict(self)
        
        logging.info("Creating object with config: %s" % cfg)
        
        response = client.create_namespaced_service(
            cfg["metadata"]["namespace"], cfg
        )
        
        pprint.pprint(response)

    def ready(self):
        
        client = get_client("CoreV1Api")
        
        cfg = util.object_as_dict(self)
        
        response = client.read_namespaced_service(
            cfg["metadata"]["name"], cfg["metadata"]["namespace"]
        )
        
        return response.status is not None

    def delete(self):
        client = get_client("CoreV1Api")
        cfg = util.object_as_dict(self)
        return client.delete_namespaced_service(
            cfg["metadata"]["name"], cfg["metadata"]["namespace"]
        )


class TensorBoardDeployment(object):
    
    def __init__(self,
                 log_dir,
                 deployment_name,
                 image="tensorflow/tensorflow:1.7.0",
                 namespace="default",
                 kind="Deployment",
                 spec=None,
                 volume_claim_id=None,
                 *args, **kwargs):

        self.apiVersion = "apps/v1beta1"
        self.kind = "Deployment"
        self.metadata = {
            "name": deployment_name,
            "namespace": namespace
        }

        command = [
            "/usr/local/bin/tensorboard",
            "--logdir=" + log_dir,
            "--port=80"
        ]

        container_args = {
            "command": command,
            "image": image,
            "name": "tensorboard",
            "ports": [{"containerPort": 80}],
        }

        attached_volume = None
        if volume_claim_id is not None:
          attached_volume = AttachedVolume(volume_claim_id)
          container_args["attached_volumes"] = [
            attached_volume
          ]

        container = Container(**container_args)

        self.spec = spec if spec is not None else {
            "replicas": 1, 
            "template": {
              "metadata": {
                "labels": {
                  "app": "tensorboard",
                  "tb-job": deployment_name,
                }, 
                "name": deployment_name,
                "namespace": namespace,
              }, 
              "spec": {
                "containers": [container]
              }
            }
        }
        
        if spec is None:
            if attached_volume is not None:
                self.spec["template"]["spec"]["volumes"] = [
                    attached_volume.volume
                ]
            
    def create(self):
        
        client = get_client("AppsV1beta1Api")
        
        cfg = util.object_as_dict(self)
        
        logging.info("Creating object with config: %s" % cfg)
        
        response = client.create_namespaced_deployment(
            cfg["metadata"]["namespace"], cfg
        )
        
        pprint.pprint(response)
        
    def ready(self):
        
        client = get_client("AppsV1beta1Api")
        
        cfg = util.object_as_dict(self)
        
        response = client.read_namespaced_deployment(
            cfg["metadata"]["name"], cfg["metadata"]["namespace"]
        )
        
        return hasattr(response, "status") and (
            response.status.ready_replicas >= 1
        )

    def delete(self):
        client = get_client("AppsV1beta1Api")
        cfg = util.object_as_dict(self)
        return client.delete_namespaced_deployment(
            cfg["metadata"]["name"], cfg["metadata"]["namespace"],
            kubernetes.client.V1DeleteOptions()
        )


class TensorBoard(object):
    
    def __init__(self, log_dir, namespace, job_name_base=None,
                 volume_claim_id=None):
    
        self.log_dir = log_dir
        self.namespace = namespace        

        if job_name_base is None:
            job_name_base = util.generate_job_name("tb")
        
        self.deployment_name = "%s-depl" % job_name_base
        self.service_name = "%s-svc" % job_name_base
        
        self.components = {
            "deployment": TensorBoardDeployment(
                log_dir=self.log_dir,
                deployment_name=self.deployment_name,
                namespace=self.namespace,
                volume_claim_id=volume_claim_id
            ),
            "service": TensorBoardService(
                service_name=self.service_name,
                namespace=self.namespace,
                deployment_name=self.deployment_name
            )
        }

    def create(self, poll_and_check=False):
        
        responses = []
        
        for component in self.components.values():
            responses.append(component.create())
        
        if not poll_and_check:
            return responses
      
        timeout = datetime.timedelta(seconds=(24*60*60))
        polling_interval = datetime.timedelta(seconds=5)
        end_time = datetime.datetime.now() + timeout
        
        while True:
            
            if datetime.datetime.now() + polling_interval > end_time:
                raise TimeoutError(
                    "Timeout waiting for tensorboard deployment "
                    "(%s) and service (%s) " % (self.deployment_name,
                                                self.service_name),
                    "in namespace %s to finish." % self.namespace)
                
            time.sleep(polling_interval.seconds)
            
            for component in self.components.values():
                if not component.ready():
                    continue

            return responses
       
    def delete(self):
        
        for component in self.components.values():
            component.delete()
