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

"""Kubernetes models and utils supporting templating Job's and TFJob's

TODO: Should consider making use of existing Kubernetes python client object
models, didn't realize these existed.

"""

import datetime
import pprint
import kubernetes
import time
import logging

from tk.util import expect_type
from tk.util import expect_path
from tk.util import object_as_dict
from tk.util import run_and_output
from tk.util import dict_prune_private


# A common base image that extends kubeflow tensorflow_notebook workspace
# image with python dependencies needed for various examples.
# TODO: In the future include various additional deps in this base image
_COMMON_BASE = "gcr.io/kubeflow-rl/common-base:0.0.1"
_TF_JOB_GROUP = "kubeflow.org"
_TF_JOB_VERSION = "v1alpha1"
_TF_JOB_PLURAL = "tfjobs"


def gen_timestamped_uid():
    """Generate a timestamped UID of form MMDD-HHMM-UUUU"""
    now = datetime.datetime.now()
    return now.strftime("%m%d-%H%M") + "-" + uuid.uuid4().hex[0:4]


def build_command(base_command, **kwargs):
    """Build a command array extending a base command with -- form kwargs.
    
    E.g. [t2t-trainer, {key: value}] -> [t2t-trainer, --key=value]
    
    """
    expect_type(base_command, str)
    command = [base_command]
    for key, value in kwargs.items():
        expect_type(key, str)
        expect_type(value, str)
        command.append("--%s=%s" % (key, value))
    return command


class AttachedVolume(object):
    """Model of information needed to attach a Kubernetes Volume

    Primarily manages correpondence of volume and volume_mount fields and
    expects objects recieving `AttachedVolume` as argument to know whether to
    access `volume` or `volume_mount` fields.

    #TODO(cwbeitel): Potentially update name of class.

    """

    def __init__(self, claim_name, mount_path=None, volume_name=None,
                 volume_type="persistentVolumeClaim"):

        if not isinstance(claim_name, str):
            raise ValueError("Expected string claim_name, saw %s" % claim_name)

        if mount_path is None:
            mount_path = "/mnt/%s" % claim_name

        if not isinstance(mount_path, str):
            raise ValueError("Expected string mount_path, saw %s" % claim_name)

        if not mount_path.startswith("/"):
            raise ValueError("Mount path should start with '/', saw %s" % mount_path)

        if volume_name is None:
            volume_name = claim_name

        if not isinstance(volume_name, str):
            raise ValueError("Expected string volume_name, saw %s" % volume_name)

        self.volume = {
            "name": volume_name,
            "persistentVolumeClaim": {
                "claimName": claim_name
            }
        }
        self.volume_mount = {
            "name": volume_name,
            "mountPath": mount_path
        }


class LocalSSD(object):

  def __init__(self, disk_id=0):

    self.volume = {
      "name": "ssd%s" % disk_id,
      "hostPath": {
        "path": "/mnt/disks/ssd%s" % disk_id
      }
    }

    self.volume_mount = {
      "name": "ssd%s" % disk_id,
      "mountPath": "/mnt/ssd%s" % disk_id
    }


class Resources(object):
    """Model of Kuberentes Container resources"""

    def __init__(self, limits=None, requests=None):

        allowed_keys = ["cpu", "memory", "nvidia.com/gpu"]

        def raise_if_disallowed_key(key):
          if key not in allowed_keys:
            raise ValueError("Saw resource request or limit key %s "
                             "which is not in allowed keys %s" % (key,
                                                                  allowed_keys))

        if limits is not None:
            self.limits = {}
            for key, value in limits.items():
              raise_if_disallowed_key(key)
              self.limits[key] = value
        
        if requests is not None:
            self.requests = {}
            for key, value in requests.items():
              raise_if_disallowed_key(key)
              self.requests[key] = value


class Container(object):
    """Model of Kubernetes Container object."""

    def __init__(self, image, name=None, args=None, command=None,
                 resources=None, volume_mounts=None,
                 allow_nameless=False, ports=None):

        if args is not None:
            self.args = args

        if command is not None:
            self.command = command

        if ports is not None:
            if not isinstance(ports, list):
                raise ValueError("ports must be a list, saw %s" % ports)
            for port in ports:
                if not isinstance(port, dict):
                    raise ValueError("ports must be a list of dict.'s, saw %s" % ports)
            self.ports = ports

        self.image = image

        if name is not None:
            self.name = name
        elif not allow_nameless:
            raise ValueError("The `name` argument must be specified "
                             "unless `allow_nameless` is True.")

        if resources is not None:
            if not isinstance(resources, Resources):
                raise ValueError("non-null resources expected to be of "
                                 "type Resources, saw %s" % type(resources))
            self.resources = resources

        if volume_mounts is not None:
            if not isinstance(volume_mounts, list):
                raise ValueError("non-null volume_mounts expected to be of "
                                 "type list, saw %s" % type(volume_mounts))
            self.volumeMounts = volume_mounts


def job_status_callback(job_response):
    """A callback to use with wait_for_job."""
    logging.info("Job %s in namespace %s; uid=%s; succeeded=%s" % (
        job_response.metadata.name,
        job_response.metadata.namespace,
        job_response.metadata.uid,
        job_response.status.succeeded
    ))


# TODO: This defines success as there having been exactly one success.
def wait_for_job(batch_api,
                 namespace,
                 name,
                 timeout=datetime.timedelta(seconds=(24*60*60)),
                 polling_interval=datetime.timedelta(seconds=30),
                 status_callback=job_status_callback):
    # ported from https://github.com/kubeflow/tf-operator/blob/master/py/tf_job_client.py
    end_time = datetime.datetime.now() + timeout
    while True:
        response = batch_api.read_namespaced_job_status(name, namespace)

        if status_callback:
            status_callback(response)
    
        if response.status.succeeded == 1:
            return response

        if datetime.datetime.now() + polling_interval > end_time:
            raise TimeoutError(
                "Timeout waiting for job {0} in namespace {1} to finish.".format(
                name, namespace))

        time.sleep(polling_interval.seconds)

    # Linter complains if we don't have a return statement even though
    # this code is unreachable.
    return None
  

#TODO: Consider pip installing additional dependencies on job startup
class Job(object):
    """Python model of a Kubernetes Job object."""

    def __init__(self,
                 job_name,
                 command="",
                 image=_COMMON_BASE,
                 restart_policy="Never",
                 namespace="default",
                 volume_claim_id=None,
                 batch=True,
                 no_wait=False,
                 api_version=None,
                 kind=None,
                 spec=None,
                 smoke=False,
                 node_selector=None,
                 additional_metadata=None,
                 pod_affinity=None,
                 num_local_ssd=0,
                 resources=None,
                 *args, **kwargs):
        """Check args for and template a Job object.

        name (str): A unique string name for the job.
        image (str): The image within which to run the job command.
        restart_policy (str): The restart policy (e.g. Never, onFailure).
        namespace (str): The namespace within which to run the Job.
        volume_claim_id (str): The ID of a persistent volume to mount at
            /mnt/`volume_claim_id`.
        api_version (str): Allow an alternative API version to be specified.
        kind (str): Allow the job kind to be overridden by subclasses.
        spec (str): Allow the job spec to be specified explicitly.

        See: https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/

        """

        # Private attributes will be ignored when converting object to dict.
        self._command = command
        self._batch = batch
        self._poll_and_check = True
        
        if not isinstance(smoke, bool):
            raise ValueError("The type of `smoke` should be boolean, "
                             "saw %s" % smoke)
        self._smoke = smoke

        logging.info("smoke: %s" % self._smoke)

        if no_wait:
            self._poll_and_check = False

        kwargs = {
            "args": command,
            "image": image,
            "name": "container",
            "resources": resources,
        }

        volumes = []

        if volume_claim_id is not None:
          volumes.append(AttachedVolume(volume_claim_id))

        if num_local_ssd > 0:
          volumes.append(LocalSSD())
          
        logging.debug("volumes: %s" % volumes)

        volume_mounts_spec = [getattr(volume, "volume_mount") for volume in volumes]
        logging.debug("volume mounts spec: %s" % volume_mounts_spec)
        
        if len(volume_mounts_spec) > 0:
          kwargs["volume_mounts"] = volume_mounts_spec

        container = Container(**kwargs)

        self.apiVersion = api_version if api_version is not None else "batch/v1"
        self.kind = kind if kind is not None else "Job"
        self.metadata = {
            "name": job_name,
            "namespace": namespace
        }
        if additional_metadata is not None:
          if not isinstance(additional_metadata, dict):
            raise ValueError("additional_metadata must be of type dict, saw "
                             "%s" % additional_metadata)
          self.metadata.update(additional_metadata)

        self.spec = spec if spec is not None else {
                "template": {
                    "spec": {
                        "containers": [container],
                        "restartPolicy": restart_policy
                    }
                },
                "backoffLimit": 4
            }

        if spec is None:
          volumes_spec = [
            getattr(volume, "volume") for volume in volumes
          ]
          if len(volumes_spec) > 0:
            self.spec["template"]["spec"]["volumes"] = volumes_spec

        self.set_node_selector(node_selector)

        self.set_pod_affinity(pod_affinity)

    def set_node_selector(self, node_selector):

      if node_selector is None:
        return

      if not isinstance(node_selector, dict):
        raise ValueError("Non-None node_selector expected to have type dict, "
                         "saw %s" % node_selector)

      for key, value in node_selector.items():
        node_selector[key] = str(value)
      self.spec["template"]["spec"]["nodeSelector"] = node_selector

    def set_pod_affinity(self, pod_affinity):

      if pod_affinity is None:
        return
      
      if not isinstance(pod_affinity, dict):
        raise ValueError("Non-None pod_affinity expected to have type dict, "
                         "saw %s" % pod_affinity)

      affinity_key = pod_affinity.keys()[0]
      affinity_values = pod_affinity[affinity_key]
      if not isinstance(affinity_values, list):
        raise ValueError("For now expecting that pod_affinity is a dict with a single "
                         "key into a list of values, saw %s" % pod_affinity)

      self.spec["template"]["spec"]["affinity"] = {
        "podAffinity":{
          "requiredDuringSchedulingIgnoredDuringExecution": [
            {
              "labelSelector": {
                "matchExpressions": [
                  {
                    "key": affinity_key,
                    "operator": "In",
                    "values": affinity_values
                  }
                ]
              }
            },
            {
              "topologyKey": "kubernetes.io/hostname"
            }
          ]
        }
      }

    def run(self):
        
        if self._smoke:
            cmd = ["echo"]
            cmd.extend(self._command)
            self._command = cmd

        if self._batch:
            self.batch_run(poll_and_check=self._poll_and_check)
        else:
            self.local_run()

    def as_dict(self):
        return dict_prune_private(object_as_dict(self))

    def batch_run(self, poll_and_check=False):
        
        kubernetes.config.load_kube_config()
        
        job_client = kubernetes.client.BatchV1Api()

        job_dict = self.as_dict()

        logging.info("Triggering batch run with job config: %s" % job_dict)
        
        response = job_client.create_namespaced_job(
            job_dict["metadata"]["namespace"],
            job_dict
        )
        
        pprint.pprint(response)
        
        if poll_and_check:
            # Poll for job completion and check status, raising exception
            # if not successful
            # TODO
            wait_for_job(job_client,
                         job_dict["metadata"]["namespace"],
                         job_dict["metadata"]["name"])
            
    def local_run(self, show=True):
        """Run the job command locally."""
        
        logging.info("Triggering local run.")
        
        output = run_and_output(self._command)


class TFJobReplica(object):
    """Python model of a kubeflow.org TFJobReplica object."""

    def __init__(self, replica_type, num_replicas, args, image,
                 resources=None,
                 attached_volumes=None,
                 restart_policy="OnFailure",
                 additional_metadata=None,
                 pod_affinity=None,
                 node_selector=None):

        self.replicas = num_replicas

        # HACK
        if attached_volumes == None:
          attached_volumes = []

        volume_mounts = [
          getattr(volume, "volume_mount") for volume in attached_volumes
        ]
        if len(volume_mounts) == 0:
          volume_mounts = None

        self.template = {
            "spec": {
                "containers": [
                    Container(args=args,
                              image=image,
                              name="tensorflow",
                              resources=resources,
                              volume_mounts=volume_mounts)
                ],
                "restartPolicy": restart_policy,
            }
        }

        self.tfReplicaType = replica_type

        attached_volume_spec = []
        for volume in attached_volumes:
          #if not isinstance(volume, AttachedVolume) and not isinstance(volume, LocalSSD):
          #  raise ValueError("attached_volumes attribute must be a list of "
          #                   "AttachedVolume or LocalSSD objects, saw %s" % volume)
          attached_volume_spec.append(volume.volume)

        if len(attached_volume_spec) > 0:
          self.template["spec"]["volumes"] = attached_volume_spec

        self.set_node_selector(node_selector)

        self.set_pod_affinity(pod_affinity)

    def set_node_selector(self, node_selector):

      if node_selector is None:
        return

      if not isinstance(node_selector, dict):
        raise ValueError("Non-None node_selector expected to have type dict, "
                         "saw %s" % node_selector)

      for key, value in node_selector.items():
        node_selector[key] = str(value)
      self.template["spec"]["nodeSelector"] = node_selector

    def set_pod_affinity(self, pod_affinity):

      if pod_affinity is None:
        return
      
      if not isinstance(pod_affinity, dict):
        raise ValueError("Non-None pod_affinity expected to have type dict, "
                         "saw %s" % pod_affinity)

      affinity_key = pod_affinity.keys()[0]
      affinity_values = pod_affinity[affinity_key]
      if not isinstance(affinity_values, list):
        raise ValueError("For now expecting that pod_affinity is a dict with a single "
                         "key into a list of values, saw %s" % pod_affinity)

      self.template["spec"]["affinity"] = {
        "podAffinity":{
          "requiredDuringSchedulingIgnoredDuringExecution": [
            {
              "labelSelector": {
                "matchExpressions": [
                  {
                    "key": affinity_key,
                    "operator": "In",
                    "values": affinity_values
                  }
                ]
              }
            },
            {
              "topologyKey": "kubernetes.io/hostname"
            }
          ]
        }
      }

# TODO: Either figure out how best to make use of existing tf_job_client in 
# https://github.com/kubeflow/tf-operator/blob/master/py/tf_job_client.py
# or use the two following, which are duplicated from there
def log_status(tf_job):
  """A callback to use with wait_for_job."""
  logging.info("Job %s in namespace %s; uid=%s; phase=%s, state=%s,",
               tf_job.get("metadata", {}).get("name"),
               tf_job.get("metadata", {}).get("namespace"),
               tf_job.get("metadata", {}).get("uid"),
               tf_job.get("status", {}).get("phase"),
               tf_job.get("status", {}).get("state"))


# Again mostly duplicated from elsewhere, see above.
def wait_for_tfjob(crd_api,
                   namespace,
                   name,
                   timeout=datetime.timedelta(minutes=10),
                   polling_interval=datetime.timedelta(seconds=30),
                   status_callback=log_status):
  """Wait for the specified job to finish.
  Args:
    client: K8s api client.
    namespace: namespace for the job.
    name: Name of the job.
    timeout: How long to wait for the job.
    polling_interval: How often to poll for the status of the job.
    status_callback: (Optional): Callable. If supplied this callable is
      invoked after we poll the job. Callable takes a single argument which
      is the job.
  """
  
  end_time = datetime.datetime.now() + timeout
  results=None
  while True:
    results = crd_api.get_namespaced_custom_object(
      _TF_JOB_GROUP, _TF_JOB_VERSION, namespace, _TF_JOB_PLURAL, name)

    if status_callback:
      status_callback(results)

    # If we poll the CRD quick enough status won't have been set yet.
    if results.get("status", {}).get("phase", {}) == "Done":
      return results

    if datetime.datetime.now() + polling_interval > end_time:
      raise util.TimeoutError(
        "Timeout waiting for job {0} in namespace {1} to finish.".format(
          name, namespace))

    time.sleep(polling_interval.seconds)

  # Linter complains if we don't have a return statement even though
  # this code is unreachable.
  return results


#TODO: So obviously this object is highly similar to the Job object.
class TFJob(Job):
    """Python model of a kubeflow.org TFJob object"""
    
    def __init__(self, replicas, *args, **kwargs):
        # Camel case to be able to conveniently display in kube-compatible
        # version with self.__dict__

        spec = {"replicaSpecs": []}
        for replica in replicas:
            spec["replicaSpecs"].append(replica.__dict__)
            
        super(TFJob, self).__init__(
            api_version="%s/%s" % (_TF_JOB_GROUP, _TF_JOB_VERSION),
            kind="TFJob",
            spec=spec,
            *args, **kwargs)

    def batch_run(self, poll_and_check=True):
        """Override Job.batch_run to run TFJob in batch via CRD api."""

        kubernetes.config.load_kube_config()
        
        crd_client = kubernetes.client.CustomObjectsApi()
        
        job_dict = self.as_dict()
        
        logging.debug("Running TFJob with name %s..." % job_dict["metadata"]["name"])
        
        response = crd_client.create_namespaced_custom_object(
            _TF_JOB_GROUP, _TF_JOB_VERSION,
            job_dict["metadata"]["namespace"],
            _TF_JOB_PLURAL, body=job_dict)
        
        # TODO: Only print this if --debug, not sure how to logging.debug with
        # pprint formatting
        #pprint.pprint(response)
        
        if poll_and_check:
            return wait_for_tfjob(crd_client,
                                  job_dict["metadata"]["namespace"],
                                  job_dict["metadata"]["name"])
    