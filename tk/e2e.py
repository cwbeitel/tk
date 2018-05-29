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

"""End to end download, datagen, training, and conditional deployment."""

import argparse
import json
import logging

from kube import Job

from jobs import DownloadJob, T2TDatagenJob, T2TExperiment, InferenceJob


class E2EJob(Job):

    def __init__(self, *args, **kwargs):

        if "app_root" not in kwargs:
            raise ValueError("app_root expected in kwargs, "
                             "saw %s" % kwargs)
        
        command = ["python", "%s/tk/e2e.py" % kwargs["app_root"],
                  "--job_config=%s" % json.dumps(kwargs)]
        
        super(E2EJob, self).__init__(command=command,
                                     *args, **kwargs)


def _fetch_kube_credentials_via_gcloud():
    
    from util import run_and_output
    
    logging.info(run_and_output([
        "curl", "-LO",
        "https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz"]))

    logging.info(run_and_output([
        "tar", "xzvf", "/app/google-cloud-sdk.tar.gz", "-C", "/app"
    ]))
    
    logging.info(run_and_output([
        "rm", "/app/google-cloud-sdk.tar.gz"
    ]))
    
    logging.info(run_and_output([
        "sh", "/app/google-cloud-sdk/install.sh",
        "--disable-installation-options",
        "--bash-completion=false",
        "--path-update=false",
        "--usage-reporting=false"
    ]))
    
    logging.info(run_and_output([
        "ln", "-sf", "/app/google-cloud-sdk/bin/gcloud",
        "/usr/local/bin/gcloud"
    ]))

    logging.info(run_and_output([
        "gcloud", "container", "clusters",
        "get-credentials", "kubeflow",
        "--zone", "us-east1-d"
    ]))  


def main(job_config):

    logging.getLogger().setLevel(logging.INFO)
    
    logging.info("Running e2e workflow with job config: %s" % job_config)

    # Launch and wait for container build job
    # TODO: Not adding this yet
    
    logging.info("type: %s" % type(job_config))
    
    # Hackity double hack
    # Right way is to mount a volume with credentials via kube secret
    if job_config["batch"]:
        _fetch_kube_credentials_via_gcloud()
        # triple hack...
    
    global_jid = job_config["job_name"]
    
    # HACK: give the job a unique ID
    job_config["job_name"] = "%s-%s" % (global_jid, 1)
    # Launch and wait for download job(s)
    download_job = DownloadJob(**job_config)
    download_job.run()

    # HACK: give the job a unique ID
    job_config["job_name"] = "%s-%s" % (global_jid, 2)
    # Launch and wait for datagen job(s)
    datagen_job = T2TDatagenJob(**job_config)
    datagen_job.run()
    
    # HACK: give the job a unique ID
    job_config["job_name"] = "%s-%s" % (global_jid, 3)
    # Launch and wait for training jobs
    train_job = T2TExperiment(**job_config)
    train_job.run()
    
    job_config["job_name"] = "%s-%s" % (global_jid, 4)
    # Launch and wait for training jobs
    job_config["data_dir"] = job_config["decode_data_dir"]
    train_job = InferenceJob(**job_config)
    train_job.run()

    # Maybe deploy a new version
    # TODO
    

def _load_job_config_from_arg(raw_arg_string):
    
    if not isinstance(raw_arg_string, str):
        raise ValueError("Expected job config arg string to be of "
                         "type str")
    
    output = {}
    
    job_config = json.loads(raw_arg_string)

    for key, value in job_config.items():
        if isinstance(key, unicode):
            key = key.encode("utf-8")
        if isinstance(value, unicode):
            value = value.encode("utf-8")
        output[key] = value
        
    return output

        
if __name__ == "__main__":
    
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_config", dest="job_config",
                        help="A JSON string job config.")

    args, _ = parser.parse_known_args()
    
    job_config = _load_job_config_from_arg(args.job_config)
    
    logging.info("Parsed job config from args: %s" % job_config)
    
    main(job_config)
    