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

import os

from tk.util import hack_dict_to_cli_args
from tk.kube import Job, TFJob, TFJobReplica, Resources, AttachedVolume


class InferenceJob(TFJob):
    """A Job that runs a training experiment.
    
    TODO: This isn't working for decoding images. For now use trainer in eval mode?
    
    """
    
    def __init__(self, app_root, image,
                 num_master_replicas=1,
                 num_ps_replicas=0,
                 cpu=1,
                 memory="1Gi",
                 *args, **kwargs):
        
        decode_output_file_path = os.path.join(app_root, "output.txt")
        
        if isinstance(num_master_replicas, str):
            num_master_replicas = int(num_master_replicas)
        
        if (not isinstance(num_master_replicas, int) or num_master_replicas <= 0):
            raise ValueError("The number of master replicas must be an "
                             "integer greater than zero.")
        
        if (not isinstance(num_ps_replicas, int) or 
            num_ps_replicas < 0):
            raise ValueError("The number of ps replicas must be an "
                             "integer greater than or equal to zero.")
        
        command = [
            "python", "%s/lib/t2t/tensor2tensor/bin/t2t-decoder" % app_root,
            "--t2t_usr_dir", "%s/tk" % app_root,
            "--decode_to_file", decode_output_file_path,
        ]
        command.extend(hack_dict_to_cli_args(kwargs))
        
        replicas = [
            TFJobReplica(replica_type="MASTER",
                         num_replicas=num_master_replicas,
                         args=command,
                         image=image,
                         resources=Resources(requests={
                             "cpu": cpu,
                             "memory": memory
                         }),
                         attached_volume=AttachedVolume("nfs-1"))
        ]
        
        super(InferenceJob, self).__init__(command=command,
                                           replicas=replicas,
                                           *args, **kwargs)
        
