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


class T2TDatagenJob(Job):
    """A Job that generates training examples from raw input data."""
    
    def __init__(self, app_root, *args, **kwargs):
        
        command = [
            "t2t-datagen"
        ]
        command.extend(hack_dict_to_cli_args(kwargs))
        
        super(T2TDatagenJob, self).__init__(command=command,
                                            *args, **kwargs)


    