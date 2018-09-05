# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#FROM gcr.io/kubeflow-images-public/tensorflow-1.7.0-notebook-gpu:v20180419-0ad94c4e
FROM gcr.io/kubeflow-rl/horovod:0.0.1

USER root

RUN apt-get update && apt-get install -y python-tk

ADD tools/install_cuda.sh /app/
RUN sh /app/install_cuda.sh

ADD tools/cudnn.deb /app/
RUN dpkg -i /app/cudnn.deb

ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt

RUN pip install -e /app/vendor/tensor2tensor[allen]
