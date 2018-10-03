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

"""Tests of modified similarity transformer model.

TODO: Currently this test assumes data_dir is /mnt/nfs-east1-d/data

python -m tk.models.similarity_transformer_export_test

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import atexit
import subprocess
import socket

import grpc
import tensorflow as tf

from tensor2tensor.utils import registry
#from tensor2tensor.serving import serving_utils
from tensor2tensor.serving import export
from tensor2tensor.utils import decoding
from tensor2tensor.utils import usr_dir
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib

from tk.models import similarity_transformer

FLAGS = tf.flags.FLAGS


def encode(input_str, output_str=None, encoders=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}


def _get_t2t_usr_dir():
  """Get the path to the t2t usr dir."""
  return os.path.join(os.path.realpath(__file__), "../")


# ---
# various things copied from
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/tensorflow_model_server_test.py
# b/c doesn't look like it's importable form tensorflow_serving


def PickUnusedPort():
  s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
  s.bind(('', 0))
  port = s.getsockname()[1]
  s.close()
  return port


def WaitForServerReady(port):
  """Waits for a server on the localhost to become ready."""

  # HACK
  sleep(5)

  """
  # TODO: I don't know why these imports don't work. They are can be imported
  # in tensor2tensor.serving.serving_utils so why not here? ...
  from tensorflow_serving.apis import predict_pb2
  #from tensorflow_serving.apis import prediction_service_pb2_grpc

  for _ in range(0, WAIT_FOR_SERVER_READY_INT_SECS):
    time.sleep(1)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'intentionally_missing_model'

    try:
      # Send empty request to missing model
      channel = grpc.insecure_channel('localhost:{}'.format(port))
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      stub.Predict(request, RPC_TIMEOUT)
    except grpc.RpcError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details():
        print('Server is ready')
        break
  """

class TensorflowModelServer(object):

  @staticmethod
  def RunServer(model_name,
                model_path,
                model_config_file=None,
                monitoring_config_file=None,
                batching_parameters_file=None,
                grpc_channel_arguments='',
                wait_for_server_ready=True,
                pipe=None):
    """Run tensorflow_model_server using test config.
    A unique instance of server is started for each set of arguments.
    If called with same arguments, handle to an existing server is
    returned.
    Args:
      model_name: Name of model.
      model_path: Path to model.
      model_config_file: Path to model config file.
      monitoring_config_file: Path to the monitoring config file.
      batching_parameters_file: Path to batching parameters.
      grpc_channel_arguments: Custom gRPC args for server.
      wait_for_server_ready: Wait for gRPC port to be ready.
      pipe: subpipe.PIPE object to read stderr from server.
    Returns:
      3-tuple (<Popen object>, <grpc host:port>, <rest host:port>).
    Raises:
      ValueError: when both model_path and config_file is empty.
    """
    args_key = TensorflowModelServerTest.GetArgsKey(**locals())
    if args_key in TensorflowModelServerTest.model_servers_dict:
      return TensorflowModelServerTest.model_servers_dict[args_key]
    port = PickUnusedPort()
    rest_api_port = PickUnusedPort()
    print('Starting test server on port: {} for model_name: '
          '{}/model_config_file: {}'.format(port, model_name,
                                            model_config_file))
    command = os.path.join(
        TensorflowModelServerTest.__TestSrcDirPath('model_servers'),
        'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --rest_api_port=' + str(rest_api_port)
    command += ' --rest_api_timeout_in_ms=' + str(HTTP_REST_TIMEOUT_MS)

    if model_config_file:
      command += ' --model_config_file=' + model_config_file
    elif model_path:
      command += ' --model_name=' + model_name
      command += ' --model_base_path=' + model_path
    else:
      raise ValueError('Both model_config_file and model_path cannot be empty!')

    if monitoring_config_file:
      command += ' --monitoring_config_file=' + monitoring_config_file

    if batching_parameters_file:
      command += ' --enable_batching'
      command += ' --batching_parameters_file=' + batching_parameters_file
    if grpc_channel_arguments:
      command += ' --grpc_channel_arguments=' + grpc_channel_arguments
    print(command)
    proc = subprocess.Popen(shlex.split(command), stderr=pipe)
    atexit.register(proc.kill)
    print('Server started')
    if wait_for_server_ready:
      WaitForServerReady(port)

    hostports = (
        proc,
        'localhost:' + str(port),
        'localhost:' + str(rest_api_port),
    )

    return hostports


class TestSimilarityTransformerExport(tf.test.TestCase):

  def test_e2e_export_and_query(self):
    """Test that we can export and query the model via tf.serving."""

    FLAGS.t2t_usr_dir = _get_t2t_usr_dir()
    FLAGS.problem = "github_function_docstring"
    FLAGS.data_dir = "/mnt/nfs-east1-d/data"
    FLAGS.tmp_dir = "/mnt/nfs-east1-d/tmp"
    FLAGS.output_dir = tempfile.mkdtemp()
    #FLAGS.export_dir = os.path.join(FLAGS.output_dir, "export")
    FLAGS.model = "similarity_transformer_dev"
    FLAGS.hparams_set = "similarity_transformer_tiny"
    #FLAGS.timeout_secs = 10
    FLAGS.train_steps = 1
    FLAGS.schedule = "train"
    
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    
    t2t_trainer.main(None)

    export.main(None)
    
    # ----
    # Start model server
    
    # Will start a tf model server on an un-used port and
    # kill process on exit.
    _, server, _ = TensorflowModelServer().RunServer(
        model,
        output_dir
    )

    # ----
    # Query the server
    
    doc_query = [1,2,3] # Dummy encoded doc query
    code_query = [1,2,3] # Dummy encoded code query

    # Alternatively for query, without going through query.main()
    # TODO: Is servable_name the same as model name?
    request_fn = serving_utils.make_grpc_request_fn(
        servable_name=model,
        server=server,
        timeout_secs=timeout_secs)

    # Compute embeddings
    # TODO: May need to customize how these queries are fed in, potentially
    #       side-stepping serving_utils.predict.
    encoded_string = serving_utils.predict([doc_query], problem_object, request_fn)
    encoded_code = serving_utils.predict([code_query], problem_object, request_fn)
    
    # TODO: Make an assertion about the result.


if __name__ == "__main__":
  tf.test.main()