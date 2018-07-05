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

import io
import os
import sys

from setuptools import find_packages, setup

NAME = 'tk'
DESCRIPTION = 'TK is a kit of tools in kit form.'
URL = ''
REQUIRED = [
    "IPython",
    "tensorflow",
    #"tensorflow-gpu",
    "kubernetes==5.0.0",
    "allensdk==0.14.4",
    "Pillow"
]

setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: Apache 2.0 License',
    ],
)
