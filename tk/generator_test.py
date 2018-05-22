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

import unittest
from generator import AllenBrainImage2image, _generator
from download import maybe_download_image_datasets, subimages_for_image_files
import tempfile
import logging
import os


class TestGenerator(unittest.TestCase):
    
    def test_null(self):
        
        tmpd = tempfile.mkdtemp()
        _generator(tmpd, 1)


class TestAllenBrainImg2img(unittest.TestCase):
    
    def test_runs(self):
        
        problem = AllenBrainImage2image()
        
        tmp_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        
        with tempfile.TemporaryDirectory() as tmp_dir:    
        
            maybe_download_image_datasets(tmp_dir,
                          section_offset=0,
                          num_sections=1,
                          images_per_section=1)

            subimages_for_image_files(tmp_dir)

            with tempfile.TemporaryDirectory() as data_dir:

                problem.generate_data(tmp_dir=tmp_dir,
                                      data_dir=data_dir)
                
                shards = os.listdir(data_dir)
                self.assertTrue(shards is not None)
                # TODO: maybe pull out example and check its shape
                

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()