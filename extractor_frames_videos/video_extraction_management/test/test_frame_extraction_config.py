"""
Project: Size Estimation
Author: Juan Carlos Miranda
Date: November 2021
Description:
Example taken from https://docs.python.org/3/library/unittest.html

...

Use: python -m unittest tests/test_something.py
"""
import unittest
import os
from video_extraction_management.frame_extraction_config import FramesManagerConfig


class TestFramesManagerConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("setUpClass(cls) -->")

    def test_FramesManagerConfigDefault(self):
        print('test_FramesManagerConfigDefault(self): -->')
        frames_extractor_config_obj = FramesManagerConfig()
        self.assertEqual(frames_extractor_config_obj.depth_data_filename, 'D')
        self.assertEqual(frames_extractor_config_obj.file_img_extesion, '.png')
        self.assertEqual(frames_extractor_config_obj.file_mat_extesion, '.mat')
        self.assertEqual(frames_extractor_config_obj.ir_data_filename, 'I')
        self.assertEqual(frames_extractor_config_obj.rgb_data_filename, 'C')
        #todo: add path_images_ouput, path_video_input, path_extractor_config_file

    def test_FramesManagerConfigLoadFile(self):
        print('test_FramesManagerConfigLoadFile(self): -->')
        BASE_DIR = os.path.abspath('')
        path_extractor_config_file = os.path.join(BASE_DIR, 'frames_extractor.conf')
        frames_extractor_config_obj = FramesManagerConfig(path_extractor_config_file)
        self.assertEqual(frames_extractor_config_obj.depth_data_filename, 'D')
        self.assertEqual(frames_extractor_config_obj.file_img_extesion, '.png')
        self.assertEqual(frames_extractor_config_obj.file_mat_extesion, '.mat')
        self.assertEqual(frames_extractor_config_obj.ir_data_filename, 'I')
        self.assertEqual(frames_extractor_config_obj.rgb_data_filename, 'C')
        #todo: add path_images_ouput, path_video_input, path_extractor_config_file

if __name__ == '__main__':
    unittest.main()