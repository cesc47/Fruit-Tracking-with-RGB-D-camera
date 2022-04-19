"""
# Project: Fruit Size Estimation
# Author: Juan Carlos Miranda
# Date: November 2021
# Description:
  Test for methods used for in extraction from Matroska files
  Iterate over a group of recorded frames in a video.

Documentation in https://docs.python.org/3/library/unittest.html

Usage:

python -m unittest $HOME/development/KA_detector/video_extraction_management/test/test_ka_real_time_video_extraction.py

"""
import unittest
import os
from os.path import expanduser
from video_extraction_management.frame_extraction_config import FramesManagerConfig
from video_extraction_management.ka_video_loop_app_2 import KALoopVideoApp


class TestRealTimeVideoExtraction(unittest.TestCase):

    def setUp(self):
        print(type(self).__name__)
        self.root_folder = expanduser("~")
        self.an_input_file = os.path.join(self.root_folder, 'recorded_video', 'motion_recording',
                                          '20210928_114406_k_r2_e_015_175_162.mkv')
        self.offset_in_seconds = 82  # seconds to start
        self.number_of_frames = 300  # number of frames to extract
        self.frames_extractor_config_obj = FramesManagerConfig()
        self.frames_extractor_obj = KALoopVideoApp(self.frames_extractor_config_obj, self.an_input_file)

    def test_video_real_time(self):
        print(self.test_video_real_time.__name__)
        [frames_checked, errors, output_folder] = self.frames_extractor_obj.go_through_frames(self.offset_in_seconds, self.number_of_frames)
        print('frames_checked ->', frames_checked)
        print('errors ->', errors)
        print('output_folder ->', output_folder)
        self.assertEqual(self.number_of_frames, frames_checked)

    def test_video_object_detection_real_time(self):
        print(self.test_video_object_detection_real_time.__name__)
        [frames_checked, errors, output_folder] = self.frames_extractor_obj.go_through_object_detection_frames_2(self.offset_in_seconds, self.number_of_frames)

        #print('frames_checked ->', frames_checked)
        #print('errors ->', errors)
        #print('output_folder ->', output_folder)
        self.assertEqual(self.number_of_frames, frames_checked)


if __name__ == '__main__':
    unittest.main()

