"""
# Project: Size Estimation
# Author: Juan Carlos Miranda
# Date: November 2021
# Description:
  Methods used for frame extraction from Matroska files

Usage:

"""
import unittest
import os
from os.path import expanduser
import numpy as np  # to convert array data
import cv2  # to show images
from video_extraction_management.frame_extraction_config import FramesManagerConfig
from video_extraction_management.frame_extraction import FramesVideoManager
from video_extraction_management.video_helpers.helpers import colorize


class TestFramesManager(unittest.TestCase):

    def setUp(self):
        print(type(self).__name__)
        self.root_folder = expanduser("~")
        self.a_video_input_file = os.path.join(self.root_folder, 'recorded_video',
                                               '20210927_121804_k_r2_e_000_150_138.mkv')
        self.offset_in_seconds = 1  # seconds to start
        self.number_of_frames = 1
        self.BASE_DIR = os.path.abspath('')
        self.output_folder = os.path.join(self.BASE_DIR, 'exported_images')
        self.a_track_file = os.path.join(self.output_folder, 'all.txt')

    def test_get_a_frame(self):
        print(self.test_get_a_frame.__name__)
        BASE_DIR = os.path.abspath('')
        output_folder = os.path.join(BASE_DIR, 'output_test')
        an_output_rgb = os.path.join(output_folder, 'result_rgb.png')
        an_output_transformed_depth_mat = os.path.join(output_folder, 'result_t_depth.mat')

        frames_extractor_config_obj = FramesManagerConfig()
        frames_extractor_obj = FramesVideoManager(frames_extractor_config_obj, self.a_video_input_file)

        [a_rgb_data, a_depth_data, a_nir_data] = frames_extractor_obj.get_a_frame(self.offset_in_seconds)
        cv2.imshow('depth', colorize(a_depth_data, (None, 5000)))
        cv2.waitKey()

        self.assertEqual(np.shape(a_rgb_data), (1080, 1920, 3))
        self.assertEqual(np.shape(a_depth_data), (1080, 1920))
        self.assertEqual(np.shape(a_nir_data), (1080, 1920))

    def test_export_frames_to_files(self):
        print(self.test_export_frames_to_files.__name__)

        frames_extractor_config_obj = FramesManagerConfig()
        frames_extractor_obj = FramesVideoManager(frames_extractor_config_obj, self.a_video_input_file)
        [frames_written, errors, output_folder] = frames_extractor_obj.export_frames_to_files(self.a_track_file,
                                                                                              self.offset_in_seconds,
                                                                                              self.number_of_frames)
        print(frames_written, errors, output_folder)
        self.assertEqual('OK', 'OK')


if __name__ == '__main__':
    unittest.main()
