"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:
    Test file name nomenclature

Use:

"""

import unittest
from dataset_management.filename_decoder import FileNameDecoder


class TestFileNameDecoder(unittest.TestCase):

    def setUp(self):
        # CONFIGURE HERE PATHS TO A TEST DATASET
        self.string_to_decode = '20210927_115932_k_r2_e_000_150_138.mkv'
        self.exp_filename = '20210927_115932_k_r2_e_000_150_138'
        self.exp_date_record = '20210927'
        self.exp_time_record = '115932'
        self.exp_time_str_record = '11:59:32'
        self.exp_sensor_record = 'k'
        self.exp_row_tree = 'r2'
        self.exp_cardinal_points = 'e'
        self.exp_speed_record = '000'
        self.exp_row_center_distance = '150'
        self.exp_height_camera_distance = '138'

    def test_filename_decode(self):
        dataset_filename = FileNameDecoder(self.string_to_decode)
        r_filename = dataset_filename.get_filename()
        r_date_record = dataset_filename.get_date_record()
        r_time_record = dataset_filename.get_time_record()
        r_time_str_record = dataset_filename.get_time_str_record()

        self.assertEqual(self.exp_filename, r_filename)
        self.assertEqual(self.exp_date_record, r_date_record)
        self.assertEqual(self.exp_time_record, r_time_record)
        self.assertEqual(self.exp_time_str_record, r_time_str_record)
