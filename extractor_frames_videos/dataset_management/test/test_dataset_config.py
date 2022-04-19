"""
# Project: Fruit Size Estimation
# Author: Juan Carlos Miranda
# Date: January 2022
# Description:
  Test for Dataset hierarchy

Documentation in https://docs.python.org/3/library/unittest.html

Usage:
python -m unittest $HOME/development/KA_detector/dataset_management/test/test_dataset_config.py

"""

import unittest
import os
from dataset_management.dataset_config import DatasetConfig


class TestDatasetConfig(unittest.TestCase):

    def setUp(self):
        # CONFIGURE HERE PATHS TO A TEST DATASET
        self.root_folder = os.path.abspath('../../')
        self.dataset_name = 'KA_Story_RGB_IR_DEPTH_dataset'  # HERE WE DEFINE THE NAME OF DATASET, WHATEVER YOU WANT
        self.dataset_folder_path = os.path.join(self.root_folder, 'test')  # link to test data
        self.dataset_manager_config_obj = DatasetConfig(self.dataset_folder_path, self.dataset_name)

    def test_dataset_manager_config(self):
        """
        Dataset structure

        dataset_root_path \
        | --- \ raw_data
        | --- \ preprocessed_data
            | --- \ dataset_images_path
            | --- \ dataset_annotations_path
            | --- \ dataset_squares_path
            | --- \ masks # todo: 01/03/2022 add tests for this option
        """
        expected_dataset_folder_img_path = os.path.join(self.dataset_folder_path, self.dataset_name,
                                                        'preprocessed_data', 'images')
        expected_dataset_folder_pv_path = os.path.join(self.dataset_folder_path, self.dataset_name, 'preprocessed_data',
                                                       'square_annotations1')
        self.assertEqual(expected_dataset_folder_img_path, self.dataset_manager_config_obj.dataset_images_path)
        self.assertEqual(expected_dataset_folder_pv_path, self.dataset_manager_config_obj.dataset_squares_path)

    def test_dataset_manager_str(self):
        print(self.dataset_manager_config_obj)
        self.assertEqual('OK', 'OK')


if __name__ == '__main__':
    unittest.main()
