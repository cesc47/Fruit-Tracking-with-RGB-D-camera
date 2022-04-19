"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:
    Configuration of simulation user interface

Use:

"""

import unittest
import os
import shutil
from dataset_management.dataset_config import DatasetConfig
from dataset_management.dataset_manager import DatasetManager


class TestDatasetManager(unittest.TestCase):

    def setUp(self):
        # CONFIGURE HERE PATHS TO A TEST DATASET
        self.root_folder = os.path.abspath('../../')
        self.dataset_name = 'KA_Story_RGB_IR_DEPTH_dataset'  # HERE WE DEFINE THE NAME OF DATASET, WHATEVER YOU WANT
        self.dataset_folder_path = os.path.join(self.root_folder, 'test')

    def test_create_hierarchy(self):
        root_folder = os.path.abspath('')
        dataset_create = 'TEST_DATASET'
        dataset_folder_path = os.path.join(root_folder)
        to_clean = os.path.join(dataset_folder_path, dataset_create)
        dataset_manager_config = DatasetConfig(dataset_folder_path, dataset_create)
        dataset_manager_obj = DatasetManager(dataset_manager_config)
        dataset_manager_obj.create_hierarchy()

        expected_dataset_folder_path = os.path.join(dataset_folder_path, dataset_create, 'preprocessed_data')
        expected_dataset_folder_ann_path = os.path.join(expected_dataset_folder_path, 'annotations')
        expected_dataset_folder_img_path = os.path.join(expected_dataset_folder_path, 'images')
        expected_dataset_folder_pv_path = os.path.join(expected_dataset_folder_path, 'square_annotations1')

        folder_path_flag = os.path.exists(expected_dataset_folder_path)
        dataset_folder_ann_path_flag = os.path.exists(expected_dataset_folder_ann_path)
        dataset_folder_img_path_flag = os.path.exists(expected_dataset_folder_img_path)
        expected_dataset_folder_pv_path_flag = os.path.exists(expected_dataset_folder_pv_path)

        self.assertEqual(folder_path_flag, True)
        self.assertEqual(dataset_folder_ann_path_flag, True)
        self.assertEqual(dataset_folder_img_path_flag, True)
        self.assertEqual(expected_dataset_folder_pv_path_flag, True)

        shutil.rmtree(to_clean, ignore_errors=True)
        pass

    def test_create_labeled_XML_files(self):
        # todo: implement create_labeled_XML_files
        self.assertEqual('OK', 'OK')
        pass

    def test_create_XML_files(self):
        # todo: implement create_XML_files
        self.assertEqual('OK', 'OK')
        pass

    def test_get_labeled_list_files(self):
        """
        Dataset structure

        dataset_root_path \
        | --- \ raw_data
        | --- \ preprocessed_data
            | --- \ dataset_images_path
                    | --- '20210927_115932_k_r2_e_000_150_138_1_0_C.png'
                    | --- '20210927_115932_k_r2_e_000_150_138_1_0_D.mat'
                    | --- '20210927_115932_k_r2_e_000_150_138_1_0_I.mat'
            | --- \ dataset_annotations_path
            | --- \ dataset_squares_path
                    | --- '20210927_115932_k_r2_e_000_150_138_1_1_C.xml'
        """
        expected_dataset_folder_path = os.path.join(self.dataset_folder_path, self.dataset_name, 'preprocessed_data')
        expected_dataset_folder_img_path = os.path.join(expected_dataset_folder_path, 'images')
        expected_dataset_folder_pv_path = os.path.join(expected_dataset_folder_path, 'square_annotations1')

        expected_pair_list = [
            ('20210927', '11:59:32',
             os.path.join(expected_dataset_folder_img_path, '20210927_115932_k_r2_e_000_150_138_1_0_C'),
             os.path.join(expected_dataset_folder_img_path, '20210927_115932_k_r2_e_000_150_138_1_0_D.mat'),
             os.path.join(expected_dataset_folder_img_path, '20210927_115932_k_r2_e_000_150_138_1_0_I.mat'),
             os.path.join(expected_dataset_folder_pv_path, '20210927_115932_k_r2_e_000_150_138_1_0_C.xml')),
            ('20210927', '11:59:33',
             os.path.join(expected_dataset_folder_img_path, '20210927_115933_k_r2_e_000_150_138_1_1_C'),
             os.path.join(expected_dataset_folder_img_path, '20210927_115933_k_r2_e_000_150_138_1_1_D.mat'),
             os.path.join(expected_dataset_folder_img_path, '20210927_115933_k_r2_e_000_150_138_1_1_I.mat'),
             os.path.join(expected_dataset_folder_pv_path, '20210927_115933_k_r2_e_000_150_138_1_1_C.xml'))
        ]

        # call to methods
        dataset_manager_config_obj = DatasetConfig(self.dataset_folder_path, self.dataset_name)
        dataset_manager_obj = DatasetManager(dataset_manager_config_obj)
        result_pair_list = dataset_manager_obj.get_labeled_list_files()
        self.assertEqual(expected_pair_list, result_pair_list)

    def test_export_data_CSV(self):
        # todo: implement export_data_CSV
        self.assertEqual('OK', 'OK')
        pass


if __name__ == '__main__':
    unittest.main()
