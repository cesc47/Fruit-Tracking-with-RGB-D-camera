"""
# Project: Size Estimation
# Author: Juan Carlos Miranda
# Date: November 2021
# Description:
  This file contains config used in frame extraction

Usage:
    BASE_DIR = os.path.abspath('.')
    path_extractor_config_file = os.path.join(BASE_DIR, 'conf', 'frames_extractor.conf')
    frames_extractor_config_obj = FramesManagerConfig(path_extractor_config_file)

"""
import logging
import os
from os.path import expanduser


class DatasetConfig:
    BASE_DIR = os.path.abspath('')
    f_config_name = os.path.join(BASE_DIR, 'conf', 'dataset.conf')
    user_path = expanduser("~")
    base_path = os.path.abspath('')
    preprocessed_sub_folder = 'preprocessed_data'

    # used in file names
    images_sub_folder = 'images'
    annotations_sub_folder = 'annotations'
    squares_sub_folder = 'square_annotations1'
    sets_sub_folder = 'sets'
    masks_sub_folder = 'masks'

    img_extension = '.png'
    ir_extension = '.mat'
    depth_extension = '.mat'
    csv_extension = '.csv'
    xml_extension = '.xml'
    class_label_name = 'Obj'

    suffix_rgb = 'C'
    suffix_depth = 'D'
    suffix_IR = 'I'

    dataset_name = None
    dataset_root_folder = None
    dataset_images_path = None
    dataset_annotations_path = None
    dataset_squares_path = None
    dataset_sets_path = None
    dataset_masks_path = None

    def __init__(self, base_path, dataset_name):
        logging.debug('%s - Constructor', type(self).__name__)
        if base_path is None or dataset_name is None:
            self.set_default()
        else:
            self.base_path = base_path
            self.dataset_name = dataset_name
            self.dataset_root_folder = os.path.join(base_path, dataset_name)
            self.dataset_root_path = os.path.join(self.dataset_root_folder, self.preprocessed_sub_folder)
            self.dataset_images_path = os.path.join(self.dataset_root_path, self.images_sub_folder)
            self.dataset_annotations_path = os.path.join(self.dataset_root_path, self.annotations_sub_folder)
            self.dataset_squares_path = os.path.join(self.dataset_root_path, self.squares_sub_folder)
            self.dataset_sets_path = os.path.join(self.dataset_root_path, self.sets_sub_folder)
            self.dataset_masks_path = os.path.join(self.dataset_root_path, self.masks_sub_folder)

    def set_default(self):
        self.dataset_root_path = os.path.join(self.base_path, self.dataset_root_folder)
        self.dataset_images_path = os.path.join(self.dataset_root_path, self.images_sub_folder)
        self.dataset_annotations_path = os.path.join(self.dataset_root_path, self.annotations_sub_folder)
        self.dataset_squares_path = os.path.join(self.dataset_root_path, self.squares_sub_folder)
        self.dataset_sets_path = os.path.join(self.dataset_root_path, self.sets_sub_folder)
        self.dataset_masks_path = os.path.join(self.dataset_root_path, self.masks_sub_folder)

    def read_config(self):
        """
        Read config from file frames_extractor.conf
        :return:
        """
        pass

    def __del__(self):
        logging.debug('%s - Finalize', type(self).__name__)

    def __str__(self):
        return "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % \
               (str(self.BASE_DIR),
                str(self.f_config_name),
                str(self.user_path),
                str(self.base_path),
                str(self.dataset_name),
                str(self.preprocessed_sub_folder),
                str(self.dataset_root_folder),
                str(self.images_sub_folder),
                str(self.annotations_sub_folder),
                str(self.squares_sub_folder),
                str(self.sets_sub_folder),
                str(self.img_extension),
                str(self.ir_extension),
                str(self.depth_extension),
                str(self.csv_extension),
                str(self.xml_extension),
                str(self.class_label_name),
                str(self.suffix_rgb),
                str(self.suffix_depth),
                str(self.suffix_IR),
                str(self.dataset_name),
                str(self.dataset_root_folder),
                str(self.dataset_images_path),
                str(self.dataset_annotations_path),
                str(self.dataset_squares_path),
                str(self.dataset_sets_path)
                )
