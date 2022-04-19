"""
Project: Fruit size estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: November 2021
Description:
Manage configurations of user interface
Use:

    ui_path_config_file = os.path.join(BASE_DIR, 'conf', 'ui_frames_extractor.conf')
    ui_frame_extractor_config = GUIFrameExtractorConfig(ui_path_config_file)

"""

import os
from os.path import expanduser
import configparser


class GUIFrameExtractorConfig:
    app_title = 'Frame Extractor app'
    width = 320
    height = 480
    geometry_about = '300x480'
    geometry_main = '320x480'
    file_extension_to_search = '.mkv'  # "*.mkv"
    split_at_timestamp = 2  # time used to split files, values in seconds
    input_test_folder = ""
    user_path = expanduser("~")
    base_folder = os.path.abspath('')
    file_browser_input_folder = os.path.join(base_folder)
    input_dataset_folder = base_folder
    label_to_search = "nothing"

    def __init__(self, file_config_name=None):
        if file_config_name is not None:
            if os.path.isfile(file_config_name):
                self.f_config_name = file_config_name
                self.read_config()

    def read_config(self):
        """
        Read config from file ui_frames_extractor.conf
        :return:
        """
        f_config = configparser.ConfigParser()
        f_config.read(self.f_config_name)
        self.width = f_config['DEFAULT']['WIDTH']
        self.height = f_config['DEFAULT']['HEIGHT']
        self.geometry_about = f_config['DEFAULT']['geometry_about']
        self.geometry_main = f_config['DEFAULT']['geometry_main']
        self.file_extension_to_search = f_config['DEFAULT']['file_extension_to_search']
        self.split_at_timestamp = int(f_config['DEFAULT']['split_at_timestamp'])
        self.base_folder = f_config['DEFAULT']['base_folder']
        self.file_browser_input_folder = f_config['DEFAULT']['file_browser_input_folder']
        self.input_dataset_folder = f_config['DEFAULT']['input_dataset_folder']
        self.label_to_search = f_config['DEFAULT']['label_to_search']
