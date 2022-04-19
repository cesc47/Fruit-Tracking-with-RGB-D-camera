"""
Project: Fruit size estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: November 2021
Description:

User interface that contains functions related to extracting video data from the Azure Kinect camera.

Use:
 python UI_frame_extractor.py
"""

import os
from os.path import expanduser
from gui_frame_ext.gui_frame_ext_console import GUIFrameExtractorConsole
from gui_frame_ext.gui_frame_ext_config import GUIFrameExtractorConfig

from dataset_management.dataset_config import DatasetConfig
from video_extraction_management.frame_extraction_config import FramesManagerConfig

if __name__ == '__main__':
    BASE_DIR = os.path.abspath('')
    ui_path_config_file = os.path.join(BASE_DIR, 'conf', 'ui_frames_extractor.conf')
    path_extractor_config_file = os.path.join(BASE_DIR, 'conf', 'frames_extractor.conf')
    user_path = expanduser("~")
    base_path = os.path.join(user_path, 'Downloads')
    dataset_name = 'KA_Story_RGB_IR_DEPTH_dataset'
    dataset_manager_config_obj = DatasetConfig(base_path, dataset_name)
    frames_extractor_config = FramesManagerConfig(path_extractor_config_file)

    ui_frame_extractor_config = GUIFrameExtractorConfig(ui_path_config_file)
    app = GUIFrameExtractorConsole(ui_frame_extractor_config, dataset_manager_config_obj, frames_extractor_config)
    app.mainloop()
    # todo: check automation of settings
    # todo:config files
    # todo:save config form
    # todo: add validations of fields
    # todo: add unit tests
    # todo: add integration tests
