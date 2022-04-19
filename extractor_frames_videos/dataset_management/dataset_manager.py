"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:
    Manage datasets functions, PASCAL VOC format conversions

Use:

Dataset structure

dataset_root_path \\
| --- \ raw_data
| --- \ preprocessed_data
    | --- \ dataset_images_path
            | --- '20210927_115932_k_r2_e_000_150_138_1_0_C.png'
            | --- '20210927_115932_k_r2_e_000_150_138_1_0_D.mat'
            | --- '20210927_115932_k_r2_e_000_150_138_1_0_I.mat'
    | --- \ dataset_annotations_path
    | --- \ dataset_squares_path
            | --- '20210927_115932_k_r2_e_000_150_138_1_1_C.xml'
    | --- \ dataset_masks_path
            | --- '20210927_115932_k_r2_e_000_150_138_1_0_C.png'
"""

import os
import logging
import PIL
import pandas as pd
import shutil
import json

from PIL import Image
from lxml import etree as ET
from dataset_management.dataset_config import DatasetConfig
from dataset_management.filename_decoder import FileNameDecoder


class DatasetManager:
    dataset_config = None
    class_label_name = "Obj_"

    def __init__(self, config_param):
        logging.debug('STARTING DatasetManager')

        if config_param is None:
            logging.debug('config_param is empty')
            self.dataset_config = DatasetConfig()
        else:
            self.dataset_config = config_param

    def create_hierarchy(self):
        """
        It creates folders with subfolders
        :return:
        """
        root_dataset = os.path.join(self.dataset_config.dataset_root_folder)
        if os.path.isdir(root_dataset):
            shutil.rmtree(root_dataset, ignore_errors=True)
        # create folders of dataset
        os.mkdir(root_dataset)
        os.mkdir(self.dataset_config.dataset_root_path)
        os.mkdir(self.dataset_config.dataset_images_path)
        os.mkdir(self.dataset_config.dataset_annotations_path)
        os.mkdir(self.dataset_config.dataset_squares_path)
        os.mkdir(self.dataset_config.dataset_sets_path)
        os.mkdir(self.dataset_config.dataset_masks_path)

    def create_labeled_XML_files(self):
        """
        Create XML files from annotated data. Parse files .csv to .xml files
        Adapted from https://github.com/GRAP-UdL-AT/RGBD_fruit_detection_faster-rcnn.pytorch
        :return:
        """
        # iterate over directory with data labelled
        annotated_csv_file_list = os.listdir(self.dataset_config.dataset_annotations_path)
        for an_annotated_csv_file_name in annotated_csv_file_list:
            if an_annotated_csv_file_name.endswith(".csv"):
                an_annotated_csv_file_path = os.path.join(self.dataset_config.dataset_annotations_path,
                                                          an_annotated_csv_file_name)
                an_annotated_base_name = an_annotated_csv_file_name.split(self.dataset_config.csv_extension)[0]
                # get image data info
                an_image_file_name = an_annotated_base_name + self.dataset_config.img_extension
                an_image_path = os.path.join(self.dataset_config.dataset_images_path, an_image_file_name)
                (width, height) = PIL.Image.open(an_image_path).size
                annotations_df = pd.read_csv(an_annotated_csv_file_path, header=None)
                # here we create an xml header
                f = ET.Element("annotations")
                ET.SubElement(f, 'filename').text = an_image_file_name
                size_element = ET.SubElement(f, 'size')
                ET.SubElement(size_element, 'width').text = str(width)
                ET.SubElement(size_element, 'height').text = str(height)
                ET.SubElement(size_element, 'depth').text = "3"  # todo: check this parameter
                # iterate over labelled objects with Pychetlabeller
                for index, a_row in annotations_df.iterrows():
                    # get coordinates of marked object in one image
                    xmin = int(a_row[1])
                    xmax = int(a_row[1] + a_row[3])
                    ymin = int(a_row[2])
                    ymax = int(a_row[2] + a_row[4])

                    # by each row make add to the XML tree
                    object_element = ET.SubElement(f, 'object')
                    ET.SubElement(object_element, 'name').text = str(
                        index) + '_' + self.dataset_config.class_label_name  # todo: check this variable
                    ET.SubElement(object_element, 'difficult').text = "0"
                    bbox = ET.SubElement(object_element, 'bbox')
                    xmin_xml = ET.SubElement(bbox, 'xmin')
                    ymin_xml = ET.SubElement(bbox, 'ymin')
                    xmax_xml = ET.SubElement(bbox, 'xmax')
                    ymax_xml = ET.SubElement(bbox, 'ymax')
                    # assign values and convert to string
                    xmin_xml.text = str(xmin)
                    ymin_xml.text = str(ymin)
                    xmax_xml.text = str(xmax)
                    ymax_xml.text = str(ymax)
                # write body of xml and close the file
                filename = os.path.join(self.dataset_config.dataset_squares_path, an_annotated_base_name + ".xml")
                tree = ET.ElementTree(f)
                tree.write(filename, pretty_print=True)


    def create_XML_files(self, label_json_path):
        """
        Create XML files from with
        Adapted from https://github.com/GRAP-UdL-AT/RGBD_fruit_detection_faster-rcnn.pytorch
        :return:
        """

        if label_json_path is not None and os.path.exists(label_json_path):
            # import simplejson as json
            with open(label_json_path, 'rb') as f:
                labelmap = json.load(f)


        # iterate over directory with data labelled
        annotated_csv_file_list = os.listdir(self.dataset_config.dataset_annotations_path)
        for an_annotated_csv_file_name in annotated_csv_file_list:
            if an_annotated_csv_file_name.endswith(".csv"):
                an_annotated_csv_file_path = os.path.join(self.dataset_config.dataset_annotations_path,
                                                          an_annotated_csv_file_name)
                an_annotated_base_name = an_annotated_csv_file_name.split(self.dataset_config.csv_extension)[0]
                # get image data info
                an_image_file_name = an_annotated_base_name + self.dataset_config.img_extension
                an_image_path = os.path.join(self.dataset_config.dataset_images_path, an_image_file_name)
                (width, height) = PIL.Image.open(an_image_path).size
                annotations_df = pd.read_csv(an_annotated_csv_file_path, header=None)
                # here we create an xml header
                f = ET.Element("annotations")
                ET.SubElement(f, 'filename').text = an_image_file_name
                size_element = ET.SubElement(f, 'size')
                ET.SubElement(size_element, 'width').text = str(width)
                ET.SubElement(size_element, 'height').text = str(height)
                ET.SubElement(size_element, 'depth').text = "3"  # todo: check this parameter
                # iterate over labelled objects with Pychetlabeller
                for index, a_row in annotations_df.iterrows():
                    # get coordinates of marked object in one image
                    xmin = int(a_row[1])
                    xmax = int(a_row[1] + a_row[3])
                    ymin = int(a_row[2])
                    ymax = int(a_row[2] + a_row[4])
                    id_class = int(a_row[5])


                    # by each row make add to the XML tree
                    object_element = ET.SubElement(f, 'object')
                    ET.SubElement(object_element, 'name').text = labelmap[id_class]['object_name']
                    ET.SubElement(object_element, 'difficult').text = "0"
                    bbox = ET.SubElement(object_element, 'bbox')
                    xmin_xml = ET.SubElement(bbox, 'xmin')
                    ymin_xml = ET.SubElement(bbox, 'ymin')
                    xmax_xml = ET.SubElement(bbox, 'xmax')
                    ymax_xml = ET.SubElement(bbox, 'ymax')
                    # assign values and convert to string
                    xmin_xml.text = str(xmin)
                    ymin_xml.text = str(ymin)
                    xmax_xml.text = str(xmax)
                    ymax_xml.text = str(ymax)
                # write body of xml and close the file
                filename = os.path.join(self.dataset_config.dataset_squares_path, an_annotated_base_name + ".xml")
                tree = ET.ElementTree(f)
                tree.write(filename, pretty_print=True)


    def get_labeled_list_files(self):
        """
        Returns a list structure used in bucles to process files
        # todo: add this for mask options. It is necessary to improve
        :return:
        """
        pair_list_files = []
        pass
        for a_filename in os.listdir(self.dataset_config.dataset_images_path):
            if a_filename.endswith(self.dataset_config.img_extension):
                dataset_filename = FileNameDecoder(a_filename)
                selected_filename = dataset_filename.get_filename()
                # get depth correspondence in another folder
                file_search_rgb = os.path.join(self.dataset_config.dataset_images_path, selected_filename)
                file_search_depth = os.path.join(self.dataset_config.dataset_images_path, selected_filename[
                                                                                          :-1] + self.dataset_config.suffix_depth + self.dataset_config.depth_extension)
                file_search_ir = os.path.join(self.dataset_config.dataset_images_path, selected_filename[
                                                                                       :-1] + self.dataset_config.suffix_IR + self.dataset_config.depth_extension)
                # get file correspondence in another folder
                file_search_labels = os.path.join(self.dataset_config.dataset_squares_path,
                                                  selected_filename + self.dataset_config.xml_extension)
                # select only files with its pairs
                if (os.path.exists(file_search_labels)) and (os.path.exists(file_search_depth)):
                    # open xml file
                    # open depth image
                    frame_captured_date = dataset_filename.get_date_record()
                    frame_captured_time = dataset_filename.get_time_str_record()
                    pair_list_files.append((
                                           frame_captured_date, frame_captured_time, file_search_rgb, file_search_depth,
                                           file_search_ir, file_search_labels))

        return pair_list_files

    def get_labeled_mask_list_files(self):
        """
                Returns a list structure used in bucles to process files. This is a copy of the method above, but works
                with masks files
                # todo: this must be improved
        :return:
        """
        pair_list_files = []
        for a_filename in os.listdir(self.dataset_config.dataset_images_path):
            if a_filename.endswith(self.dataset_config.img_extension):
                dataset_filename = FileNameDecoder(a_filename)
                selected_filename = dataset_filename.get_filename()
                # get depth correspondence in another folder
                file_search_rgb = os.path.join(self.dataset_config.dataset_images_path, selected_filename)
                file_search_depth = os.path.join(self.dataset_config.dataset_images_path, selected_filename[
                                                                                          :-1] + self.dataset_config.suffix_depth + self.dataset_config.depth_extension)
                file_search_ir = os.path.join(self.dataset_config.dataset_images_path, selected_filename[
                                                                                       :-1] + self.dataset_config.suffix_IR + self.dataset_config.depth_extension)
                # get file correspondence in another folder
                file_search_labels = os.path.join(self.dataset_config.dataset_squares_path,
                                                  selected_filename + self.dataset_config.xml_extension)

                file_search_mask = os.path.join(self.dataset_config.dataset_masks_path, selected_filename + '.png')

                # select only files with its pairs
                if (os.path.exists(file_search_labels)) and (os.path.exists(file_search_depth)) and (os.path.exists(file_search_mask)):
                    # open xml file
                    # open depth image
                    frame_captured_date = dataset_filename.get_date_record()
                    frame_captured_time = dataset_filename.get_time_str_record()
                    pair_list_files.append((
                                           frame_captured_date, frame_captured_time, file_search_rgb, file_search_depth,
                                           file_search_ir, file_search_labels, file_search_mask))

        return pair_list_files




    def load_template_in_files(self, template_path):
        """
        Put template in files. This is used to create files with one template data
        :return:
        """
        # iterate over directory with data labelled
        f_template = open(template_path, "r")

        # TEMPLATE_STR = '0,1129.0,419.0,42.0,42.0,1'
        annotated_csv_file_list = os.listdir(self.dataset_config.dataset_annotations_path)
        for an_annotated_csv_file_name in annotated_csv_file_list:
            if an_annotated_csv_file_name.endswith(".csv"):
                an_annotated_csv_file_path = os.path.join(self.dataset_config.dataset_annotations_path,
                                                          an_annotated_csv_file_name)
                f_template = open(template_path, "r")
                f2 = open(an_annotated_csv_file_path, "w")
                f2.write(f_template.read())
                f2.close()
                f_template.close()



    def export_data_csv(self):
        # todo: implement export_data_CSV(self):
        raise Exception("NOT IMPLEMENTED")
        pass