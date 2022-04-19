"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:


Use:

"""
import os

def remove_files(folder_to_search, extension_file_to_search):
    # ---------------------------------------------------
    label_training_files_list = []
    for a_filename in os.listdir(folder_to_search):
        if a_filename.endswith(extension_file_to_search):
            os.remove(folder_to_search + a_filename)
    # ---------------------------------------------------