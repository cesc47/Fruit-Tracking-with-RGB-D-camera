"""
Project: Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:
    Suit to test all classes dataset management, don't forget to add new tests HERE!

Use:
    python -m unittest mass_estimation/test/test_suite_dataset_management.py
"""

import unittest
from test_dataset_config import TestDatasetConfig
from test_dataset_manager import TestDatasetManager
from test_filename_decoder import TestFileNameDecoder
from test_pascal_voc_parser import TestPASCALVOCParser


def dataset_management_suite():
    """
        Gather all the tests to process data from detected objects in all levels
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDatasetConfig))
    test_suite.addTest(unittest.makeSuite(TestDatasetManager))
    test_suite.addTest(unittest.makeSuite(TestFileNameDecoder))
    test_suite.addTest(unittest.makeSuite(TestPASCALVOCParser))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(dataset_management_suite())
