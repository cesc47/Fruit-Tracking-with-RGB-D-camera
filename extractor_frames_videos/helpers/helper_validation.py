"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:


Use:

"""
def digit_validation(inStr,acttyp):
    if acttyp == '1':
        if not inStr.isdigit():
            return False
    return True