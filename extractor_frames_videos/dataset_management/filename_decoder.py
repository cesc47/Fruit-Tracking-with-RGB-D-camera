"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:
    Decoder of filenames, get information about sensor, hour, date

Use:

"""

class FileNameDecoder:
    sentence_separator = '_'
    ext_separator = '.'
    time_separator = ':'
    raw_str_filename = None
    filename_splitted = None
    file_first = None
    file_extension = None

    date_record = None
    time_record = None
    sensor_record = None
    row_tree = None
    cardinal_points = None
    speed_record = None
    row_center_distance = None
    height_camera_distance = None


    def __init__(self, str_filename):
        self.raw_str_filename = str_filename
        self.decode()


    def decode(self):
        first_split = self.raw_str_filename.split(self.ext_separator)
        self.filename = first_split[0]
        self.file_extension = '.'+first_split[1]
        self.filename_splitted = self.raw_str_filename.split(self.sentence_separator)
        self.date_record = self.filename_splitted[0]
        self.time_record = self.filename_splitted[1]
        self.time_str_record = self.filename_splitted[1][0] + self.filename_splitted[1][1] + self.time_separator + self.filename_splitted[1][2] + self.filename_splitted[1][3] + self.time_separator+ self.filename_splitted[1][4] + self.filename_splitted[1][5]
        self.sensor_record = self.filename_splitted[2]
        self.row_tree = self.filename_splitted[3]
        self.cardinal_points = self.filename_splitted[4]
        self.speed_record = self.filename_splitted[5]
        self.row_center_distance = self.filename_splitted[6]
        self.height_camera_distance = self.filename_splitted[0]
        pass

    def get_filename(self):
        return self.filename

    def get_time_record(self):
        return self.time_record

    def get_time_str_record(self):
        return self.time_str_record

    def get_date_record(self):
        return self.date_record

    def get_sensor(self):
        return self.sensor_record

    def get_row_tree(self):
        return self.row_tree

    def get_cardinal_points(self):
        return self.cardinal_points

    def get_speed_record(self):
        return self.speed_record

    def get_row_center_distance(self):
        return self.row_tree

    def get_height_camera_distance(self):
        return self.height_camera_distance

# raise NotImplementedError("Can't use yet!")