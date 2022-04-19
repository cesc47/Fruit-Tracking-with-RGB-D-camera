"""
# Project: Size Estimation
# Author: Juan Carlos Miranda
# Date: November 2021
# Description:
  Methods used for frame extraction from Matroska files

Usage:

    BASE_DIR = os.path.abspath('.')
    path_extractor_config_file = os.path.join(BASE_DIR, 'conf', 'frames_extractor.conf')
    frames_extractor_config_obj = FramesManagerConfig(path_extractor_config_file)
    frames_extractor_obj = FramesVideoManager(frames_extractor_config_obj)
    #remote_obj.run()

"""

import cv2
import os
import numpy as np
import scipy.io as sio
import logging
from video_extraction_management.frame_extraction_config import FramesManagerConfig
from pyk4a import PyK4APlayback
from video_extraction_management.video_helpers.helpers import convert_to_bgra_if_required, colorize


class FramesVideoManager:
    _frame_video_config = None
    _a_matroska_file = None

    def __init__(self, config_param, a_matroska_file):
        logging.debug('STARTING Frames Video Manager')

        if config_param is None:
            logging.debug('config_param is empty')
            self._frame_video_config = FramesManagerConfig()
        else:
            self._frame_video_config = config_param

        if a_matroska_file is None:
            # todo: raise exception
            pass
        else:
            self._a_matroska_file = a_matroska_file

    def get_info(self, playback: PyK4APlayback):
        """
        Given a file get info used to record the video
        :param playback:
        :return:
        """
        # Record configuration: {'color_format': <ImageFormat.COLOR_MJPG: 0>, 'color_resolution': <ColorResolution.RES_1080P: 2>, 'depth_mode': <DepthMode.NFOV_UNBINNED: 2>, 'camera_fps': <FPS.FPS_30: 2>, 'color_track_enabled': True, 'depth_track_enabled': True, 'ir_track_enabled': True, 'imu_track_enabled': False, 'depth_delay_off_color_usec': 0, 'wired_sync_mode': <WiredSyncMode.STANDALONE: 0>, 'subordinate_delay_off_master_usec': 0, 'start_timestamp_offset_usec': 280777}
        # todo: add test, if this is necessary
        # todo: convert this to data field if it is necessary
        print(f"Record length: {playback.length / 1000000: 0.2f} sec")
        print(f"Record path: {playback.path}")
        print(f"Record configuration: {playback.configuration}")
        print(f"color_format: {playback.configuration['color_format']}")
        print(f"color_resolution: {playback.configuration['color_resolution']}")
        print(f"depth_mode: {playback.configuration['depth_mode']}")
        print(f"camera_fps: {playback.configuration['camera_fps']}")
        print(f"imu_track_enabled: {playback.configuration['imu_track_enabled']}")

    def get_a_frame(self, start_offset):
        """
        Given a Matroska file, gets from one frame as following: RGB, DEPTH, NIR.
        In the method, image ndparray is converted to image data
        This data is used with OpenCV library to visualise images.

        :param a_matroska_file: a path with Matroska file name
        :param start_offset: number of seconds from the beginning
        :return:
        """
        rgb_data = None
        depth_data = None
        nir_data = None

        playback = PyK4APlayback(self._a_matroska_file)
        playback.open()

        if start_offset != 0.0:
            playback.seek(int(start_offset * 1000000))

        try:
            capture = playback.get_next_capture()

            if capture.color is not None:
                rgb_data = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)

            if capture.transformed_depth is not None:
                depth_data = capture.transformed_depth

            if capture.transformed_ir is not None:
                nir_data = capture.transformed_ir
        except EOFError:
            pass

        playback.close()

        return rgb_data, depth_data, nir_data

    def export_frames_to_files(self, track_file, start_offset, number_of_frames=None, filename=None):
        """
        From one Matroska file, extract frames and save them into a directory
        Read every frame and export this to files.

        :param a_matroska_file: a path with Matroska file name
        :param start_offset: number of seconds from the beginning
        :param number_of_frames: number of frames to extract
        :return:
        """
        playback = PyK4APlayback(self._a_matroska_file)
        playback.open()

        if start_offset != 0.0:
            playback.seek(int(start_offset * 1000000))

        if number_of_frames is None:
            number_of_frames = playback.length  #

        if filename is None:
            filename_with_ext = os.path.splitext(self._a_matroska_file)[0]
            filename = os.path.basename(filename_with_ext)

        # todo: add test
        # todo: add a range check for start_offset
        try:
            # todo: track file could be out of this file
            f1 = open(track_file, "w")

            frames_written = 0
            while frames_written < number_of_frames:
                print("frames_written=", frames_written)
                capture = playback.get_next_capture()
                # ------------------------------------
                base_name = filename + '_' + str(start_offset) + '_' + str(frames_written) + '_'
                img_write_filename_RGB = base_name + self._frame_video_config.rgb_data_filename + self._frame_video_config.file_img_extension
                img_write_filename_depth = base_name + self._frame_video_config.depth_data_filename + self._frame_video_config.file_img_extension
                img_write_filename_ir = base_name + self._frame_video_config.ir_data_filename + self._frame_video_config.file_img_extension
                img_write_filename_depth_mat = base_name + self._frame_video_config.depth_data_filename + self._frame_video_config.file_mat_extension
                img_write_filename_ir_mat = base_name + self._frame_video_config.ir_data_filename + self._frame_video_config.file_mat_extension

                f1.write(base_name + '\n')  # todo: check this if it is necessary to put out of this class

                # TODO: TEMPORAL SOLUTION THAT solves .csv annotations for PychetLabeller
                img_annotation_RGB = base_name + self._frame_video_config.rgb_data_filename + self._frame_video_config.file_csv_extension
                f2 = open(os.path.join(self._frame_video_config.path_annotations_ouput, img_annotation_RGB), "w")
                f2.write(base_name + '\n')
                f2.close()

                # ------------------------------------
                if capture.color is not None:
                    cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_RGB),
                                convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))

                if capture.transformed_depth is not None:
                    # cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_depth), colorize(capture.transformed_depth, (None, 5000)))
                    depth_dict = {}
                    depth_dict['transformed_depth'] = capture.transformed_depth
                    sio.savemat(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_depth_mat),
                                depth_dict)

                if capture.transformed_ir is not None:
                    # cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_ir), colorize(capture.transformed_ir, (None, 500), colormap=cv2.COLORMAP_JET))
                    ir_dict = {}
                    ir_dict['transformed_ir'] = capture.transformed_ir
                    sio.savemat(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_ir_mat),
                                ir_dict)

                frames_written = frames_written + 1
        except EOFError:
            pass

        playback.close()
        errors = "STRING_ERROR"
        output_folder = self._frame_video_config.path_images_ouput
        return frames_written, errors, output_folder

    def export_raw_frames_to_files(self, start_offset, number_of_frames=None, filename=None):
        """
        From one Matroska file, extract raw frames and save them into a directory without transformation
        This is useful when is needed to see original images
        Read every frame and export this to files.

        :param start_offset: number of seconds from the beginning
        :param number_of_frames: number of frames to extract
        :return:
        """
        playback = PyK4APlayback(self._a_matroska_file)
        playback.open()

        if start_offset != 0.0:
            playback.seek(int(start_offset * 1000000))

        if number_of_frames is None:
            number_of_frames = playback.length  #

        if filename is None:
            filename_with_ext = os.path.splitext(self._a_matroska_file)[0]
            filename = os.path.basename(filename_with_ext)

        # todo: add test
        try:
            frames_written = 0
            while frames_written < number_of_frames:
                print("frames_written=", frames_written)
                capture = playback.get_next_capture()
                # ------------------------------------
                img_write_filename_RGB = filename + '_' + self._frame_video_config.rgb_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + self._frame_video_config.file_img_extesion
                img_write_filename_depth = filename + '_' + self._frame_video_config.depth_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + self._frame_video_config.file_img_extesion
                img_write_filename_ir = filename + '_' + self._frame_video_config.ir_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + self._frame_video_config.file_img_extesion

                img_write_filename_depth_mat = filename + '_' + self._frame_video_config.depth_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + '_' + self._frame_video_config.file_mat_extesion
                img_write_filename_ir_mat = filename + '_' + self._frame_video_config.ir_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + '_' + self._frame_video_config.file_mat_extesion
                # ------------------------------------

                if capture.color is not None:
                    cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_RGB),
                                convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))

                if capture.depth is not None:
                    cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_depth),
                                colorize(capture.depth, (None, 5000)))
                    depth_dict = {}
                    depth_dict['transformed_depth'] = capture.depth
                    sio.savemat(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_depth_mat),
                                depth_dict)

                if capture.ir is not None:
                    cv2.imwrite(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_ir),
                                colorize(capture.ir, (None, 500), colormap=cv2.COLORMAP_JET))
                    ir_dict = {}
                    ir_dict['transformed_ir'] = capture.ir
                    sio.savemat(os.path.join(self._frame_video_config.path_images_ouput, img_write_filename_ir_mat),
                                ir_dict)

                frames_written = frames_written + 1

        except EOFError:
            pass

        playback.close()
        errors = "STRING_ERROR"
        output_folder = self._frame_video_config.path_images_ouput
        return frames_written, errors, output_folder

    def go_through_frames(self, start_offset, number_of_frames=None):
        """
        From one Matroska file, go through frames and make something with every frame

        :param start_offset: number of seconds from the beginning
        :param number_of_frames: number of frames to extract. If this is None, we take lenght as value
        :return:
        """
        playback = PyK4APlayback(self._a_matroska_file)
        playback.open()

        if start_offset != 0.0:
            playback.seek(int(start_offset * 1000000))

        if number_of_frames is None:
            number_of_frames = playback.length  #

        # todo: add test

        try:
            frames_checked = 0
            while frames_checked < number_of_frames:
                print("frames_checked=", frames_checked)
                capture = playback.get_next_capture()
                # ------------------------------------
                # ------------------------------------
                if capture.color is not None:
                    pass

                if capture.transformed_color is not None:
                    print('capture.transformed_color-> Not None')

                if capture.depth is not None:
                    cv2.imshow("Transformed Depth", colorize(capture.depth, (None, 5000)))
                    # pass
                # todo: uncomment this if is needed
                # if capture.transformed_depth is not None:
                #    cv2.imshow("Transformed Depth", colorize(capture.transformed_depth, (None, 5000)))
                # pass

                # if capture.transformed_ir is not None:
                #    pass
                # ------------------------------------
                # here we will put a function to process something
                # ------------------------------------
                key = cv2.waitKey(10)
                if key != -1:
                    cv2.destroyAllWindows()
                    break
                frames_checked = frames_checked + 1

        except EOFError:
            pass
            # break

        playback.close()

        errors = "STRING_ERROR"
        output_folder = self._frame_video_config.path_images_ouput

        return frames_checked, errors, output_folder

    def export_frames_to_mesh(self, start_offset, number_of_frames=None, filename=None):

        """
        From one Matroska file, extract frames and save them into a directory as mesh file

        :param a_matroska_file: a path with Matroska file name
        :param start_offset: number of seconds from the beginning
        :param number_of_frames: number of frames to extract
        :return:
        """
        playback = PyK4APlayback(self._a_matroska_file)
        playback.open()

        if start_offset != 0.0:
            playback.seek(int(start_offset * 1000000))

        if number_of_frames is None:
            number_of_frames = playback.length

        if filename is None:
            filename_with_ext = os.path.splitext(self._a_matroska_file)[0]
            filename = os.path.basename(filename_with_ext)
        try:
            print("a_matroska_file -->", self._a_matroska_file)
            print("start_offset -->", start_offset)
            print("number_of_frames -->", number_of_frames)
            print("self._frame_video_config.path_images_ouput -->", self._frame_video_config.path_images_ouput)
            frames_written = 0
            while frames_written < number_of_frames:
                print("frames_written=", frames_written)
                capture = playback.get_next_capture()
                # ------------------------------------
                img_write_filename_mesh = filename + '_' + self._frame_video_config.rgb_data_filename + '_' + str(
                    start_offset) + '_' + str(frames_written) + self._frame_video_config.file_mesh_extesion
                # ------------------------------------
                if capture.depth is not None:
                    # todo:add mesh data function to save
                    points = capture.depth_point_cloud.reshape((-1, 3))
                    np.savetxt(os.path.join(self._frame_video_config.path_mesh_ouput, img_write_filename_mesh), points,
                               delimiter=' ', fmt='%u')

                frames_written = frames_written + 1
        except EOFError:
            pass
        playback.close()
        errors = "STRING_ERROR"
        output_folder = self._frame_video_config.path_mesh_ouput

        return frames_written, errors, output_folder

    def export_frames_to_colorized_mesh(self, start_offset, number_of_frames=None, filename=None):
        # todo: will be implemented
        frames_written = 0
        errors = "STRING_ERROR"
        output_folder = self._frame_video_config.path_mesh_ouput
        return frames_written, errors, output_folder

# TODO: 17/11/2021 Colorized cloud points functions. At this time we have implemented a cloud point without color.
#  To generate a colorized cloud point the image format color must be K4A_IMAGE_FORMAT_COLOR_BGRA32.
#  With  K4A_IMAGE_FORMAT_COLOR_MJPG, at this momment can´t be made.
#  But the Azure viewer tool made this, I think that this could be made with C/C++
#  Transform to point cloud data -> https://github.com/etiennedub/pyk4a/issues/21
