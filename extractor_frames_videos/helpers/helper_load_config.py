"""
Project: Fruit Size Estimation
Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
Date: February 2022
Description:


Use:

"""
import configparser
from pyk4a import Config, ColorResolution, ImageFormat, DepthMode
from pyk4a import FPS, WiredSyncMode


def convert_color_resolution(convert_param):
    color_resolution = ColorResolution.RES_720P
    if convert_param == 'OFF':
        color_resolution = ColorResolution.OFF
    elif convert_param == 'RES_720P':
        color_resolution = ColorResolution.RES_720P
    elif convert_param == 'RES_1080P':
        color_resolution = ColorResolution.RES_1080P
    elif convert_param == 'RES_1440P':
        color_resolution = ColorResolution.RES_1440P
    elif convert_param == 'RES_1536P':
        color_resolution = ColorResolution.RES_1536P
    elif convert_param == 'RES_2160P':
        color_resolution = ColorResolution.RES_2160P
    elif convert_param == 'RES_3072P':
        color_resolution = ColorResolution.RES_3072P
    return color_resolution

def convert_color_format(convert_param):
    color_format = ImageFormat.COLOR_BGRA32
    if convert_param == 'COLOR_MJPG':
        color_format = ImageFormat.COLOR_MJPG
    elif convert_param == 'COLOR_NV12':
        color_format = ImageFormat.COLOR_NV12
    elif convert_param == 'COLOR_YUY2':
        color_format = ImageFormat.COLOR_YUY2
    elif convert_param == 'COLOR_BGRA32':
        color_format = ImageFormat.COLOR_BGRA32
    elif convert_param == 'DEPTH16':
        color_format = ImageFormat.DEPTH16
    elif convert_param == 'IR16':
        color_format = ImageFormat.IR16
    elif convert_param == 'CUSTOM8':
        color_format = ImageFormat.CUSTOM8
    elif convert_param == 'CUSTOM16':
        color_format = ImageFormat.CUSTOM16
    elif convert_param == 'CUSTOM':
        color_format = ImageFormat.CUSTOM
    return color_format


def convert_depth_mode(convert_param):
    depth_mode = DepthMode.NFOV_UNBINNED
    if convert_param == 'OFF':
        depth_mode = DepthMode.OFF
    elif convert_param == 'NFOV_2X2BINNED':
        depth_mode = DepthMode.NFOV_2X2BINNED
    elif convert_param == 'NFOV_UNBINNED':
        depth_mode = DepthMode.NFOV_UNBINNED
    elif convert_param == 'WFOV_2X2BINNED':
        depth_mode = DepthMode.WFOV_2X2BINNED
    elif convert_param == 'WFOV_UNBINNED':
        depth_mode = DepthMode.WFOV_UNBINNED
    elif convert_param == 'PASSIVE_IR':
        depth_mode = DepthMode.PASSIVE_IR
    return depth_mode

def convert_camera_fps(convert_param):
    camera_fps = FPS.FPS_30
    if convert_param == 'FPS_5':
        camera_fps = FPS.FPS_5
    elif convert_param == 'FPS_15':
        camera_fps = FPS.FPS_15
    elif convert_param == 'FPS_30':
        camera_fps = FPS.FPS_30
    return camera_fps

def convert_wired_sync_mode(convert_param):
    wired_sync_mode = WiredSyncMode.STANDALONE
    if convert_param == 'MASTER':
        wired_sync_mode = WiredSyncMode.MASTER
    elif convert_param == 'SUBORDINATE':
        wired_sync_mode = WiredSyncMode.SUBORDINATE
    elif convert_param == 'STANDALONE':
        wired_sync_mode = WiredSyncMode.STANDALONE
    return wired_sync_mode


def load_config_from_file(f_config_name: object) -> object:
    """
    Read config from file settings.conf
    :return:
    """
    f_config = configparser.ConfigParser()
    f_config.read(f_config_name)
    dev_conf = Config() # confid pyk4a
    dev_conf.color_resolution = convert_color_resolution(f_config['DEFAULT']['color_resolution'])
    dev_conf.color_format = convert_color_format(f_config['DEFAULT']['color_format'])
    dev_conf.depth_mode = convert_depth_mode(f_config['DEFAULT']['depth_mode'])
    dev_conf.camera_fps = convert_camera_fps(f_config['DEFAULT']['camera_fps'])
    dev_conf.synchronized_images_only = bool(f_config['DEFAULT']['synchronized_images_only'])
    dev_conf.depth_delay_off_color_usec = int(f_config['DEFAULT']['depth_delay_off_color_usec'])
    dev_conf.wired_sync_mode = convert_wired_sync_mode(f_config['DEFAULT']['wired_sync_mode'])
    dev_conf.subordinate_delay_off_master_usec = int(f_config['DEFAULT']['subordinate_delay_off_master_usec'])
    dev_conf.disable_streaming_indicator = bool(f_config['DEFAULT']['disable_streaming_indicator'])
    return dev_conf
