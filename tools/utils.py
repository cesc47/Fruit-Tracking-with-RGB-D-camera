import os
import numpy as np

GLOBAL_PATH_DB = '../data/Apple_Tracking_db'


def rotate_and_transform_bbox(xtl, ytl, xbr, ybr, clockwise):
    """
    Rotate the bounding box 90 degrees and transform it to the new coordinates because the image has been rotated
    :param xtl: x coordinate of the top left corner of the bounding box
    :param ytl: y coordinate of the top left corner of the bounding box
    :param xbr: x coordinate of the bottom right corner of the bounding box
    :param ybr: y coordinate of the bottom right corner of the bounding box
    :param clockwise: boolean, if the rotation is clockwise or counterclockwise
    :return: the new coordinates of the bounding box
    """
    if clockwise:
        xtl_new = 1080 - ytl
        ytl_new = xtl
        xbr_new = 1080 - ybr
        ybr_new = xbr
    else:
        xtl_new = ytl
        ytl_new = 1920 - xtl
        xbr_new = ybr
        ybr_new = 1920 - xbr
    return int(xbr_new), int(ybr_new), int(xtl_new), int(ytl_new)


def convert_bbox_to_yolo_format(size_img, bbox):
    """
    Convert the bounding box to the yolo format
    :param size_img: the size of the image
    :param bbox: the bounding box
    :return: the bounding box in the yolo format
    """

    dw = 1./size_img[0]
    dh = 1./size_img[1]
    x = (bbox[0] + bbox[1])/2.0
    y = (bbox[2] + bbox[3])/2.0
    w = abs(bbox[1] - bbox[0])
    h = abs(bbox[3] - bbox[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return x, y, w, h


def convert_bbox_from_yolo_format(x, y, w, h, size_img=(1080, 1920)):
    """
    Convert the bounding box from the yolo format to the original format
    """

    x_min = x - w/2
    y_min = y - h/2
    x_max = x + w/2
    y_max = y + h/2

    x_min = x_min * size_img[0]
    y_min = y_min * size_img[1]
    x_max = x_max * size_img[0]
    y_max = y_max * size_img[1]

    return round(x_min), round(y_min), round(x_max), round(y_max)


def get_gt_range_index_imgs(video_name):
    """
    Get the ground truth range of the images of a video. The ground truth range is the range of the images that have
    a ground truth annotation.
    :param video_name: the name of the video
    :return: the ground truth range of the images of the video, and the name of the video
    """
    path_images = os.path.join(GLOBAL_PATH_DB, video_name, 'images')

    # read all path_images sorted
    images_sorted = sorted(os.listdir(path_images))
    # get the number before the C.png string
    list_frame_numbers = []
    for image in images_sorted:
        frame_num_str = image.split('.')[0]
        frame_num_str = frame_num_str.split('_')[-2]
        frame_num_str = int(frame_num_str)
        list_frame_numbers.append(frame_num_str)

    str_video = images_sorted[0].split('.')[0]
    str_video = str_video.split('_')[:-2]
    str_video = '_'.join(str_video)

    return str_video, np.min(list_frame_numbers)


def compute_centroids_bboxes_from_gt_yolo(ground_truth):
    """
    Compute the centroids of the bounding boxes from the ground truth from the yolo format
    :param ground_truth: the ground truth in the yolo format
    :return: the centroids of the bounding boxes from the ground truth from the yolo format in the format (x, y)
    """
    gt_centers = []
    # compute centroid of bbox
    for bbox in ground_truth['bboxes']:
        gt_centers.append((int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)))
    return gt_centers


def convert_gt_to_readable_detections(ground_truth):
    """
    Convert the ground truth to the detections format. gets the ground truth in a dict format ('id', 'bboxes' and
    'id_video') and returns the detections in a list format ('bbox', 1) for each detection
    :param ground_truth: the ground truth
    :return: the detections
    """
    detections = []
    for bbox in ground_truth['bboxes']:
        bbox_detection = (bbox[0], bbox[1], bbox[2], bbox[3], 1)
        detections.append(bbox_detection)
    return detections


def compute_area_bbox_tlbr(bbox):
    """
    Compute the area of a bbox in tlbr format (top left and bottom right) and return it.
    :param bbox: bbox in tlbr format
    """
    return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])