import os
import numpy as np
from tools.read_segmentation import read_segmentation
from tqdm import tqdm
import cv2
from tools.dataset_gestions import read_depth_or_infrared_file

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
    :param x: x coordinate of the top left corner of the bounding box
    :param y: y coordinate of the top left corner of the bounding box
    :param w: width of the bounding box
    :param h: height of the bounding box
    :param size_img: the size of the image
    :return: the bounding box in the original format
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


def get_gt_range_index_imgs(video_name, path=GLOBAL_PATH_DB):
    """
    Get the ground truth range of the images of a video. The ground truth range is the range of the images that have
    a ground truth annotation.
    :param video_name: the name of the video
    :param path: the path to the database
    :return: the ground truth range of the images of the video, and the name of the video
    """
    path_images = os.path.join(path, video_name, 'images')

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


def compute_centroids_bboxes_tlbr(ground_truth):
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


def compute_sizes_all_gts():
    """
    Compute the sizes of the bboxes in all the dataset. prints the sizes of the bboxes in all the dataset, one min area
    and one max area per video.
    """
    video = sorted(os.listdir(GLOBAL_PATH_DB))
    for video_name in video:
        if not (video_name.endswith('.txt') or video_name.endswith('.xlsx')):
            print(f'sizes for video {video_name} are:')
            annotations = read_segmentation(video_name)

            max_bbox = None
            min_bbox = None

            for annotation in annotations['frames'][:]:
                for idx_figure, figure in enumerate(annotation['figures']):
                    xtl, ytl = figure['geometry']['points']['exterior'][0]
                    xbr, ybr = figure['geometry']['points']['exterior'][1]

                    size_bbox = abs((xbr - xtl) * (ybr - ytl))

                    if max_bbox is None:
                        max_bbox = size_bbox
                    else:
                        max_bbox = max(max_bbox, size_bbox)

                    if min_bbox is None:
                        min_bbox = size_bbox
                    else:
                        min_bbox = min(min_bbox, size_bbox)

            print(f'min area bbox: {min_bbox}, max area bbox: {max_bbox}')


def filter_detections_by_size(detections, detection_file):
    """
    Filter the detections by size. gets the detections in a list format ('bbox', 1) and returns the detections in a
    list format ('bbox', 1) with the detections that have a size bigger than a threshold. the threshold is defined
    thanks to compute_sizes_all_gts().
    :param detections: the detections in a list format ('bbox', 1)
    :param detection_file: the name of the detection file
    :return: the detections in a list format ('bbox', 1) with the detections that have a size bigger than a threshold
    """

    meters_to_apples = detection_file.split('_')[6]
    if meters_to_apples == '225':
        min_area = 100
    elif meters_to_apples == '175':
        min_area = 221
    elif meters_to_apples == '125':
        min_area = 624
    else:
        raise ValueError(f'unknown size {meters_to_apples}')

    for detection in detections:
        bbox = detection[:4]
        if compute_area_bbox_tlbr(bbox) < min_area:
            detections.remove(detection)

    return detections


def augment_size_of_bboxes(detections, percentage_to_augment=0.075, size_img=(1080, 1920)):
    """
    Augment the size of the bboxes. gets the detections in a list format ('bbox', 1) and returns the detections in a
    list format ('bbox', 1) with the detections enlarged. the size to augment is defined thanks to the size_to_augment
    parameter.
    :param detections: the detections in a list format ('bbox', 1)
    :param percentage_to_augment: the percentage to augment the size of the bboxes
    :param size_img: the size of the image
    :return: the detections in a list format ('bbox', 1) with the detections enlarged
    """
    detections_augmented = []
    for detection in detections:
        # convert tuple into list
        detection = list(detection)

        height = detection[3] - detection[1]
        width = detection[2] - detection[0]

        # augment the size of the bbox
        # augment the size of the bbox
        detection[0] -= percentage_to_augment * width
        if detection[0] < 0:
            detection[0] = 0
        detection[1] -= percentage_to_augment * height
        if detection[1] < 0:
            detection[1] = 0

        detection[2] += percentage_to_augment * width
        if detection[2] > size_img[0]:
            detection[2] = int(size_img[0])

        detection[3] += percentage_to_augment * height
        if detection[3] > size_img[1]:
            detection[3] = int(size_img[1])

        # convert the all the elements to int except the last one (4)
        detection_augm = [int(element) for element in detection[:-1]]

        # add the last element
        detection_augm.append(detection[-1])

        # convert list into tuple
        detection_augm = tuple(detection_augm)

        detections_augmented.append(detection_augm)

    return detections_augmented


def augment_size_of_bboxes_in_crops(bbox_tlbr, percentage_to_augment=0.15, size_img=(1080, 1920)):
    """
    Augment the size of the bboxes.
    :param bbox_tlbr: the bbox in tlbr format
    :param percentage_to_augment: the percentage to augment, in percentage of the size of the image
    :param size_img: the size of the image
    :return: the bbox in tlbr format with the size augmented
    """
    height = bbox_tlbr[3] - bbox_tlbr[1]
    width = bbox_tlbr[2] - bbox_tlbr[0]

    # augment the size of the bbox
    bbox_tlbr[0] -= percentage_to_augment * width
    if bbox_tlbr[0] < 0:
        bbox_tlbr[0] = 0
    bbox_tlbr[1] -= percentage_to_augment * height
    if bbox_tlbr[1] < 0:
        bbox_tlbr[1] = 0

    bbox_tlbr[2] += percentage_to_augment * width
    if bbox_tlbr[2] > size_img[0]:
        bbox_tlbr[2] = size_img[0]

    bbox_tlbr[3] += percentage_to_augment * height
    if bbox_tlbr[3] > size_img[1]:
        bbox_tlbr[3] = size_img[1]

    return [int(x) for x in bbox_tlbr]


def reduce_size_of_bbox(detection, percentage_to_reduce=0.075, size_img=(1080, 1920)):
    """
    Reduce the size of a bbox according to a percentage to reduce (must match the percentage in the
    augment_size_of_bboxes)
    :param detection: the detection in a list format (bbox)
    :param percentage_to_reduce: the percentage to reduce
    :param size_img: the size of the image
    :return: the detection in a list format (bbox) with the size reduced
    """
    height = detection[3] - detection[1]
    width = detection[2] - detection[0]

    # augment the size of the bbox
    # augment the size of the bbox
    detection[0] += percentage_to_reduce * width
    if detection[0] < 0:
        detection[0] = 0
    detection[1] += percentage_to_reduce * height
    if detection[1] < 0:
        detection[1] = 0

    detection[2] -= percentage_to_reduce * width
    if detection[2] > size_img[0]:
        detection[2] = int(size_img[0])

    detection[3] -= percentage_to_reduce * height
    if detection[3] > size_img[1]:
        detection[3] = int(size_img[1])

    # convert the all the elements to int except the last one (4)
    detection_reduced = [int(element) for element in detection]

    return detection_reduced


def reduce_size_of_bboxes_in_tracking_results(all_tracking_results, percentage_to_reduce=0.075):
    """
    Reduce the size of the bboxes in the tracking predictions.
    :param all_tracking_results: the tracking predictions in a dict format
    :param percentage_to_reduce: the percentage to reduce the size of the bboxes
    :return: the tracking predictions in a dict format with the bboxes reduced
    """
    print('reducing the size of the bboxes')
    for idx_frame, tracking_results in tqdm(enumerate(all_tracking_results)):
        for idx_bbox, bbox in enumerate(tracking_results['bboxes']):
            all_tracking_results[idx_frame]['bboxes'][idx_bbox] = reduce_size_of_bbox(bbox, percentage_to_reduce)


def search_in_dataset_an_image_from_yolo_dataset(framename):
    """
    Search in the dataset an image from the yolo dataset. gets the framename and returns the path of the image.
    :param framename: the framename
    :return: the path of the image
    """
    for video_name in os.listdir(GLOBAL_PATH_DB):
        if not (video_name.endswith('.txt') or video_name.endswith('.xlsx')):
            path_video = os.path.join(GLOBAL_PATH_DB, video_name, 'images')
            for image_name in os.listdir(path_video):
                # if strings are equal, return the path of the image
                if image_name == framename + '.png':
                    path_rgb = os.path.join(path_video, image_name)
                    # remove the last 5 characters of the path to get the path of the image (_C.png)
                    path = path_rgb[:-5]

                    return path


def skip_bbox_if_outside_map(bbox_tlbr, top_limit=600, bottom_limit=1450):
    """
    Skip the bbox if it is outside the map. gets the bbox in tlbr format and returns True if the bbox is outside the map
    and False otherwise.
    :param bbox_tlbr: the bbox in tlbr format
    :param top_limit: the top limit of the map
    :param bottom_limit: the bottom limit of the map
    :return: True if the bbox is outside the map and False otherwise
    """
    skip = False
    if bbox_tlbr[1] > bottom_limit or bbox_tlbr[3] < top_limit:
        skip = True

    return skip


def order_detections_folder_nums(path):
    """
    Order the detections in the folder. gets the path of the folder and returns the number of the detections ordered.
    :param path: the path of the folder
    :return: the number of the detections ordered
    """
    numbers = [int(file.split('_')[-2]) for file in os.listdir(path)]
    numbers.sort()
    return numbers


def filter_detections_by_depth_map(all_detections):
    """
    Filters the detections (list of frames and inside each frame => list of (bbox, conf)) by the size of the depth
    map. The depth and Ir map, does not work with all the image (1920, 1080), it only gives us information in the
    center of the image (around 600 and 1400).
    :param all_detections: the detections in a list format
    :return: the detections in a list format with the detections filtered by the size of the depth map
    """
    for idx, detections_frame in enumerate(all_detections):
        detections_frame = [detection for detection in detections_frame if not skip_bbox_if_outside_map([detection[0], detection[1], detection[2], detection[3]])]
        all_detections[idx] = detections_frame


def filter_gt_by_depth_map(all_gts):
    """
    Does the same as 'filter_detections_by_depth_map' but now working with gt. It reads a list of dicts instead of
    reading a list of lists.
    :param all_gts: the gt in a list
    :return: the gt in a list with the gt filtered by the size of the depth map
    """
    for idx, gt_frame in enumerate(all_gts):
        bboxes = []
        ids = []
        id_video = gt_frame['id_video']

        for bbox, id in zip(gt_frame['bboxes'], gt_frame['ids']):
            if not skip_bbox_if_outside_map([bbox[0], bbox[1], bbox[2], bbox[3]]):
                bboxes.append(bbox)
                ids.append(id)

        all_gts[idx] = {'id_video': id_video, 'bboxes': bboxes, 'ids': ids}


def read_img_5_channels(img_file_name):
    ori_img = cv2.imread(img_file_name + 'C.png')
    # split string in /
    img_file_name = img_file_name.split('/')
    videoname = img_file_name[3]
    filename = img_file_name[-1]
    ori_img_d = read_depth_or_infrared_file(videoname, filename + 'D', normalization=4000)
    ori_img_i = read_depth_or_infrared_file(videoname, filename + 'I', normalization=2000)

    img = np.array((ori_img[:, :, 0], ori_img[:, :, 1], ori_img[:, :, 2], ori_img_d * 255, ori_img_i * 255))

    # transpose channels (a, b, c) => (b, c, a). in img we have the same as before but now 5 channels
    ori_img = img.transpose(1, 2, 0)

    return ori_img
if __name__ == '__main__':
    compute_sizes_all_gts()


