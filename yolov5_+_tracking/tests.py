# objective: call track function with bytetrack => precision 1 => no apples into the rows that are behind the row that
#            is being tracked. with that we can compute the minimum apple size.
# knowing that, a function can be done later in order to filter all the detections by a minimum size.
import os
from tools.utils import convert_bbox_from_yolo_format, compute_area_bbox_tlbr
from track import track_yolo_results


def compute_min_apple_size(all_tracking_predictions):
    """
    Compute the minimum apple size based on the predictions of the tracking function.
    :param all_tracking_predictions: list of predictions from the tracking function
    :return: minimum apple size
    """
    area_min_size = None

    for tracking_predictions in all_tracking_predictions:
        for bbox in tracking_predictions['bboxes']:
            if area_min_size is None:
                area_min_size = compute_area_bbox_tlbr(bbox)
            else:
                area_size = compute_area_bbox_tlbr(bbox)
                if area_size < area_min_size:
                    area_min_size = area_size

    return area_min_size


def compute_min_apple_size_gts():
    """
    this function reads from where is located the db used for yolo and gets the area of all the bounding boxes from the
    ground truth files. With that we will have the area of the apple that is the minimum size.
    :return: minimum apple size
    """
    area_min_size = None

    for split in os.listdir(os.path.join('datasets', 'Apple_Tracking_db_yolo')):
        for file in os.listdir(os.path.join('datasets', 'Apple_Tracking_db_yolo', split, 'labels')):
            if file.endswith('.txt'):
                with open(os.path.join('datasets', 'Apple_Tracking_db_yolo', split, 'labels', file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_split = line.split(' ')
                        x_min, y_min, x_max, y_max = convert_bbox_from_yolo_format(float(line_split[1]),
                                                              float(line_split[2]),
                                                              float(line_split[3]),
                                                              float(line_split[4]),
                                                              size_img=(1080, 1920))
                        area_size = (x_max - x_min) * (y_max - y_min)
                        if area_min_size is None:
                            area_min_size = area_size
                        else:
                            if area_size < area_min_size:
                                area_min_size = area_size

    return area_min_size

if __name__ == '__main__':
    """
    tracking_predictions = track_yolo_results(dataset_name='Apple_Tracking_db_yolo', exp_name='yolov5s',
                                              tracker_type='bytetrack', partition='valid', img_size=(1080, 1920),
                                              tracker_evaluation=False, visualize_results=False)

    area_min_size = compute_min_apple_size(tracking_predictions)
    print(f'size of the minimum apple: {area_min_size}')
    """
    print(f'size of the minimum apple: {compute_min_apple_size_gts()}')