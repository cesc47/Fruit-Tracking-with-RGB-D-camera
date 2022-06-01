import os
from sort import sort
import motmetrics as mm
import numpy as np
import json
from bytetrack import byte_tracker


def track(path_detections, video_name_gt, tracker_type='sort', img_size=(1920, 1080)):
    """
    Performs the tracking of the detections in a video.
    params:
        path_detections: path to the detections
        video_name_gt: name of the video (without the extension) for the ground truth
        tracker_type: type of the tracker (default: Sort)
        img_size: size of the image (default: (1920, 1080))
    returns:
        tracking_predictions: list of the tracking predictions of the tracker

    """
    # Create the accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    # read detections from yolo output file
    all_detections = read_from_yolo(path_detections, img_size)

    # read ground truth from ground truth file
    ground_truths = read_segmentation(video_name_gt)

    # process ground truths
    annotations = process_ground_truths(ground_truths)

    # where will be stored the predictions of the tracker
    tracking_predictions = []

    # create the tracker
    if tracker_type == 'sort':
        tracker = sort.Sort()

    # for each frame:
    for frame_num, detections in enumerate(all_detections):
        # perform the tracking
        det_centers, det_ids, tracking_predictions = track_detections_frame(tracking_predictions, detections,
                                                                            tracker, tracker_type=tracker_type)

        # update the accumulator with the detections
        accumulator.update(
            annotations[frame_num]['id'],  # Ground truth objects in this frame
            det_ids,  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(annotations[frame_num]['centroid'], det_centers)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )
        
    # Compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)

    return tracking_predictions


def track_detections_frame(tracking_predictions, detections, tracker, tracker_type, img_size=(1920, 1080)):
    """
    Performs the tracking of the detections in a frame.
    params:
        tracking_predictions: list of the tracking predictions of the tracker
        detections: list of the detections in the frame
        tracker: the tracker
        frame_num: the number of the frame
        tracker_type: type of the tracker (default: Sort)
        img_size: size of the image (default: (1920, 1080))
    returns:
        tracking_predictions: list of the tracking predictions of the tracker
        det_centers: list of the centers of the detections
        det_ids: list of the ids of the detections
    """
    det_centers = []
    det_ids = []

    # if there are no detections in the frame, skip it (all zeros)
    if tracker_type == 'sort':
        if len(detections) == 0:
            trackers = tracker.update(np.empty((0, 5)))
        else:
            # update the tracker with the detections
            trackers = tracker.update(np.array(detections))

        for t in trackers:
            det_centers.append((int(t[0] + t[2] / 2), int(t[1] + t[3] / 2)))
            det_ids.append(int(t[4]))
            tracking_predictions.append(
                [int(t[4]), int(t[0]), int(t[1]), int(t[2] - t[0]), int(t[3] - t[1])])

    elif tracker_type == 'bytetrack':
        if len(detections) == 0:
            trackers = tracker.update(np.empty((0, 5)), img_info=img_size, img_size=img_size)
        else:
            # update the tracker with the detections
            trackers = tracker.update(np.array(detections), img_info=img_size, img_size=img_size)

        for t in trackers:
            det_centers.append((int(t.tlbr[0] + t.tlbr[2] / 2), int(t.tlbr[1] + t.tlbr[3] / 2)))
            det_ids.append(int(t.track_id))
            tracking_predictions.append(
                [int(t.track_id), int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2] - t.tlbr[0]), int(t.tlbr[3] - t.tlbr[1])])

    return det_centers, det_ids, tracking_predictions


def process_ground_truths(annotations):
    """
    Processes the ground truth annotations. Assigns the id to each object and computes the centroid of each object.
    params:
        annotations: list of the annotations
    returns:
        all_labels: list of the annotations processed
    """

    # process the ground truth annotations and store all of them in a list
    all_labels = []

    for annotations_frame in annotations['frames']:
        # for each frame:
        labels = {
            'centroid': [],
            'id': [],
        }
        for figure in annotations_frame['figures']:
            # get the bounding box for each figure
            xtl, ytl = figure['geometry']['points']['exterior'][0]
            xbr, ybr = figure['geometry']['points']['exterior'][1]
            # get the centroid of the bounding box
            centroid = (int((xtl + xbr) / 2), int((ytl + ybr) / 2))
            # save the centroid and the id of the figure
            labels['centroid'].append(centroid)
            labels['id'].append(figure['objectKey'])

        all_labels.append(labels)

    # convert the id's (are stranges codes...) to integers
    list_ids = []
    for labels in all_labels:
        for unique_id in labels['id']:
            if unique_id not in list_ids:
                list_ids.append(unique_id)

    for labels in all_labels:
        for i in range(len(labels['id'])):
            labels['id'][i] = list_ids.index(labels['id'][i])

    return all_labels


def read_from_yolo(path_detections, ground_truth=False, img_size=(1920, 1080)):
    """
    Reads detections from yolo output file.
    :param path_detections: path to yolo output file
    :param ground_truth: if True, the ground truth annotations are read instead of the detections
    :param img_size: size of the image
    :return: list of detections/ground truths (for all frames)
    """
    # this is where all the detections from all the frames will be stored
    all_detections = []

    # for all the files in the folder read the detections
    for detections_file in os.listdir(path_detections):
        # read detections from frame
        with open(os.path.join(path_detections, detections_file), 'r') as f:
            lines = f.readlines()

        # convert each line to float numbers
        detections = [list(map(float, line.split(' '))) for line in lines]

        # get the bounding boxes and the detections
        detections = [detection[1:] for detection in detections]

        # unnormalize the bounding boxes by image size (1920, 1080) and convert to int (x, y, w, h)
        # if ground truth, the .txt hasn't got last col of scores
        if ground_truth:
            detections = [
                (int(detection[0] * img_size[0]), int(detection[1] * img_size[1]), int(detection[2] * img_size[0]),
                 int(detection[3] * img_size[1])) for detection in detections]

            # convert from (x, y, w, h) to (x1, y1, x2, y2)
            detections = [
                (detection[0], detection[1], detection[0] + detection[2], detection[1] + detection[3])
                for detection in detections]

        else:
            detections = [
                (int(detection[0] * img_size[0]), int(detection[1] * img_size[1]), int(detection[2] * img_size[0]),
                 int(detection[3] * img_size[1]), detection[4]) for detection in detections]

            # convert from (x, y, w, h) to (x1, y1, x2, y2)
            detections = [
                (detection[0], detection[1], detection[0] + detection[2], detection[1] + detection[3], detection[4])
                for detection in detections]

        # add detections to all_detections list
        all_detections.append(detections)

    return all_detections


def read_segmentation(video_name):
    """
    Reads the segmentation file and returns the segmentation.
    params:
        video_name: name of the video
    returns:
        segmentation: parsed json containing the annotations
    """

    GLOBAL_PATH_DB = './../data/Apple_Tracking_db/'

    path_ann = os.path.join(GLOBAL_PATH_DB, video_name, 'segmentation', 'ann.json')
    with open(path_ann) as fh:
        data = fh.read()
    annotations = json.loads(data)

    return annotations


def track_yolo_results(dataset_name, exp_name, tracker_type='sort', partition='test', img_size=(1920, 1080)):
    """
    Performs the tracking in the test dataset from yolo. It is simmilar to track() function but now it does not take as
    ground truth the labels from supervisely (.json) but the labels from yolo (.txt)
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment (in yolo folder)
    :param tracker_type: type of tracker (e.g. sort, bytetrack)
    :param partition: partition where the results are computed => test, train or val
    :param img_size: size of the image (1920, 1080)
    """
    if partition != ('test' or 'train' or 'valid'):
        raise AssertionError('partition should be named: test, train or valid')

    # if tracker_type != ('sort' or 'bytetrack' or 'deepsort'):
    #     raise AssertionError('tracker_type should be named: sort, bytetrack or deepsort')

    # by default the dataset must be located in ./dataset (outside the yolov5 folder)
    path_ground_truths = os.path.join('datasets', dataset_name, partition, 'labels')

    # path_detections is the folder where the detections from yolo are stored
    path_detections = os.path.join('yolov5', 'runs', 'detect', exp_name, 'labels')

    # Create the accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    # create the tracker
    if tracker_type == 'sort':
        tracker = sort.Sort()
    elif tracker_type == 'bytetrack':
        tracker = byte_tracker.BYTETracker()

    # where will be stored the predictions of the tracker
    tracking_predictions = []

    # process ground truths from yolo input files
    ground_truths = read_from_yolo(path_ground_truths, ground_truth=True, img_size=img_size)

    # read detections from yolo output files
    all_detections = read_from_yolo(path_detections, ground_truth=False, img_size=img_size)

    # iterate for each img:
    for ground_truth, detections in zip(ground_truths, all_detections):
        # perform the tracking
        det_centers, det_ids, tracking_predictions = track_detections_frame(tracking_predictions, detections,
                                                                            tracker, tracker_type)


# todo: prova bytetrack amb ground truths


if __name__ == '__main__':
    # path_detections = 'yolov5/runs/detect/exp3/labels'
    # video_name_gt = '210928_165030_k_r2_w_015_125_162'
    # track(path_detections, video_name_gt=video_name_gt)

    track_yolo_results(dataset_name='SegmentacioPomes_v2.v2i.yolov5pytorch', exp_name='prova_yolo_small_nms',
                       tracker_type='bytetrack', partition='test', img_size=(1920, 1080))
