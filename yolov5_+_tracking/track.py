import os
from sort import sort
import motmetrics as mm
import numpy as np
from bytetrack import byte_tracker
from tools.utils import compute_centroids_bboxes_from_gt_yolo


def track_detections_frame(tracking_predictions, detections, tracker, tracker_type, img_size=(1080, 1920)):
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
            det_centers.append((int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2)))
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
            det_centers.append((int((t.tlbr[0] + t.tlbr[2]) / 2), int((t.tlbr[1] + t.tlbr[3]) / 2)))
            det_ids.append(int(t.track_id))
            tracking_predictions.append(
                [int(t.track_id), int(t.tlbr[0]), int(t.tlbr[1]), int(t.tlbr[2] - t.tlbr[0]), int(t.tlbr[3] - t.tlbr[1])])

    return det_centers, det_ids, tracking_predictions


def read_from_yolo(path, ground_truth=False, img_size=(1080, 1920)):
    """
    Reads detections from yolo output file.
    :param path: path to yolo output file
    :param ground_truth: if True, the ground truth annotations are read instead of the detections
    :param img_size: size of the image
    :return: list of detections/ground truths (for all frames)
    """
    # this is where all the results (labels or detections) from all the frames will be stored
    all_results = []

    # path to db
    path_to_db = '../data/Apple_Tracking_db'

    # for each .txt we have to go and search for the ids of the objects that are in ./data/Apple_Tracking_db/
    if ground_truth:
        # for all the files in the folder read the detections
        for gt_file in sorted(os.listdir(path)):
            # read the ground truth annotations
            results = {
                'ids': [],
                'bboxes': [],
                'id_video': ''
            }
            id_video = (gt_file.split('.')[0]).split('_')[:-2]
            results['id_video'] = '_'.join(id_video)

            for videoname in os.listdir(path_to_db):
                # if file_path does not end with .txt or .xlsx, continue
                if not videoname.endswith('.txt') and not videoname.endswith('.xlsx'):
                    # check if exists detections_file in this folder
                    if os.path.exists(os.path.join(path_to_db, videoname, 'segmentation', 'labels_yolo_format+ids', gt_file)):
                        # if exists, read the detections
                        with open(os.path.join(
                                path_to_db, videoname, 'segmentation', 'labels_yolo_format+ids', gt_file), 'r') as f:
                            lines = f.readlines()

                        # convert each line to float numbers
                        ground_truths = [list(map(float, line.split(' '))) for line in lines]

                        # get the ids of the objects
                        results['ids'] = [int(ground_truth[1]) for ground_truth in ground_truths]

                        # get the bounding boxes and the detections
                        results['bboxes'] = [ground_truth[2:] for ground_truth in ground_truths]

                        # unnormalize the bounding boxes by image size (1920, 1080) and convert to int (x, y, w, h)
                        # if ground truth, the .txt hasn't got last col of scores
                        results['bboxes'] = [(int(bbox[0] * img_size[0]), int(bbox[1] * img_size[1]),
                                              int(bbox[2] * img_size[0]), int(bbox[3] * img_size[1]))
                                             for bbox in results['bboxes']]

                        # convert from (x, y, w, h) to (x1, y1, x2, y2)
                        results['bboxes'] = [(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                                             for bbox in results['bboxes']]

                        # add detections to all_detections list
                        all_results.append(results)

    # not gt, detections
    else:
        # for all the files in the folder read the detections
        for detections_file in sorted(os.listdir(path)):
            # read detections from frame
            with open(os.path.join(path, detections_file), 'r') as f:
                lines = f.readlines()

            # convert each line to float numbers
            detections = [list(map(float, line.split(' '))) for line in lines]

            # get the bounding boxes and the detections
            detections = [detection[1:] for detection in detections]

            # unnormalize the bounding boxes by image size (1920, 1080) and convert to int (x, y, w, h)
            # if ground truth, the .txt hasn't got last col of scores
            detections = [(int(detection[0] * img_size[0]), int(detection[1] * img_size[1]),
                                 int(detection[2] * img_size[0]), int(detection[3] * img_size[1]), detection[4])
                                 for detection in detections]

            # convert from (x, y, w, h) to (x1, y1, x2, y2)
            detections = [(detection[0], detection[1], detection[0] + detection[2], detection[1] + detection[3],
                           detection[4]) for detection in detections]

            # add detections to all_detections list
            all_results.append(detections)

    return all_results


def create_tracker(tracker_type):
    """
    Create a tracker based on the tracker_type. The tracker_type can be either 'sort', 'deepsort' or 'bytetrack'. The
    tracker_type 'sort' is based on the SORT tracker. The tracker_type 'deepsort' is based on the DeepSORT tracker. The
    tracker_type 'bytetrack' is based on the Bytetrack tracker.
    :param tracker_type: the tracker type
    :return: the tracker
    """
    # todo: deepsort implementation
    # create the tracker
    if tracker_type == 'sort' or tracker_type == 'Sort' or tracker_type == 'SORT':
        tracker = sort.Sort()
    elif tracker_type == 'bytetrack' or tracker_type == 'Bytetrack' \
            or tracker_type == 'ByteTrack' or tracker_type == 'BYTETRACK':
        tracker = byte_tracker.BYTETracker()
    else:
        raise AssertionError('tracker_type should be named: sort, bytetrack or deepsort')

    # Create the accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    return tracker, accumulator


def reset_tracker(accumulator, tracker_type, tracker_evaluation, anterior_video_id):
    """
    Reset the tracker and the accumulator.
    :param accumulator: the accumulator to reset (show metrics of tracking)
    :param tracker_type: the tracker type (sort, deepsort or bytetrack)
    :param tracker_evaluation: the tracker evaluation (show metrics of tracking => boolean)
    :param anterior_video_id: the id of the previous video
    """
    if tracker_evaluation:
        # compute the metrics (results)
        tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id)

    tracker, accumulator = create_tracker(tracker_type)

    return tracker, accumulator


def tracking_evaluation_update_params(accumulator, ground_truth, det_ids, det_centers, tracker_evaluation):
    """
    Updates the accumulator with the ground truth and the detections of the current frame
    :param accumulator: accumulator to update
    :param ground_truth: ground truth of the current frame
    :param det_ids: ids of the detections of the current frame
    :param det_centers: centers of the detections of the current frame
    :param tracker_evaluation: tracker update params of the current frame if true
    """
    if tracker_evaluation:
        # update the accumulator with the detections
        accumulator.update(
            ground_truth['ids'],  # Ground truth objects in this frame
            det_ids,  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(compute_centroids_bboxes_from_gt_yolo(ground_truth), det_centers)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )


def tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id):
    """
    Computes the metrics (results) of the tracking
    :param accumulator: accumulator to compute the metrics
    :param tracker_evaluation: if True, the metrics are computed for the tracker
    :param anterior_video_id: id of the video
    """
    if tracker_evaluation:
        # Compute the metrics
        print('printing results for video {}'.format(anterior_video_id))
        mh = mm.metrics.create()
        summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idf1'], name='acc')
        print(summary)
        print('\n')


def track_yolo_results(dataset_name, exp_name, tracker_type='sort', partition='test', img_size=(1080, 1920),
                       tracker_evaluation=True):
    """
    Performs the tracking in the test dataset from yolo. It is simmilar to track() function but now it does not take as
    ground truth the labels from supervisely (.json) but the labels from yolo (.txt)
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment (in yolo folder)
    :param tracker_type: type of tracker (e.g. sort, bytetrack)
    :param partition: partition where the results are computed => test, train or val
    :param img_size: size of the image (1920, 1080)
    :param tracker_evaluation: if True, the metrics are computed for the tracker
    """
    if partition != ('test' or 'train' or 'valid'):
        raise AssertionError('partition should be named: test, train or valid')

    # where will be stored the predictions of the tracker
    # tracking_predictions = []

    # process ground truths from yolo input files
    # the ground truths are in the ./data/Apple_Tracking_db dataset
    ground_truths = read_from_yolo(os.path.join('datasets', dataset_name, partition, 'labels'),
                                   ground_truth=True, img_size=img_size)

    # read detections from yolo output files
    # path_detections is the folder where the detections from yolo are stored
    all_detections = read_from_yolo(os.path.join('yolov5', 'runs', 'detect', exp_name, 'labels'),
                                    ground_truth=False, img_size=img_size)

    anterior_video_id = None
    # iterate for each img:
    for ground_truth, detections in zip(ground_truths, all_detections):
        # if video_id is not the same as the current video_id, then we have to reset the tracker
        if anterior_video_id is None:
            # create the tracker
            tracker, accumulator = create_tracker(tracker_type)
            anterior_video_id = ground_truth['id_video']

        elif anterior_video_id != ground_truth['id_video']:
            # reset and create the tracker and print results if tracker_evaluation is True
            tracker, accumulator = reset_tracker(accumulator, tracker_type, tracker_evaluation, anterior_video_id)
            anterior_video_id = ground_truth['id_video']

        tracking_predictions = []

        # perform the tracking
        det_centers, det_ids, tracking_predictions = track_detections_frame(tracking_predictions, detections,
                                                                            tracker, tracker_type)

        # todo: visualization of the tracking results

        # update the accumulator
        tracking_evaluation_update_params(accumulator, ground_truth, det_ids, det_centers, tracker_evaluation)

    # print the results for the last video
    tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id)


if __name__ == '__main__':
    track_yolo_results(dataset_name='Apple_Tracking_db_yolo', exp_name='yolov5s',
                       tracker_type='bytetrack', partition='test', img_size=(1080, 1920),
                       tracker_evaluation=True)
