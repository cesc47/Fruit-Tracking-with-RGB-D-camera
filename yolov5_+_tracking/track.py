import os
from sort import sort
import motmetrics as mm
import numpy as np
from bytetrack import byte_tracker
from deepsort import deepsort
from tools.utils import compute_centroids_bboxes_from_gt_yolo, convert_bbox_from_yolo_format, \
    convert_gt_to_readable_detections
from tools.visualization import visualize_tracking_results
import csv


def track_detections_frame(predictions, detections, tracker, tracker_type, anterior_video_id, frame_name,
                           dataset_name, partition, img_size=(1080, 1920)):
    """
    Performs the tracking of the detections in a frame.
    params:
        predictions: list of the tracking predictions of the tracker
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
    results = {
        'video_id': anterior_video_id,
        'ids': [],
        'bboxes': []
    }

    # if there are no detections in the frame, skip it (all zeros)
    if len(detections) == 0:
        trackers = tracker.update(np.empty((0, 5)))

    # if there are detections in the frame, track them
    else:
        # update the tracker with the detections
        if tracker_type == 'sort':
            trackers = tracker.update(np.array(detections))

        elif tracker_type == 'bytetrack':
            trackers = tracker.update(np.array(detections), img_info=img_size, img_size=img_size)

        elif tracker_type == 'deepsort':
            path_to_img = os.path.join('datasets', dataset_name, partition, 'images', f'{frame_name}.png')
            trackers = tracker.update(np.array(detections), path_to_img)

        for t in trackers:
            # prepare data to be all the same format for all trackers
            if tracker_type == 'bytetrack':
                t_id = t.track_id
                t = t.tlbr
                t = np.append(t, t_id)

            det_centers.append((int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2)))
            det_ids.append(int(t[4]))
            results['bboxes'].append([int(t[0]), int(t[1]), int(t[2]), int(t[3])])
            results['ids'].append(int(t[4]))

    predictions.append(results)

    return det_centers, det_ids, predictions


def read_from_yolo(path, ground_truth=False):
    """
    Reads detections from yolo output file.
    :param path: path to yolo output file
    :param ground_truth: if True, the ground truth annotations are read instead of the detections
    :return: list of detections/ground truths (for all frames)
    """
    # this is where all the results (labels or detections) from all the frames will be stored
    all_results = []

    # where all the names of the videos are stored
    videonames = []

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
                    if os.path.exists(os.path.join(path_to_db, videoname, 'segmentation', 'labels_yolo_format+ids',
                                                   gt_file)):
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
                        results['bboxes'] = [convert_bbox_from_yolo_format(bbox[0], bbox[1], bbox[2], bbox[3])
                                             for bbox in results['bboxes']]

                        # add detections to all_detections list
                        all_results.append(results)

                        # add the name of the video to the list
                        videonames.append(gt_file.split('.')[0])

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
            bboxes = [convert_bbox_from_yolo_format(detection[0], detection[1], detection[2], detection[3])
                      for detection in detections]

            detections = [(bbox[0], bbox[1], bbox[2], bbox[3], detection[4])
                          for (detection, bbox) in zip(detections, bboxes)]

            # add detections to all_detections list
            all_results.append(detections)

            # add the name of the video to the list
            videonames.append(detections_file.split('.')[0])

    return all_results, videonames


def create_tracker(tracker_type):
    """
    Create a tracker based on the tracker_type. The tracker_type can be either 'sort', 'deepsort' or 'bytetrack'. The
    tracker_type 'sort' is based on the SORT tracker. The tracker_type 'deepsort' is based on the DeepSORT tracker. The
    tracker_type 'bytetrack' is based on the Bytetrack tracker.
    :param tracker_type: the tracker type
    :return: the tracker
    """
    # create the tracker
    if tracker_type == 'sort':
        tracker = sort.Sort()

    elif tracker_type == 'bytetrack':
        tracker = byte_tracker.BYTETracker()

    elif tracker_type == 'deepsort':
        tracker = deepsort.DeepSort(model_path=os.path.join('deepsort', 'checkpoints', 'ckpt.t7'))

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
    results = None

    if tracker_evaluation:
        # compute the metrics (results)
        results, _ = tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id)

    tracker, accumulator = create_tracker(tracker_type)

    return tracker, accumulator, results


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
        gt_ids = ground_truth['ids']
        gt_centers = compute_centroids_bboxes_from_gt_yolo(ground_truth)
        accumulator.update(
            gt_ids,  # Ground truth objects in this frame
            det_ids,  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(gt_centers, det_centers, max_d2=100)
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
        metrics = ['num_frames', 'num_matches', 'num_misses', 'num_objects', 'num_false_positives', 'num_switches',
                   'num_predictions', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'num_fragmentations',
                   'mota', 'precision', 'recall', 'idf1', ]
        summary = mh.compute(accumulator, metrics=metrics, name='acc')

        # print all the values of the summary
        for key, value in summary.items():
            print('{}: {}'.format(key, value[0]))
        print('\n')

        return summary, metrics


def save_tracking_results(results, dataset_name, exp_name, tracker_type, partition, metrics):
    """
    Save the tracking results in a csv file. The tracking results are saved in the folder 'results_tracking'
    :param results: tracking results to save
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment
    :param tracker_type: type of the tracker
    :param partition: partition of the dataset
    :param metrics: metrics of the tracking
    """
    # create new folder for the results of the tracking if it does not exist
    if not os.path.exists(os.path.join(os.getcwd(), 'results_tracking')):
        os.mkdir(os.path.join(os.getcwd(), 'results_tracking'))

    # save the results in a csv file
    path_file = os.path.join('results_tracking', f'{dataset_name}_{exp_name}_{tracker_type}_{partition}.csv')

    # if file already exists, delete it
    if os.path.exists(path_file):
        os.remove(path_file)

    with open(path_file, 'a') as f:
        writer = csv.writer(f)
        metrics.insert(0, 'video_id')
        writer.writerow(metrics)
        for result, id_video in results:
            to_write = [value[0] for _, value in result.items()]
            to_write.insert(0, id_video)
            writer.writerow(to_write)


def track_yolo_results(dataset_name, exp_name, tracker_type='sort', partition='test', tracker_evaluation=True,
                       visualize_results=False, save_results=False):
    """
    Performs the tracking in the test dataset from yolo. It is simmilar to track() function but now it does not take as
    ground truth the labels from supervisely (.json) but the labels from yolo (.txt)
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment (in yolo folder)
    :param tracker_type: type of tracker (e.g. sort, bytetrack)
    :param partition: partition where the results are computed => test, train or val
    :param tracker_evaluation: if True, the metrics are computed for the tracker
    :param visualize_results: if True, the results are visualized in the images
    :param save_results: if True, the results are saved in a csv file
    """
    if partition != ('test' or 'train' or 'valid'):
        raise AssertionError('partition should be named: test, train or valid')

    # where will be stored the predictions of the tracker
    all_tracking_predictions = []

    # results of the tracking (metrics)
    all_tracking_results = []

    # tracker and accumulator are referenced here
    accumulator, tracker = None, None

    # process ground truths from yolo input files
    # the ground truths are in the ./data/Apple_Tracking_db dataset
    ground_truths, video_names_gt = read_from_yolo(os.path.join('datasets', dataset_name, partition, 'labels'),
                                                   ground_truth=True)

    # read detections from yolo output files
    # path_detections is the folder where the detections from yolo are stored
    all_detections, video_names_det = read_from_yolo(os.path.join('yolov5', 'runs', 'detect', exp_name, 'labels'),
                                                     ground_truth=False)

    anterior_video_id = None

    # iterate for each img:
    for idx_frame, (ground_truth, detections) in enumerate(zip(ground_truths, all_detections)):
        # if video_id is not the same as the current video_id, then we have to reset the tracker
        if anterior_video_id is None:
            # create the tracker
            tracker, accumulator = create_tracker(tracker_type)
            anterior_video_id = ground_truth['id_video']

        elif anterior_video_id != ground_truth['id_video']:
            # reset and create the tracker and print results if tracker_evaluation is True
            tracker, accumulator, results = reset_tracker(accumulator, tracker_type, tracker_evaluation,
                                                          anterior_video_id)
            anterior_video_id = ground_truth['id_video']
            # append the results to the list of results
            all_tracking_results.append([results, anterior_video_id])

        # perform the tracking
        det_centers, det_ids, all_tracking_predictions = track_detections_frame(all_tracking_predictions,
                                                                                detections,
                                                                                tracker,
                                                                                tracker_type,
                                                                                anterior_video_id,
                                                                                video_names_det[idx_frame],
                                                                                dataset_name,
                                                                                partition)

        # update the accumulator
        tracking_evaluation_update_params(accumulator, ground_truth, det_ids, det_centers, tracker_evaluation)

    # print the results for the last video
    results, metrics = tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id)

    # append the results to the list of results
    all_tracking_results.append([results, anterior_video_id])

    # if save_results is True, then we save the results of the tracker and the detections
    if save_results:
        print('saving tracking results ...')
        save_tracking_results(all_tracking_results, dataset_name, exp_name, tracker_type, partition, metrics)

    # if visualize_results is True, then we visualize the results of the tracker and the detections
    if visualize_results:
        visualize_tracking_results(all_tracking_predictions, ground_truths, partition, dataset_name)

    return all_tracking_predictions, all_tracking_results


def track_gt_files(dataset_name, exp_name='prueba_groundTruths', tracker_type='sort', partition='test',
                   tracker_evaluation=True, visualize_results=False, save_results=False):
    """
       Performs the tracking in the test dataset from yolo. It is simmilar to track() function but now it does not take
       as ground truth the labels from supervisely (.json) but the labels from yolo (.txt)
       :param dataset_name: name of the dataset
       :param exp_name: name of the experiment (in yolo folder)
       :param tracker_type: type of tracker (e.g. sort, bytetrack)
       :param partition: partition where the results are computed (WHERE THE INFERENCE HAS BEEN DONE) =>
       test, train or val
       :param tracker_evaluation: if True, the metrics are computed for the tracker
       :param visualize_results: if True, the results are visualized in the images
       :param save_results: if True, the results are saved in a csv file
       return: tracking_predictions: list of the predictions of the tracker
       return: tracking_results: list of the results of the tracker
       """
    if partition != ('test' or 'training' or 'valid'):
        raise AssertionError('partition should be named: test, training or valid')

    # where will be stored the predictions of the tracker
    all_tracking_predictions = []

    # results of the tracking (metrics)
    all_tracking_results = []

    # tracker and accumulator are referenced here
    accumulator, tracker = None, None

    # process ground truths from yolo input files
    # the ground truths are in the ./data/Apple_Tracking_db dataset
    ground_truths, video_names_gt = read_from_yolo(os.path.join('datasets', dataset_name, partition, 'labels'),
                                                   ground_truth=True)

    anterior_video_id = None
    # iterate for each img:
    for idx_frame, ground_truth in enumerate(ground_truths):
        # if video_id is not the same as the current video_id, then we have to reset the tracker
        if anterior_video_id is None:
            # create the tracker
            tracker, accumulator = create_tracker(tracker_type)
            anterior_video_id = ground_truth['id_video']

        elif anterior_video_id != ground_truth['id_video']:
            # reset and create the tracker and print results if tracker_evaluation is True
            tracker, accumulator, results = reset_tracker(accumulator, tracker_type, tracker_evaluation,
                                                          anterior_video_id)
            anterior_video_id = ground_truth['id_video']
            # append the results to the list of results
            all_tracking_results.append([results, anterior_video_id])

        gt_detections = convert_gt_to_readable_detections(ground_truth)

        # perform the tracking
        det_centers, det_ids, all_tracking_predictions = track_detections_frame(all_tracking_predictions,
                                                                                gt_detections,
                                                                                tracker,
                                                                                tracker_type,
                                                                                anterior_video_id,
                                                                                video_names_gt[idx_frame],
                                                                                dataset_name,
                                                                                partition)

        # update the accumulator
        tracking_evaluation_update_params(accumulator, ground_truth, det_ids, det_centers, tracker_evaluation)

    # print the results for the last video
    results, metrics = tracking_evaluation_results(accumulator, tracker_evaluation, anterior_video_id)

    # append the results to the list of results
    all_tracking_results.append([results, anterior_video_id])

    # if save_results is True, then we save the results of the tracker and the detections
    if save_results:
        print('saving tracking results ...')
        save_tracking_results(all_tracking_results, dataset_name, exp_name, tracker_type, partition, metrics)

    # if visualize_results is True, then we visualize the results of the tracker and the detections
    if visualize_results:
        visualize_tracking_results(all_tracking_predictions, ground_truths, partition, dataset_name)

    return all_tracking_predictions, all_tracking_results


if __name__ == '__main__':

    tracking_predictions, tracking_results = track_yolo_results(dataset_name='Apple_Tracking_db_yolo',
                                                                exp_name='yolov5s',
                                                                tracker_type='sort',
                                                                partition='test',
                                                                tracker_evaluation=True,
                                                                visualize_results=True,
                                                                save_results=True)
    """
    tracking_predictions, tracking_results = track_gt_files(dataset_name='Apple_Tracking_db_yolo',
                                                            tracker_type='sort',
                                                            partition='test',
                                                            tracker_evaluation=True,
                                                            visualize_results=True,
                                                            save_results=True)
    """