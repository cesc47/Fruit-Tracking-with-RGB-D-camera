import os
import csv

import motmetrics as mm
import numpy as np

from tools.utils import compute_centroids_bboxes_tlbr
from tools.TrackEval.trackeval.metrics import HOTA


def compute_distance_matrix_frame(gt_centers, det_centers, max_d2=2000):
    """
    Computes the distance matrix between the ground truth and the detections. The distance matrix is computed for each
    frame. The distance matrix is computed using the norm2 squared distance.
    :param gt_centers: ground truth centers
    :param det_centers: detections centers
    :param max_d2: maximum distance squared
    :return: distance matrix
    """
    return mm.distances.norm2squared_matrix(gt_centers, det_centers, max_d2)


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
        accumulator.update(
            ground_truth['ids'],  # Ground truth objects in this frame
            det_ids,  # Detector hypotheses in this frame
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
            compute_distance_matrix_frame(compute_centroids_bboxes_tlbr(ground_truth), det_centers)
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


def get_num_ids_from_sequence(sequence):
    """
    recieves as input a list of lists of ids and returns the number of ids in the sequence.
    :param sequence: list of lists of ids
    :return: number of ids in the sequence
    """
    ids_list = []
    for frame in sequence:
        # search if the id is in the list
        for id in frame['ids']:
            if id not in ids_list:
                ids_list.append(id)
    return len(ids_list)


def get_num_dets(sequence):
    """
    recieves as input a list of lists of detections and returns the number of detections in the sequence.
    :param sequence: list of lists of detections
    :return: number of detections in the sequence
    """
    num_dets = 0
    for frame in sequence:
        num_dets += len(frame['bboxes'])
    return num_dets


def get_ids(sequence):
    """
    recieves as input a list of lists of detections and returns the ids in the sequence.
    :param sequence: list of lists of detections
    :return: ids in the sequence
    """
    ids_list = []
    for frame in sequence:
        # search if the id is in the list
        ids_list.append(np.array(frame['ids']))
    return ids_list


def compute_similarity_scores_for_sequence(tracking_predictions_sequence, ground_truths_sequence):
    """
    computes the similarity scores for the current sequence. This function is used to compute the similarity scores for
    each sequence. The similarity scores are computed for each frame.
    :param tracking_predictions_sequence: list of lists of tracking predictions
    :param ground_truths_sequence: list of lists of ground truth
    :return: similarity scores for the current sequence
    """

    distance_matrix = []
    for predictions, ground_truths in zip(tracking_predictions_sequence, ground_truths_sequence):
        distance_matrix.append(mm.distances.iou_matrix(ground_truths['bboxes'], predictions['bboxes']))
    return distance_matrix


def compute_data_for_hota_metric(tracking_predictions_sequence, ground_truths_sequence):
    data = {'num_tracker_dets': get_num_dets(tracking_predictions_sequence),
            'num_gt_dets': get_num_dets(ground_truths_sequence),
            'num_tracker_ids': get_num_ids_from_sequence(tracking_predictions_sequence),
            'num_gt_ids': get_num_ids_from_sequence(ground_truths_sequence),
            'gt_ids': get_ids(ground_truths_sequence),
            'tracker_ids': get_ids(tracking_predictions_sequence),
            'similarity_scores': compute_similarity_scores_for_sequence(tracking_predictions_sequence,
                                                                        ground_truths_sequence)
            }
    return data


def reinitialise_hota_metric_variables(id_past_frame=None):
    """
    reinitialise the variables used to compute the hota metric. This is necessary because the metric is computed for
    each sequence. This function is used to reinitialise the variables when a new sequence starts.
    :return: the variables used to compute the hota metric
    """
    id_past_frame = id_past_frame
    tracking_predictions_sequence = []
    ground_truths_sequence = []

    return id_past_frame, tracking_predictions_sequence, ground_truths_sequence


def refactor_ids_in_sequence(sequence):
    """
    refactor the ids in the sequence. This function is used to refactor the ids in the sequence in order to work with
    the metric.
    :param sequence: list of lists of detections
    :return: list of lists of detections with refactored ids
    """
    # todo: hay que ordenar la lista de ids y mover las bboxes en acorde. esto es para que no hayan saltos entre los ids,
    #  e.g. que en una secuencia, el id 3 no aparezca nunca. esto es para que no pete el codigo del hota.py

    """
    # 1. get all the ids in the sequence and sort it
    ids_list = sorted(get_ids(sequence))

    new_bboxes = []
    new_ids = []
    for frame in sequence:
        # see if the id is in the list
        for id in frame['ids']:
            if id in ids_list:
                # get the index of the id in the list
                index = ids_list.index(frame['ids'])
                # get the bounding box of the id
                bbox = frame['bboxes'][index]
                # add the bounding box to the new list
                new_bboxes.append(bbox)
    """
    print('todo...')

    return sequence


def evaluate_sequences_hota_metric(all_tracking_predictions, all_ground_truths):
    """
    This function is used to compute the hota metric for all the sequences.
    :param all_tracking_predictions: list of lists of tracking predictions
    :param all_ground_truths: list of lists of ground truth
    :return: the results of computing the hota metric for all the sequences
    """
    # where all the metrics will be stored
    all_results = []

    # variables that will be used to compute the metrics
    id_past_frame, tracking_predictions_sequence, ground_truths_sequence = reinitialise_hota_metric_variables()

    # refactor ids and bboxes (being coherent) of the ground truth and tracking predictions of the sequence
    all_tracking_predictions = refactor_ids_in_sequence(all_tracking_predictions)
    all_ground_truths = refactor_ids_in_sequence(all_ground_truths)

    for predictions, ground_truths in zip(all_tracking_predictions, all_ground_truths):
        if id_past_frame is None:
            id_past_frame = predictions['video_id']

        elif id_past_frame != predictions['video_id']:
            # create other accumulator for computing hota metric
            hota_metric = HOTA()

            # generate the necessary data to compute the metrics
            data = compute_data_for_hota_metric(tracking_predictions_sequence, ground_truths_sequence)

            # compute the metrics
            res = hota_metric.eval_sequence(data)

            # append the result to the list of results
            all_results.append(res)

            # means that there is a change of sequence, reset the variables
            id_past_frame, tracking_predictions_sequence, ground_truths_sequence = reinitialise_hota_metric_variables(
                id_past_frame=predictions['video_id']
            )

        else:
            # add the predictions to the sequence
            tracking_predictions_sequence.append(predictions)
            ground_truths_sequence.append(ground_truths)

    return all_results


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