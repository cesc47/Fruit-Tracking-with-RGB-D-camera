import os
import csv

import motmetrics as mm
import numpy as np

from tools.utils import compute_centroids_bboxes_tlbr
from tools.TrackEval.trackeval.metrics import HOTA
from tools.visualization import visualize_tracking_results


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

    else:
        return None, None


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
    """
    computes the data for the hota metric. This function is used to compute the data for the hota metric. The data is
    computed for each frame.
    :param tracking_predictions_sequence: list of lists of tracking predictions
    :param ground_truths_sequence: list of lists of ground truth
    :return: data for the hota metric
    """
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


def refactor_ids_in_sequence(all_sequences, gt=False):
    """
    refactor the ids in the sequence (ALL sequences). This function is used to refactor the ids in the sequence in order
    to work with the metric.
    :param all_sequences: list of lists of detections
    :param gt: if True, the ids are refactored for the ground truth. If False, the ids are refactored for the tracking
    :return: list of lists of detections with refactored ids
    """

    sequence_ordered = []
    ids = []

    # get the ids in the sequence (there are some jumps between ids)
    for frame in all_sequences:
        for id in frame['ids']:
            if id not in ids:
                ids.append(id)

    # sort the ids
    ids.sort()

    # refactor the ids in the sequence. search for the id in the list and move the bbox to the right position
    for frame in all_sequences:
        if gt:
            frame_ordered = {
                'video_id': frame['id_video'],
                'bboxes': [],
                'ids': []
            }
        else:
            frame_ordered = {
                'video_id': frame['video_id'],
                'bboxes': [],
                'ids': []
            }
        for id in frame['ids']:
            frame_ordered['ids'].append(ids.index(id))
            frame_ordered['bboxes'].append(frame['bboxes'][frame['ids'].index(id)])
        sequence_ordered.append(frame_ordered)

    return sequence_ordered


def separate_all_sequences_by_video(all_sequences):
    """
    separate the sequences by video. This function is used to separate the sequences by video.
    :param all_sequences: list of lists of detections
    :return: list of lists of sequences
    """
    sequences_separated = []
    same_sequence = []
    past_id = None

    for sequence in all_sequences:
        if past_id is None:
            past_id = sequence['video_id']
            same_sequence.append(sequence)
        elif past_id != sequence['video_id']:
            sequences_separated.append(same_sequence)
            past_id = sequence['video_id']
            same_sequence = [sequence]
        else:
            same_sequence.append(sequence)

    sequences_separated.append(same_sequence)

    return sequences_separated


def refactor_ids_sequences_v2(all_sequences, gt=False):
    """
    refactor the ids in the sequence (for each sequence). This function is used to refactor the ids in the sequence in
    order to work with the metric.
    :param all_sequences: list of lists of detections
    :param gt: if True, the ids are refactored for the ground truth. If False, the ids are refactored for the tracking
    :return: list of lists of detections with refactored ids
    """
    sequence_refactored = []
    sequences_separated = separate_all_sequences_by_video(all_sequences)
    for sequence in sequences_separated:
        sequence = refactor_ids_in_sequence(sequence, gt)
        for sequence_frame in sequence:
            sequence_refactored.append(sequence_frame)

    return sequence_refactored


def reset_ids_sequence(sequence):
    """
    reset the ids in the sequence (for each sequence). This function is used to reset the ids in the sequence in order
    to work with the metric. ids must start in 0 and be progressive.
    :param sequence: list of lists of detections (sequence)
    """
    # get the value of the minimum id of the sequence
    min_id = 1000000000000
    for frame in sequence:
        for id in frame['ids']:
            if id < min_id:
                min_id = id

    # reset the ids in the sequence
    for idx, frame in enumerate(sequence):
        for id in frame['ids']:
            sequence[idx]['ids'][frame['ids'].index(id)] = id - min_id

    return sequence


def evaluate_sequences_hota_metric(all_tracking_predictions, all_ground_truths):
    """
    todo: clean this function... it is a mess. do it better by sequences (based as in refactor_ids_sequences_v2)
    todo: clean bug for when using multiplier of frames (it crashes here)
    This function is used to compute the hota metric for all the sequences.
    :param all_tracking_predictions: list of lists of tracking predictions
    :param all_ground_truths: list of lists of ground truth
    :return: the results of computing the hota metric for all the sequences
    """
    # where all the metrics will be stored
    all_results = []

    # variables that will be used to compute the metrics
    id_past_frame, tracking_predictions_sequence, ground_truths_sequence = reinitialise_hota_metric_variables()

    # apaño por si usamos deepsort (copiamos lo del frame posterior)
    for idx, tracking_predictions in enumerate(all_tracking_predictions):
            if (idx < len(all_tracking_predictions) - 1) and \
                    (not all_tracking_predictions[idx]['ids'] and not all_tracking_predictions[idx]['bboxes']):
                all_tracking_predictions[idx]['ids'] = all_tracking_predictions[idx+1]['ids']
                all_tracking_predictions[idx]['bboxes'] = all_tracking_predictions[idx+1]['bboxes']

    # refactor ids and bboxes (being coherent) of the ground truth and tracking predictions of the sequence
    all_tracking_predictions = refactor_ids_in_sequence(all_tracking_predictions, gt=False)
    all_ground_truths = refactor_ids_in_sequence(all_ground_truths, gt=True)
    all_tracking_predictions_refactorised = refactor_ids_sequences_v2(all_tracking_predictions, gt=False)

    for idx, (predictions, ground_truths) in enumerate(zip(all_tracking_predictions_refactorised, all_ground_truths)):
        if id_past_frame is None:
            id_past_frame = predictions['video_id']

        elif id_past_frame != predictions['video_id'] or idx == len(all_tracking_predictions_refactorised) - 1:
            # create other accumulator for computing hota metric
            hota_metric = HOTA()

            tracking_predictions_sequence = reset_ids_sequence(tracking_predictions_sequence)
            ground_truths_sequence = reset_ids_sequence(ground_truths_sequence)

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

        # add the predictions to the sequence
        tracking_predictions_sequence.append(predictions)
        ground_truths_sequence.append(ground_truths)

    return all_results


def save_tracking_results(results, results_hota, dataset_name, exp_name, tracker_type, partition, metrics, reid):
    """
    Save the tracking results in a csv file. The tracking results are saved in the folder 'results_tracking'
    :param results: tracking results to save (motmetrics)
    :param results_hota: tracking results to save (hota metric)
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment
    :param tracker_type: type of the tracker
    :param partition: partition of the dataset
    :param metrics: metrics of the tracking
    :param reid: if we use reid or not (string)
    """
    # create new folder for the results of the tracking if it does not exist
    if not os.path.exists(os.path.join(os.getcwd(), 'results_tracking')):
        os.mkdir(os.path.join(os.getcwd(), 'results_tracking'))

    # save the results in a csv file
    if reid is None:
        reid = 'no_reid'
    path_file = os.path.join('results_tracking', f'{dataset_name}_{exp_name}_{tracker_type}_{partition}_{reid}.csv')

    # if file already exists, delete it
    if os.path.exists(path_file):
        os.remove(path_file)

    videos_and_frames, total_frames = compute_frames_for_partition(partition)

    with open(path_file, 'a') as f:
        writer = csv.writer(f)
        metrics.insert(0, 'video_id')
        metrics.insert(1, 'nframes')
        metrics.append('hota')
        writer.writerow(metrics)
        for (result, _), hota, video_name, nframes in zip(results, results_hota, videos_and_frames["video_id"], videos_and_frames["nframes"]):
            # with 4 decimal places
            to_write = [round(value[0], 4) for _, value in result.items()]
            to_write.insert(0, video_name)
            to_write.insert(1, nframes)
            to_write.append(hota['HOTA'][0])
            writer.writerow(to_write)

    compute_mean_for_metrics(path_file, total_frames)
    delete_duplicated_row(path_file)


def compute_frames_for_partition(partition):
    """
    This function is used to compute the number of frames for each video in the yolo partition.
    :param partition: the partition of the dataset e.g train, val, test
    :return: a dictionary with the video_id and the number of frames for each video
    """

    # dict where all the number of frames for each video that are used in that specific partition will be stored.
    videos_and_frames = {
        'video_id': [],
        'nframes': [],
    }
    # read the yolo db and get the number of frames for each video
    path_to_db = os.path.join('../yolov5_+_tracking', 'datasets', 'Apple_Tracking_db_yolo', partition, 'images')

    past_img = None
    count = 0

    for img in sorted(os.listdir(path_to_db)):
        # get the name of the video
        img = img.split('_')[:-2]
        img = '_'.join(img)

        if past_img is None:
            past_img = img
            count += 1
        elif past_img == img:
            count += 1
        else:
            videos_and_frames['video_id'].append(past_img)
            videos_and_frames['nframes'].append(count)
            past_img = img
            count = 1

    # last video
    videos_and_frames['video_id'].append(past_img)
    videos_and_frames['nframes'].append(count)

    total_frames = sum(videos_and_frames['nframes'])

    return videos_and_frames, total_frames


def compute_mean_for_metrics(path_to_csv, total_frames):
    """
    This function modifies the .csv file to compute the mean of the metrics for each video taking the nframes as a
    weight.
    """
    # this row will have the mean of the metrics for each video
    row_to_append = []

    # read the csv file
    with open(path_to_csv, 'r') as f:
        reader = csv.reader(f)
        # get the headers
        headers = next(reader)
        # get the data
        data = list(reader)
        for row in data:
            row_to_append.append(row[1:])
        # transform the row_to_append to a numeric array
        row_to_append = np.array(row_to_append, dtype=float)
        # compute the mean of the metrics for each video
        mean_videos = row_to_append.mean(axis=0)
        # divide each element of row by the first element of the row (nframes) and ponderate the result by the total
        nframes = row_to_append[:, 0]
        for col in range(1, len(row_to_append[0])):
            row_to_append[:, col] = row_to_append[:, col] * nframes / total_frames
        # do the sum of the columns
        row_to_append = np.sum(row_to_append, axis=0)
        # convert the mean to a list
        row_to_append = row_to_append.tolist()
        mean_videos = mean_videos.tolist()
        # convert the list to string
        row_to_append = [str(i) for i in row_to_append]
        mean_videos = [str(i) for i in mean_videos]
        # add 'mean_video' to the fist element of the list
        row_to_append.insert(0, 'mean_video (weighted by nframes in each video)')
        mean_videos.insert(0, 'mean_video (without weight => division by number of videos)')
        # write the mean of the metrics for each video in the csv file
        writer = csv.writer(open(path_to_csv, 'a'))
        writer.writerow(row_to_append)
        writer.writerow(mean_videos)


def delete_duplicated_row(path_to_csv):
    """
    This function deletes the duplicated rows in the .csv file.
    :param path_to_csv: path to the .csv file
    """
    # read the csv file
    with open(path_to_csv, 'r') as f:
        reader = csv.reader(f)
        # get the headers
        headers = next(reader)
        # delete the second element of the headers
        headers.pop(1)
        # get the data
        data = list(reader)
        # delete the second column of the data (as i've seen that it is duplicated)
        for row in data:
            row.pop(1)
        # write the data in the csv file
        writer = csv.writer(open(path_to_csv, 'w'))
        writer.writerow(headers)
        writer.writerows(data)


def save_and_visualize(save_results, all_tracking_results, hota_metric_results, dataset_name, exp_name, tracker_type,
                       partition, metrics, visualize_results, all_tracking_predictions, ground_truths, reid):
    """
    Save the results of the tracking and visualize them. If save_results is True, the results are saved in a .csv file.
    :param save_results: save the results of the tracking (boolean)
    :param all_tracking_results: the results of the tracking (metrics)
    :param hota_metric_results: the results of the tracking (metrics)
    :param dataset_name: the name of the dataset
    :param exp_name: the name of the experiment
    :param tracker_type: the tracker type (sort, deepsort or bytetrack)
    :param partition: the partition of the dataset (train, test or val)
    :param metrics: the metrics to save (list of strings)
    :param visualize_results: visualize the results of the tracking (boolean)
    :param all_tracking_predictions: the predictions of the tracker (list of lists)
    :param ground_truths: the ground truths of the dataset (list of lists)
    :param reid: the reid (string or None)
    """
    # if save_results is True, then we save the results of the tracker and the detections
    if save_results:
        print('saving tracking results ...')
        save_tracking_results(all_tracking_results, hota_metric_results, dataset_name, exp_name, tracker_type,
                              partition, metrics, reid)

    # if visualize_results is True, then we visualize the results of the tracker and the detections
    if visualize_results and save_results:
        visualize_tracking_results(all_tracking_predictions, ground_truths, partition, dataset_name, plot_gts=False,
                                   plot_preds=True, save_video=True, path_to_save_video='results_tracking')

    elif visualize_results and not save_results:
        visualize_tracking_results(all_tracking_predictions, ground_truths, partition, dataset_name)






