import os
from sort import sort
import motmetrics as mm
import json
from track import read_from_yolo, track_detections_frame

# !!!!!!! the functions have no utility now, it is just to look at pieces of code that I have already written !!!!!!!


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