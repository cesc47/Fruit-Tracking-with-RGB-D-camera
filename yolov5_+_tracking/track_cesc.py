import os
from sort import sort
import motmetrics as mm
import numpy as np

def track(path_detections, path_gt, tracker_type='Sort', img_size=(1920, 1080)):

    # Create the accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    # read detections from yolo output file
    all_detections = read_detections_from_yolo(path_detections, img_size)

    # create the tracker
    if tracker_type == 'Sort':
        tracker = sort.Sort()

    for detections in all_detections:
        # if there are no detections in the frame, skip it (all zeros)
        if len(detections) == 0:
            trackers = tracker.update(np.empty((0, 5)))
        else:
            # update the tracker with the detections
            trackers = tracker.update(output_results=np.array(detections), img_info=img_size, img_size=img_size)
        """
        accumulator.update(
            gt_ids,  # Ground truth objects in this frame
            det_ids,  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(gt_centers, det_centers)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )
        
    # Compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)
    """

def read_detections_from_yolo(path_detections, img_size=(1920, 1080)):
    """
    Reads detections from yolo output file.
    :param path_detections: path to yolo output file
    :param img_size: size of the image
    :return: list of detections
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
        detections = [
            (int(detection[0] * img_size[0]), int(detection[1] * img_size[1]), int(detection[2] * img_size[0]),
             int(detection[3] * img_size[1])) for detection in detections]

        # convert from (x, y, w, h) to (x1, y1, x2, y2)
        detections = [(detection[0], detection[1], detection[0] + detection[2], detection[1] + detection[3])
                      for detection in detections]

        # add detections to all_detections list
        all_detections.append(detections)

        return all_detections


if __name__ == '__main__':
    path_detections = 'yolov5/runs/detect/exp3/labels'
    path
    track(path_detections)
    print('Done')




