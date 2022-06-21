import json
import os

# located in data/Apple_tracking_db
GLOBAL_PATH_DB = '../data/Apple_Tracking_db/'


def read_segmentation(video_name, path_db=GLOBAL_PATH_DB):
    """
    Reads the segmentation file and returns the segmentation.
    params:
        video_name: name of the video
    returns:
        segmentation: parsed json containing the annotations
    """

    path_ann = os.path.join(path_db, video_name, 'segmentation', 'ann.json')
    with open(path_ann) as fh:
        data = fh.read()
    annotations = json.loads(data)

    return annotations

