from tqdm import tqdm
import cv2
from track import create_tracker, track_detections_frame
from tools.metrics import *
from tools.utils import convert_bbox_from_yolo_format, filter_detections_by_size, augment_size_of_bboxes, \
    order_detections_folder_nums, reduce_size_of_bboxes_in_tracking_results
from tools.dataset_gestions import delete_depth_maps, rotate_imgs


def visualize_tracking_results_frame(img_name, path_to_imgs, prediction, plot_preds=True):
    """
    Visualize the tracking results for a single frame. The ground truth and the prediction are dicts.
    :param img_name: name of the image
    :param path_to_imgs: path to the images
    :param prediction: prediction: dict with keys 'id' and 'bboxes' from the detections
    :param plot_preds: if True, plot the prediction
    :return: the image with the tracking results
    """
    # load the image
    img = cv2.imread(os.path.join(path_to_imgs, img_name))
    # initialize 20 different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)
                , (0, 0, 0), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
                , (128, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64)]

    if plot_preds:
        for bbox, id in zip(prediction['bboxes'], prediction['ids']):
            color = colors[id % len(colors)]
            # get the bounding box coordinates
            x_min, y_min, x_max, y_max = bbox
            # plot the bounding box: prediction in red
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
            # plot the label: prediction in red
            cv2.putText(img, str(id), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # press 0 to show images
    cv2.imshow('img', img)
    key = cv2.waitKey(0)

    return img


def visualize_tracking_results(tracking_predictions, dataset_name,  plot_preds=True, save_video=True,
                               path_to_save_video=None):
    """
    Visualize the tracking results for all frames The ground truth and the prediction are a list of dicts.
    :param tracking_predictions: list of predictions for all frames (dicts)
    :param dataset_name: name of the dataset
    :param plot_preds: True if you want to plot the predictions
    :param save_video: True if you want to save the video
    :param path_to_save_video: path to save the video
    """
    # get the path to the images
    # images are located in the yolov5+tracking folder
    path_to_imgs = os.path.join('../data', 'videos_demostratius', dataset_name)

    # where the imgs with the tracking results will be saved
    imgs = []

    numbers = order_detections_folder_nums(path_to_imgs)

    # read all images from the folder
    for idx, number in enumerate(numbers):

        img_name = os.listdir(path_to_imgs)[0].split('_')[:-2]
        img_name = '_'.join(img_name)
        img_name += f'_{number}_C.png'

        img = visualize_tracking_results_frame(img_name, path_to_imgs, tracking_predictions[idx], plot_preds=plot_preds)
        if save_video:
            imgs.append(img)

    if save_video:
        save_frames_to_video(imgs, path_to_save_video)


def save_frames_to_video(imgs, path):
    """
    Save a video from a list of images.
    :param imgs: list of images
    :param path: path to the video
    """
    # the path
    path = os.path.join(path, 'tracking_results.mp4')

    # create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30, (imgs[0].shape[1], imgs[0].shape[0]))

    # write the images to the video0
    for img in imgs:
        out.write(img)

    # release the video writer
    out.release()


def read_from_yolo(path, filter_detections=True, augment_bboxes=0.075):
    """
    Reads detections from yolo output file.
    :param path: path to yolo output file
    :param filter_detections: if True, the detections are filtered by size (size hardcoded in utils.py, in function of
    the distance between the camera and the apples: 125, 175 or 225), are hardcoded thanks to compute_sizes_all_gts()
    :param augment_bboxes: increment the size of bboxes by this value (augment_bboxes * 100 (%))
    :param ground_truth: if True, the ground truth annotations are read instead of the detections
    :return: list of detections/ground truths (for all frames)
    """
    # this is where all the results (labels or detections) from all the frames will be stored
    all_results = []

    # where all the names of the videos are stored
    videonames = []

    # path to db
    path_to_db = '../data/Apple_Tracking_db'

    # for all the files in the folder read the detections
    # sort the files by name
    numbers = order_detections_folder_nums(path)

    for number in numbers:
        # read detections from frame
        detections_file = os.listdir(path)[0].split('_')[:-2]
        detections_file = '_'.join(detections_file)
        detections_file += f'_{number}_C.txt'

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

        # filter detections with size (size hardcoded in utils.py, in function of the distance between the camera
        # and the apples: 125, 175 or 225), are hardcoded thanks to compute_sizes_all_gts() function in utils.py
        if filter_detections:
            detections = filter_detections_by_size(detections, detections_file)

        detections = augment_size_of_bboxes(detections, percentage_to_augment=augment_bboxes)

        # add detections to all_detections list
        all_results.append(detections)

        # add the name of the video to the list
        videonames.append(detections_file.split('.')[0])

    return all_results, videonames


def initialise_data(exp_name):
    """
    Initialise the data in order to perform the tracking loop.
    :param exp_name: the name of the experiment
    :return: the data inicialised
    """
    # where will be stored the predictions of the tracker
    all_tracking_predictions = []

    # tracker referenced here
    tracker = None

    # read detections from yolo output files
    # path_detections is the folder where the detections from yolo are stored
    all_detections, video_names_det = read_from_yolo(os.path.join('yolov5', 'runs', 'detect', exp_name, 'labels'),
                                                     filter_detections=True)

    return all_tracking_predictions, all_detections, video_names_det, tracker


def track_yolo_results(exp_name, dataset_name, tracker_type='sort', reid=None):
    """
    Performs the tracking in the test dataset from yolo. It is simmilar to track() function but now it does not take as
    ground truth the labels from supervisely (.json) but the labels from yolo (.txt)
    :param dataset_name: name of the dataset
    :param exp_name: name of the experiment (in yolo folder)
    :param tracker_type: type of tracker (e.g. sort, bytetrack)
    :param reid: use reid network to track. If None, no reid network is used or use reid by default in deepsort case.
    """

    # initialise the data
    all_tracking_predictions, all_detections, video_names_det, tracker = initialise_data(exp_name)

    # create the tracker
    tracker, accumulator = create_tracker(tracker_type=tracker_type, reid=reid)

    # iterate for each img:
    print('performing the tracking loop...')
    for idx_frame, detections in tqdm(enumerate(all_detections)):
        # if video_id is not the same as the current video_id, then we have to reset the tracker

        # perform the tracking
        det_centers, det_ids, all_tracking_predictions = track_detections_frame(predictions=all_tracking_predictions,
                                                                                detections=detections,
                                                                                tracker=tracker,
                                                                                tracker_type=tracker_type,
                                                                                anterior_video_id='None',
                                                                                frame_name=video_names_det[idx_frame],
                                                                                reid=reid)

    #reduce_size_of_bboxes_in_tracking_results(all_tracking_predictions, percentage_to_reduce=0.0375)

    visualize_tracking_results(all_tracking_predictions, dataset_name, plot_preds=True, save_video=True,
                               path_to_save_video='../data')


    return all_tracking_predictions


if __name__ == "__main__":
    path = os.path.join('../data', 'videos_demostratius')
    delete_depth_maps(path)
    # rotate_imgs(path)
    track_yolo_results(exp_name='video2_demostratiu_yolov5x',
                       dataset_name='210928_165030_k_r2_w_015_125_162',
                       tracker_type='bytetrack',
                       reid=None)