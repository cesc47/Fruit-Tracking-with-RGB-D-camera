import os
import cv2
import numpy as np
import random
#from utils import convert_bbox_from_yolo_format

# located in data/Apple_tracking_db
GLOBAL_PATH_DB = './data/Apple_Tracking_db'


def plot_gt_bboxes_in_img(annotations, video_name, frame_num, draw_centroids=False, show_img=False):
    """
    Plot the ground truth bounding boxes in an image.
    :param annotations: loaded .json file from supervisely
    :param video_name: name of the video
    :param frame_num: frame number to plot
    :param draw_centroids: if True, draw centroids of the bounding boxes
    :param show_img: if True, show the image
    :return:
    rgb_img: rgb image
    labels: list of labels
    """


    # load the annotations for that specific frame
    annotation = annotations['frames'][frame_num]
    path_images = os.path.join(GLOBAL_PATH_DB, video_name, 'images')

    # read all path_images sorted
    images_sorted = sorted(os.listdir(path_images))
    # get the number before the C.png string
    list_frame_numbers = []
    for image in images_sorted:
        frame_num_str = image.split('.')[0]
        frame_num_str = frame_num_str.split('_')[-2]
        frame_num_str = int(frame_num_str)
        list_frame_numbers.append(frame_num_str)

    image_name = '_'.join(images_sorted[0].split('_')[:-2]) + '_' + str(min(list_frame_numbers)+frame_num) + '_C.png'

    # rgb images are sorted by order and appear every 3 (rgb, depth, IR):
    # read the selected image
    path_rgb_img = os.path.join(path_images, image_name)
    rgb_img = cv2.imread(path_rgb_img)

    labels = {
        'centroid': [],
        'id': [],
    }

    for figure in annotation['figures']:
        # get the bounding box for each figure
        xtl, ytl = figure['geometry']['points']['exterior'][0]
        xbr, ybr = figure['geometry']['points']['exterior'][1]
        # draw the bounding box
        cv2.rectangle(rgb_img, (xtl, ytl), (xbr, ybr), (255, 0, 0), 2)
        # get the centroid of the bounding box
        centroid = (int((xtl + xbr) / 2), int((ytl + ybr) / 2))
        # draw the centroid
        if draw_centroids:
            cv2.circle(rgb_img, centroid, 2, (0, 0, 255), -1)
        # save the centroid and the id of the figure
        labels['centroid'].append(centroid)
        labels['id'].append(figure['objectKey'])

    if show_img:
        cv2.imshow('image', rgb_img)
        cv2.waitKey(0)

    return rgb_img, labels


def plot_gt_bboxes_in_video(annotations, video_name, init_frame, end_frame, plot_tail=True):
    """
    Plot the ground truth bounding boxes in a video.
    :param annotations: loaded .json file from supervisely
    :param video_name: name of the video
    :param init_frame: first frame to plot
    :param end_frame: last frame to plot
    :param plot_tail: if True, plot the tail of tracklets
    """
    # make sure that init_frame is lower than end_frame
    if init_frame > end_frame:
        raise AssertionError('init_frame must be lower than end_frame')

    # make sure that end_frame is in the range of the video
    if end_frame > len(annotations['frames']):
        raise AssertionError('end_frame must be lower than the number of frames')

    # process imgs (draw bounding boxes in the image)
    rgb_imgs = []
    all_labels = []
    if plot_tail:
        tails = {
            'id': [],
            'centroids': [],
            'color': [],
        }

    for idx_frame in range(init_frame, end_frame+1):
        rgb_img, labels = plot_gt_bboxes_in_img(annotations, video_name, frame_num=idx_frame)
        rgb_imgs.append(rgb_img)
        if plot_tail:
            for centroid, id in zip(labels['centroid'], labels['id']):
                # if the id is not in the tails, add it
                if id not in tails['id']:
                    tails['id'].append(id)
                    tails['centroids'].append([centroid])
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    tails['color'].append(color)    # random color for each tail
                else:
                    # if the id is in the tails, add the centroid
                    idx = tails['id'].index(id)
                    tails['centroids'][idx].append(centroid)

            # plot the tails
            for _, centroids, color in zip(tails['id'], tails['centroids'], tails['color']):
                # unify the centroids with cv2.polylines (plot the tail)
                cv2.polylines(rgb_imgs[-1], np.int32([centroids]), False, color, 1)

            # delete tails that are not in the current frame
            for idx, id in enumerate(tails['id']):
                if id not in labels['id']:
                    del tails['id'][idx]
                    del tails['centroids'][idx]
                    del tails['color'][idx]


    # press 0 to show images
    for img in rgb_imgs:
        # img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


def plot_bboxes_in_img_yolo(img_name, split='training'):
    """
    Plot the bounding boxes in an image using YOLO format for the labels.
    :param img_name: name of the image
    :param split: 'training' or 'valid' or 'test'
    """
    path = os.path.join(os.getcwd(), 'yolov5_+_tracking', 'datasets', 'Apple_Tracking_db_yolo', split)

    # see if the image exists
    path_to_img = os.path.join(path, 'images', img_name + '.png')
    if not os.path.isfile(path_to_img):
        raise AssertionError('Image not found')

    path_to_label = os.path.join(path, 'labels', img_name + '.txt')
    if not os.path.isfile(path_to_img):
        raise AssertionError('Label not found')

    # load the image
    img = cv2.imread(path_to_img)

    # load the labels
    with open(path_to_label, 'r') as f:
        labels = f.readlines()

    # process labels
    for label in labels:
        label = label.split()
        label = [float(x) for x in label]
        _, x, y, h, w = label
        x_min, y_min, x_max, y_max = convert_bbox_from_yolo_format(x, y, h, w)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # press 0 to show images
    cv2.imshow('img', img)
    key = cv2.waitKey(0)


def visualize_tracking_results_frame(img_name, path_to_imgs, ground_truth, prediction, plot_gts=True, plot_preds=True):
    """
    Visualize the tracking results for a single frame. The ground truth and the prediction are dicts.
    :param img_name: name of the image
    :param path_to_imgs: path to the images
    :param ground_truth: ground truth: dict with keys 'id' and 'bboxes' from the ground truth
    :param prediction: prediction: dict with keys 'id' and 'bboxes' from the detections
    :param plot_gts: if True, plot the ground truth
    :param plot_preds: if True, plot the prediction
    :return: the image with the tracking results
    """
    # load the image
    img = cv2.imread(os.path.join(path_to_imgs, img_name))

    # iterate over the ground truth and prediction dicts
    if plot_gts:
        for bbox, id in zip(ground_truth['bboxes'], ground_truth['ids']):
            # get the bounding box coordinates
            x_min, y_min, x_max, y_max = bbox
            # plot the bounding box: ground truth in green
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # plot the label: ground truth in green
            cv2.putText(img, str(id), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if plot_preds:
        for bbox, id in zip(prediction['bboxes'], prediction['ids']):
            # get the bounding box coordinates
            x_min, y_min, x_max, y_max = bbox
            # plot the bounding box: prediction in red
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            # plot the label: prediction in red
            cv2.putText(img, str(id), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # press 0 to show images
    cv2.imshow('img', img)
    key = cv2.waitKey(0)

    return img


def visualize_tracking_results(tracking_predictions, ground_truths, partition, dataset_name, plot_gts=True,
                               plot_preds=True, save_video=False, path_to_save_video=None):
    """
    Visualize the tracking results for all frames The ground truth and the prediction are a list of dicts.
    :param tracking_predictions: list of predictions for all frames (dicts)
    :param ground_truths: list of ground truth for all frames (dicts)
    :param partition: 'training' or 'valid' or 'test'
    :param dataset_name: name of the dataset
    :param plot_gts: True if you want to plot the ground truth
    :param plot_preds: True if you want to plot the predictions
    :param save_video: True if you want to save the video
    :param path_to_save_video: path to save the video
    """
    # get the path to the images
    # images are located in the yolov5+tracking folder
    path_to_imgs = os.path.join('../yolov5_+_tracking', 'datasets', dataset_name, partition, 'images')

    # where the imgs with the tracking results will be saved
    imgs = []

    # read all images from the folder
    for idx, img_name in enumerate(sorted(os.listdir(path_to_imgs))):
        img = visualize_tracking_results_frame(img_name, path_to_imgs, ground_truths[idx], tracking_predictions[idx],
                                               plot_gts=plot_gts, plot_preds=plot_preds)
        if save_video:
            imgs.append(img)

    if save_video:
        save_frames_to_video(imgs, path_to_save_video)


def save_frames_to_video(imgs, path):
    """
    Save a video from a list of images.
    :param imgs: list of images
    :param path: path to the video
    :param partition: 'training' or 'valid' or 'test'
    :param dataset_name: name of the dataset
    """
    # the path
    path = os.path.join(path, 'tracking_results.mp4')

    # create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 4, (imgs[0].shape[1], imgs[0].shape[0]))

    # write the images to the video0
    for img in imgs:
        out.write(img)

    # release the video writer
    out.release()


if __name__ == "__main__":
    """
    video_name = '210928_094225_k_r2_e_015_175_162'
    annotations = read_segmentation(video_name)
    plot_gt_bboxes_in_video(annotations, video_name, init_frame=0, end_frame=20)
    """
    """
    video_name = '210906_121930_k_r2_e_015_125_162'
    annotations = read_segmentation(video_name)
    # plot_gt_bboxes_in_img(annotations, video_name, frame_num=0, show_img=True)
    plot_gt_bboxes_in_video(annotations, video_name, init_frame=0, end_frame=13)
    """

    img = '210726_170244_k_r2_a_015_225_162_697_172_C'
    plot_bboxes_in_img_yolo(img, split='test')
