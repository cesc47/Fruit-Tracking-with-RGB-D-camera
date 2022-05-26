import os
from read_segmentation import read_segmentation
import cv2
import numpy as np
import random

# located in data/Apple_tracking_db
GLOBAL_PATH_DB = './data/Apple_Tracking_db'


def plot_gt_bboxes_in_img(annotations, video_name, frame_num, draw_centroids=False, show_img=False):
    """
    Plot the ground truth bounding boxes in an image.
    :param annotations: loaded .json file from supervisely
    :param video_name: name of the video
    :param frame_num: frame number to plot
    :param show_img: if True, show the image
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
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

if __name__ == "__main__":
    """
    video_name = '210928_094225_k_r2_e_015_175_162'
    annotations = read_segmentation(video_name)
    plot_gt_bboxes_in_video(annotations, video_name, init_frame=0, end_frame=20)
    """

    video_name = '210726_180555_k_r2_a_060_225_162'
    annotations = read_segmentation(video_name)
    # plot_gt_bboxes_in_img(annotations, video_name, frame_num=0, show_img=True)
    plot_gt_bboxes_in_video(annotations, video_name, init_frame=0, end_frame=13)
