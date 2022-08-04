import os

import cv2
import json
import shutil
import pickle
import random
import torch

import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import torchvision.transforms as transforms
from tools.utils import *
import sys
import matplotlib.pyplot as plt

def refactor_id_frames_extractor():
    """
    This function deletes the frames (goten by 'frame extractor') that are not labeled.
    """

    # path to db
    path = '../data/Apple_Tracking_db'

    # read excel that contains all the information about the frames
    df = pd.read_excel(os.path.join(path, 'db_info.xlsx'))

    # Important info from the excel
    videos_names = df['Video'].unique()

    for idx, video_name in enumerate(videos_names):
        if str(video_name) != 'nan':
            path_to_frames = os.path.join(path, video_name, 'images')
            path_to_segmentation = os.path.join(path, video_name, 'segmentation')
            # if frames are extracted and labeling is done
            if os.path.exists(path_to_segmentation) and os.path.exists(path_to_frames):
                # get row index from the df
                row_idx = df[df['Video'] == video_name].index.values[0]
                # the initial frame on which the extractor starts
                frame_init = df.loc[row_idx, 'Frame inicial extractor frames']
                # number of frames labeled
                nframes = df.loc[row_idx, 'NFrames'] - 1  # -1: apaño por que he cambiado una cosa...!
                # get the path files inside the folder
                path_frames = os.listdir(path_to_frames)

                # only refactor when the number of frames labeled is different from the number of frames extracted.
                # we divide by 3 because we have 3 channels, color, depth and IR
                if len(path_frames)/3 != nframes + 1:
                    print(f'Refactoring frames from => {video_name}...')
                    for path_frame in path_frames:
                        # split the string with _ amd get the number of the frame
                        num = path_frame.split('_')[-2]
                        if int(num) < int(frame_init) or int(num) > int(frame_init + nframes):
                            # remove the file
                            os.remove(os.path.join(path_to_frames, path_frame))


def rotate_images(path_to_images, clockwise=True, test=False):
    """
    This function rotates the images in the folder.
    :param path_to_images: path to the folder with the images
    :param clockwise: if True, the image will be rotated clockwise, otherwise counterclockwise
    :param test: if True, the function will not rotate the images, but will just print the path to the images
    """

    # path to db
    path = '../data/Apple_Tracking_db'

    # read all files in that folder
    path_files = os.path.join(path, path_to_images, 'images')

    for file in tqdm(os.listdir(path_files)):
        # check if the extension is .png or .mat
        if file.endswith('.png'):
            # read the image
            img = cv2.imread(os.path.join(path_files, file))
            # rotate the image. can be rotated clockwise or counterclockwise
            if clockwise:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            else:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            if test:
                # see if everything is ok
                cv2.imshow('image', img)
                cv2.waitKey(0)
            else:
                # delete the file
                os.remove(os.path.join(path_files, file))
                # save the image rotated in the same folder
                cv2.imwrite(os.path.join(path_files, file), img)
        else:
            # we are dealing with a .mat file
            file_name = os.path.join(path_files, file)
            # read .mat file
            mat = sio.loadmat(file_name)
            # rotate the mat file

            # if the filename ends with D.mat
            if file.endswith('D.mat'):
                if clockwise:
                    mat = np.rot90(mat['transformed_depth'], k=3)
                else:
                    mat = np.rot90(mat['transformed_depth'], k=1)

            # filename ends with I.mat
            else:
                if clockwise:
                    mat = np.rot90(mat['transformed_ir'], k=3)
                else:
                    mat = np.rot90(mat['transformed_ir'], k=1)

            if test:
                # show the image
                cv2.imshow('image', mat)
                cv2.waitKey(0)
            else:
                # delete the file
                os.remove(os.path.join(path_files, file))
                # save the mat file rotated
                if file.endswith('D.mat'):
                    sio.savemat(os.path.join(path_files, file), {'transformed_depth': mat})
                else:
                    sio.savemat(os.path.join(path_files, file), {'transformed_ir': mat})


def rotate_segmentation():
    """
    This function rotates the segmentation of the images. takes into account all the videos on the dataset and the
    type of rotation that is done (clockwise or counterclockwise). It just modifies the json file ann.json for each
    video.
    """

    # read info_rotations.txt, it has the info if the rotations of the video are clockwise or counterclockwise
    with open('../data/Apple_Tracking_db/info_rotations.txt', 'r') as f:
        info = f.readlines()

    # get the info
    for line in info:
        # get the path to the folder with the images
        videoname = line.split(' ')[0]
        print(f'refactoring {videoname} ...')

        # get the info if the rotation is clockwise or counterclockwise
        clockwise = line.split(' ')[1] == 'True\n'

        # read segmentation from the function in read segmentation.py
        annotations = read_segmentation(videoname)

        # modify 'size' in the annotations, harcoded for the apple tracking dataset (1920, 1080), only if json has
        # not been modified

        if annotations['size']['height'] == 1080 and annotations['size']['width'] == 1920:
            annotations['size']['height'] = 1920
            annotations['size']['width'] = 1080

            # rotate the bounding box 90 degrees according to the rotation of the image too
            for idx_annotation, annotation in enumerate(annotations['frames'][:]):
                for idx_figure, figure in enumerate(annotation['figures']):
                    xtl, ytl = figure['geometry']['points']['exterior'][0]
                    xbr, ybr = figure['geometry']['points']['exterior'][1]
                    # rotate the bounding box
                    xtl_new, ytl_new, xbr_new, ybr_new = rotate_and_transform_bbox(xtl, ytl, xbr, ybr, clockwise)
                    # modify the bounding box in the annotations
                    annotations['frames'][idx_annotation]['figures'][idx_figure]['geometry']['points']['exterior'][0] = \
                        [xtl_new, ytl_new]
                    annotations['frames'][idx_annotation]['figures'][idx_figure]['geometry']['points']['exterior'][1] = \
                        [xbr_new, ybr_new]

            # to test if everything is ok
            # plot_gt_bboxes_in_video(annotations, videoname, init_frame=0, end_frame=13)

            # save the annotations in the new folder
            path_to_save = os.path.join('../data/Apple_Tracking_db/', videoname, 'segmentation')

            # delete ann.json (the annotations that were not rotated)
            os.remove(os.path.join(path_to_save, 'ann.json'))

            # save the annotations in the new folder
            with open(os.path.join(path_to_save, 'ann.json'), 'w') as f:
                json.dump(annotations, f)


def change_name_imgs_in_folder(video_folder):
    """
    This function changes a character in the name of the images in a folder.
    :param video_folder: the folder where the images are
    """
    for file in os.listdir(os.path.join('../data/Apple_Tracking_db/', video_folder, 'images')):
            # change the name of the image: the 13th character change from '0' to '1'
            os.rename(os.path.join('../data/Apple_Tracking_db/', video_folder, 'images', file),
                        os.path.join('../data/Apple_Tracking_db/', video_folder, 'images', file[:12] + '1' + file[13:]))


def delete_depth_maps(path):
    """
    delete all depth maps in the path. This is used to clean up the directory before running the code.
    :param path: path to the directory
    """
    for video in os.listdir(path):
        for frame in os.listdir(os.path.join(path, video)):
            if frame.endswith(".mat"):
                os.remove(os.path.join(path, video, frame))


def rotate_imgs(path):
    """
    Rotate the images in the path, 90 degrees counterclockwise.
    :param path: path to the images
    """
    for video in os.listdir(path):
        for file in os.listdir(os.path.join(path, video)):
            if file.endswith('.png'):
                # read the image
                img = cv2.imread(os.path.join(path, video, file))
                # rotate the image. can be rotated clockwise or counterclockwise
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                # delete the file
                os.remove(os.path.join(path, video, file))
                # save the image rotated in the same folder
                cv2.imwrite(os.path.join(path, video, file), img)


def generate_yolo_labels():
    """
    This function generates the labels for the yolo format. It takes into account all the videos on the dataset. It
    generates the labels for each video and saves them in the folder 'labels_yolo_format' of each video.
    Thanks to that we can use the labels in the yolo format to train the yolo model, and ingest it into Roboflow
    (img+label for each frame). Then, roboflow (https://app.roboflow.com/) will ingest the labels and the images in the
    same way and it will generate the dataset for us.
    """

    # img params (of the refactorized dataset, imgs in vertical)
    img_size = [1080, 1920]

    # path to db
    path = '../data/Apple_Tracking_db'

    for video_name in os.listdir(path):
        if not (video_name.endswith('.xlsx') or video_name.endswith('.txt')):
            # path to the video
            path_to_annotations = os.path.join(path, video_name, 'segmentation')

            # delete the old labels
            if os.path.exists(os.path.join(path_to_annotations, 'labels_yolo_format')):
                shutil.rmtree(os.path.join(path_to_annotations, 'labels_yolo_format'))

            # create folder in path_to_annotations to save the labels if it does not exist
            if not os.path.exists(os.path.join(path_to_annotations, 'labels_yolo_format')):
                os.makedirs(os.path.join(path_to_annotations, 'labels_yolo_format'))

            # read segmentation
            annotations = read_segmentation(video_name, path_db=path)

            # get the index of the images in the video (to save them in .txt in the same format)
            # todo: arreglar esto por si alguna vez se cogen los frames que no sean consecutivos
            str_video, index = get_gt_range_index_imgs(video_name, path=path)

            for annotation in annotations['frames'][:]:
                # variable where the bboxes of that frame in yolo format will be stored
                figures_list = []

                for figure in annotation['figures']:
                    xtl, ytl = figure['geometry']['points']['exterior'][0]
                    xbr, ybr = figure['geometry']['points']['exterior'][1]

                    # convert the bounding box to the yolo format
                    x, y, w, h = convert_bbox_to_yolo_format(size_img=img_size, bbox=[xtl, xbr, ybr, ytl])

                    # add the bounding box to the list
                    figures_list.append([x, y, w, h])

                # write the labels in the txt file
                text = ""
                for cx, cy, w, h in figures_list:
                    text = text + "0 " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + "\n"
                text.strip()

                # save the labels in the new folder
                with open(os.path.join(path_to_annotations, 'labels_yolo_format',
                                       str_video + '_' + str(index) + '_C' + '.txt'), 'w') as f:
                    f.write(text)

                index += 1


def create_custom_db_for_yolo(path='../data/Apple_Tracking_db',
                              path_to_new_db='../yolov5_+_tracking/datasets/Apple_Tracking_db_yolo',
                              percentage_training=0.6, percentage_space=0.2,
                              percentage_validation=0.1, percentage_test=0.1):
    """
    This function creates a custom dataset from the Apple_tracking_db for the yolo model. It will create a folder
    with the images and the labels in the same format as the dataset for the yolo model. It will also create a
    .txt file with the ids of the images in the db. The images will be saved in the same folder as the labels.
    :param path: path to the db
    :param path_to_new_db: path to the new db
    :param percentage_training: percentage of the dataset that will be used for training
    :param percentage_space: percentage of the dataset that will be used for space
    :param percentage_validation: percentage of the dataset that will be used for validation
    :param percentage_test: percentage of the dataset that will be used for test
    """

    # make sure that the percentages sum 1
    if percentage_test + percentage_validation + percentage_space + percentage_training != 1:
        raise AssertionError('percentages are not equal to 1')

    # --------FOLDER CREATION----------
    # create the new folder if it does not exist
    if not os.path.exists(path_to_new_db):
        os.makedirs(path_to_new_db)

    partitions = ['training', 'valid', 'test']

    # create the new folders for the new db
    for partition in partitions:
        if not os.path.exists(os.path.join(path_to_new_db, partition)):
            os.makedirs(os.path.join(path_to_new_db, partition))
            os.makedirs(os.path.join(path_to_new_db, partition, 'images'))
            os.makedirs(os.path.join(path_to_new_db, partition, 'labels'))

    # --------IMAGES AND LABELS CREATION----------
    print(os.listdir(path))
    for video_name in os.listdir(path):
        if not (video_name.endswith('.xlsx') or video_name.endswith('.txt')):
            print(f'Processing video {video_name} ...')
            # needed to order the images in the same order as the labels
            list_nums = []
            img = None
            for img in sorted(os.listdir(os.path.join(path, video_name, 'images'))):
                # sorting by string appears to not work, so we sort by the number of the image
                if img.endswith('.png'):
                    frame_num_str = img.split('.')[0]
                    frame_num_str = frame_num_str.split('_')[-2]
                    frame_num_str = int(frame_num_str)
                    list_nums.append(frame_num_str)

            # sort the list of numbers
            list_nums.sort()
            # split the string of img and remove the last element (the extension)
            str_video = '_'.join(img.split('_')[:-2])

            # iterate in the list only in the first 60% of the images
            for index, frame_num in enumerate(list_nums):
                # image goes into the training folder
                if index < int(len(list_nums) * percentage_training):
                    # copy the image to the new folder
                    shutil.copy(os.path.join(path, video_name, 'images', str_video + '_' + str(frame_num) + '_C.png'),
                                os.path.join(path_to_new_db, 'training', 'images',
                                             str_video + '_' + str(frame_num) + '_C.png'))
                    # copy the label to the new folder
                    shutil.copy(os.path.join(path, video_name, 'segmentation', 'labels_yolo_format',
                                             str_video + '_' + str(frame_num) + '_C.txt'),
                                os.path.join(path_to_new_db, 'training', 'labels',
                                             str_video + '_' + str(frame_num) + '_C.txt'))

                # image goes into the validation folder (leaving out the percentage_space)
                elif int(len(list_nums) * (percentage_training + percentage_space)) < index < \
                        int(len(list_nums) * (percentage_training + percentage_space + percentage_validation)):
                    # copy the image to the new folder
                    shutil.copy(os.path.join(path, video_name, 'images', str_video + '_' + str(frame_num) + '_C.png'),
                                os.path.join(path_to_new_db, 'valid', 'images',
                                             str_video + '_' + str(frame_num) + '_C.png'))
                    # copy the label to the new folder
                    shutil.copy(os.path.join(path, video_name, 'segmentation', 'labels_yolo_format',
                                             str_video + '_' + str(frame_num) + '_C.txt'),
                                os.path.join(path_to_new_db, 'valid', 'labels',
                                             str_video + '_' + str(frame_num) + '_C.txt'))

                # image goes into the test folder
                elif index >= int(len(list_nums) * (percentage_training + percentage_space + percentage_validation)):
                    # copy the image to the new folder
                    shutil.copy(os.path.join(path, video_name, 'images', str_video + '_' + str(frame_num) + '_C.png'),
                                os.path.join(path_to_new_db, 'test', 'images',
                                             str_video + '_' + str(frame_num) + '_C.png'))
                    # copy the label to the new folder
                    shutil.copy(os.path.join(path, video_name, 'segmentation', 'labels_yolo_format',
                                             str_video + '_' + str(frame_num) + '_C.txt'),
                                os.path.join(path_to_new_db, 'test', 'labels',
                                             str_video + '_' + str(frame_num) + '_C.txt'))


def generate_yolo_labels_and_ids():
    """
    This function does the same than generate_yolo_labels() but writes also the ids of the images too.
    """

    # img params (of the refactorized dataset, imgs in vertical)
    img_size = [1080, 1920]

    # path to db
    path = '../data/Apple_Tracking_db'

    # list_ids is a list of all the ids of the videos
    list_ids = []

    for video_name in os.listdir(path):
        if not (video_name.endswith('.xlsx') or video_name.endswith('.txt')):
            # path to the video
            path_to_annotations = os.path.join(path, video_name, 'segmentation')

            # delete the old labels
            if os.path.exists(os.path.join(path_to_annotations, 'labels_yolo_format+ids')):
                shutil.rmtree(os.path.join(path_to_annotations, 'labels_yolo_format+ids'))

            # create folder in path_to_annotations to save the labels if it does not exist
            if not os.path.exists(os.path.join(path_to_annotations, 'labels_yolo_format+ids')):
                os.makedirs(os.path.join(path_to_annotations, 'labels_yolo_format+ids'))

            # read segmentation from the function in read segmentation.py
            annotations = read_segmentation(video_name)

            # get the index of the images in the video (to save them in .txt in the same format)
            # todo: arreglar esto por si alguna vez se cogen los frames que no sean consecutivos
            str_video, index = get_gt_range_index_imgs(video_name, path)

            for annotation in annotations['frames'][:]:
                # variable where the bboxes and the ids of that frame in yolo format will be stored
                figures_list = []

                for figure in annotation['figures']:
                    # add objectkey to the list if it is not already there
                    if figure['objectKey'] not in list_ids:
                        list_ids.append(figure['objectKey'])
                    # get the index of the objectKey in the list
                    index_id = list_ids.index(figure['objectKey'])

                    xtl, ytl = figure['geometry']['points']['exterior'][0]
                    xbr, ybr = figure['geometry']['points']['exterior'][1]

                    # convert the bounding box to the yolo format
                    x, y, w, h = convert_bbox_to_yolo_format(size_img=img_size, bbox=[xtl, xbr, ybr, ytl])

                    # add the bounding box to the list
                    figures_list.append([index_id, x, y, w, h])

                # write the labels in the txt file
                text = ""
                for id, cx, cy, w, h in figures_list:
                    text = text + "0 " + str(id) + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + "\n"
                text.strip()

                # save the labels in the new folder
                with open(os.path.join(path_to_annotations, 'labels_yolo_format+ids',
                                       str_video + '_' + str(index) + '_C' + '.txt'), 'w') as f:
                    f.write(text)

                index += 1


def extract_frames_zed_camera():
    """
    This function extracts the frames from the zed camera.
    """

    # path to the zed camera
    path_to_zed_camera = '../data/Zed_dataset'

    for video_name in os.listdir(path_to_zed_camera):
        print('-------------------------------------')
        print(f'Extracting frames from {video_name}')
        print('-------------------------------------')
        for file in os.listdir(os.path.join(path_to_zed_camera, video_name, 'segmentation')):
            if file.endswith('.mp4'):
                path_video = os.path.join(path_to_zed_camera, video_name, 'segmentation', file)
                # read the video
                video = cv2.VideoCapture(path_video)
                # extract the frames
                index = 0
                while video.isOpened():
                    # read the frame
                    ret, frame = video.read()
                    # if the frame is not empty
                    if ret:
                        # save the frame
                        cv2.imwrite(os.path.join(path_to_zed_camera, video_name, 'images',
                                                 video_name + '_' + str(index) + '_C.png'), frame)
                        index += 1
                        print(f'Frame {index} extracted')
                    else:
                        video.release()
                        cv2.destroyAllWindows()
                        break


def read_depth_or_infrared_file(videoname, file_name, normalization=None, show_img=False):
    """
    This function reads the depth or infrared file and returns the depth or infrared image. It also shows the image if
    show_img is True.
    :param videoname: name of the video
    :param file_name: name of the file
    :param normalization: if it is not None, it is the normalization of the depth or infrared image
    :param show_img: if True, the image is shown
    :return: the depth or infrared image
    """
    # path to the depth file
    path_to_file = os.path.join('../data', 'Apple_Tracking_db', videoname, 'images', f'{file_name}.mat')

    # read .mat file
    mat = sio.loadmat(path_to_file)

    # if the file_name ends with D => depth image, else I => IR image
    if file_name.endswith('D'):
        img = mat['transformed_depth'].astype(np.float32)
    elif file_name.endswith('I'):
        img = mat['transformed_ir'].astype(np.float32)
    else:
        raise ValueError('The file name must end with D or I')

    if normalization is not None:
        # divide the image by the normalization factor, as type float
        img = img / normalization
        # if there are values greater than 1, set them to 1
        img[img > 1] = 1

    if show_img:
        # divide all img values by the max value of the img to get the values between 0 and 255
        img_show = img / img.max()
        img_show = img_show * 255
        img_show = img_show.astype(np.uint8)

        # show the image (1 channel => hxw) in a color scale to have a better representation of the depth or IR image
        # apply colormap to the img
        img_show = cv2.applyColorMap(img_show, cv2.COLORMAP_JET)
        # include a bar with the color scale
        cv2.imshow('image', img_show)
        cv2.waitKey(0)

        # save image in the folder in the actual path
        #cv2.imwrite(os.path.join(os.getcwd(), file_name + '.png'), img_show)

    return img


def compute_max_value_depth_and_infrared_crops(crops):
    """
    This function computes the max value of the depth and infrared crops.
    :param crops: list of crops
    """

    max_d = 0
    max_i = 0

    print('-------------------------------------')
    print('Computing max value of the depth and infrared crops')
    print('-------------------------------------')

    for crop in tqdm(crops):
        # get the video folder name
        video_folder_name = crop['file_name'].split('_')[:-2]
        video_folder_name = '_'.join(video_folder_name)

        # get the img path to files
        img_d = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_D')
        img_i = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_I')

        # crop the images
        img_d = img_d[crop['bbox_tlbr'][1]:crop['bbox_tlbr'][3], crop['bbox_tlbr'][0]:crop['bbox_tlbr'][2]]
        img_i = img_i[crop['bbox_tlbr'][1]:crop['bbox_tlbr'][3], crop['bbox_tlbr'][0]:crop['bbox_tlbr'][2]]

        if img_d.max() > max_d:
            max_d = img_d.max()
        if img_i.max() > max_i:
            max_i = img_i.max()

    print(f'Max value of the depth: {max_d}')
    print(f'Max value of the infrared: {max_i}')

    return max_d, max_i


def get_crops(skip_crops=True, only125=False):
    """
    This function extracts the crops from the dataset. Returns a list of dicts with the crops where for every crop:
    {
    'filename': filename, represents the name of the file where the apple is,
    'id': id, represents the id of the apple,
    'bbox_tlbr': bbox_tlbr, represents the bounding box in the format [xtl, ytl, xbr, ybr] (img == (width, height))
    }
    :param: -
    :return: list of dicts with the crops
    """
    # read in yolo datasets folder
    path_db = os.path.join('../data', 'Apple_Tracking_db')
    crops = []
    for video_name in os.listdir(path_db):
        if not (video_name.endswith('.txt') or video_name.endswith('.xlsx')):
            # get the distance in cm between the apple and the camera
            distance = float(video_name.split('_')[-2])
            # if the distance is greater than 125 cm, skip the file
            if only125:
                if distance > 125:
                    continue

            path_to_frames = os.path.join(path_db, video_name, 'segmentation', 'labels_yolo_format+ids')
            for file in os.listdir(path_to_frames):
                # read the file
                with open(os.path.join(path_to_frames, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # get the bounding box
                        bbox = line.split()
                        # get the id of the object
                        id = int(bbox[1])
                        # remove the .txt from the string
                        file_name = file.split('.')[0]
                        # remove the last two characters from the string (_C)
                        file_name = file_name[:-2]
                        # get the coordinates of the bounding box
                        x, y, w, h = float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])
                        x_min, y_min, x_max, y_max = convert_bbox_from_yolo_format(x, y, w, h)
                        # skip crop if outside the image
                        if skip_bbox_if_outside_map([x_min, y_min, x_max, y_max], top_limit=600, bottom_limit=1450) \
                                and skip_crops:
                            continue
                        crop = {
                            'file_name': file_name,
                            'id': id,
                            'bbox_tlbr': [x_min, y_min, x_max, y_max]
                        }
                        crops.append(crop)
    # ids list
    ids = [crop['id'] for crop in crops]
    # for the ids that appear only one time, delete the crop
    for id in ids:
        if ids.count(id) == 1:
            # get the index of the crop with the id
            index = ids.index(id)
            # delete the crop
            crops.pop(index)
            # delete the index from the ids list
            ids.pop(index)

    return crops


def generate_csv_from_crops():
    """
    This function generates a csv file with the crops from the dataset. The csv file is in the folder
    crops_of_Apple_Tracking_db. The csv file has the following columns:
    filename, id, x_min, y_min, x_max, y_max
    """
    crops = get_crops()
    # create folder crops_info if it doesn't exist
    path_to_crops_info = os.path.join('../data', 'crops_info')
    if not os.path.exists(path_to_crops_info):
        os.makedirs(path_to_crops_info)

    # save the crops in a csv file in the folder 'crops', create it if it doesn't exist
    path_to_csv = os.path.join(path_to_crops_info, 'crops_train.csv')
    with open(path_to_csv, 'a') as f:
        for crop in crops:
            if crop['id'] % 5 != 0:
                f.write(f'{crop["file_name"]}_{crop["id"]}_C.png,'
                        f'{crop["file_name"]}_{crop["id"]}_D.png,'
                        f'{crop["file_name"]}_{crop["id"]}_I.png,'
                        f'{crop["id"]}\n')

    path_to_csv = os.path.join(path_to_crops_info, 'crops_test.csv')
    with open(path_to_csv, 'a') as f:
        for crop in crops:
            if crop['id'] % 5 == 0:
                f.write(f'{crop["file_name"]}_{crop["id"]}_C.png,'
                        f'{crop["file_name"]}_{crop["id"]}_D.png,'
                        f'{crop["file_name"]}_{crop["id"]}_I.png,'
                        f'{crop["id"]}\n')


def generate_crops_numpy(add_D_and_I=True, only125=False):
    """
    This function generates a numpy file with the crops from the dataset. The numpy file is in the folder
    crops_of_Apple_Tracking_db_numpy.
    :param add_D_and_I: if True, the numpy file will contain the depth and infrared images. If False, the numpy file
    """
    crops = get_crops(skip_crops=True, only125=only125)

    # get max value of depth and infrared crops
    # max_d, max_i = compute_max_value_depth_and_infrared_crops(crops)

    # save the crops in a pickle file in the folder 'crops', create it if it doesn't exist
    path_to_crops = os.path.join('../data', 'crops_of_Apple_Tracking_db_numpy')
    if not os.path.exists(path_to_crops):
        os.makedirs(path_to_crops)

    # get the max id of the crops
    max_id = 0
    for crop in crops:
        if crop['id'] > max_id:
            max_id = crop['id']

    # create list of lists empty of the size of the max id
    crops_list = [[] for i in range(max_id + 1)]
    for crop in crops:
        crops_list[crop['id']].append(crop)

    print('generating crops of the images...')
    crops_pickles = [[] for i in range(max_id + 1)]
    for idx, crops in tqdm(enumerate(crops_list)):
        if len(crops) == 0:
            continue
        for crop in crops:
            # get the video folder name
            video_folder_name = crop['file_name'].split('_')[:-2]
            video_folder_name = '_'.join(video_folder_name)

            # get the img path to files
            img_rgb = cv2.imread(
                os.path.join('../data', 'Apple_Tracking_db', video_folder_name, 'images', crop['file_name'] + '_C.png'))
            # normalization numbers computed with compute_max_value_depth_and_infrared_crops(crops)
            img_d = None
            img_i = None
            if add_D_and_I:
                img_d = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_D', normalization=4000)
                img_i = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_I', normalization=2000)

            # augment the bounding boxes by a percentage
            bbox_tlbr = crop['bbox_tlbr']
            crop_bbox_modified = augment_size_of_bboxes_in_crops(bbox_tlbr, percentage_to_augment=0.15)

            # crop the images
            img_rgb = img_rgb[crop_bbox_modified[1]:crop_bbox_modified[3], crop_bbox_modified[0]:crop_bbox_modified[2]]
            if add_D_and_I:
                img_d = img_d[crop_bbox_modified[1]:crop_bbox_modified[3], crop_bbox_modified[0]:crop_bbox_modified[2]]
                img_i = img_i[crop_bbox_modified[1]:crop_bbox_modified[3], crop_bbox_modified[0]:crop_bbox_modified[2]]

            if not add_D_and_I:
                img = np.array((img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]))
            else:
                img = np.array((img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2], img_d, img_i))

            crops_pickles[crop['id']].append(img)

    train_crops = [[] for i in range(max_id + 1)]
    test_crops = [[] for i in range(max_id + 1)]

    print('distributing the crops in train and test randomly with a 80% prob in train and then saving them...')
    for idx, crops in tqdm(enumerate(crops_pickles)):
        if len(crops) != 0:
            # select randomly the 80% of the elements in a list
            train_crops[idx] = random.sample(crops, int(len(crops) * 0.8))
            # get the indexes of the arrays selected
            indexes = [i for i in range(len(crops)) if i not in [j for j in range(int(len(crops) * 0.8))]]
            # select the remaining 20% of the elements in a list
            test_crops[idx] = [crops[i] for i in indexes]

    # save the crops in a pickle file, named train and test respectively
    if add_D_and_I and only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_125.pkl')
    elif add_D_and_I and not only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops.pkl')
    elif not add_D_and_I and only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_without_D_and_I_125.pkl')
    else:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_without_D_and_I.pkl')

    with open(path_to_train_crops, 'wb') as f:
        pickle.dump(train_crops, f)

    if add_D_and_I and only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_125.pkl')
    elif add_D_and_I and not only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops.pkl')
    elif not add_D_and_I and only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_without_D_and_I_125.pkl')
    else:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_without_D_and_I.pkl')

    with open(path_to_test_crops, 'wb') as f:
        pickle.dump(test_crops, f)

    #redistribute_crops_numpy()


def redistribute_crops_numpy(add_D_and_I=True, only125=False):
    """
    This function redistributes the crops in the train and test numpy files in the folder
    crops_of_Apple_Tracking_db_numpy. Basically puts all the crops in a list with a id as index and the crops as value.
    :param add_D_and_I: if True, the numpy file with the crops will have the depth and infrared images.
    """
    path_to_crops = os.path.join('../data', 'crops_of_Apple_Tracking_db_numpy')

    if not os.path.exists(path_to_crops):
        print('the folder crops_of_Apple_Tracking_db_numpy does not exist')
        generate_crops_numpy()

    # load the crops in a pickle file
    if add_D_and_I and only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_125.pkl')
    elif add_D_and_I and not only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops.pkl')
    elif not add_D_and_I and only125:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_without_D_and_I_125.pkl')
    else:
        path_to_train_crops = os.path.join(path_to_crops, 'train_crops_without_D_and_I.pkl')

    with open(path_to_train_crops, 'rb') as f:
        train_crops = pickle.load(f)

    if add_D_and_I and only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_125.pkl')
    elif add_D_and_I and not only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops.pkl')
    elif not add_D_and_I and only125:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_without_D_and_I_125.pkl')
    else:
        path_to_test_crops = os.path.join(path_to_crops, 'test_crops_without_D_and_I.pkl')

    with open(path_to_test_crops, 'rb') as f:
        test_crops = pickle.load(f)

    # delete the crops and get the indexes of the crops deleted
    indexes_deleted = []
    for idx, crops in enumerate(train_crops):
        if len(crops) < 6:
            indexes_deleted.append(idx)
    # delete the indexes of the crops deleted
    train_crops = [train_crops[i] for i in range(len(train_crops)) if i not in indexes_deleted]
    test_crops = [test_crops[i] for i in range(len(test_crops)) if i not in indexes_deleted]

    for idx, train_crop in enumerate(train_crops):
        # if the crop is empty print the index of the crop
        if len(train_crop) == 1:
            print(idx)

    # for each item in the list of lists, assign the index of the list and the item, as a tuple
    train = []
    for idx, crops in enumerate(train_crops):
        for crop in crops:
            train.append((crop, idx))
    test = []
    for idx, crops in enumerate(test_crops):
        for crop in crops:
            test.append((crop, idx))

    # delete the old pickle file
    os.remove(path_to_train_crops)
    os.remove(path_to_test_crops)

    # save the new pickle file
    with open(path_to_train_crops, 'wb') as f:
        pickle.dump(train, f)
    with open(path_to_test_crops, 'wb') as f:
        pickle.dump(test, f)


def get_info_crops():
    """
    This function gets the information of the crops in the folder crops_of_Apple_Tracking_db_numpy. It gets the
    percentage of zeros in the depth and infrared images (images that are all 0!). Also gets the mean and std of the
    distance and infrared of the images for each video.
    """

    crops = get_crops()

    zeros_in_depth = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    zeros_in_infrared = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    mean_distance_depth = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    std_distance_depth = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    mean_distance_infrared = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    std_distance_infrared = {
        '125': 0,
        '175': 0,
        '225': 0,
    }
    total_imgs = {
        '125': 0,
        '175': 0,
        '225': 0,
    }

    information = {
        'zeros_in_depth': zeros_in_depth,
        'zeros_in_infrared': zeros_in_infrared,
        'mean_distance_depth': mean_distance_depth,
        'std_distance_depth': std_distance_depth,
        'mean_distance_infrared': mean_distance_infrared,
        'std_distance_infrared': std_distance_infrared,
        'total_imgs': total_imgs,
    }

    print('getting information about the crops...')
    for crop in tqdm(crops):
        # augment size bbox by percentage
        crop_bbox_modified = augment_size_of_bboxes_in_crops(crop['bbox_tlbr'], percentage_to_augment=0.15)

        # skip crop if it is outside the bbox of the video (600, 1450)
        if skip_bbox_if_outside_map(crop_bbox_modified):
            continue

        video_folder_name = crop['file_name'].split('_')[:-2]
        video_folder_name = '_'.join(video_folder_name)

        # get the distance from the camera
        distance = video_folder_name.split('_')[-2]

        # get the img path to files
        img_d = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_D')
        img_i = read_depth_or_infrared_file(video_folder_name, crop['file_name'] + '_I')

        # crop the images
        img_d = img_d[crop_bbox_modified[1]:crop_bbox_modified[3], crop_bbox_modified[0]:crop_bbox_modified[2]]
        img_i = img_i[crop_bbox_modified[1]:crop_bbox_modified[3], crop_bbox_modified[0]:crop_bbox_modified[2]]

        # if img_d contains all zeros
        if np.sum(img_d) == 0:
            zeros_in_depth[distance] += 1
        # if img_i contains all zeros
        if np.sum(img_i) == 0:
            zeros_in_infrared[distance] += 1
        # update num total of images
        information['total_imgs'][distance] += 1

        # compute mean and std for each distance but skipping zeros in the depth and infrared images
        # compute the mean distance
        if np.sum(img_d != 0) != 0:
            mean_distance_depth[distance] = (mean_distance_depth[distance] + np.sum(img_d) / np.sum(img_d != 0)) / 2
        if np.sum(img_i != 0) != 0:
            mean_distance_infrared[distance] = (mean_distance_infrared[distance] + np.sum(img_i) / np.sum(img_i != 0)) / 2
        # compute the std distance but skipping zeros in the depth and infrared images
        if np.sum(img_d != 0) != 0:
            # get the non zero values of imd_d
            non_zero_values = img_d[img_d != 0]
            # compute the std
            std_distance_depth[distance] = (std_distance_depth[distance] + np.std(non_zero_values)) / 2
        if np.sum(img_i != 0) != 0:
            # get the non zero values of imd_i
            non_zero_values = img_i[img_i != 0]
            # compute the std
            std_distance_infrared[distance] = (std_distance_infrared[distance] + np.std(non_zero_values)) / 2

    print(f'zeros in depth 125 (%): {information["zeros_in_depth"]["125"] / information["total_imgs"]["125"] * 100}')
    print(f'zeros in depth 175 (%): {information["zeros_in_depth"]["175"] / information["total_imgs"]["175"] * 100}')
    print(f'zeros in depth 225 (%): {information["zeros_in_depth"]["225"] / information["total_imgs"]["225"] * 100}')
    print(f'zeros in infrared 125 (%): {information["zeros_in_infrared"]["125"] / information["total_imgs"]["125"] * 100}')
    print(f'zeros in infrared 175 (%): {information["zeros_in_infrared"]["175"] / information["total_imgs"]["175"] * 100}')
    print(f'zeros in infrared 225 (%): {information["zeros_in_infrared"]["225"] / information["total_imgs"]["225"] * 100}')
    print(f'mean distance depth 125: {information["mean_distance_depth"]["125"]}')
    print(f'mean distance depth 175: {information["mean_distance_depth"]["175"]}')
    print(f'mean distance depth 225: {information["mean_distance_depth"]["225"]}')
    print(f'std distance depth 125: {information["std_distance_depth"]["125"]}')
    print(f'std distance depth 175: {information["std_distance_depth"]["175"]}')
    print(f'std distance depth 225: {information["std_distance_depth"]["225"]}')
    print(f'mean distance infrared 125: {information["mean_distance_infrared"]["125"]}')
    print(f'mean distance infrared 175: {information["mean_distance_infrared"]["175"]}')
    print(f'mean distance infrared 225: {information["mean_distance_infrared"]["225"]}')
    print(f'std distance infrared 125: {information["std_distance_infrared"]["125"]}')
    print(f'std distance infrared 175: {information["std_distance_infrared"]["175"]}')
    print(f'std distance infrared 225: {information["std_distance_infrared"]["225"]}')


def stack_imgs_for_embeddings(db, visualize_images=False):
    """
    This function stacks the RGB depth and infrared images of the crops from the variable ids. It saves the stacked
    images in the variable imgs_stacked in order to create the embeddings by doing inference to a model in another
    function.
    :param db: the database
    :param visualize_images: if True, the stacked images will be visualized.
    :return:
    imgs_stacked - a list of the stacked images, with the same index as the ids.
    ids - a list of the ids of the crops.
    id_map - a list that maps the id of the crops to the index of the stacked images.
    """
    ids = [100, 200, 350, 450, 500, 600, 700, 800, 850]  # get random ids to plot

    # use seed 42
    #random.seed(43)
    random.seed(0)
    ids = np.arange(914)
    random.shuffle(ids)
    ids = ids[:9]
    ids = ids.tolist()

    files_by_id = db.files_by_id

    # transformations pytorch
    if files_by_id[0][0].shape[-1] == 5: # rgb-d-i dataset
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.114, 0.073],  # last 2 values of the vector computed with function
                std=[0.229, 0.224, 0.225, 0.135, 0.0643]),
        ])
    else:
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # last 2 values of the vector computed with function
                std=[0.229, 0.224, 0.225]),
        ])

    # create a list of numbers, each number is the index of the stacked image
    id_map = []
    for id in ids:
        for _ in range(len(files_by_id[id])):
            id_map.append(id)


    # put all the imgs in a list
    imgs = []
    for id in ids:
        for frame_apple in files_by_id[id]:
            imgs.append(frame_apple)

    # stack the imgs to be the input of a model
    imgs_stacked = None
    for idx, img in enumerate(imgs):
        if visualize_images:
            image_rgb = img[:, :, :3]
            # convert img to int
            image_rgb = image_rgb.astype(int)
            # swap bgr to rgb
            image_rgb = image_rgb[:, :, [2, 1, 0]]
            # transpose the image to be (a, b, c) => (c, a, b)
            # plot the image with plt
            if files_by_id[0][0].shape[-1] == 5:
                # create subplot
                fig, ax = plt.subplots(1, 3, figsize=(15, 15))
                # plot the image
                ax[0].imshow(image_rgb)
                ax[0].set_title('id: ' + str(id_map[idx]))
                ax[0].axis('off')
            else:
                plt.imshow(image_rgb)
                plt.title('id: ' + str(id_map[idx]))
                plt.show()

            # rgb-d-i dataset
            if files_by_id[0][0].shape[-1] == 5:
                # in subplot
                cmap = plt.cm.get_cmap("jet")
                # print legend for the color map
                ax[1].imshow(img[:, :, 3], cmap=cmap)
                ax[1].set_title('dept0h')
                ax[1].axis('off')
                # save image with the legend
                ax[2].imshow(img[:, :, 4], cmap=cmap)
                ax[2].set_title('infrared')
                ax[2].axis('off')
                # wait key button 0
                plt.waitforbuttonpress(0)
                # close the figure
                plt.close()

        # transform the img
        img = transformations(img)

        # stack the img (repeated 3 times because it is the input of the model)
        if imgs_stacked is None:
            imgs_stacked = torch.stack([img.repeat(3, 1, 1, 1)])
        else:
            imgs_stacked = torch.cat([imgs_stacked, img.repeat(3, 1, 1, 1).unsqueeze(0)])

    return imgs_stacked, ids, id_map


if __name__ == "__main__":
    # refactor_id_frames_extractor()
    # change_name_imgs_in_folder('210928_165031_k_r2_w_015_125_162')
    # rotate_images(path_to_images='210906_121931_k_r2_e_015_125_162', clockwise=False, test=False)
    # rotate_segmentation()
    # generate_yolo_labels()
    # create_custom_db_for_yolo()
    # generate_yolo_labels_and_ids()
    # extract_frames_zed_camera()
    # read_depth_or_infrared_file('210928_094225_k_r2_e_015_175_162', '210928_094225_k_r2_e_015_175_162_61_59_I', normalization=None, show_img=True)

    generate_crops_numpy(add_D_and_I=True, only125=True)
    # redistribute_crops_numpy(add_D_and_I=True, only125=False)

    # generate_csv_from_crops()
    # compute_max_value_depth_and_infrared_crops()
    # get_info_crops()
    # show imported packages sys.path
    # add path '/home/francesc/PycharmProjects/Fruit-Tracking-with-RGB-D-camera/yolov5_+_tracking'

    print('finished')