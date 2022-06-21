import cv2
import json
import shutil

import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from utils import *


def refactor_id_frames_extractor():
    """
    This function deletes the frames (goten by 'frame extractor') that are not labeled.
    """

    # path to db
    path = './data/Apple_Tracking_db'

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
    path = './data/Apple_Tracking_db'

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
    type of rotation that is done (clockwise or counterclockwise). It just modifies the json file ann.json for each video.
    """

    # read info_rotations.txt, it has the info if the rotations of the video are clockwise or counterclockwise
    with open('./data/Apple_Tracking_db/info_rotations.txt', 'r') as f:
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
            path_to_save = os.path.join('./data/Apple_Tracking_db/', videoname, 'segmentation')

            # delete ann.json (the annotations that were not rotated)
            os.remove(os.path.join(path_to_save, 'ann.json'))

            # save the annotations in the new folder
            with open(os.path.join(path_to_save, 'ann.json'), 'w') as f:
                json.dump(annotations, f)


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
    # path = '../data/Apple_Tracking_db'
    path = '../data/Zed_dataset'

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
    for video_name in os.listdir(path):
        if not (video_name.endswith('.xlsx') or video_name.endswith('.txt')):
            print(f'Processing video {video_name} ...')
            # needed to order the images in the same order as the labels
            list_nums = []
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


if __name__ == "__main__":
    # refactor_id_frames_extractor()
    # rotate_images(path_to_images='210928_165030_k_r2_w_015_125_162', clockwise=False, test=True)
    # rotate_segmentation()
    # generate_yolo_labels()
    create_custom_db_for_yolo(path='../data/Zed_dataset',
                              path_to_new_db='../yolov5_+_tracking/datasets/Zed_dataset_yolo')
    # generate_yolo_labels_and_ids()
    # extract_frames_zed_camera()
