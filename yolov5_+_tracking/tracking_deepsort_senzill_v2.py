from email.policy import default
import numpy as np
import cv2
import sys
from yolov5.detect_pol_v2 import yolo_detector
from deepsort_lab.deepsort import *

from base64 import b64encode
import os
import glob
import warnings
import torch
import random
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/runs/train/yolov5s_sis_videos_results2/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./tmp/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='./yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--video', type=str, default=None, help='path to the video file')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(1920, 1080), help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opt = parse_opt()
    video_file = opt.video
    temporal_folder = "./tmp"
    if not os.path.exists(temporal_folder):
        os.makedirs(temporal_folder)

    # Check a video path is passed
    if video_file == None:
        print ('A path to a video is required!!')
        sys.exit

    # Read Video sequence
    if not os.path.isfile(video_file):
        print ('Can not find {}'.format(video_file))
        sys.exit

    # cap = cv2.VideoCapture('videos/race.mp4')
    cap = cv2.VideoCapture(video_file)
    # Exit if video not opened.
    if not cap.isOpened():
        print('Could not open {} video'.format(video_file))
        sys.exit()

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(os.path.split(video_file)[-1].replace(".mp4", "_tracked.mp4"), fourcc, 20, (1080, 1920), True)

    frames_to_process = []

    nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)
    for c_frame in range(0, nb_frame):
        frames_to_process.append(c_frame)

    last_frame_to_process = max(frames_to_process)

    frame_ID = 0

    #Initialize deep sort. 
    deepsort = deepsort_rbc()
    detections = None
    colors = {}

    detector = yolo_detector(weights=opt.weights, imgsz=opt.imgsz, data=opt.data, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, half=opt.half, dnn=opt.dnn, max_det=opt.max_det)

    while (frame_ID < last_frame_to_process):
        print('Processing frame {}'.format(frame_ID))
        ret,frame = cap.read()
        
        # frame = frame.astype(np.uint8)

        if ret and frame_ID in frames_to_process:
            if "_k_" in os.path.split(video_file)[-1]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            detections, out_scores = detector.run(frame)

            if len(detections) == 0:
                print("No dets")
                frame_ID+=1
                continue
            
            detections = np.array(detections)
            out_scores = np.array(out_scores)
            
            # frame = img.astype(np.uint8)
            tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box                                                   
                id_num = str(track.track_id) #Get the ID for the particular track.                                                 
                features = track.features #Get the feature vector corresponding to the detection.
                if id_num not in colors.keys():
                    colors[id_num] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))                                  

                # Overlay bbox from tracker.                                                                                           
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[id_num], 3)
                cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,130,255), 3)

        frame_ID += 1
        writer.write(frame)

    # When everything done, release the capture
    writer.release()
    print("FINISHED")