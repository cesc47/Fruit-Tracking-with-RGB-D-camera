# Fruit-Tracking-with-RGB-D-camera
In the recent years, Precision Agriculture has become a fast-growing research topic including its potential applications thanks
to the arise of Deep Learning. Subsequently, more precise solutions involving fruit-counting algorithms are being implemented with
respect to the classical methods launched previous years. In this work, an apple counting model based on Deep Learning techniques
using tracking algorithms is proposed in order to face problems when recounting as occlusions. In this repo, a modification of
state-of-the art algorithms is performed by including distance and depth information to the videos.


## Motivation
In the near future, more fruit production will be manufactured on less land due to the overpopulation of the planet. In
order to achieve sustainable agriculture in the coming years to reduce production costs, it will be necessary to monitor
the plantations and extract interesting information from them. The food generating sector is one of the leading occupations
among the people in rural areas lacks due to underdeveloped methodologies or use of outdated know-how. For that reason,
PA (Precision Agriculture) is going to play an important role in the following years, as it will make possible to optimise the
harvest thanks to new technologies and algorithms. For that reason, the development of object counting methods are a relevant topic in computer vision and in PA.

### Goal
Overall, the goal of this repo is to develop a robust tracking methodology for fruit counting. The principal contribution will
be to modify the state-of-the art tracking algorithms to include depth and infrared data. Apart of that, a benefit from this study will be taken to analyze and compare the algorithms to see which fruit
tracking algorithm of those currently being used performs best.


## Method and results

### Dataset 
The Dataset can be downloadable [here](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing). About its structure, it is divided in several folders, each one represents the title of a video. The videos are recorded with a vehicle that has the camera appended to it
An example of a title of a folder is:

'**videoname folder**': 210726_170244_k_r2_a_015_225_162. The information parsed from this example is:

- **210726**      - the date of the video
- **170244**      - the time of the video
- **k**           - kinect camera
- **r2**          - number of row of the apples in the field, r2 represents row 2
- **a**           - all (e: east part of the trees, w: west part of the trees, a: both parts)
- **015**         - Velocity of the vehicle scanning the apples (in km/h)
- **225**         - Distance from the camera to the trees
- **162**         - Distance of the sensor to the floor


Then, if we enter into a videoname folder it always follows the same structure, where we have two folders:
- '**images**': contains all the images of the dataset that are labeled. The camera used is an Azure Kinect, it provides a color image a depth image and an infrared image for each frame (frame rate is 30fps).
Images are stored in a folder named images, and they are placed in order (the first image in the folder is the first image labeled in the segmentation part).

- '**segmentation**': contains the annotations of the dataset. It has the information about the apple's positions in the images (bboxes). The labels have been generated using the Supervisely web page, using a semiautomatic labeler (TransT).
It has 6 archives:
    - '**labels_yolo_format**' folder: contains the labels in the yolo format in case they want to be used directly (bounding boxes in yolo format for each frame in .txt).
    - '**labels_yolo_format+ids**' folder: contains the labels in the yolo format and the ids of the apples in the frame in case they want to be used directly (bounding boxes in yolo format for each frame in .txt).
    - **ann.json**: contains the annotations of the dataset. It has the information about the apple's positions in the images (bboxes). It should be the one that is used to perform the training, evaluation and testing, as it has all the needed information of the labeling done. The 'frame' variable refers to the frame labeled of the 'video.mp4' file.
    - **key_id_map.json**: contains the mapping between the key of the annotations and the id of the images.
    - **meta.json**: contains the metadata of the dataset.
    - **video.mp4**: contains the video of the dataset.


The structure of the dataset is the following:

    - Video_name_folder_1/
    
        ...

    - Video_name_folder_i/

        - images/
            - image1_C.png
            - image1_D.mat
            - image1_I.mat
            - ...
            - imageN_C.png
            - imageN_D.mat
            - imageN_I.mat

        - segmentation/
            - labels_yolo_format/
                - image1_C.txt
                - image2_C.txt
                - ...
                - imageN_C.txt
            - labels_yolo_format+ids/
                - image1_C.txt
                - image2_C.txt
                - ...
                - imageN_C.txt
            - ann.json
            - key_id_map.json
            - meta.json
            - video.mp4



## Repository overview

A design of a modifiable and easy-to-use system is implemented. Two steps are needed to perform the analysis:
1) **Object detector**: YOLOv5 is going to be used, due to its simplicity and the good results that provides. A training with the Apple tracking dataset
will be performed in order to fine-tune the model to our domain. After performing the inference in YOLOv5 a set of detections
are obtained with their respective confidences.
2) **Tracker**: Every tracker takes a set of detections and confidences for each frame, and they are processed according to
several algorithms. Three trackers are going to be analysed: SORT, deepSORT and Bytetrack. These trackers are widely-used in this field and they both
use a motion model (Kalman) to track the objects in a video stream, with the difference that a Re-ID network is used in
deepSORT. In the other hand, ByteTrack, follows the philosophy these two trackers but with
one difference: the objects with low detection scores, e.g. occluded objects, are not thrown away. 

Structured repository overview **after** cloning and performing the installations in the next section:

    .
    ├── data
    │   └── Apple_Tracking_db           # Dataset to replicate the results of the article if desired
    ├── tools        
    │   ├── TrackEval                   # Cloned github repo to evaluate the trackers (Hota metric)
    │   │   ├── ...
    │   ├── dataset_gestions.py         # Methods to manage the dataset, read and write files, etc.   
    │   ├── metrics.py                  # Methods to evaluate the trackers
    │   ├── read_segmentation.py        # Read the segmentation files from dataset
    │   ├── utils.py                    # Utilities
    │   └── visualization.py            # Visualization of the results, dataset and trackers
    └── yolov5_+_tracking
        ├── bytetrack                   # Cloned & modified github repo to perform the tracking of the apples using ByteTrack
        ├── models                      # Models uploaded here to use custom Re-ID networks
        │   │   ├── ...
        │   ├── byte_tracker.py         # Bytetrack files (some of them modified)
        │   └── ...                     
        ├── deepsort                    # Cloned & modified github repo to perform the tracking of the apples using deepSORT
        │   ├── deepsort.py             # DeepSORT files (some of them modified)
        │   └── ...
        ├── sort                        # Cloned & modified github repo to perform the tracking of the apples using SORT
        │   └── sort.py
        ├── datasets                    # Yolo dataset to generate inference
        ├── results tracking            # Results of the tracking of the apples 
        ├── yolov5                      # Cloned yolov5 github repo to perform the inference of the apples
        │   ├── data
        │   │   ├── ....yaml            # Yaml file to perform the inference of the apples
        │   ├── runs
        │   │   ├── detect              # Where inference will be automatically stored
        │   │   └── train               # Where yolo model should be stored
        │   └── tools
        └── track.py                    # Main file to perform the tracking of the apples




## Installation & requirements
These are the requirements needed to replicate the results of the article. If you just want to use the detection + tracking without the evaluation, less steps are needed:

1) Install requirements and create data folder.
    ```
    pip install -r requirements.txt
    mkdir data
    ```

2) [Download dataset](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing) and put it inside data/ folder.


3) Clone TrackEval repo:
    ```
    cd tools
    git clone https://github.com/JonathonLuiten/TrackEval.git
    cd ..
    ```
4) Add Re-ID models:
    ```
    mkdir yolov5_+_tracking/bytetrack/models
    ```
   [Download models](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing) and put it inside yolov5_+_tracking/bytetrack/models/ folder.


5) Prepare yolo dataset
    ```
    mkdir yolov5_+_tracking/datasets
    python3 tools/dataset_gestions.py --create_yolo_dataset
    ```
6) Clone yolov5 repo:
    ```
    cd yolov5_+_tracking
    git clone https://github.com/ultralytics/yolov5.git
    cd ..
   
[Download .yaml](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing) and put it inside yolov5_+_tracking/yolov5/data/ folder.

[Download trained model](https://drive.google.com/file/d/192nAusRE4WtHu5nNrnaJxE5V61AYxt7f/view?usp=sharing) and put it inside yolov5_+_tracking/yolov5/runs/train/ folder.


## Running instructions

To perform inference in yolo with trained model, run the following command:
```
python3 --weights runs/train/yolov5x_results/weights/best.pt --source ../datasets/Apple_Tracking_db_yolo/test/images --data data/segmentacio_pomes.yaml --save-txt --save-conf
```

To perform the tracking once the inference is performed, run the following command:

Required arguments:
```
 --tracker_type 	       Type of tracker (e.g. sort, bytetrack, deepsort)
```

Optional arguments: 

| Parameter          | Default       | Description                                                                                                                                                   |	
|:-------------------|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| reid               | None          | Use reid network to** track. If None, no reid network is used or use reid by default in deepsort case.                                                        |
| partition          | test        | Partition where the results are computed => test, train or val. careful that this relates to what db you have done inference in yolo (name of the experiment) |
| multiplier_frames  | 1    | Number of frames to skip between each frame                                                                                                                   |
| tracker_evaluation | False | If True, the metrics are computed for the tracker                                                                                                             |
| visualize_results  | False | If True, the results are visualized in the images                                                                                                             |
| save_results       | False     | If True, the results are saved in a csv file                                                                                                                  |


An example using deepsort and the Re-ID: reid_applenet_resnet_triplet, evaluating and saving the results: 
```
--tracker_type deepsort --reid reid_applenet_resnet_triplet --tracker_evaluation True --save_results True
```

The final results of the tracking should have this look:

![](video_tracking.gif)

## More resources

The article is avaliable at ....................


