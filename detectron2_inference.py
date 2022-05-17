"""
Apply pre-trained MaskRCNN or FasterRCNN on COCO in Out-Of-Context Dataset
"""
import os
import glob
import cv2
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

model_id = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
frames_path = 'frames_videos_azure_kinect/210928_160015_k_r2_w_015_225_162'

if __name__ == "__main__":

    # CONFIGURATION
    # Model config
    cfg = get_cfg()

    # Run a model in detectron2's core library: get file and weights
    cfg.merge_from_file(model_zoo.get_config_file(model_id))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)

    # Hyper-params
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25 # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)   # Initialize predictor

    os.makedirs('inference_detectron2', exist_ok=True)
    # Iterate through all the images of the dataset

    print(f'Doing inference on {frames_path}...')
    for idx, img_path in tqdm(enumerate(sorted(glob.glob(f'{frames_path}/*.png')))):
        if idx < 30:
            im = cv2.imread(img_path)

            outputs = predictor(im)

            # Drop predictions that are not 'apple':
            #for idx, idx_class in enumerate(outputs["instances"].pred_classes):
            #    if idx_class != 47: # 47 is the class index for 'apple' np.where(a == a.min())
            #        outputs["instances"].pred_classes[idx]

            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
            out = v.draw_instance_predictions(outputs["instances"].to('cpu'))



            name_img = img_path.split('/')[-1].split('.')[0]
            cv2.imwrite(f'inference_detectron2/{name_img}.png', out.get_image()[:, :, ::-1])