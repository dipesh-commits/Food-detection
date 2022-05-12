import matplotlib.pyplot as plt

import numpy as np
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8 # Number of output classes
    cfg.MODEL.WEIGHTS = "outputs/model_final.pth"  # path to the model we just trained
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

def visualize_mask(img: np.array):
    predictor = get_predictor()
    predictions = predictor(img)
    v = Visualizer(img[:, :, ::-1],scale=1.2)
    annotated_image = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    plt.imshow(annotated_image.get_image())

if __name__ == "__main__":
    img_path = "/home/gvc/Desktop/dev/Food-detection-main/Food_Segmentation/dataset/apple/frame_00071.jpg"  # Input image
    img = cv2.imread(img_path)
    pred = visualize_mask(img)
    print(pred)