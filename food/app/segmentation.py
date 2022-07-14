
try:
    import matplotlib.pyplot as plt
except:
    pass

import config
import numpy as np
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.SEGMENTATION_NETWORK))
    cfg.DATALOADER.NUM_WORKERS = config.NUM_WORKERS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES
    cfg.MODEL.WEIGHTS = config.SEGMENTATION_MODEL  # path to the model we just trained
    cfg.SOLVER.IMS_PER_BATCH = config.BATCH_SIZE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.ROI_THRESHOLD # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

def predict_mask(img: np.array):
    predictor = get_predictor()
    predictions = predictor(img)
    v = Visualizer(img[:, :, ::-1],scale=1.2)
    annotated_image = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    plt.imshow(annotated_image.get_image())