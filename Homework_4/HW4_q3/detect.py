import random
import cv2
import os
import argparse
import numpy as np 
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from maskrcnn import cnn, roi_head

class CustomPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])
            return predictions

    def extract_features(self, boxes):
        return self.model.extract_features(boxes)

def inference_second_stream(model, image):
    outputs = model(image)
    return outputs 

def prepare_second_stream():
    cfg2 = get_cfg()
    cfg2.merge_from_file('./configs/modified_config.yaml')
    cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    if not torch.cuda.is_available():
        cfg2.MODEL.DEVICE = 'cpu'
    model2 = CustomPredictor(cfg2)
    return model2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    
    parser.add_argument('--image_dir', required=True, metavar='path to images', help='path to images')
    parser.add_argument('--ROI_SCORE_THRESH', required=False, metavar='threshold for hand detections', \
        help='hand detection score threshold', default=0.7)
    
    args = parser.parse_args()
    images_path = args.image_dir
    roi_score_thresh = float(args.ROI_SCORE_THRESH)
    model2 = prepare_second_stream()

    images = sorted(os.listdir(images_path))
    count = 0
    for img in images:
        count += 1
        print(count)
        boxes = np.array([[0, 0, 10, 10], [0, 0, 20, 20]])
        im = cv2.imread(os.path.join(images_path, img))
        # These are the final features. 1024 dimensional vectors.
        # These are of shape [N_boxes, 1024], where N_boxes 
        # are the number of boxes
        image_features = inference_second_stream(model2, im)
        box_features = model2.extract_features(boxes)
        box_features = box_features.detach().to('cpu').numpy()
        print(box_features.shape)
        