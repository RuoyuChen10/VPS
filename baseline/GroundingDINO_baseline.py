import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
plt.rc('font', family="Arial")

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

import tensorflow as tf
from xplique.attributions import (Rise, KernelShap, HsicAttributionMethod)
from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions

from torchvision.ops import box_convert
from utils import *

import argparse
from tqdm import tqdm

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048*5)]
)

data_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/lvis_v1',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/lvis_v1_rare_groundingdino_misdetect.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/grounding-dino-lvis-misdetect/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image)
    image_transformed, _ = data_transform(image, None)
    return image_transformed

class GroundingDino_Adaptation_Xplique(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        self.detection_model.to(self.device)
        
        self.caption = None
        
        self.h = None
        self.k = None
        
        self.target_box = None
        self.target_label = None
    
    def calculate_iou(self, batched_boxes, target_box):
        # batched_boxes: [batch, np, 4]
        # target_box: [4]

        # Separation coordinates
        x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
        tx1, ty1, tx2, ty2 = torch.tensor(target_box)

        # Calculate intersection area
        inter_x1 = torch.maximum(x1, tx1)
        inter_y1 = torch.maximum(y1, ty1)
        inter_x2 = torch.minimum(x2, tx2)
        inter_y2 = torch.minimum(y2, ty2)

        # 计算相交区域的面积
        inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)

        # Calculate the area of ​​the intersection
        box_area = (x2 - x1) * (y2 - y1)
        target_area = (tx2 - tx1) * (ty2 - ty1)

        # Calculating IoU
        union_area = box_area + target_area - inter_area
        iou = inter_area / union_area

        return iou
    
    def forward_results(self, images):
        """_summary_

        Args:
            images (tensor): torch.Size([batch, 3, 773, 1332])
        """
        batch = images.shape[0]
        captions = [self.caption for i in range(batch)]
        
        with torch.no_grad():
            outputs = self.detection_model(images, captions=captions)
            
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (batch, nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (batch, nq, 4)

        boxes = prediction_boxes * torch.Tensor([self.w, self.h, self.w, self.h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        return xyxy, prediction_logits
    
    def forward(self, images):
        # images (100, 773, 1332, 3)
        batch_input_images = images.to(self.device)
        
        with torch.no_grad():
            bounding_boxes, logits = self.forward_results(batch_input_images) # [batch, np, 4] [batch, np, 256]
        
        ious = self.calculate_iou(bounding_boxes, self.target_box)
        cls_score = logits[:,:,torch.tensor(self.target_label)].max(dim=-1)[0]
        
        scores = (ious * cls_score).max(dim=-1, keepdim=True)[0]
        
        del bounding_boxes
        del logits
        del ious
        del cls_score
        
        return scores
    
def main(args):
    # set batch size parameter
    batch_size = 12
    
    # model init
    # Load the model
    model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
    
    detection_model = GroundingDino_Adaptation_Xplique(model)
    # caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
    
    if "coco" in args.eval_list:
        caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
        classes_grounding_idx = coco_classes_grounding_idx
        detection_model.caption = caption
    
    elif "lvis" in args.eval_list:
        caption1 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_1)
        caption2 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_2)
        caption3 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_3)
        caption4 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_4)
        caption5 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_5)
    
    print("Load Grounding DINO model!")
    
    # wrap the torch model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapped_model = TorchWrapper(detection_model.eval(), device)
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    # build the explainers
    explainers = [
                # Rise(wrapped_model, nb_samples=5000, grid_size=16, batch_size=batch_size),
                HsicAttributionMethod(wrapped_model, nb_design=1000, grid_size=16, batch_size=batch_size),
    ]
    
    for explainer in explainers:
        save_dir = os.path.join(
            os.path.join(args.save_dir, explainer.__class__.__name__), "npy")
        mkdir(save_dir)
        
        select_infos = val_file["case3"]
        for info in tqdm(select_infos[300:]):
            if os.path.exists(
                os.path.join(save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.npy".format(info["id"])))
            ):
                continue
            
            if "lvis" in args.eval_list:
                if info["category"] in lvis_classes_split_1:
                    caption = caption1
                    classes_grounding_idx = lvis_classes_grounding_idx_split1
                elif info["category"] in lvis_classes_split_2:
                    caption = caption2
                    classes_grounding_idx = lvis_classes_grounding_idx_split2
                elif info["category"] in lvis_classes_split_3:
                    caption = caption3
                    classes_grounding_idx = lvis_classes_grounding_idx_split3
                elif info["category"] in lvis_classes_split_4:
                    caption = caption4
                    classes_grounding_idx = lvis_classes_grounding_idx_split4
                elif info["category"] in lvis_classes_split_5:
                    caption = caption5
                    classes_grounding_idx = lvis_classes_grounding_idx_split5
            
            image_path = os.path.join(args.Datasets, info["file_name"])
            image = cv2.imread(image_path)
            
            X = [image]
            X_preprocessed = torch.stack([transform_vision_data(x) for x in X])
            X_preprocessed4explainer = np.moveaxis(X_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])
            label = tf.keras.utils.to_categorical(0, 1)
            Y = np.array([label])
            
            explainer.model.model.caption = caption
            explainer.model.model.target_label = classes_grounding_idx[info["category"]]
            explainer.model.model.target_box = info["bbox"]
            explainer.model.model.h, explainer.model.model.w = image.shape[:2]
            
            explanations = explainer(X_preprocessed4explainer, Y)
            
            saliency_map = cv2.resize(explanations[0].numpy(), (image.shape[1], image.shape[0]))
            
            # np.save(os.path.join(save_dir, info["file_name"].replace(".jpg", "_{}.npy").format(info["category"])), saliency_map)
            
            if "coco" in args.eval_list:
                np.save(os.path.join(save_dir, info["file_name"].replace(".jpg", ".npy")), saliency_map)
            elif "lvis" in args.eval_list:
                np.save(
                    os.path.join(save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.npy".format(info["id"]))),
                    saliency_map
            )
            
if __name__ == "__main__":
    args = parse_args()
    
    main(args)
            