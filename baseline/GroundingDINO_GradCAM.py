import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
plt.rc('font', family="Arial")

import torch
import torch.nn.functional as F

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

from torchvision.ops import box_convert
from utils import COCO_TEXT_PROMPT, coco_classes, coco_classes_grounding_idx, mkdir

import argparse
from tqdm import tqdm

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
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_groundingdino_misdetect.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/grounding-dino-misdetect/',
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

def calculate_iou(batched_boxes, target_box):
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

def main(args):
    feature_level = 2
    
    # model init
    eps = torch.finfo(torch.float32).eps
    device = "cuda"
    # Load the model
    model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
    model.to(device)
    
    caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
    print("Load Grounding DINO model!")
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    save_dir = os.path.join(
        os.path.join(args.save_dir, "gradcam"), "npy")
    mkdir(save_dir)
    
    select_infos = val_file["case3"]
    for info in tqdm(select_infos[:]):
        if os.path.exists(
            os.path.join(save_dir, info["file_name"].replace(".jpg", ".npy"))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, info["file_name"])
        image = cv2.imread(image_path)
        
        target_box = info["bbox"]
        target_label = coco_classes_grounding_idx[info["category"]]
        h,w = image.shape[:2]
        
        image_proccess = transform_vision_data(image)
        out = model(image_proccess.unsqueeze(0).to(device), captions=[caption], unset_image_tensor=False)
        
        prediction_boxes = out["pred_boxes"].cpu()
        boxes = prediction_boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        ious = calculate_iou(xyxy, target_box)
        
        idx = ious.argmax()
        
        out_score = out['pred_logits'][0,idx,target_label[0]]
        out_left = out["pred_boxes"][0,idx,0]
        out_top = out["pred_boxes"][0,idx,1]
        out_right = out["pred_boxes"][0,idx,2]
        out_bottom = out["pred_boxes"][0,idx,3]
        
        model.features[0].tensors.retain_grad()
        model.features[1].tensors.retain_grad()
        model.features[2].tensors.retain_grad()
        
        tmp = []
        feature = model.features[feature_level].tensors
        for score in [out_score, out_left, out_top, out_right, out_bottom]:
            model.zero_grad()
            score.retain_grad()
            score.backward(retain_graph=True)
            
            grad = model.features[feature_level].tensors.grad
            grad = grad.mean([-1,-2], keepdim=True)
            odam_map = F.relu_((grad * feature).sum(1))
            odam_map = odam_map.detach().squeeze(0)
            odam_map = (odam_map - odam_map.min()) / (odam_map.max() - odam_map.min()).clamp(min=eps)
            
            saliency_map = odam_map.cpu().numpy()
            saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
            
            tmp.append(saliency_map)
            
        tmp = np.array(tmp)
        saliency_map_final = np.max(tmp, axis=0)
        
        model.unset_image_tensor()
        
        np.save(os.path.join(save_dir, info["file_name"].replace(".jpg", ".npy")), saliency_map_final)

if __name__ == "__main__":
    args = parse_args()
    
    main(args)