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
from utils import *

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
                        default='datasets/lvis_v1',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/lvis_v1_rare_groundingdino_correct_detection.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/grounding-dino-lvis-correctly/',
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
    
    print("Load Grounding DINO model!")
    
    if "coco" in args.eval_list:
        caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
        classes_grounding_idx = coco_classes_grounding_idx
    
    elif "lvis" in args.eval_list:
        caption1 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_1)
        caption2 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_2)
        caption3 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_3)
        caption4 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_4)
        caption5 = preprocess_caption(caption=LVIS_RARE_TEXT_PROMPT_SPLIT_5)
        
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    save_dir = os.path.join(
        os.path.join(args.save_dir, "ODAM"), "npy")
    mkdir(save_dir)
    
    id = 1
    select_infos = val_file["case1"]
    for info in tqdm(select_infos[id-1:]):
        if "coco" in args.eval_list:
            if os.path.exists(
                os.path.join(save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.npy").format(id))
            ):
                id+=1
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
        
        target_box = info["bbox"]
        target_label = classes_grounding_idx[info["category"]]
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
        for score in [out_score, out_left, out_top, out_right, out_bottom]:
            model.zero_grad()
            score.retain_grad()
            score.backward(retain_graph=True)
            
            for feature_level in [0,1,2]:
                feature = model.features[feature_level].tensors
            
                grad = model.features[feature_level].tensors.grad
                odam_map = F.relu_((grad * feature).sum(1))
                odam_map = odam_map.detach().squeeze(0)
                odam_map = (odam_map - odam_map.min()) / (odam_map.max() - odam_map.min()).clamp(min=eps)
                
                saliency_map = odam_map.cpu().numpy()
                saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
            
                tmp.append(saliency_map)
            
        tmp = np.array(tmp)
        saliency_map_final = np.max(tmp, axis=0)
        
        model.unset_image_tensor()
        
        if "coco" in args.eval_list:
            np.save(os.path.join(save_dir, info["file_name"].replace(".jpg", "_{}.npy").format(id)), saliency_map_final)
        elif "lvis" in args.eval_list:
            np.save(
                os.path.join(save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.npy".format(info["id"]))),
                saliency_map_final
        )
            
        id += 1

if __name__ == "__main__":
    args = parse_args()
    
    main(args)