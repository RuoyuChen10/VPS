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

from torchvision.ops import box_convert
from utils import *

import argparse
from tqdm import tqdm

from utils import *

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
    parser.add_argument('--steps',
                        type=int,
                        default=49,
                        help='steps.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/lvis_v1_rare_groundingdino_misdetect.json',
                        help='Datasets.')
    parser.add_argument('--eval-dir', 
                        type=str, default='./baseline_results/grounding-dino-lvis-misdetect/HsicAttributionMethod/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

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

def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

def main(args):
    # model init
    device = "cuda"
    # Load the model
    model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
    model.to(device)
    # caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
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
    
    json_save_dir = os.path.join(args.eval_dir, "json")
    mkdir(json_save_dir)
    npy_dir = os.path.join(args.eval_dir, "npy")
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    select_infos = val_file["case3"]
    for info in tqdm(select_infos[:]):
        # if os.path.exists(
        #     os.path.join(json_save_dir, info["file_name"].replace(".jpg", ".json"))
        # ):
        #     continue
        if os.path.exists(
            os.path.join(json_save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.json").format(info["id"]))
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
        
        if "coco" in args.eval_list:
            saliency_map = np.load(
                os.path.join(npy_dir, info["file_name"].replace(".jpg", "_{}.npy").format(info["category"]))
            )
        elif "lvis" in args.eval_list:
            saliency_map = np.load(
                os.path.join(npy_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.npy").format(info["id"]))
            )
        
        target_box = info["bbox"]
        target_label = classes_grounding_idx[info["category"]]
        h,w = image.shape[:2]
        
        json_file = {}
        json_file["insertion_score"] = []
        json_file["deletion_score"] = []
        json_file["insertion_iou"] = []
        json_file["insertion_box"] = []
        json_file["insertion_cls"] = []
        json_file["deletion_iou"] = []
        json_file["deletion_box"] = []
        json_file["deletion_cls"] = []
        json_file["region_area"] = []
        json_file["target_box"] = target_box
        json_file["target_label"] = target_label
        
        for i in range(1, args.steps+1):
            perturbed_rate = i / args.steps
            json_file["region_area"].append(perturbed_rate)
            
            target_label = torch.tensor(target_label)
            
            # insertion
            insertion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "insertion")
            cv2.imwrite("insertion.jpg", insertion_image)
            image_proccess = transform_vision_data(insertion_image)
            with torch.no_grad():
                out = model(image_proccess.unsqueeze(0).to(device), captions=[caption])
                
                prediction_boxes = out["pred_boxes"].cpu()
                logits = out["pred_logits"].cpu().sigmoid()
                boxes = prediction_boxes * torch.Tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                ious = calculate_iou(xyxy, target_box)
                cls_score = logits[:,:,target_label].max(dim=-1)[0]
                
                insertion_scores = (ious * cls_score).max(dim=-1)[0]
                
                insertion_idx = (ious * cls_score)[0].argmax().cpu().item()
                json_file["insertion_score"].append(insertion_scores[0].cpu().item())
                json_file["insertion_iou"].append(ious[0][insertion_idx].cpu().item())
                json_file["insertion_box"].append(xyxy[0][insertion_idx].cpu().numpy().astype(int).tolist())
                json_file["insertion_cls"].append(cls_score[0][insertion_idx].cpu().item())
            
            # deletion
            deletion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "deletion")
            cv2.imwrite("deletion.jpg", deletion_image)
            
            image_proccess = transform_vision_data(deletion_image)
            with torch.no_grad():
                out = model(image_proccess.unsqueeze(0).to(device), captions=[caption])
                
                prediction_boxes = out["pred_boxes"].cpu()
                logits = out["pred_logits"].cpu().sigmoid()
                boxes = prediction_boxes * torch.Tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                ious = calculate_iou(xyxy, target_box)
                cls_score = logits[:,:,target_label].max(dim=-1)[0]
                
                deletion_scores = (ious * cls_score).max(dim=-1)[0]
                
                deletion_idx = (ious * cls_score)[0].argmax().cpu().item()
                
                json_file["deletion_score"].append(deletion_scores[0].cpu().item())
                json_file["deletion_iou"].append(ious[0][deletion_idx].cpu().item())
                json_file["deletion_box"].append(xyxy[0][deletion_idx].cpu().numpy().astype(int).tolist())
                json_file["deletion_cls"].append(cls_score[0][deletion_idx].cpu().item())
        
        # Save json file
        if "coco" in args.eval_list:
            with open(
                os.path.join(json_save_dir, info["file_name"].replace(".jpg", "_{}.json").format(info["category"])), "w") as f:
                f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
        elif "lvis" in args.eval_list:
            with open(
                os.path.join(json_save_dir, info["file_name"].replace("/", "_").replace(".jpg", "_{}.json").format(info["id"])), "w") as f:
                f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)