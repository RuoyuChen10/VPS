"""
Created on 2024/11/7

@author: Ruoyu Chen
Florence-2 Attribution Inference
"""
import os
import json
import cv2
import clip
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
import argparse
plt.rc('font', family="Arial")

from torchvision.ops import box_convert

from transformers import AutoProcessor, AutoModelForCausalLM

from tqdm import tqdm
from utils import mkdir

model_id = 'microsoft/Florence-2-large-ft'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/',
                        help='Datasets.')
    parser.add_argument('--steps',
                        type=int,
                        default=49,
                        help='steps.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/refcoco_val_florence-2_correct.json',
                        help='Datasets.')
    parser.add_argument('--eval-dir', 
                        type=str, default='./baseline_results/florence-2-refcoco-correct/HsicAttributionMethod/',
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

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image).convert('RGB')
    image_transformed = processor(images=image, return_tensors="pt")
    return image_transformed['pixel_values'][0]

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

def florence2(model, task_prompt, image, text_input=None, device="cuda"):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))

    return parsed_answer

def main(args):
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True).eval()
    
    device = "cuda"
    model.to(device)
    
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    print("ok")

    json_save_dir = os.path.join(args.eval_dir, "json")
    mkdir(json_save_dir)
    npy_dir = os.path.join(args.eval_dir, "npy")
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    select_infos = val_file["case1"]
    for info in tqdm(select_infos[250:]):
        if os.path.exists(
            os.path.join(json_save_dir, info["file_name"].split("/")[-1].replace(".jpg", "_{}.json").format(info["id"]))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, info["file_name"])
        image = cv2.imread(image_path)
        
        saliency_map = np.load(
                os.path.join(npy_dir, info["file_name"].split("/")[-1].replace(".jpg", "_{}.npy".format(info["id"])))
            )
        
        target_box = info["bbox"]
        target_label = info["category"]
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
            
            # target_label = torch.tensor(target_label)
            
            # insertion
            insertion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "insertion")
            insertion_image = Image.fromarray(cv2.cvtColor(insertion_image, cv2.COLOR_BGR2RGB)) 
            # cv2.imwrite("insertion.jpg", insertion_image)
            # image_proccess = transform_vision_data(insertion_image)
            with torch.no_grad():
                results = florence2(model, task_prompt, insertion_image, target_label)
                boxes = torch.tensor(results[task_prompt]['bboxes'])
                phrases = results[task_prompt]['bboxes_labels']
                
                if boxes.shape[0] == 0:
                    json_file["insertion_score"].append(0)
                    json_file["insertion_iou"].append(0)
                    json_file["insertion_box"].append([0,0,0,0])
                    json_file["insertion_cls"].append(0)
                
                else:
                    ious = calculate_iou(boxes, target_box)
                    cls_score = float(phrases[0] == target_label)
                    
                    insertion_scores = ious[0] * cls_score
                    
                    insertion_idx = 0
                    json_file["insertion_score"].append(insertion_scores.cpu().item())
                    json_file["insertion_iou"].append(ious[insertion_idx].cpu().item())
                    json_file["insertion_box"].append(boxes[insertion_idx].cpu().numpy().astype(int).tolist())
                    json_file["insertion_cls"].append(cls_score)
            
            # deletion
            deletion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "deletion")
            deletion_image = Image.fromarray(cv2.cvtColor(deletion_image, cv2.COLOR_BGR2RGB)) 
            # image_proccess = transform_vision_data(deletion_image)
            with torch.no_grad():
                results = florence2(model, task_prompt, deletion_image, target_label)
                boxes = torch.tensor(results[task_prompt]['bboxes'])
                phrases = results[task_prompt]['bboxes_labels']
                
                if boxes.shape[0] == 0:
                    json_file["deletion_score"].append(0)
                    json_file["deletion_iou"].append(0)
                    json_file["deletion_box"].append([0,0,0,0])
                    json_file["deletion_cls"].append(0)
                
                else:
                    ious = calculate_iou(boxes, target_box)
                    cls_score = float(phrases[0] == target_label)
                    
                    deletion_scores = ious[0] * cls_score
                    
                    deletion_idx = 0
                    json_file["deletion_score"].append(deletion_scores.cpu().item())
                    json_file["deletion_iou"].append(ious[deletion_idx].cpu().item())
                    json_file["deletion_box"].append(boxes[deletion_idx].cpu().numpy().astype(int).tolist())
                    json_file["deletion_cls"].append(cls_score)
                    
        with open(
                os.path.join(json_save_dir, info["file_name"].split("/")[-1].replace(".jpg", "_{}.json").format(info["id"])), "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    args = parse_args()
    main(args)