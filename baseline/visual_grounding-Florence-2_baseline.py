"""
Created on 2024/11/07

@author: Ruoyu Chen
Florence-2 Attribution Baseline
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

import tensorflow as tf
from xplique.attributions import (Rise, KernelShap, HsicAttributionMethod)
from xplique.wrappers import TorchWrapper
from xplique.plots import plot_attributions

from torchvision.ops import box_convert

from transformers import AutoProcessor, AutoModelForCausalLM

from tqdm import tqdm
from utils import mkdir

tf.config.run_functions_eagerly(True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048*5)]
)

model_id = 'microsoft/Florence-2-large-ft'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/refcoco_val_florence-2_correct.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/florence-2-refcoco-correct/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

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

class Florence_2_Adaptation(torch.nn.Module):
    def __init__(self, 
                 foundation_model,
                 task_prompt=None,
                 device = "cuda",
                 text_similarity=False):
        super().__init__()
        self.foundation_model = foundation_model
        self.device = device
        
        self.task_prompt = task_prompt
        self.target_class = ""
        self.target_box = None
        
        self.input_ids = None
        
        self.padding_size = 50
        self.h = None
        self.k = None
    
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
    
    def forward(self, images):
        """_summary_

        Args:
            images (tensor): torch.Size([batch, 3, 768, 768])
        """
        batch = images.shape[0]
        input_ids = torch.stack([self.input_ids for i in range(batch)])
        
        with torch.no_grad():
            generated_ids = self.foundation_model.generate(
                input_ids=input_ids.to(self.device),
                pixel_values=images,
                max_new_tokens=1024,
                early_stopping=True,
                do_sample=False,
                num_beams=3,
            )
        
        generated_texts = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)
        
        prediction_logits = []
        prediction_boxes = []
        
        for generated_text in generated_texts:
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(self.w, self.h))

            logits = torch.from_numpy(np.array(parsed_answer[self.task_prompt]['bboxes_labels'])==self.target_class).float().to(self.device)
            
            logits_padding = torch.zeros(self.padding_size-logits.shape[0]).float().to(self.device)
            logits_padding = torch.cat((logits, logits_padding),dim=0)
            prediction_logits.append(logits_padding.unsqueeze(1))   # add (self.padding_size, 1)
            
            boxes = torch.tensor(parsed_answer[self.task_prompt]['bboxes']).to(self.device)  # [np, 4]
            boxes_padding = torch.zeros((self.padding_size-logits.shape[0], 4)).to(self.device)
            boxes_padding = torch.cat((boxes, boxes_padding),dim=0) # (self.padding_size, 4)
            prediction_boxes.append(boxes_padding)
        
        prediction_logits = torch.stack(prediction_logits) # prediction_logits.shape = (batch, nq, 1)
        prediction_boxes = torch.stack(prediction_boxes) # prediction_boxes.shape = (batch, nq, 4)
        
        ious = self.calculate_iou(prediction_boxes, self.target_box)
        scores = (ious * prediction_logits[:,:,0]).max(dim=-1, keepdim=True)[0]
        
        return scores
    
def main(args):
    # set batch size parameter
    batch_size = 12
    
    # model init
    # Load the model
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True).eval()
    model.to(device)
    
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    grounding_model = Florence_2_Adaptation(model, 
                                        task_prompt=task_prompt)
    print("Load Florence-2 model!")
    
    # wrap the torch model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapped_model = TorchWrapper(grounding_model.eval(), device)
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    # build the explainers
    explainers = [
                # Rise(wrapped_model, nb_samples=1500, grid_size=16, batch_size=batch_size),
                HsicAttributionMethod(wrapped_model, nb_design=800, grid_size=16, batch_size=batch_size),
    ]
    
    for explainer in explainers:
        save_dir = os.path.join(
            os.path.join(args.save_dir, explainer.__class__.__name__), "npy")
        mkdir(save_dir)
        
        select_infos = val_file["case1"]
        for info in tqdm(select_infos[:250]):
            if os.path.exists(
                os.path.join(save_dir, info["file_name"].split("/")[-1].replace(".jpg", "_{}.npy".format(info["id"])))
            ):
                continue
            
            image_path = os.path.join(args.Datasets, info["file_name"])
            image = cv2.imread(image_path)
            
            X = [image]
            X_preprocessed = torch.stack([transform_vision_data(x) for x in X])
            X_preprocessed4explainer = np.moveaxis(X_preprocessed.numpy(), [1, 2, 3], [3, 1, 2])
            label = tf.keras.utils.to_categorical(0, 1)
            Y = np.array([label])
            
            explainer.model.model.input_ids = processor(text=task_prompt+info["category"], images=image, return_tensors="pt")['input_ids'][0].to(device)
            explainer.model.model.target_class = info["category"]
            explainer.model.model.target_box = info["bbox"]
            explainer.model.model.h, explainer.model.model.w = image.shape[:2]
            
            explanations = explainer(X_preprocessed4explainer, Y)
            
            saliency_map = cv2.resize(explanations[0].numpy(), (image.shape[1], image.shape[0]))
            
            np.save(os.path.join(save_dir, info["file_name"].split("/")[-1].replace(".jpg", "_{}.npy".format(info["id"]))), saliency_map)

if __name__ == "__main__":
    args = parse_args()
    main(args)