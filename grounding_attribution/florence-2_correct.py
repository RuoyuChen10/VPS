"""
Created on 2024/11/7

@author: Ruoyu Chen
Florence-2 Attribution
"""

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
import argparse
plt.rc('font', family="Arial")
import clip

from torchvision.ops import box_convert

from transformers import AutoProcessor, AutoModelForCausalLM


from interpretation.submodular_detection import DetectionSubModularExplanation

from tqdm import tqdm
from utils import mkdir

model_id = 'microsoft/Florence-2-large-ft'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/refcoco_val_florence-2_correct.json',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--division-number', 
                        type=int, default=50,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/florence-2-refcoco-correctly/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def SubRegionDivision(image, mode="slico", region_size=30):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler = 20.0) 
        slic.iterate(20)     # The number of iterations, the larger the better the effect
        label_slic = slic.getLabels()        # Get superpixel label
        number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

        for i in range(number_slic):
            img_copp = (label_slic == i)[:,:, np.newaxis].astype(int)
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image,10)  # The input image size must be the same as the initialization shape and the number of iterations is 10
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = (label_slic == i)[:,:, np.newaxis].astype(int)
            element_sets_V.append(img_copp)
    return element_sets_V

def transform_vision_data(image, text):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    image = Image.fromarray(image).convert('RGB')
    image_transformed = processor(images=image, text=text, return_tensors="pt")
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
        
        self.input_ids = None
        
        self.padding_size = 20
        
        self.text_similarity = text_similarity
        if text_similarity:
            self.text_model, preprocess = clip.load("ViT-B/32")
            self.text_model.to(self.device)
            self.text_model.eval()
    
    def forward(self, images, h, w):
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
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                use_cache=False
            )
        
        generated_texts = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)
        
        prediction_logits = []
        prediction_boxes = []
            
        for generated_text in generated_texts:
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(w, h))
            
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
        
        return prediction_boxes, prediction_logits
    
def main(args):
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
    
    # Submodular
    smdl = DetectionSubModularExplanation(
        grounding_model,
        transform_vision_data,
        device=device,
        batch_size=20
    )
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))  
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    target_class = [0]
    
    end = args.end
    if end == -1:
        end = None
    select_infos = val_file["case1"][args.begin : end]
    
    for info in tqdm(select_infos):
        if os.path.exists(
            os.path.join(save_json_root_path, info["file_name"].split("/")[-1].replace(".jpg", "_{}.json".format(info["id"])))
        ):
            continue
        
        target_box = info["bbox"]
        x1, y1, x2, y2 = target_box
        image_path = os.path.join(args.Datasets, info["file_name"])
        
        image = cv2.imread(image_path)
        empty = np.zeros((image.shape[0], image.shape[1]))
        empty[y1:y2, x1:x2] = 1
        
        # Sub-region division
        image_proccess = transform_vision_data(image, task_prompt+info["category"])
        image_seg = cv2.resize(image, image_proccess.shape[1:][::-1])
        
        region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)

        V_set = SubRegionDivision(image_seg, region_size = region_size)
        
        # center distance
        empty = cv2.resize(empty, image_proccess.shape[1:][::-1])
        y, x = np.where(empty == 1)
        box_center = np.array([np.mean(y), np.mean(x)])
        
        # element center
        element_centers = []
        for element in V_set:
            y, x = np.where(element[:,:,0]==1)
            element_center = [np.mean(y), np.mean(x)]
            element_centers.append(element_center)
        element_centers = np.array(element_centers) # (49,2)
        
        distances = np.sqrt(np.sum((element_centers - box_center) **2, axis=1))
        sorted_indices = np.argsort(distances)
        
        V_set = np.array(V_set)[sorted_indices]
        V_set = [element for element in V_set]
        
        smdl.detection_model.input_ids = processor(text=task_prompt + info["category"], images=image, return_tensors="pt")['input_ids'][0].to(device)
        smdl.detection_model.target_class = info["category"]
        
        S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
        
        saved_json_file["category"] = info["category"]
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, info["file_name"].split("/")[-1].replace(".jpg", "_{}.npy".format(info["id"]))),
            np.array(S_set)
        )
        
        # Save json file
        with open(
            os.path.join(save_json_root_path, info["file_name"].split("/")[-1].replace(".jpg", "_{}.json".format(info["id"]))), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

if __name__ == "__main__":
    args = parse_args()
    
    main(args)