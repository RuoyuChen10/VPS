"""
Created on 2024/11/05

@author: Ruoyu Chen
Grounding DINO Attribution (Visual Grounding)
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

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

from torchvision.ops import box_convert

from interpretation.submodular_detection import DetectionSubModularExplanation

from tqdm import tqdm
from utils import mkdir

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
                        default='datasets',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/refcoco_val_groundingdino_correct.json',
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
                        type=int, default=24,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/grounding-dino-refcoco-correctly/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

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

class GroundingDino_Visual_Grounding_Adaptation(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        self.caption = None
    
    def forward(self, images, h, w):
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

        boxes = prediction_boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        logits = prediction_logits.max(-1, keepdim=True)[0]
        
        return xyxy, logits
    
def main(args):
    # model init
    # Load the model
    model = load_model("config/GroundingDINO_SwinB_cfg.py", "ckpt/groundingdino_swinb_cogcoor.pth")

    detection_model = GroundingDino_Visual_Grounding_Adaptation(model).to("cuda")
    # caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
    # detection_model.caption = caption
    print("Load Grounding DINO model!")
    
    # Submodular
    smdl = DetectionSubModularExplanation(
        detection_model,
        transform_vision_data,
        device="cuda",
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
        image_path = os.path.join(args.Datasets, info["file_name"])
        TEXT_PROMPT = info["category"]
        caption = preprocess_caption(caption=TEXT_PROMPT)
        # detection_model.caption = caption
        
        image = cv2.imread(image_path)
        
        # Sub-region division
        image_proccess = transform_vision_data(image)
        image_seg = cv2.resize(image, image_proccess.shape[1:][::-1])

        region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)

        V_set = SubRegionDivision(image_seg, region_size = region_size)
        
        smdl.detection_model.caption = caption
        S_set, saved_json_file = smdl(image, image_proccess, V_set, [0], target_box)
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
        
        # break

if __name__ == "__main__":
    args = parse_args()
    
    main(args)