# -- coding: utf-8 --**

"""
Created on 2024/10/06

@author: Ruoyu Chen
"""

import argparse

import os
import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from tqdm import tqdm

from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Faithfulness Metric')
    parser.add_argument('--explanation-dir', 
                        type=str, 
                        default='baseline_results/grounding-dino-lvis-misclassification/HsicAttributionMethod',
                        help='Save path for saliency maps generated by our methods.')
    args = parser.parse_args()
    return args

def main(args):
    print(args.explanation_dir)
    
    json_root_file = os.path.join(args.explanation_dir, "json")
    json_file_names = os.listdir(json_root_file)
    
    insertion_aucs = []
    insertion_iou_aucs = []
    insertion_cls_aucs = []
    
    deletion_aucs = []
    deletion_iou_aucs = []
    deletion_cls_aucs = []
    
    highest_cls_score_50 = []
    highest_cls_score_75 = []
    
    for json_file_name in tqdm(json_file_names):
        json_file_path = os.path.join(json_root_file, json_file_name)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            saved_json_file = json.load(f)
            
        insertion_area = np.array([0] + saved_json_file["region_area"])
        deletion_area = 1 - insertion_area
        
        insertion_score = np.array([saved_json_file["deletion_score"][-1]] + saved_json_file["insertion_score"])
        insertion_iou_score = np.array([saved_json_file["deletion_iou"][-1]] + saved_json_file["insertion_iou"])
        insertion_cls_score = np.array([saved_json_file["deletion_cls"][-1]] + saved_json_file["insertion_cls"])
        
        deletion_score = np.array([saved_json_file["insertion_score"][-1]] + saved_json_file["deletion_score"])
        deletion_iou_score = np.array([saved_json_file["insertion_iou"][-1]] + saved_json_file["deletion_iou"])
        deletion_cls_score = np.array([saved_json_file["insertion_cls"][-1]] + saved_json_file["deletion_cls"])
        
        insertion_auc = metrics.auc(insertion_area, insertion_score)
        deletion_auc = metrics.auc(deletion_area, deletion_score)
        
        insertion_iou_auc = metrics.auc(insertion_area, insertion_iou_score)
        deletion_iou_auc = metrics.auc(deletion_area, deletion_iou_score)
        
        insertion_cls_auc = metrics.auc(insertion_area, insertion_cls_score)
        deletion_cls_auc = metrics.auc(deletion_area, deletion_cls_score)
        
        insertion_aucs.append(insertion_auc)
        deletion_aucs.append(deletion_auc)
        
        insertion_iou_aucs.append(insertion_iou_auc)
        deletion_iou_aucs.append(deletion_iou_auc)
        
        insertion_cls_aucs.append(insertion_cls_auc)
        deletion_cls_aucs.append(deletion_cls_auc)
        
        # highest cls
        highest_cls_score_50.append(
            ((insertion_iou_score>0.5) * insertion_cls_score).max()
        ) 
        highest_cls_score_75.append(
            ((insertion_iou_score>0.5) * insertion_cls_score).max()
        )
    
    insertion_auc_score = np.array(insertion_aucs).mean()
    deletion_auc_score = np.array(deletion_aucs).mean()
    
    insertion_iou_auc_score = np.array(insertion_iou_aucs).mean()
    deletion_iou_auc_score = np.array(deletion_iou_aucs).mean()
    
    insertion_cls_auc_score = np.array(insertion_cls_aucs).mean()
    deletion_cls_auc_score = np.array(deletion_cls_aucs).mean()
    
    average_highest_cls_score_50 = np.array(highest_cls_score_50).mean()
    average_highest_cls_score_75 = np.array(highest_cls_score_75).mean()
    
    debug_success_rate_50 = (np.array(highest_cls_score_50)>0.35).mean()
    debug_success_rate_75 = (np.array(highest_cls_score_75)>0.25).mean()
    
    print("Insertion AUC Score: {:.4f}\nDeletion AUC Score: {:.4f}".format(insertion_auc_score, deletion_auc_score))
    print("Insertion CLS AUC Score: {:.4f}\nDeletion CLS AUC Score: {:.4f}".format(insertion_cls_auc_score, deletion_cls_auc_score))
    print("Insertion IOU AUC Score: {:.4f}\nDeletion IOU AUC Score: {:.4f}".format(insertion_iou_auc_score, deletion_iou_auc_score))
    
    print("Average highest confidence, IOU@0.50: {:.4f}, IOU@0.75: {:.4f}".format(average_highest_cls_score_50, average_highest_cls_score_75))
    print("Debug successful rate, IOU@0.50: {:.4f}, IOU@0.75: {:.4f}".format(debug_success_rate_50, debug_success_rate_75))
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)