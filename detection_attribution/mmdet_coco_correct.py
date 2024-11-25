"""
Created on 2024/10/04

@author: Ruoyu Chen
Grounding DINO Attribution
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

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

from interpretation.submodular_mm_detection import DetectionSubModularExplanation

from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg
from mmdet.models.utils import select_single_mlvl, filter_scores_and_topk
from mmdet.structures.bbox import bbox2roi
from mmcv.transforms import Compose
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)

from utils import COCO_TEXT_PROMPT, coco_classes, coco_classes_grounding_idx, mkdir

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_mask_rcnn_correct.json',
                        help='Datasets.')
    parser.add_argument('--detector',
                        type=str,
                        default='mask_rcnn',
                        help='Object detector.')
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
                        type=int, default=64,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/tradition-detector-correctly/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class Mask_RCNN_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.target_class = None
        
    def forward(self, images, h, w):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes+1)
        """
        prediction_boxes = []
        prediction_logits = []
        
        for image in images:
            data_ = dict(img=image, img_id=0)
            data_ = self.test_pipeline(data_)
            
            data_['inputs'] = [data_['inputs']]
            data_['data_samples'] = [data_['data_samples']]
            
            with torch.no_grad():
                img_shape = data_['data_samples'][0].metainfo['img_shape']
                scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
                
                data_preproccess = self.detection_model.data_preprocessor(data_, False)
                
                # backbone
                x = self.detection_model.extract_feat(data_preproccess['inputs'])

                rpn_results_list = self.detection_model.rpn_head.predict(
                        x, data_preproccess['data_samples'], rescale=False)
                
                proposals = [res.bboxes for res in rpn_results_list]
                rois = bbox2roi(proposals)
                
                roi_outs = self.detection_model.roi_head.forward(x, rpn_results_list,
                                         data_preproccess['data_samples'])
        
                bbox_pred = roi_outs[1]
                # classification
                cls_score_ = F.softmax(roi_outs[0], dim=-1)
                
                # bounding boxes
                num_classes = self.detection_model.roi_head.bbox_head.num_classes
                rois = rois.repeat_interleave(num_classes, dim=0)
                bbox_pred = bbox_pred.view(-1, self.detection_model.roi_head.bbox_head.bbox_coder.encode_size)
                bboxes = self.detection_model.roi_head.bbox_head.bbox_coder.decode(
                        rois[..., 1:], bbox_pred, max_shape=img_shape)
                
                bboxes = bboxes.view(-1, num_classes, self.detection_model.roi_head.bbox_head.bbox_coder.encode_size)
                
                bboxes = scale_boxes(bboxes, scale_factor)
                
                if bboxes.shape[0] != 1000:
                    pad_size = 1000 - bboxes.shape[0]
                    bboxes = F.pad(bboxes, (0, 0, 0, 0, 0, pad_size))
                    
                    cls_score_ = F.pad(cls_score_, (0, 0, 0, pad_size))
                prediction_boxes.append(bboxes)
                prediction_logits.append(cls_score_)
            
        prediction_logits = torch.stack(prediction_logits)
        # prediction_logits = prediction_logits[:,:,self.target_class].unsqueeze(-1)
        prediction_boxes = torch.stack(prediction_boxes)[:,:,self.target_class,:]

        return prediction_boxes, prediction_logits
    
class YOLO_V3_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.top_k = 5000
        
    def forward(self, images, h, w):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        prediction_boxes = []
        prediction_logits = []
        
        for image in images:
            cv2.imwrite("yolo.jpg", image)
            data_ = dict(img=image, img_id=0)
            data_ = self.test_pipeline(data_)
            
            data_['inputs'] = [data_['inputs']]
            data_['data_samples'] = [data_['data_samples']]
            
            with torch.no_grad():
                data_preproccess = self.detection_model.data_preprocessor(data_, False)
                scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
                
                pred_maps = self.detection_model.forward(data_preproccess['inputs'])[0]
        
                featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]
                mlvl_anchors = self.detection_model.bbox_head.prior_generator.grid_priors(
                    featmap_sizes, device=pred_maps[0].device)
                flatten_preds = []
                flatten_strides = []
                for pred, stride in zip(pred_maps, self.detection_model.bbox_head.featmap_strides):
                    pred = pred.permute(0, 2, 3, 1).reshape(1, -1,
                                                            self.detection_model.bbox_head.num_attrib)
                    pred[..., :2].sigmoid_()
                    flatten_preds.append(pred)
                    flatten_strides.append(
                        pred.new_tensor(stride).expand(pred.size(1)))

                flatten_preds = torch.cat(flatten_preds, dim=1)
                flatten_bbox_preds = flatten_preds[..., :4]
                flatten_objectness = flatten_preds[..., 4].sigmoid()
                flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
                flatten_anchors = torch.cat(mlvl_anchors)
                flatten_strides = torch.cat(flatten_strides)
                flatten_bboxes = self.detection_model.bbox_head.bbox_coder.decode(flatten_anchors,
                                                        flatten_bbox_preds,
                                                        flatten_strides.unsqueeze(-1))  # 不定长，可以选top 400
                bboxes = scale_boxes(flatten_bboxes, scale_factor)
                
                # 获取前topk个最大值的索引
                # _, top_indices = torch.topk(flatten_objectness, self.top_k, dim=-1)
                
                # scores = torch.gather(flatten_cls_scores, 1, top_indices.unsqueeze(-1).expand(-1, -1, flatten_cls_scores.size(-1)))[0]
                # bboxes = torch.gather(bboxes, 1, top_indices.unsqueeze(-1).expand(-1, -1, bboxes.size(-1)))[0]
                prediction_boxes.append(bboxes[0])
                prediction_logits.append(flatten_cls_scores[0])
        
        prediction_logits = torch.stack(prediction_logits)
        prediction_boxes = torch.stack(prediction_boxes)

        return prediction_boxes, prediction_logits
    
class FCOS_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.top_k = 10000
        
    def forward(self, images, h, w):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        prediction_boxes = []
        prediction_logits = []
        
        for image in images:
            data_ = dict(img=image, img_id=0)
            data_ = self.test_pipeline(data_)
            
            data_['inputs'] = [data_['inputs']]
            data_['data_samples'] = [data_['data_samples']]
                
            with torch.no_grad():
                data_preproccess = self.detection_model.data_preprocessor(data_, False)
                scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
                cls_scores, bbox_preds, score_factors = self.detection_model.forward(data_preproccess['inputs'])
                
                num_levels = len(cls_scores)
                featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
                mlvl_priors = self.detection_model.bbox_head.prior_generator.grid_priors(
                    featmap_sizes,
                    dtype=cls_scores[0].dtype,
                    device=cls_scores[0].device)
                
                cls_score_list = select_single_mlvl(
                    cls_scores, 0, detach=True)
                bbox_pred_list = select_single_mlvl(
                    bbox_preds, 0, detach=True)
                score_factor_list = select_single_mlvl(
                    score_factors, 0, detach=True)
                    
                mlvl_bbox_preds = []
                mlvl_valid_priors = []
                mlvl_scores = []
                
                for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
                    assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                
                    dim = self.detection_model.bbox_head.bbox_coder.encode_size
                    
                    bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
                    score_factor = score_factor.permute(1, 2,
                                                        0).reshape(-1).sigmoid()
                    
                    cls_score = cls_score.permute(1, 2,
                                                0).reshape(-1, self.detection_model.bbox_head.cls_out_channels)
                    scores = cls_score.sigmoid()
                    mlvl_bbox_preds.append(bbox_pred)
                    mlvl_scores.append(scores)
                    mlvl_valid_priors.append(priors)
                    
            bbox_pred = torch.cat(mlvl_bbox_preds)
            priors = cat_boxes(mlvl_valid_priors)
            
            bboxes = self.detection_model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=data_['data_samples'][0].metainfo["img_shape"])
            scores = torch.cat(mlvl_scores)
            # # 获取前topk个最大值的索引
            # object_scores, _ = scores.max(-1)
            # _, top_indices = torch.topk(object_scores, self.top_k, dim=-1)
            
            # bboxes = bboxes[top_indices]
            # scores = scores[top_indices]
            
            bboxes = scale_boxes(bboxes, scale_factor)
            
            prediction_boxes.append(bboxes)
            prediction_logits.append(scores)

        prediction_logits = torch.stack(prediction_logits)
        prediction_boxes = torch.stack(prediction_boxes)
        
        return prediction_boxes, prediction_logits

class SSD_R(torch.nn.Module):
    def __init__(self, 
                 detection_model,
                 device = "cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        
        # 定义测试图像的预处理流程
        test_pipeline = get_test_pipeline_cfg(self.detection_model.cfg)
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline)
        
        self.top_k = 10000
        
    def forward(self, images, h, w):
        """_summary_

        Args:
            images (_type_): 一个列表，里面是原始的图片
            h (_type_): _description_
            w (_type_): _description_
            
        Return:
            prediction_boxes: (batch, 1000, 4)
            prediction_logits: (batch, 1000, num_classes)
        """
        prediction_boxes = []
        prediction_logits = []
        
        for image in images:
            data_ = dict(img=image, img_id=0)
            data_ = self.test_pipeline(data_)
            
            data_['inputs'] = [data_['inputs']]
            data_['data_samples'] = [data_['data_samples']]
            
            with torch.no_grad():
                data_preproccess = self.detection_model.data_preprocessor(data_, False)
                scale_factor = [1 / s for s in data_['data_samples'][0].metainfo['scale_factor']]
                cls_scores, bbox_preds = self.detection_model.forward(data_preproccess['inputs'])
                num_levels = len(cls_scores)
                featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
                mlvl_priors = self.detection_model.bbox_head.prior_generator.grid_priors(
                    featmap_sizes,
                    dtype=cls_scores[0].dtype,
                    device=cls_scores[0].device)
                
                cls_score_list = select_single_mlvl(
                    cls_scores, 0, detach=True)
                bbox_pred_list = select_single_mlvl(
                    bbox_preds, 0, detach=True)
                score_factor_list = [None for _ in range(num_levels)]
                # score_factor_list = select_single_mlvl(
                #     score_factors, 0, detach=True)
                    
                mlvl_bbox_preds = []
                mlvl_valid_priors = []
                mlvl_scores = []
                
                for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
                    assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                
                    dim = self.detection_model.bbox_head.bbox_coder.encode_size
                    
                    bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
                    # score_factor = score_factor.permute(1, 2,
                    #                                     0).reshape(-1).sigmoid()
                    
                    cls_score = cls_score.permute(1, 2,
                                                0).reshape(-1, self.detection_model.bbox_head.cls_out_channels)
                    scores = cls_score.softmax(-1)[:, :-1]
                    mlvl_bbox_preds.append(bbox_pred)
                    mlvl_scores.append(scores)
                    mlvl_valid_priors.append(priors)
                    
            bbox_pred = torch.cat(mlvl_bbox_preds)
            priors = cat_boxes(mlvl_valid_priors)
            
            bboxes = self.detection_model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=data_['data_samples'][0].metainfo["img_shape"])
            scores = torch.cat(mlvl_scores)
            # # 获取前topk个最大值的索引
            # object_scores, _ = scores.max(-1)
            # _, top_indices = torch.topk(object_scores, self.top_k, dim=-1)
            
            # bboxes = bboxes[top_indices]
            # scores = scores[top_indices]
            
            bboxes = scale_boxes(bboxes, scale_factor)
            
            prediction_boxes.append(bboxes)
            prediction_logits.append(scores)

        prediction_logits = torch.stack(prediction_logits)
        prediction_boxes = torch.stack(prediction_boxes)

        return prediction_boxes, prediction_logits

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

def main(args):
    device = "cuda"
    
    assert args.detector in args.eval_list
    
    # model init
    # Load the model
    if args.detector == "mask_rcnn":
        config = 'config/mask-rcnn_r50_fpn_2x_coco.py'
        checkpoint = 'ckpt/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = Mask_RCNN_R(model.eval(), device)
    elif args.detector == "yolo_v3":
        config = 'config/yolov3_d53_8xb8-ms-608-273e_coco.py'
        checkpoint = 'ckpt/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = YOLO_V3_R(model.eval(), device)
    elif args.detector == "fcos":
        config = 'config/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py'
        checkpoint = 'ckpt/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
        model = init_detector(config, checkpoint, device)
        detection_model = FCOS_R(model.eval(), device)
    elif args.detector == "ssd":
        config = "config/ssd300_coco.py"
        checkpoint = "ckpt/ssd300_coco_20210803_015428-d231a06e.pth"
        model = init_detector(config, checkpoint, device)
        model.bbox_head.loss_cls = False
        detection_model = SSD_R(model.eval(), device)
    
    print("Load {} model!".format(args.detector))
    
    # Submodular
    smdl = DetectionSubModularExplanation(
        detection_model,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device="cuda",
        batch_size=100
    )
    
    # Read datasets
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
        
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(args.detector, args.lambda1, args.lambda2, args.division_number))  
    
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
            os.path.join(save_json_root_path, info["file_name"].replace(".jpg", "_{}.json".format(info["id"])))
        ):
            continue
        
        target_class = coco_classes.index(info["category"])
        target_box = info["bbox"]
        image_path = os.path.join(args.Datasets, info["file_name"])
        
        image = cv2.imread(image_path)
        
        # Sub-region division
        # image_proccess = transform_vision_data(image)
        # image_seg = cv2.resize(image, image_proccess.shape[1:][::-1])

        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)

        V_set = SubRegionDivision(image, region_size = region_size)
        
        smdl.detection_model.target_class = target_class
        
        S_set, saved_json_file = smdl(image, V_set, [target_class], target_box)
        saved_json_file["category"] = info["category"]
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, info["file_name"].replace(".jpg", "_{}.npy".format(info["id"]))),
            np.array(S_set)
        )
        
        # Save json file
        with open(
            os.path.join(save_json_root_path, info["file_name"].replace(".jpg", "_{}.json".format(info["id"]))), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
         
    return

if __name__ == "__main__":
    args = parse_args()
    
    main(args)