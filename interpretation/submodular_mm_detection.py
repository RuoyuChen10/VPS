import numpy as np

from tqdm import tqdm

import torch
import cv2

import time

class DetectionSubModularExplanation(object):
    """
    Instance-level interpretability of object detection 
    based on submodular subset selection
    """
    def __init__(self, 
                 detection_model,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 batch_size = 4,    # Suggestion: [2080Ti: 4], [3090: 16]
                 mode = "object",   # object, iou, cls
                 device = "cuda"):
        """_summary_

        Args:
            detection_model (_type_): _description_
            lambda1 (float, optional): _description_. Defaults to 1.0.
            lambda2 (float, optional): _description_. Defaults to 1.0.
            device (str, optional): _description_. Defaults to "cuda".
        """
        super(DetectionSubModularExplanation, self).__init__()
        
        # Parameters of the submodular
        self.detection_model = detection_model.to(device)
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.device = device
        
        self.batch_size = batch_size
        self.mode = mode
        
    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["insertion_score"] = []
        self.saved_json_file["deletion_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["insertion_iou"] = []
        self.saved_json_file["insertion_box"] = []
        self.saved_json_file["insertion_cls"] = []
        self.saved_json_file["deletion_iou"] = []
        self.saved_json_file["deletion_box"] = []
        self.saved_json_file["deletion_cls"] = []
        self.saved_json_file["region_area"] = []
        self.saved_json_file["target_box"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["mode"] = self.mode
    
    def process_in_batches(self, images, batch_size, detection_model, h, w):
        all_bounding_boxes = []
        all_logits = []

        # 将输入图像拆分为 batch_size 批次
        num_batches = (len(images) + batch_size - 1) // batch_size  # 计算需要的批次数

        for i in range(num_batches):
            # 获取当前批次的图像
            batch_images = images[i * batch_size:(i + 1) * batch_size]

            # 将当前批次传递到检测模型
            bounding_boxes, logits = detection_model(batch_images, h, w)

            # 将结果收集到列表中
            all_bounding_boxes.append(bounding_boxes)
            all_logits.append(logits)

        # 将所有批次的结果拼接成一个完整的张量
        all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        return all_bounding_boxes, all_logits
    
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
    
    def generate_masked_input(self, alpha_batch):
        alpha_batch = np.array(alpha_batch) # [33, 714, 979, 1]
        
        source_image_process = self.source_image[np.newaxis, :, :, :]  # [1, 773, 1332, 3]
        
        return alpha_batch * source_image_process
        
    def evaluation_maximun_sample(self, S_set):
        # timer = time.time()
        V_set_tem = np.array(self.V_set) # (100, 773, 1332, 1)
        
        alpha_batch = (V_set_tem + self.refer_baseline[np.newaxis,...]).astype(np.uint8) # (100, 773, 1332, 1)
        
        # print("Stage 1 time comsume: {}".format(time.time()-timer))
        # timer = time.time()
        
        batch_input_images = self.generate_masked_input(alpha_batch)
        batch_input_images_reverse = self.generate_masked_input(1-alpha_batch)
        
        # print("Stage 2 time comsume: {}".format(time.time()-timer))
        # timer = time.time()
        
        with torch.no_grad():
            # Insertion
            bounding_boxes, logits = self.process_in_batches(batch_input_images, self.batch_size, self.detection_model, self.h, self.w) # [batch, np, 4] [batch, np, 256]
            
            # print("Stage 3.1 time comsume: {}".format(time.time()-timer))
            # timer = time.time()
            
            ious = self.calculate_iou(bounding_boxes, self.target_box)
            if self.mode == "cls":
                ious_clip = (ious>0.5).int()
            elif self.mode == "object":
                ious_clip = ious
            
            cls_score = logits[:,:,self.target_label].max(dim=-1)[0]   # torch.Size([170, 900])
            
            insertion_scores = (ious_clip * cls_score).max(dim=-1)[0]
            
            # print("Stage 3 time comsume: {}".format(time.time()-timer))
            # timer = time.time()
            
            # Deletion
            bounding_boxes_reverse, logits_reverse = self.process_in_batches(batch_input_images_reverse, self.batch_size, self.detection_model, self.h, self.w) # [batch, np, 4] [batch, np, 256]
            
            ious_reverse = self.calculate_iou(bounding_boxes_reverse, self.target_box)
            if self.mode == "cls":
                ious_reverse_clip = (ious_reverse>0.5).int()
            elif self.mode == "object":
                ious_reverse_clip = ious_reverse
            
            cls_score_reverse = logits_reverse[:,:,self.target_label].max(dim=-1)[0]   # torch.Size([170, 900])
            
            deletion_scores = (ious_reverse_clip * cls_score_reverse).max(dim=-1)[0]
            
            # print("Stage 4 time comsume: {}".format(time.time()-timer))
            # timer = time.time()
            
            #Overall submodular score
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1-deletion_scores)
            arg_max_index = smdl_scores.argmax().cpu().item()
            
            # print("Stage 5 time comsume: {}".format(time.time()-timer))
            # timer = time.time()
            
            # Save intermediate results
            insertion_boxer = bounding_boxes[arg_max_index].cpu().numpy()
            insertion_box_id = (ious[arg_max_index] * cls_score[arg_max_index]).argmax().cpu().item()
            insertion_box = insertion_boxer[insertion_box_id].astype(int).tolist()
            insertion_iou = ious[arg_max_index][insertion_box_id].cpu().item()
            insertion_cls = cls_score[arg_max_index][insertion_box_id].cpu().item()
            self.saved_json_file["insertion_iou"].append(insertion_iou)
            self.saved_json_file["insertion_box"].append(insertion_box)
            self.saved_json_file["insertion_cls"].append(insertion_cls)
            
            deletion_boxer = bounding_boxes_reverse[arg_max_index].cpu().numpy()
            deletion_box_id = (ious_reverse[arg_max_index] * cls_score_reverse[arg_max_index]).argmax().cpu().item()
            deletion_box = deletion_boxer[deletion_box_id].astype(int).tolist()
            deletion_iou = ious_reverse[arg_max_index][deletion_box_id].cpu().item()
            deletion_cls = cls_score_reverse[arg_max_index][deletion_box_id].cpu().item()
            self.saved_json_file["deletion_iou"].append(deletion_iou)
            self.saved_json_file["deletion_box"].append(deletion_box)
            self.saved_json_file["deletion_cls"].append(deletion_cls)
            
            # Update
            S_set.append(self.V_set[arg_max_index])
            self.refer_baseline = self.refer_baseline+self.V_set[arg_max_index]
            del self.V_set[arg_max_index]
            
            self.saved_json_file["region_area"].append(
                self.refer_baseline.sum() / self.region_area
            )
            
            self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].cpu().item())
            self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].cpu().item())
            self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())

        return S_set
    
    def get_merge_set(self):
        # define a subset
        S_set = []
        self.refer_baseline = np.zeros_like(self.V_set[0])
        
        for i in tqdm(range(self.saved_json_file["sub-region_number"])):
            S_set = self.evaluation_maximun_sample(S_set)
        
        self.saved_json_file["org_score"] = self.saved_json_file["insertion_score"][-1]
        self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_score"][-1]
        
        return S_set
    
    def __call__(self, image, V_set, class_id, given_box):
        """_summary_

        Args:
            image (cv2 format): (h, w, 3)
            V_set (_type_): (n, h, w, 3)
            class_id (List [int, ...]): which classes?
            given_box (xyxy): which boxes?
        """
        self.save_file_init()
        self.saved_json_file["target_box"] = given_box
        self.saved_json_file["sub-region_number"] = len(V_set)
        
        self.source_image = image
        self.h, self.w, _ = self.source_image.shape
        self.region_area = self.h * self.w
        
        self.V_set = V_set.copy()
        self.target_label = torch.tensor(class_id)
        self.target_box = given_box
        self.saved_json_file["target_label"] = class_id
        
        Submodular_Subset = self.get_merge_set()
        
        self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
        self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
        return Submodular_Subset, self.saved_json_file