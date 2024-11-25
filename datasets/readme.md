## Prepare the datasets

Download [MS COCO 2017](https://cocodataset.org/#download) dataset, [LVIS V1](https://www.lvisdataset.org/dataset) dataset and [RefCOCO+](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/refcoco%2B.zip) dataset and organize as follows:

```
--datasets
  |--coco
  |  |--annotations
  |  |--val2017
  |--lvis_v1
  |  |--annotations
  |  |--train2017
  |  |--val2017
  |--refcoco+
  |  |--image
  |  |--testA.tsv
  |  |--testB.tsv
  |  |--train.tsv
  |  |--val.tsv
  |--coco_groundingdino_correct_detection.json
  |--coco_groundingdino_misclassification.json
  |--coco_groundingdino_misdetect.json
  |--coco_florence-2_correct_detection.json
  |--coco_mask_rcnn_correct.json
  |--coco_yolo_v3_correct.json
  |--coco_fcos_correct.json
  |--coco_ssd_correct.json
  |--lvis_v1_rare_groundingdino_correct_detection.json
  |--lvis_v1_rare_groundingdino_misclassification.json
  |--lvis_v1_rare_groundingdino_misdetect.json
  |--refcoco_val_groundingdino_correct.json
  |--refcoco_val_groundingdino_mistake.json
  |--refcoco_val_florence-2_correct.json
```