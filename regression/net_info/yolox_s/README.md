## Yolox-s

<https://github.com/Megvii-BaseDetection/YOLOX>

## Dataset

* [COCO_2017](http://images.cocodataset.org/zips/val2017.zip )
* [label instances_val2017.json](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
* please put `coco 2017` in folder `$DATASET_PATH/coco/`

## Eval Code

eval code reference to [eval.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/eval.py)

## Accuracy Results

* official mAP = **40.5** by COCO

- 2021-12-22
  imageset: COCO image num 5000
  for eval:
  score_threshold: 0.01, overlap threshold: 0.65
  for draw:
  score_threshold: 0.1, overlap threshold: **0.45**

  | mode            | mAPval<br/>0.5:0.95 |
  | --------------- | ------------------- |
  | torch           | 40.5                |
  | onnx            | 40.4                |
  | int8 multiplier | 38.7                |