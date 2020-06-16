## MobileNet-SSD

<https://github.com/chuanqi305/MobileNet-SSD>

## Dataset

[VOC2012](http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)
[labelmap_voc.prototxt](https://github.com/sfzhang15/RefineDet/blob/master/data/VOC0712/labelmap_voc.prototxt)

**please put `VOC2012` in `$DATASET_PATH` **

## Eval Code

eval code reference to [voc_eval.py](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py)

## Accuracy Results

* official mAP: 72.7

- 20200616
  imageset: trainval; image num: 11540; over threshold: 0.5

  | mode            | mAP  |
  | --------------- | ---- |
  | caffe           | 78.0 |
  | fp32            | 78.0 |
  | int8 multiplier | 73.1 |
