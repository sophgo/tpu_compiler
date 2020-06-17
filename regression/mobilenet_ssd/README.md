## MobileNet-SSD

<https://github.com/chuanqi305/MobileNet-SSD>

## Dataset

* [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
* [labelmap_voc.prototxt](https://github.com/sfzhang15/RefineDet/blob/master/data/VOC0712/labelmap_voc.prototxt)
* please put `VOC2012` and `VOC2007` in folder `$DATASET_PATH/VOCdevkit/`

## Eval Code

eval code reference to [voc_eval.py](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py)

## Accuracy Results

* official mAP = **72.7** by VOC0712

- 2020-06-17
  imageset: VOC2012 trainval, image num 11540; VOC2007 trainval, image num 5011
  overlap threshold: 0.5

  | mode            | mAP (VOC2012) | mAP (VOC2007) |
  | --------------- | ------------- | ------------- |
  | caffe           | 78.0          | 77.9          |
  | fp32            | 78.0          | 77.9          |
  | int8 multiplier | 73.1          | 73.0          |
