#!/usr/bin/env python3
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import cv2
import shutil
import caffe
import xml.etree.ElementTree as ET
import os
import argparse
import numpy as np
import uuid
import pymlir

# over_threshold = 0.5

voc_class = ('__background__',  # always index 0
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')


class pascal_voc():
    def __init__(self, image_set, year, devkit_path):
        self._year = year
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._class_to_ind = dict(zip(voc_class, range(len(voc_class))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._image_num = len(self._image_index)
        # Default to roidb handler
        self._salt = str(uuid.uuid4())

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # /data/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        image_index = []
        with open(image_set_file) as f:
            for x in f.readlines():
                image_index.append(x.strip())
                if args.count > 0 and len(image_index) == args.count:
                    break
        return image_index

    def _get_voc_results_file_template(self, cls):
        # /output_dir/<uuid>_aeroplane.txt
        filename = self._salt + '_' + self._image_set + '_{:s}.txt'.format(cls)
        path = os.path.join(args.output_dir, filename)
        return path

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def voc_eval(self,
                 detpath,
                 annopath,
                 classname,
                 ovthresh=0.5,
                 use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        classname: Category name (duh)
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name

        # first load gt
        recs = {}
        for i, imagename in enumerate(self._image_index):
            recs[imagename] = self.parse_rec(annopath.format(imagename))

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in self._image_index:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        detfile = detpath.format(classname)
        if os.path.isfile(detfile) == False:
            return 0, 0, 0
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        if len(image_ids) == 0:
            return 0, 0, 0
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def _do_python_eval(self, output_dir='output', over_threshold=0.5):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        print('~~~~~~~~')
        for i, cls in enumerate(voc_class):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(
                filename, annopath, cls, over_threshold,
                use_07_metric=use_07_metric)
            if ap == 0:
                continue
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('mAP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')

    def evaluate_detections(self, output_dir):
        self._do_python_eval(output_dir)
        # cleanup
        for cls in voc_class:
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(cls)
            if os.path.isfile(filename):
                os.remove(filename)

    def parse_top_detection(self, image_index, resolution, detections, conf_threshold=0.6):
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(
            det_conf) if conf >= conf_threshold]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].astype(int).tolist()
        top_xmin = det_xmin[top_indices]
        top_xmin[top_xmin < 0] = 0
        top_ymin = det_ymin[top_indices]
        top_ymin[top_ymin < 0] = 0
        top_xmax = det_xmax[top_indices]
        top_xmax[top_xmax > 1] = 1.0
        top_ymax = det_ymax[top_indices]
        top_ymax[top_ymax > 1] = 1.0

        bboxs = np.zeros((top_conf.shape[0], 4), dtype=int)
        for i in range(top_conf.shape[0]):
            bboxs[i][0] = int(round(top_xmin[i] * resolution[1]))
            bboxs[i][1] = int(round(top_ymin[i] * resolution[0]))
            bboxs[i][2] = int(round(top_xmax[i] * resolution[1]))
            bboxs[i][3] = int(round(top_ymax[i] * resolution[0]))
        for i, cls in enumerate(top_label_indices):
            filename = self._get_voc_results_file_template(voc_class[cls])
            with open(filename, 'a+') as f:
                f.write('{:s} {:.4f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(image_index, top_conf[i],
                               bboxs[i][0], bboxs[i][1], bboxs[i][2], bboxs[i][3]))
        return top_label_indices, top_conf, bboxs


def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Caffe model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from caffemodel, and eval by Caffe.')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
    parser.add_argument('--mlir', type=str, default='',
                        help='load mlir file, and eval by mlir')
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_scale", type=float,
                        help="Multiply input features by this scale.", default=1.0)
    parser.add_argument("--mean", help="Per Channel image mean values")
    parser.add_argument("--dataset", type=str, default='',
                        help="dataset path")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Result output dir")
    parser.add_argument("--count", type=int, default=-1)
    args = parser.parse_args()
    return args


def eval_voc_main():
    voc = pascal_voc('trainval', '2012', args.dataset)
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]

    mean = np.array([float(s) for s in args.mean.split(',')], dtype=np.float32)
    transformer = caffe.io.Transformer({'data': (1,3,net_input_dims[0],net_input_dims[1])})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', mean)
    transformer.set_input_scale('data', args.input_scale)
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR

    if not args.mlir.strip():
        # eval by caffe
        net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
        for i in range(voc._image_num):
            if i % 1000 == 0:
                print("caffe inference image:{}/{}".format(i, voc._image_num))
            image = caffe.io.load_image(voc.image_path_at(i))
            net.blobs['data'].reshape(1, 3, net_input_dims[0], net_input_dims[1])
            net.blobs['data'].data[...] = transformer.preprocess('data', image)
            detections = net.forward()['detection_out']
            voc.parse_top_detection(
                voc._image_index[i], image.shape, detections, conf_threshold=args.confidence_threshold)
    else:
        module = pymlir.module()
        module.load(args.mlir)
        for i in range(voc._image_num):
            if i % 1000 == 0:
                print("mlir inference image:{}/{}".format(i, voc._image_num))
            image = caffe.io.load_image(voc.image_path_at(i))
            x = transformer.preprocess('data', image)
            x = np.expand_dims(x, axis=0)
            _ = module.run(x)
            data = module.get_all_tensor()
            detections = data['detection_out']
            # print(x.shape, detections.shape)
            voc.parse_top_detection(
                voc._image_index[i], image.shape, detections, conf_threshold=args.confidence_threshold)
    voc.evaluate_detections(args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    eval_voc_main()

