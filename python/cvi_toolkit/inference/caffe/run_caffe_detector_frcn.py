#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from cvi_toolkit.model import CaffeModel

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def check_files(args):
    if not os.path.isfile(args.model_def):
        print("cannot find the file %s", args.model_def)
        sys.exit(1)

    if not os.path.isfile(args.pretrained_model):
        print("cannot find the file %s", args.pretrained_model)
        sys.exit(1)

    if not os.path.isfile(args.input_file):
        print("cannot find the file %s", args.input_file)
        sys.exit(1)


def draw(image, output,verbose):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # https://github.com/amikelive/coco-labels

    for i  in range(len(output)):
        x1, y1, x2, y2, index, score = output[i]
        cls = CLASSES[int(index)]

        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(cls, score),
                        (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 1, cv2.LINE_AA)

        if verbose:
            print('class: {0}, score: {1:.2f}'.format(cls, score))
            print('box coordinate x, y, w, h: {0}'.format(bboxs[i]))

    return image


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')

    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--dump_blobs_with_inplace",
                        type=bool, default=False,
                        help="Dump all blobs including inplace blobs (takes much longer time)")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")

    args = parser.parse_args()
    check_files(args)
    return args


if __name__ == '__main__':
    args = parse_args()

    net_input_dims = [int(x) for x in args.net_input_dims.split(',')]

    prototxt = args.model_def
    caffemodel = args.pretrained_model

    im = cv2.imread(args.input_file)
    im_orig = im
    im_shape = im_orig.shape

    scale = min(float(net_input_dims[1]) / im_shape[1], float(net_input_dims[0]) / im_shape[0])
    rescale_w = int(im_shape[1] * scale)
    rescale_h = int(im_shape[0] * scale)

    resized_img = cv2.resize(im, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((net_input_dims[0], net_input_dims[1], 3), 0, dtype=np.float32)
    paste_w = (net_input_dims[1] - rescale_w) // 2
    paste_h = (net_input_dims[0] - rescale_h) // 2

    new_image[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img

    new_image = new_image.astype(np.float32, copy=True)
    new_image -= np.array([[[102.9801, 115.9465, 122.7717]]])

    new_image = np.transpose(new_image, [2,0,1])
    image_x = np.expand_dims(new_image, axis=0)
    inputs = image_x
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, image_x, axis=0)

    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    caffemodel.inference(inputs)
    outputs = caffemodel.net.blobs

    all_tensor_dict = caffemodel.get_all_tensor(inputs, args.dump_blobs_with_inplace)
    np.savez(args.dump_blobs, **all_tensor_dict)

    output = outputs['output'].data
    output[:,:,:,0:4] = output[:,:,:,0:4] / scale
    output = output[0][0].tolist()
    if args.draw_image:
      result = draw(im_orig, output, False)
      cv2.imwrite(args.draw_image, result)