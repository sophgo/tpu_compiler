
import numpy as np
import cv2
import argparse
from enum import Enum
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('preprocess')


def center_crop(img, crop_dim):
    # Take center crop.
    center = np.array(img.shape[1:]) / 2.0
    crop_dim = np.array(crop_dim)
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -(crop_dim / 2.0),
        crop_dim / 2.0
    ])
    crop = crop.astype(int)
    img = img[:, crop[0]:crop[2], crop[1]:crop[3]]
    return img

def add_preprocess_parser(parser):
    if not isinstance(parser, argparse.ArgumentParser):
        raise RuntimeError("parser is invaild")
    parser.add_argument("--image_resize_dims", type=str, default='256,256')
    parser.add_argument("--net_input_dims", type=str, default='224,224')
    parser.add_argument("--raw_scale", type=float, help="Multiply raw input image data by this scale.", default=255)
    parser.add_argument("--mean", help="Per Channel image mean values")
    parser.add_argument("--std", help="Per Channel image std values", default='1,1,1')
    parser.add_argument("--mean_file", type=str, help="the resized ImageNet dataset mean file.")
    parser.add_argument("--input_scale", type=float, help="Multiply input features by this scale.", default=1.0)
    parser.add_argument("--model_channel_order", type=str, help="channel order of model inference used, default: bgr", default="bgr")
    parser.add_argument("--data_format", type=str, help="input image data dim order, default: nchw", default="nchw")
    parser.add_argument("--bgray", type=int, default=0, help="whether the input image is gray, channel size is 1")
    return parser


def get_preprocess_parser(existed_parser=None):
    if existed_parser:
        parser = existed_parser
    else:
        parser = argparse.ArgumentParser(description="Image Preprocess.")
    return add_preprocess_parser(parser)

class preprocess(object):
    def __init__(self):
        pass

    def config(self, net_input_dims='224,224',
                     resize_dims=None,
                     mean=None,
                     mean_file=None,
                     std=None,
                     input_scale=1.0,
                     raw_scale=255.0,
                     data_format="nchw",
                     rgb_order='bgr',
                     npy_input=None,
                     letter_box=False,
                     batch=1,
                     bgray=0):
        print("preprocess :\n         \
            \tnet_input_dims: {}\n    \
            \tresize_dims   : {}\n    \
            \tmean          : {}\n    \
            \tmean_file     : {}\n    \
            \tstd           : {}\n    \
            \tinput_scale   : {}\n    \
            \traw_scale     : {}\n    \
            \tdata_format   : {}\n    \
            \trgb_order     : {}\n    \
            \tnpy_input     : {}\n    \
            \tletter_box    : {}\n    \
            \tbatch_size    : {}\n    \
            \tbgray         : {}\n    \
        ".format(net_input_dims, resize_dims, mean, \
                mean_file, std, input_scale, raw_scale, \
                 data_format, rgb_order, npy_input,
                letter_box, batch, bgray
        ))
        self.npy_input = npy_input
        self.letter_box = letter_box
        self.batch = batch
        self.bgray = bgray

        self.net_input_dims = [int(s) for s in net_input_dims.split(',')]
        if resize_dims != None :
            self.resize_dims = [int(s) for s in resize_dims.split(',')]
            self.resize_dims = [ max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]
        else :
            self.resize_dims = self.net_input_dims

        self.raw_scale = raw_scale

        if mean:
            self.mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
            if self.resize_dims != None :
                self.mean = self.mean[:, np.newaxis, np.newaxis]
            self.mean_file = np.array([])
        else:
            if mean_file != None :
                self.mean_file = np.load(mean_file)
            else:
                self.mean = np.array([])
                self.mean_file = np.array([])
        if std:
            self.std = np.array([float(s) for s in std.split(',')], dtype=np.float32)
        else:
            self.std = None

        if bgray:
            self.mean = self.mean[:1]
            self.std = self.std[:1]

        self.input_scale = float(input_scale)

        self.data_format = data_format[1:]

        self.rgb_order = rgb_order
        self.ori_channel_order = None

    def to_dict(self):
        return {
            'net_input_dims': self.net_input_dims,
            'resize_dims': self.resize_dims,
            'mean': self.mean,
            'std': self.std,
            'input_scale': self.input_scale,
            'raw_scale': self.raw_scale,
            'data_format': "n{}".format(self.data_format),
            'rgb_order': self.rgb_order,
        }

    def run(self, input, output_npz=None, pfunc=None,
            input_name=None, input_type='file',
            input_channel_order="rgb", output_channel_order="bgr",
            input_data_format="chw", output_data_format="chw"):

        if input_type == 'file':
            logger.debug("origin order is bgr(OpenCV), output channel order is {}".format(output_channel_order))
            if self.npy_input != None :
                x = np.load(str(self.npy_input).rstrip())
                if output_npz:
                    np.savez(output_npz, **{input_name if input_name else "input": x})
                return x

            image = cv2.imread(str(input).rstrip(), cv2.IMREAD_GRAYSCALE if self.bgray else cv2.IMREAD_COLOR)

            self.ori_channel_order = "bgr"
            if image is None:
                print("not existed {}".format(str(input).rstrip()))
                return None

            image = cv2.resize(image, (self.resize_dims[1], self.resize_dims[0])) # w,h
            image = image.astype(np.float32)

            if not self.bgray:
                # opencv read image data format is hwc
                # tranpose to chw
                image = np.transpose(image, (2, 0, 1))
            else:
                # if grapscale image,
                # expand dim to (1, h, w)
                image = np.expand_dims(image, axis=0)

            input_data_format="chw"

        elif input_type == 'tensor':
            if not isinstance(input, np.ndarray):
                raise RuntimeError("input type {} is wrong format, np.ndarray is expected".format(type(input_data)))
            logger.debug("input channel order is {}, output channel order is {}".format(input_channel_order, output_channel_order))
            # Default is rgb in
            self.ori_channel_order = input_channel_order
            if input_data_format == "chw":
                # input tensor shape is CHW
                if self.resize_dims != self.net_input_dims:
                    # CHW to HWC, then use cv2 resize
                    input = np.transpose(input, (1, 2, 0))
                    input = cv2.resize(input, (self.resize_dims[1], self.resize_dims[0])) # w,h
                    # turn back
                    input =  np.transpose(input, (2, 0, 1))
            elif input_data_format == "hwc":
                if self.resize_dims != self.net_input_dims:
                    input = cv2.resize(
                        input, (self.resize_dims[1], self.resize_dims[0]))  # w,h
            image = input

        # Do preprocess if with call back function
        if pfunc is not None:
            output = pfunc(image)
        else:
            x = image
            # we default use data format CHW do preprocess
            # if data format is HWC,  we still turn it to CHW
            # turn back to HWC after preprcessing
            if input_data_format == "hwc":
                x = np.transpose(x, (2, 0, 1))

            # if source data order is different with handle order
            # swap source data order
            # only handle rgb to bgr , bgr to rgb
            if self.rgb_order != self.ori_channel_order:
                logger.debug("ori image channel order is {}, but we handle order is {}, swap it".format(self.ori_channel_order, self.rgb_order))
                x = x[[2,1,0], :, :]


            x = x * (self.raw_scale / 255.0)
            # preprocess
            if self.mean_file.size != 0 :
                x -= self.mean_file
            elif self.mean.size != 0:
                x -= self.mean

            if self.input_scale != 1.0:
                x *= self.input_scale
            if self.std is not None:
                x /= self.std[:,np.newaxis, np.newaxis]

            # Take center crop.
            x = center_crop(x, self.net_input_dims)

            # if We need output order is not the same with preprocess order
            # swap it
            if output_channel_order != self.rgb_order:
                logger.debug("handle order is {}, but output order need {}, swap it".format(self.rgb_order, output_channel_order))
                x = x[[2,1,0], :, :]

            if input_data_format == "hwc" and self.data_format == "hwc":
                # if data format is HWC,  turn it to CHW at first
                # here we turn back to HWC
                x = np.transpose(x, (1, 2, 0))
            elif input_data_format == "chw" and self.data_format == "hwc":
                x = np.transpose(x, (1, 2, 0))
            elif input_data_format == "hwc" and self.data_format == "chw":
                # input data foramt is hwc
                # data format is chw
                # we transpose it before, just return
                pass
            else:
                # input data foramt is chw, data format is chw
                # no need to do anything
                pass

            output = np.expand_dims(x, axis=0)
        if self.bgray:
            # if is grayscale, then output only one channel
            output=output[:,0:1,:,:]

        if output_npz:
            # Must convert to npz file as input
            np.savez(output_npz, **{input_name if input_name else "input": output})


        return output