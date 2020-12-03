
import numpy as np
import cv2
import argparse
from enum import Enum
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('preprocess')


class CropMethod(Enum):
    CENTOR = "centor"
    ASPECT_RATIO = "aspect_ratio"


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


def get_aspect_ratio_img(img, net_h, net_w, img_h, img_w, no_pad=False):
    rescale_h, rescale_w = get_aspect_ratio_wh(net_h, net_w, img_h, img_w)
    resized_img = _get_aspect_ratio_img(img, net_h, net_w, img_h, img_w)
    if no_pad:
        return resized_img
    new_image = np.full((net_h, net_w, 3), 0, dtype=np.float32)
    offset_w = (net_w - rescale_w) // 2
    offset_h = (net_h - rescale_h) // 2

    new_image[offset_h: offset_h + rescale_h,
              offset_w: offset_w + rescale_w, :] = resized_img
    return new_image


def get_aspect_ratio_pads(net_h, net_w, rescale_h, rescale_w):

    offset_w = (net_w - rescale_w) // 2
    offset_h = (net_h - rescale_h) // 2
    pad_l = offset_w
    pad_r = net_w - offset_w - rescale_w
    pad_t = offset_h
    pad_b = net_h - offset_h - rescale_h
    return [0, 0, pad_t, pad_l, 0, 0, pad_b, pad_r]

def _get_aspect_ratio_img(img, net_h, net_w, img_h, img_w):
    rescale_h, rescale_w = get_aspect_ratio_wh(
         net_h, net_w, img_h, img_w)
    return cv2.resize(img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)

def get_aspect_ratio_wh(net_h, net_w, img_h, img_w):
    scale = min(float(net_w) / img_w, float(net_h) / img_h)
    rescale_w = int(img_w * scale)
    rescale_h = int(img_h * scale)
    return rescale_h, rescale_w

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
    parser.add_argument("--astype", type=str, help="store npz type, default is float32", default="float32")
    parser.add_argument("--crop_method", type=str,
                        help="crop method when image_resize_dims not same with net_input_dims, \
                        ,ex: centor or aspect_ratio, if aspect_ratio the flag will ignore --image_resize_dims value", default="centor",
                        choices=["centor", "aspect_ratio"])
    parser.add_argument("--only_aspect_ratio_img",
                        help="get aspect ratio data only, no padding", type=int, default=0)

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
                     batch=1,
                     bgray=0,
                     crop_method="centor",
                     only_aspect_ratio_img=0,
                     astype="float32"):
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
            \tbatch_size    : {}\n    \
            \tbgray         : {}\n    \
            \tcrop_method   : {}\n    \
            \tastype        : {}\n    \
        ".format(net_input_dims, resize_dims, mean, \
                mean_file, std, input_scale, raw_scale, \
                data_format, rgb_order, npy_input, \
                batch, bgray, crop_method, astype,
        ))
        self.npy_input = npy_input
        self.batch = batch
        self.bgray = bgray

        self.net_input_dims = [int(s) for s in net_input_dims.split(',')]
        if resize_dims != None and crop_method != "aspect_ratio":
            self.resize_dims = [int(s) for s in resize_dims.split(',')]
            self.resize_dims = [ max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]
        else :
            self.resize_dims = self.net_input_dims

        self.raw_scale = raw_scale
        self.astype = astype
        if crop_method == "centor":
            self.crop_method = CropMethod.CENTOR
        elif crop_method == "aspect_ratio":
            self.crop_method = CropMethod.ASPECT_RATIO
            self.only_aspect_ratio_img = only_aspect_ratio_img
        else:
            raise RuntimeError("Not Existed crop method {}".format(crop_method))
        if mean:
            self.mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
            if self.resize_dims != None :
                self.mean = self.mean[:, np.newaxis, np.newaxis]
            self.mean_file = np.array([])
        else:
            if mean_file != None :
                self.mean_file = np.load(mean_file)
            else:
                self.mean = np.array([0,0,0], dtype=np.float32)
                self.mean = self.mean[:, np.newaxis, np.newaxis]
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

    def to_dict(self, input_w=0, input_h=0, preprocess_input_data_format="nhwc"):
        if self.crop_method is CropMethod.CENTOR:
            return {
                'net_input_dims': self.net_input_dims,
                'resize_dims': self.resize_dims,
                'mean': self.mean,
                'std': self.std,
                'input_scale': self.input_scale,
                'raw_scale': self.raw_scale,
                'data_format': "n{}".format(self.data_format),
                'rgb_order': self.rgb_order,
                'crop_offset': self.get_center_crop_offset(),
                'pads': [0,0,0,0],
                'pad_const_val': 0,
                'crop_method': CropMethod.CENTOR.value,
                'preprocess_input_data_format': preprocess_input_data_format
            }
        elif self.crop_method is CropMethod.ASPECT_RATIO:
            return {
                'net_input_dims': self.net_input_dims,
                'resize_dims': self.resize_dims,
                'mean': self.mean,
                'std': self.std,
                'input_scale': self.input_scale,
                'raw_scale': self.raw_scale,
                'data_format': "n{}".format(self.data_format),
                'rgb_order': self.rgb_order,
                'crop_offset': [0,0,0,0],
                'pads': get_aspect_ratio_pads(self.net_input_dims[0], self.net_input_dims[1], input_h, input_w),
                'pad_const_val': 0,
                'crop_method': CropMethod.ASPECT_RATIO.value,
                'input_shape': [input_h, input_w],
                'preprocess_input_data_format': preprocess_input_data_format
            }
        else:
            raise RuntimeError(
                "Not Existed crop method {}".format(self.crop_method))

    def get_center_crop_offset(self):
        w, h = self.resize_dims
        cropx,cropy = self.net_input_dims
        startx = w//2-(cropx//2)
        starty = h//2-(cropy//2)
        return [0, 0, startx, starty]


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
            if self.crop_method is CropMethod.CENTOR:
                image = cv2.resize(image, (self.resize_dims[1], self.resize_dims[0])) # w,h

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

            if self.crop_method is CropMethod.ASPECT_RATIO:
                # get aspect ratio resize
                x = np.transpose(x, (1, 2, 0))
                x = get_aspect_ratio_img(
                    x, self.net_input_dims[0], self.net_input_dims[1], x.shape[0], x.shape[1], no_pad=self.only_aspect_ratio_img)
                x = np.transpose(x, (2, 0, 1))
            else:
                # Take center crop.
                x = center_crop(x, self.net_input_dims)

            # if source data order is different with handle order
            # swap source data order
            # only handle rgb to bgr , bgr to rgb
            if self.rgb_order != self.ori_channel_order:
                logger.debug("ori image channel order is {}, but we handle order is {}, swap it".format(self.ori_channel_order, self.rgb_order))
                x = x[[2,1,0], :, :]

            x = x.astype(np.float32)
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

        if self.astype == "float32":
            pass
        elif self.astype == "uint8":
            output = output.astype(np.uint8)
        else:
            raise RuntimeError("Not support {} type".format(self.astype))

        if output_npz:
            if self.batch > 1:
                output = np.repeat(output, self.batch, axis=0)
            # Must convert to npz file as input
            np.savez(output_npz, **{input_name if input_name else "input": output})


        return output
