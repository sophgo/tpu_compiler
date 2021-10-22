import os
import PIL
import numpy as np
import cv2
import argparse
import mlir
from enum import Enum
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root', log_level="INFO")

class YuvType(Enum):
  YUV420_PLANAR = 1
  YUV_NV12 = 2
  YUV_NV21 = 3

supported_pixel_formats = [
    'RGB_PLANAR',
    'RGB_PACKED',
    'BGR_PLANAR',
    'BGR_PACKED',
    'GRAYSCALE',
    'YUV420_PLANAR',
    'YUV_NV21',
    'YUV_NV12',
    'RGBA_PLANAR',
    None
]

pixel_format_attributes = {
    'RGB_PLANAR':    ('rgb', 'nchw'),
    'RGB_PACKED':    ('rgb', 'nhwc'),
    'BGR_PLANAR':    ('bgr', 'nchw'),
    'BGR_PACKED':    ('bgr', 'nhwc'),
    'GRAYSCALE':     ('bgr', 'nchw'),
    'YUV420_PLANAR': ('bgr', 'nchw'),
    'YUV_NV12':      ('bgr', 'nchw'),
    'YUV_NV21':      ('bgr', 'nchw'),
    'RGBA_PLANAR':   ('rgba', 'nchw')
}

# fix bool bug of argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "1")

class ImageResizeTool:
    @staticmethod
    def stretch_resize(image, h, w):
        return cv2.resize(image, (w, h)) # w,h

    @staticmethod
    def letterbox_resize(image, h, w):
        ih = image.shape[0]
        iw = image.shape[1]
        scale = min(float(w) / iw, float(h) / ih)
        rescale_w = int(iw * scale)
        rescale_h = int(ih * scale)
        resized_img = cv2.resize(image, (rescale_w, rescale_h))
        paste_w = (w - rescale_w) // 2
        paste_h = (h - rescale_h) // 2
        if image.ndim == 3 and image.shape[2] == 3:
            new_image = np.full((h, w, 3), 0, dtype=image.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w, :] = resized_img
            return new_image
        elif image.ndim == 2:
            new_image = np.full((h, w), 0, dtype=image.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w] = resized_img
            return new_image
        raise RuntimeError("invalid image shape:{}".format(image.shape))

def add_preprocess_parser(parser):
    parser.add_argument("--net_input_dims", type=str,
                         help="'h,w', model's input heigh/width dimension")
    parser.add_argument("--resize_dims", type=str,
                        help="Image was resize to fixed 'h,w', default is same as net_input_dims")
    parser.add_argument("--channel_num", type=int,  default=3,
                        help="channel number of inputed image")
    parser.add_argument("--keep_aspect_ratio", type=str2bool, default=False,
                        help="Resize image by keeping same ratio, any areas which" +
                             "are not taken are filled with 0")
    parser.add_argument("--crop_method", choices=['center', 'centor', 'right'], default='center')
    parser.add_argument("--raw_scale", type=float, default=255.0,
                        help="Multiply raw input image data by this scale.")
    parser.add_argument("--mean", default='0,0,0,0', help="Per Channel image mean values")
    parser.add_argument("--std", default='1,1,1,1', help="Per Channel image std values")
    parser.add_argument("--input_scale", type=float, default=1.0,
                        help="Multiply input features by this scale.")
    parser.add_argument("--channel_order", choices=['bgr', 'rgb', 'rgba'], default='bgr',
                        help="channel order of model inference used")
    parser.add_argument("--pixel_format", choices=supported_pixel_formats, default=None,
                        help='fixel format of output data that sent into model')
    parser.add_argument("--aligned", type=str2bool, default=False,
                        help='if the fixel format is aligned')
    parser.add_argument("--data_format", choices=['nchw', 'nhwc'], default='nchw',
                        help='data layout of output data, ' + \
                             'this value will be ignored if pixel_format is set')
    parser.add_argument("--gray", type=str2bool, default=False,
                        help='set pixel_format to GRAYSCALE')
    parser.add_argument('--model_channel_order', dest='channel_order',
                        help="alias to --channel_order, deprecated")
    parser.add_argument('--image_resize_dims', dest='resize_dims',
                         help="alias to --resize_dims, deprecated")
    return parser

def get_preprocess_parser(existed_parser=None):
    if existed_parser:
        if not isinstance(existed_parser, argparse.ArgumentParser):
            raise RuntimeError("parser is invaild")
        parser = existed_parser
    else:
        parser = argparse.ArgumentParser(description="Image Preprocess.")
    return add_preprocess_parser(parser)


class preprocess(object):
    def __init__(self):
        pass

    def config(self, net_input_dims=None,
               resize_dims=None, crop_method='center', keep_aspect_ratio=False,
               raw_scale=255.0, mean='0,0,0,0', std='1,1,1,1', input_scale=1.0,
               channel_order='bgr', pixel_format=None, data_format='nchw',
               aligned=False, gray=False, channel_num=3, chip="", **ignored):
        if not net_input_dims and not resize_dims:
            raise RuntimeError("net_input_dims or resize_dims should be set")

        if net_input_dims:
            self.net_input_dims = [int(s) for s in net_input_dims.split(',')]
            if not resize_dims:
                self.resize_dims = self.net_input_dims
        if resize_dims:
            self.resize_dims = [int(s) for s in resize_dims.split(',')]
            if not net_input_dims:
                self.net_input_dims = self.resize_dims
            self.resize_dims = [max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]

        self.crop_method = crop_method
        self.keep_aspect_ratio = keep_aspect_ratio

        self.channel_order = channel_order
        self.pixel_format = pixel_format
        self.aligned = aligned
        self.channel_num = channel_num

        if not self.pixel_format:
            if gray or self.channel_num == 1:
                self.pixel_format = 'GRAYSCALE'
            elif self.channel_num == 4:
                self.pixel_format = 'RGBA_PLANAR'
                assert(data_format == 'nchw')
            elif data_format == 'nchw':
                self.pixel_format = 'BGR_PLANAR' if self.channel_order == 'bgr' else \
                                    'RGB_PLANAR'
            else:
                self.pixel_format = 'BGR_PACKED' if self.channel_order == 'bgr' else \
                                    'RGB_PACKED'
        if self.pixel_format not in supported_pixel_formats:
            raise RuntimeError("{} unsupported pixel format".format(pixel_format))

        if self.pixel_format == "RGBA_PLANAR":
            self.channel_order = 'rgba'
            self.channel_num = 4
        elif self.pixel_format == "GRAYSCALE":
            self.channel_num = 1
        elif self.pixel_format == "YUV420_PLANAR" or self.pixel_format == "YUV_NV12" or self.pixel_format == "YUV_NV21":
            self.channel_num = 3
            self.aligned = True
        elif self.pixel_format.startswith('RGB'):
            self.channel_num = 3
            self.channel_order = 'rgb'
        elif self.pixel_format.startswith('BGR'):
            self.channel_num = 3
            self.channel_order = 'bgr'
        else:
            self.channel_num = 3

        if str(chip).lower().endswith('183x'):
            self.VPSS_W_ALIGN = 32
            self.VPSS_Y_ALIGN = 32
            self.VPSS_CHANNEL_ALIGN = 4096
            if self.pixel_format == "YUV420_PLANAR":
              self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2
        else:
            self.VPSS_W_ALIGN = 64
            self.VPSS_Y_ALIGN = 64
            self.VPSS_CHANNEL_ALIGN = 64
            if self.pixel_format == "YUV420_PLANAR":
              self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2

        self.data_format = 'nchw' if self.pixel_format.endswith('PLANAR') else 'nhwc'
        self.input_name = 'input'

        _raw_scale = raw_scale
        _mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
        assert(_mean.size >= self.channel_num)
        _mean = _mean[:self.channel_num]
        _mean = _mean[:, np.newaxis, np.newaxis]
        _std = np.array([float(s) for s in std.split(',')], dtype=np.float32)
        assert(_std.size >= self.channel_num)
        _std = _std[:self.channel_num]
        _std = _std[:, np.newaxis, np.newaxis]
        _input_scale = float(input_scale)

        # preprocess:
        #   (x * (raw_scale / 255) - mean) * input_scale / std
        # => x * raw_scale / 255 * input_scale / std - mean * input_scale / std
        # => x * perchannel_scale - perchannel_mean
        # so: perchannel_scale = raw_scale / 255 * input_scale / std
        #     perchannel_mean  = mean * input_scale / std
        sa = _raw_scale / 255
        sb = _input_scale / _std
        self.perchannel_scale = sa * sb
        self.perchannel_mean = _mean * sb

        info_str = \
            "\n\t _______________________________________________________________________ \n" + \
            "\t| preprocess:                                                           |\n" + \
            "\t|   (x * (raw_scale / 255) - mean) * input_scale / std                  |\n" + \
            "\t| => x * raw_scale / 255 * input_scale / std - mean * input_scale / std |\n" + \
            "\t| => x * perchannel_scale - perchannel_mean                             |\n" + \
            "\t| so: perchannel_scale = raw_scale / 255 * input_scale / std            |\n" + \
            "\t|     perchannel_mean  = mean * input_scale / std                       |\n" + \
            "\t'-----------------------------------------------------------------------'\n"

        format_str = "  Preprocess args : \n" + \
               "\tnet_input_dims        : {}\n" + \
               "\tresize_dims           : {}\n" + \
               "\tcrop_method           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\t--------------------------\n" + \
               "\tchannel_order         : {}\n" + \
               "\tchannel_num          : {}\n" + \
               "\tperchannel_scale      : {}\n" + \
               "\tperchannel_mean       : {}\n" + \
               "\t   raw_scale          : {}\n" + \
               "\t   mean               : {}\n" + \
               "\t   std                : {}\n" + \
               "\t   input_scale        : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n" + \
               "\taligned               : {}\n"
        info_str += format_str.format(
                self.net_input_dims, self.resize_dims, self.crop_method,
                self.keep_aspect_ratio, self.channel_order, self.channel_num,
                list(self.perchannel_scale.flatten()), list(self.perchannel_mean.flatten()),
                _raw_scale, list(_mean.flatten()), list(_std.flatten()), _input_scale,
                self.pixel_format, self.aligned)
        logger.info(info_str)

    def get_input_num(self, model_file):
        with open(model_file, 'r') as f:
            context = f.read()
        ctx = mlir.ir.Context()
        ctx.allow_unregistered_dialects = True
        m = mlir.ir.Module.parse(context, ctx)
        body = m.body.operations[0].regions[0].blocks[0]
        input_ops = []
        for op in body.operations:
            if op.operation.name == 'tpu.input':
                input_ops.append(op)
        return len(input_ops)

    def load_config(self, model_file, idx, chip=""):
        with open(model_file, 'r') as f:
            context = f.read()
        ctx = mlir.ir.Context()
        ctx.allow_unregistered_dialects = True
        m = mlir.ir.Module.parse(context, ctx)
        body = m.body.operations[0].regions[0].blocks[0]
        input_ops = []
        for op in body.operations:
            if op.operation.name == 'tpu.input':
                input_ops.append(op)
        assert(len(input_ops) >= idx + 1)
        target = input_ops[idx]
        shape_type = mlir.ir.ShapedType(target.operands[0].type)
        shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
        if len(shape) >= 3:
            self.net_input_dims = shape[-2:]
            self.channel_num = shape[-3]
        self.input_name = mlir.ir.StringAttr(target.attributes['name']).value
        if 'preprocess' not in target.attributes:
            return
        attrs = mlir.ir.DictAttr(target.attributes['preprocess'])
        self.pixel_format = mlir.ir.StringAttr(attrs['pixel_format']).value
        self.channel_order = mlir.ir.StringAttr(attrs['channel_order']).value
        self.keep_aspect_ratio = mlir.ir.BoolAttr(attrs['keep_aspect_ratio']).value
        self.resize_dims = [mlir.ir.IntegerAttr(x).value for x in mlir.ir.ArrayAttr(attrs['resize_dims'])]
        self.perchannel_mean = np.array([mlir.ir.FloatAttr(x).value for x \
                                         in mlir.ir.ArrayAttr(attrs['mean'])]).astype(np.float32)
        self.perchannel_mean = self.perchannel_mean[:,np.newaxis, np.newaxis]
        self.perchannel_scale = np.array([mlir.ir.FloatAttr(x).value for x \
                                          in mlir.ir.ArrayAttr(attrs['scale'])]).astype(np.float32)
        self.perchannel_scale = self.perchannel_scale[:,np.newaxis, np.newaxis]
        self.crop_method = 'center'
        self.aligned = False
        if self.pixel_format == "YUV420_PLANAR" or self.pixel_format == "YUV_NV12" or self.pixel_format == "YUV_NV21" :
            self.aligned = True
        if str(chip).lower().endswith('183x'):
            self.VPSS_W_ALIGN = 32
            self.VPSS_Y_ALIGN = 32
            self.VPSS_CHANNEL_ALIGN = 4096
            if self.pixel_format == "YUV420_PLANAR":
              self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2
        else:
            self.VPSS_W_ALIGN = 64
            self.VPSS_Y_ALIGN = 64
            self.VPSS_CHANNEL_ALIGN = 64
            if self.pixel_format == "YUV420_PLANAR":
              self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2
        self.data_format = pixel_format_attributes[self.pixel_format][1]

        format_str = "\n  Preprocess args : \n" + \
               "\tnet_input_dims        : {}\n" + \
               "\tresize_dims           : {}\n" + \
               "\tcrop_method           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\t--------------------------\n" + \
               "\tchannel_order         : {}\n" + \
               "\tchannel_num           : {}\n" + \
               "\tperchannel_scale      : {}\n" + \
               "\tperchannel_mean       : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n" + \
               "\taligned               : {}\n"
        logger.info(format_str.format(
                self.net_input_dims, self.resize_dims, self.crop_method,
                self.keep_aspect_ratio, self.channel_order, self.channel_num,
                list(self.perchannel_scale.flatten()),
                list(self.perchannel_mean.flatten()),
                self.pixel_format, self.aligned))

    def to_dict(self):
        return {
            'resize_dims': self.resize_dims,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'crop_offset': self.__get_center_crop_offset(),
            'channel_order': self.channel_order,
            'perchannel_mean': list(self.perchannel_mean.flatten()),
            'perchannel_scale': list(self.perchannel_scale.flatten()),
            'pixel_format': self.pixel_format,
            'aligned': self.aligned
        }

    def __get_center_crop_offset(self):
        h, w = self.resize_dims
        crop_h, crop_w = self.net_input_dims
        start_h = (h // 2) -(crop_h // 2)
        start_w = (w // 2) - (crop_w // 2)
        return [0, 0, start_h, start_w]

    def __right_crop(self, img, crop_dim):
        ih, iw = img.shape[1:]
        oh, ow = crop_dim
        img = img[:, ih-oh:, iw-ow:]
        return img

    def __center_crop(self, img, crop_dim):
        # Take center crop.
        _, h, w = img.shape
        crop_h, crop_w = crop_dim
        start_h = (h // 2) -(crop_h // 2)
        start_w = (w // 2) - (crop_w // 2)
        img = img[:, start_h : (start_h + crop_h),
                     start_w : (start_w + crop_w)]
        return img

    def align_up(self, x, n):
        return x if n == 0 else ((x + n - 1)// n) * n

    # Y = 0.2569 * R + 0.5044 * G + 0.0979 * B + 16
    # U = -0.1483 * R - 0.2911 * G + 0.4394 * B + 128
    # V = 0.4394 * R - 0.3679 * G - 0.0715 * B + 128
    def rgb2yuv420(self, input, pixel_type):
        # every 4 y has one u,v
        # vpss format, w align is 32, channel align is 4096
        h, w, c = input.shape
        y_w_aligned = self.align_up(w, self.VPSS_Y_ALIGN)
        y_offset = 0
        if pixel_type == YuvType.YUV420_PLANAR:
          uv_w_aligned = self.align_up(int(w/2), self.VPSS_W_ALIGN)
          u_offset = self.align_up(y_offset + h * y_w_aligned, self.VPSS_CHANNEL_ALIGN)
          v_offset = self.align_up(u_offset + int(h/2) * uv_w_aligned, self.VPSS_CHANNEL_ALIGN)
        else :
          uv_w_aligned = self.align_up(w, self.VPSS_W_ALIGN)
          u_offset = self.align_up(y_offset + h * y_w_aligned, self.VPSS_CHANNEL_ALIGN)
          v_offset = u_offset
          
        total_size = self.align_up(v_offset + int(h/2) * uv_w_aligned, self.VPSS_CHANNEL_ALIGN)
        yuv420 = np.zeros(int(total_size), np.uint8)
        for h_idx in range(h):
            for w_idx in range(w):
                r, g, b = input[h_idx][w_idx]
                y = int(0.2569 * r + 0.5044 * g + 0.0979 * b + 16)
                u = int(-0.1483 * r - 0.2911 * g + 0.4394 * b + 128)
                v = int(0.4394 * r - 0.3679 * g - 0.0715 * b + 128)
                y = max(min(y, 255), 0)
                u = max(min(u, 255), 0)
                v = max(min(v, 255), 0)
                yuv420[y_offset + h_idx * y_w_aligned + w_idx] = y
                if (h_idx % 2 == 0 and w_idx % 2 == 0):
                  if pixel_type == YuvType.YUV420_PLANAR:
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2))
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2))
                  elif pixel_type == YuvType.YUV_NV12:
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2)
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2 + 1)
                  else :
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2 + 1)
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2)
                    
                  yuv420[u_idx] = u
                  yuv420[v_idx] = v
        return yuv420.reshape(int(total_size), 1, 1)

    def __load_image_and_resize(self, input):
        image = None
        if type(input) == str:
            image_path = str(input).rstrip()
            if not os.path.exists(image_path):
                raise RuntimeError("{} doesn't existed !!!".format(image_path))

            if self.channel_num == 1:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            elif self.channel_num == 3:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            elif self.channel_num == 4:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image.shape[-1] != 4:
                    image = PIL.Image.open(image_path).convert('RGBA')
                    image = np.array(image)
        elif isinstance(input, np.ndarray):
            assert(input.shape[-1] == 3)
            if self.channel_num == 1:
                image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
            else:
                image = input
        else:
            raise RuntimeError("invalid input type:{}".format(type(input)))

        if self.keep_aspect_ratio:
            image = ImageResizeTool.letterbox_resize(
                image, self.resize_dims[0], self.resize_dims[1])
        else:
            image = ImageResizeTool.stretch_resize(
                image, self.resize_dims[0], self.resize_dims[1])

        if self.channel_num == 1:
            # if grapscale image,
            # expand dim to (1, h, w)
            image = np.expand_dims(image, axis=0)
        else:
            # opencv read image data format is hwc
            # tranpose to chw
            image = np.transpose(image, (2, 0, 1))
        return image

    def align_packed_frame(self, x, aligned):
        if not aligned:
            return x
        h, w, c = x.shape
        w = w * c
        x = np.reshape(x, (1, h, w))
        x_tmp = np.zeros((1, h, self.align_up(w, self.VPSS_W_ALIGN)), x.dtype)
        x_tmp[:, :, : w] = x
        return x_tmp

    def align_planar_frame(self, x, aligned):
        if not aligned:
            return x
        c, h, w = x.shape
        x_tmp = np.zeros((c, h, self.align_up(w, self.VPSS_W_ALIGN)), x.dtype)
        x_tmp[:, :, :w] = x
        return x_tmp

    def run(self, input, batch=1):
        # load and resize image, the output image is chw format.
        x = self.__load_image_and_resize(input)

        # take center crop if needed
        if self.resize_dims != self.net_input_dims:
            if self.crop_method == "right":
                x = self.__right_crop(x, self.net_input_dims)
            else:
                x = self.__center_crop(x, self.net_input_dims)

        x = x.astype(np.float32)

        if self.pixel_format == 'GRAYSCALE':
            self.perchannel_mean = self.perchannel_mean[:1]
            self.perchannel_scale = self.perchannel_scale[:1]
            x = x * self.perchannel_scale - self.perchannel_mean
            x = self.align_planar_frame(x, self.aligned)
            x = np.expand_dims(x, axis=0)
        elif self.pixel_format == 'YUV420_PLANAR' or self.pixel_format == 'YUV_NV12' or self.pixel_format == "YUV_NV21":
            # swap to 'rgb'
            pixel_type = YuvType.YUV420_PLANAR; 
            if self.pixel_format == 'YUV420_PLANAR':
              pixel_type = YuvType.YUV420_PLANAR
            elif self.pixel_format == 'YUV_NV12':
              pixel_type = YuvType.YUV_NV12
            else:
              pixel_type = YuvType.YUV_NV21
            x = x[[2, 1, 0], :, :]
            x = x * self.perchannel_scale - self.perchannel_mean
            x = np.transpose(x, (1, 2, 0))
            x = self.rgb2yuv420(x, pixel_type)
            x = x.astype(np.float32)
            assert(batch == 1)
        elif self.pixel_format == 'RGBA_PLANAR':
            x = x * self.perchannel_scale - self.perchannel_mean
            x = np.expand_dims(x, axis=0)
        else:  # RGB_PLANAR|PACKED or  BGR_PLANAR|PACKED
            if self.channel_order == "rgb":
                x = x[[2, 1, 0], :, :]
                x = x * self.perchannel_scale - self.perchannel_mean
            else:
                x = x * self.perchannel_scale - self.perchannel_mean
            if self.data_format == 'nhwc':
                x = np.transpose(x, (1, 2, 0))
                x = self.align_packed_frame(x, self.aligned)
            else:
                x = self.align_planar_frame(x, self.aligned)
            x = np.expand_dims(x, axis=0)

        if batch > 1:
            x = np.repeat(x, batch, axis=0)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    preprocesser = preprocess()

    preprocesser.config(net_input_dims='244,224', pixel_format='BGR_PLANAR')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (224, 244)) # w,h
    y=np.transpose(y, (2, 0, 1))
    if np.any(x != y):
        raise Exception("1. BGR PLANAR test failed")
    logger.info("1. BGR PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='RGB_PLANAR')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    y=np.transpose(y, (2, 0, 1))
    if np.any(x != y):
        raise Exception("2. RGB PLANAR test failed")
    logger.info("2. RGB PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='BGR_PACKED')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("3. BGR PACKED test failed")
    logger.info("3. BGR PACKED test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='RGB_PACKED')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("RGB PACKED test failed")
    logger.info("4. RGB PACKED test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='GRAYSCALE')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("5. GRAYSCALE test failed")
    logger.info("5. GRAYSCALE test passed!!")

    preprocesser.config(net_input_dims='244,224', resize_dims='443,424',
                        crop_method='center', pixel_format='BGR_PACKED')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (424, 443))
    h_offset = (443 - 244) // 2
    w_offset = (424 - 224) // 2
    y = y[h_offset:h_offset + 244, w_offset : w_offset + 224]
    if np.any(x != y):
        raise Exception("6. Center crop test failed")
    logger.info("6. Center Crop test passed!!")

    preprocesser.config(net_input_dims='244,224', keep_aspect_ratio=True,
                        pixel_format='BGR_PACKED')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image)
    ih, iw, _ = y.shape
    w, h = (224, 244)
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    y0 = cv2.resize(y, (nw,nh))
    y = np.full((h, w, 3), 0, dtype='uint8')
    y[(h - nh) // 2:(h - nh) // 2 + nh,
            (w - nw) // 2:(w - nw) // 2 + nw, :] = y0
    if np.any(x != y):
        raise Exception("6. keep ratio resize test failed")
    logger.info("6. keep ratio resize test passed!!")
