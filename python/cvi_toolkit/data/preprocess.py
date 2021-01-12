
import numpy as np
import cv2
import argparse

supported_pixel_formats = [
    'RGB_PLANAR', 'RGB_PACKAGE',
    'BGR_PLANAR', 'BGR_PACKAGE',
    'GRAYSCALE',
    'YUV420_PLANAR',
    None
]

pixel_format_attributes = {
    'RGB_PLANAR':    ('rgb', 'nchw'),
    'RGB_PACKAGE':   ('rgb', 'nhwc'),
    'BGR_PLANAR':    ('bgr', 'nchw'),
    'BGR_PACKAGE':   ('bgr', 'nhwc'),
    'GRAYSCALE':     ('bgr', 'nchw'),
    'YUV420_PLANAR': ('bgr', 'nchw')
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
        new_image = np.full((h, w, 3), 0, dtype=image.dtype)
        paste_w = (w - rescale_w) // 2
        paste_h = (h - rescale_h) // 2

        new_image[paste_h:paste_h + rescale_h,
                paste_w: paste_w + rescale_w, :] = resized_img
        return new_image

def add_preprocess_parser(parser):
    parser.add_argument("--net_input_dims", type=str, default='0,0',
                         help="'h,w', model's input heigh/width dimension")
    parser.add_argument("--resize_dims", type=str,
                        help="Image was resize to fixed 'h,w', default is same as net_input_dims")
    parser.add_argument("--keep_aspect_ratio", type=str2bool, default=False,
                        help="Resize image by keeping same ratio, any areas which" +
                             "are not taken are filled with 0")
    parser.add_argument("--crop_method", choices=['center', 'centor'], default='center')
    parser.add_argument("--raw_scale", type=float, default=255.0,
                        help="Multiply raw input image data by this scale.")
    parser.add_argument("--mean", default='0,0,0', help="Per Channel image mean values")
    parser.add_argument("--std", default='1,1,1', help="Per Channel image std values")
    parser.add_argument("--input_scale", type=float, default=1.0,
                        help="Multiply input features by this scale.")
    parser.add_argument("--channel_order", choices=['bgr', 'rgb'], default='bgr',
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

    def config(self, net_input_dims='0,0',
               resize_dims=None, crop_method='center', keep_aspect_ratio=False,
               raw_scale=255.0, mean='0,0,0', std='1,1,1', input_scale=1.0,
               channel_order='bgr', pixel_format=None, data_format='nchw',
               aligned=False, gray=False, **ignored):

        self.net_input_dims = [int(s) for s in net_input_dims.split(',')]
        if resize_dims:
            self.resize_dims = [int(s) for s in resize_dims.split(',')]
            self.resize_dims = [max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]
        else :
            self.resize_dims = self.net_input_dims

        self.crop_method = crop_method
        self.keep_aspect_ratio = keep_aspect_ratio

        self.raw_scale = raw_scale
        self.mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
        self.mean = self.mean[:, np.newaxis, np.newaxis]
        self.std = np.array([float(s) for s in std.split(',')], dtype=np.float32)
        self.std = self.std[:,np.newaxis, np.newaxis]
        self.input_scale = float(input_scale)

        # preprocess method:
        # (x * raw_scale / 255 - mean) * input_scale / std
        # => (x - mean * 255 / raw_scale) * (input_scale / std) * (raw_scale / 255)
        # => (x - perchannel_mean) * perchannel_scale
        # so: perchannel_mean = mean * 255 / raw_scale
        #     perchannel_scale = (input_scale / std) * (raw_scale / 255)
        sa = self.raw_scale / 255
        sb = self.input_scale / self.std
        self.perchannel_mean = self.mean / sa
        self.perchannel_scale = sb * sa

        self.channel_order = channel_order

        self.pixel_format = pixel_format
        self.aligned = aligned

        if not self.pixel_format:
            if gray:
                self.pixel_format = 'GRAYSCALE'
            elif data_format == 'nchw':
                self.pixel_format = 'BGR_PLANAR' if self.channel_order == 'bgr' else \
                                    'RGB_PLANAR'
            else:
                self.pixel_format = 'BGR_PACKAGE' if self.channel_order == 'bgr' else \
                                    'RGB_PACKAGE'
        if self.pixel_format not in supported_pixel_formats:
            raise RuntimeError("{} unsupported pixel format".format(pixel_format))

        self.data_format = 'nchw' if self.pixel_format.endswith('PLANAR') else 'nhwc'
        self.gray = True if self.pixel_format == 'GRAYSCALE' else False

        if self.data_format == "YUV420_PLANAR":
            self.aligned = True

        format_str = "  Preprocess args : \n" + \
               "\tnet_input_dims        : {}\n" + \
               "\tresize_dims           : {}\n" + \
               "\tcrop_method           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\t--------------------------\n" + \
               "\tchannel_order         : {}\n" + \
               "\tperchannel_mean       : {}\n" + \
               "\tperchannel_scale      : {}\n" + \
               "\t   raw_scale          : {}\n" + \
               "\t   mean               : {}\n" + \
               "\t   std                : {}\n" + \
               "\t   input_scale        : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n" + \
               "\taligned               : {}\n"
        print(format_str.format(
                self.net_input_dims, self.resize_dims, self.crop_method, self.keep_aspect_ratio, self.channel_order,
                list(self.perchannel_mean.flatten()), list(self.perchannel_scale.flatten()),
                self.raw_scale, list(self.mean.flatten()), list(self.std.flatten()), self.input_scale,
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
        w, h = self.resize_dims
        cropx,cropy = self.net_input_dims
        startx = w//2-(cropx//2)
        starty = h//2-(cropy//2)
        return [0, 0, startx, starty]

    def __center_crop(self, img, crop_dim):
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

    def rgb2yuv420(self, input):
        def align_up(x, n):
            return x if n == 0 else ((x + n - 1)// n) * n
        # every 4 y has one u,v
        # vpss format, w align is 32, channel align is 4096
        h, w, c = input.shape
        y_w_aligned = align_up(w, 32)
        uv_w_aligned = align_up(int(w/2), 32)
        y_offset = 0
        u_offset = align_up(y_offset + h * y_w_aligned, 4096)
        v_offset = align_up(u_offset + int(h/2) * uv_w_aligned, 4096)
        total_size = align_up(v_offset + int(h/2) * uv_w_aligned, 4096)
        yuv420 = np.zeros(int(total_size), np.uint8)
        for h_idx in range(h):
            for w_idx in range(w):
                r, g, b = input[h_idx][w_idx]
                y = int(0.299*r + 0.587*g + 0.114*b)
                u = int(-0.1687*r - 0.3313*g + 0.5*b + 128)
                v = int(0.5*r - 0.4187*g - 0.0813*b + 128)
                yuv420[y_offset + h_idx * y_w_aligned + w_idx] = y
                if (h_idx % 2 == 0 and w_idx % 2 == 0):
                    u_idx = int(u_offset + h_idx/2 * uv_w_aligned + w_idx / 2)
                    v_idx = int(v_offset + h_idx/2 * uv_w_aligned + w_idx / 2)
                    yuv420[u_idx] = u
                    yuv420[v_idx] = v
        return yuv420.reshape(int(total_size), 1, 1)

    def __load_image_and_resize(self, input):
        image = None
        if type(input) == str:
            image_path = str(input).rstrip()
            mode = cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR
            image = cv2.imread(image_path, mode)
            if image is None:
                raise RuntimeError("{} doesn't existed !!!".format(image_path))
        elif isinstance(input, np.ndarray):
            assert(input.shape[-1] == 3)
            if self.gray:
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

        if self.gray:
            # if grapscale image,
            # expand dim to (1, h, w)
            image = np.expand_dims(image, axis=0)
        else:
            # opencv read image data format is hwc
            # tranpose to chw
            image = np.transpose(image, (2, 0, 1))
        return image


    def run(self, input, batch=1):
        # load and resize image, the output image is chw format.
        x = self.__load_image_and_resize(input)

        # take center crop if needed
        if self.resize_dims != self.net_input_dims:
            x = self.__center_crop(x, self.net_input_dims)

        # if color order for preprocessing is not "bgr",
        # swap it to correct order
        if self.channel_order != "bgr":
            x = x[[2,1,0], :, :]

        # convert from uint8 to fp32
        x = x.astype(np.float32)
        x = x * (self.raw_scale / 255.0)
        # preprocess
        if self.gray:
            self.mean = self.mean[:1]
            self.std = self.std[:1]
        if self.mean.size != 0:
            x -= self.mean
        if self.input_scale != 1.0:
            x *= self.input_scale
        if self.std is not None:
            x /= self.std

        # if not 'bgr', swap back
        if self.channel_order != 'bgr':
            x = x[[2, 1, 0], :, :]

        if self.pixel_format == 'YUV420_PLANAR':
            # swap to 'rgb'
            x = x[[2,1,0], :, :]
            x = np.transpose(x, (1, 2, 0))
            x = self.rgb2yuv420(x)
            x = x.astype(np.float32)
            assert(batch == 1)
        elif self.pixel_format.startswith('RGB') or \
              self.pixel_format.startswith('BGR'):
            if self.pixel_format.startswith('RGB'):
                # swap to 'rgb'
                x = x[[2, 1, 0], :, :]
            if self.data_format == 'nhwc':
                x = np.transpose(x, (1, 2, 0))
            # expand to 4 dimensions
            x = np.expand_dims(x, axis=0)
        else: # GRAYSCALE
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
    print("1. BGR PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='RGB_PLANAR')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    y=np.transpose(y, (2, 0, 1))
    if np.any(x != y):
        raise Exception("2. RGB PLANAR test failed")
    print("2. RGB PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='BGR_PACKAGE')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("3. BGR PACKAGE test failed")
    print("3. BGR PACKAGE test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='RGB_PACKAGE')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("RGB PACKAGE test failed")
    print("4. RGB PACKAGE test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='GRAYSCALE')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("5. GRAYSCALE test failed")
    print("5. GRAYSCALE test passed!!")

    preprocesser.config(net_input_dims='244,224', resize_dims='443,424',
                        crop_method='center', pixel_format='BGR_PACKAGE')
    x=preprocesser.run(args.image)
    print("x", x, x.shape)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (424, 443))
    h_offset = (443 - 244) // 2
    w_offset = (424 - 224) // 2
    y = y[h_offset:h_offset + 244, w_offset : w_offset + 224]
    print("y", y, y.shape)
    if np.any(x != y):
        raise Exception("6. Center crop test failed")
    print("6. Center Crop test passed!!")

    preprocesser.config(net_input_dims='244,224', keep_aspect_ratio=True,
                        pixel_format='BGR_PACKAGE')
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
    print("6. keep ratio resize test passed!!")
