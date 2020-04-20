
import numpy as np
import cv2
from enum import Enum


class InputType(Enum):
    FILE = 'FILE'
    NDARRAY = 'NDARRAY'

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
                     transpose="2,0,1",
                     rgb_order='bgr',
                     npy_input=None,
                     letter_box=False):
        print("preprocess :\n         \
            \tnet_input_dims: {}\n    \
            \tresize_dims   : {}\n    \
            \tmean          : {}\n    \
            \tmean_file     : {}\n    \
            \tstd           : {}\n    \
            \tinput_scale   : {}\n    \
            \traw_scale     : {}\n    \
            \ttranspose     : {}\n    \
            \trgb_order     : {}\n    \
            \tnpy_input     : {}\n    \
            \tletter_box    : {}\n    \
        ".format(net_input_dims, resize_dims, mean, \
                mean_file, std, input_scale, raw_scale, \
                transpose, rgb_order, npy_input, \
                letter_box
        ))
        self.npy_input = npy_input
        self.letter_box = letter_box
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
        self.input_scale = float(input_scale)

        if transpose != None:
            self.transpose = tuple([int(s)for s in transpose.split(",")])
        else :
            self.transpose = None
        self.rgb_order = rgb_order

    def run(self, input, output_npz=None, pfunc=None, input_name=None, input_type=InputType.FILE):
        if input_type == InputType.FILE:
            if self.npy_input != None :
                x = np.load(str(self.npy_input).rstrip())
                if output_npz:
                    np.savez(output_npz, **{input_name if input_name else "input": x})
                return x

            image = cv2.imread(str(input).rstrip())
            if image is None:
                print("not existed {}".format(str(input).rstrip()))
                return None

            image = image.astype(np.float32)
            
        elif input_type == InputType.NDARRAY:
            image = input

        # Do preprocess if with call back function
        if pfunc is not None:
            output = pfunc(image)
        else:
            if self.resize_dims != None:
                x = cv2.resize(image, (self.resize_dims[1], self.resize_dims[0])) # w,h
                if self.rgb_order == 'rgb' :
                    x = x[[2,1,0], :, :]

                # transpose
                if self.transpose != None :
                    x = np.transpose(x, self.transpose)
                x = x * self.raw_scale / 255.0
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

            else :
                if self.letter_box :
                    bgr_img = cv2.imread(str(input_file).rstrip())
                    yolo_w = self.net_input_dims[1]
                    yolo_h = self.net_input_dims[0]
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    rgb_img = rgb_img / 255.0

                    ih = rgb_img.shape[0]
                    iw = rgb_img.shape[1]

                    scale = min(float(yolo_w) / iw, float(yolo_h) / ih)
                    rescale_w = int(iw * scale)
                    rescale_h = int(ih * scale)

                    resized_img = cv2.resize(rgb_img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
                    new_image = np.full((yolo_h, yolo_w, 3), 0, dtype=np.float32)
                    paste_w = (yolo_w - rescale_w) // 2
                    paste_h = (yolo_h - rescale_h) // 2

                    new_image[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img
                    new_image = np.transpose(new_image, (2, 0, 1))      # row to col, (HWC -> CHW)
                    x = new_image

                else :
                    if self.rgb_order == 'rgb' :
                        image[:,:,0], image[:,:,2] = image[:,:,2], image[:,:,0]

                    if self.mean.size != 0:
                        image -= self.mean
                    if self.input_scale != 1.0:
                        image *= self.input_scale


                    if self.mean_file != None:
                        image -= self.mean_file

                    x = cv2.resize(image, (self.net_input_dims[1], self.net_input_dims[0]))

                    if self.transpose != None :
                        x = np.transpose(x, self.transpose)


            output = np.expand_dims(x, axis=0)
        if output_npz:
            # Must convert to npz file as input
            np.savez(output_npz, **{input_name if input_name else "input": output})

        return output
