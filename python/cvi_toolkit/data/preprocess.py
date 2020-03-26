
import numpy as np
import cv2

def center_crop(img,crop_dim):
      print(img.shape)
  h,w,_ = img.shape
  cropy,cropx = crop_dim
  startx = w//2-(cropx//2)
  starty = h//2-(cropy//2)
  return img[starty:starty+cropy, startx:startx+cropx, :]


class preprocess(object):
    def __init__(self):
        pass

    def config(self, net_input_dims='224,224',
                     resize_dim="256,256",
                     mean=None,
                     mean_file=None,
                     input_scale=1.0,
                     raw_scale=255.0,
                     channel_swap='2,1,0',
                     letter_box=None):

        self.resize_dims = [int(s) for s in resize_dims.split(',')]
        self.net_input_dim = [int(s) for s in net_input_dims.split(',')]
        self.resize_dims = [ max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dim)]
        self.raw_scale = raw_scale
        if mean:
            self.mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
        else:
            if mean_file:
                self.mean = np.load(mean_file)
            else:
                self.mean = np.array([])

        self.input_scale = float(input_scale)
        self.channel_swap = tuple([float(s)for s in channel_swap.split(",")])

    def run(self, input_file):
        image = cv2.imread(str(input_file).rstrip())
        image = image.astype(np.float32)
        # resize
        x = cv2.resize(image, (self.resize_dims[1], self.resize_dims[0])) # w,h
        # Take center crop.
        x = center_crop(x, self.net_input_dims)
        # transpose
        x = np.transpose(x, self.channel_swap)
        # preprocess
        x = x * self.raw_scale /255.0
        if self.mean.size != 0:
            x -= mean
        if self.input_scale != 1.0:
            x *= input_scale
        x = np.expand_dims(x, axis=0)
        return x