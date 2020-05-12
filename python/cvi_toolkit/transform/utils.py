from functools import partial, wraps
from math import floor, ceil
import numpy as np
import torch
import logging
from ..utils.log_setting import setup_logger


logger = setup_logger('root')
log_flag = logger.level <= logging.INFO

def calcConv2DSpatial(i, kernel, stride, padding, dilation):
    #[i + 2*p - k - (k-1)*(d-1)]/s + 1
    return int(floor(i + 2*padding - dilation * (kernel - 1) - 1)/stride + 1)

def calcPool2DFloor(i, kernel, stride, padding):
    return int(floor((i + 2 * padding - kernel) / stride) + 1)

def calcPool2DCeil(i, kernel, stride, padding):
    return int(ceil((i + 2 * padding - kernel) / stride) + 1)


def get_TF_SAME_Padding(input_spatial_shape, output_spatial_shape, kernel, stride):
    """
    If padding == "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    """
    return ((output_spatial_shape - 1) * stride + kernel - input_spatial_shape) / 2

def get_shape_size(shape):
    size = 1
    for i in shape:
        size*=i
    return size

def turn_shape_nhwc_to_nchw(shape):
    if not isinstance(shape, list):
        raise RuntimeError("Shape is wrong type with {}, it's must be list".format(type(shape)))
    if len(shape) != 4:
        raise RuntimeError("Shape length is {}, it's must be 4".format(len(shape)))
    return [shape[0], shape[3], shape[1], shape[2]]


def turn_data_hwio_to_oihw(data):
    if not isinstance(data, np.ndarray):
        raise RuntimeError(
            "Shape is wrong type with {}, it's must be np.ndarray".format(type(data)))
    if len(data.shape) != 4:
        logger.warning("Shape length is {}, change hwio to oihw dim must be 4".format(len(data.shape)))
        return data

    data = np.transpose(data, (3, 2, 0, 1))
    data = np.ascontiguousarray(data)
    return data

# generate onnx model from torch
def to_onnx(torch_model, input, model_path, inputs_list, outputs_list):
    torch.onnx.export(torch_model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = inputs_list,   # the model's input names
                  output_names = outputs_list, # the model's output names
                  )
class Color:
    # Foreground:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # End colored text
    END = '\033[0m'
    NC = '\x1b[0m'  # No Color



def docolor(color):
    def decorator(func):
        def wrap(*args, **kwargs):
            if color == 'blue':
                print(Color.OKBLUE)
            elif color == 'green':
                print(Color.OKGREEN)
            elif color == 'yellow':
                print(Color.WARNING)
            elif color == 'red':
                print(Color.FAIL)
            f = func(*args)
            print(Color.END)
            return f
        return wrap
    return decorator

