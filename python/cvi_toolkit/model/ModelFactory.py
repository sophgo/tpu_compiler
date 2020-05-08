import pymlir
import abc
import numpy as np
from enum import Enum
from .caffe_model import CaffeModel
from .onnx_model import OnnxModel
from .mlir_model import MLIRModel
from .tensorflow_lite_model import TFLiteModel
from .tensorflow_model import TFModel


ModelType = [
    'caffe',
    'mlir',
    'onnx',
    'tflite',
    'tensorflow',
]


class ModelFactory(object):
    def __init__(self):
        self.model_type = None
        self.model = None

    def load_model(self, model_type, model_file=None, weight_file=None, mlirfile=None):
        if model_type not in ModelType:
            raise RuntimeError("Model Type {} not support. support {}".format(model_type, ModelType))
        else:
            self.model_type = model_type
            if model_type == 'caffe':
                self.model = CaffeModel()
                self.model.load_model(model_file, weight_file)
            elif model_type == 'onnx':
                self.model = OnnxModel()
                self.model.load_model(model_file)
            elif model_type == 'mlir':
                self.model = MLIRModel()
                self.model.load_model(mlirfile)
            elif model_type == 'tflite':
                self.model = TFLiteModel()
                self.model.load_model(model_file)
            elif model_type == 'tensorflow':
                self.model = TFModel()
                self.model.load_model(model_file)

    def inference(self, input):
       return self.model.inference(input)

    def get_all_tensor(self, input_data=None, npz_file=None):
        tensor_dict = self.model.get_all_tensor(input_data=input_data)
        if npz_file:
            np.savez(npz_file, **tensor_dict)
        return tensor_dict

    def get_op_info(self):
        return self.model.get_op_info()

    def get_input_shape(self):
        return self.model.get_input_shape()