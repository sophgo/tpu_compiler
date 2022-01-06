import pymlir
import abc
import numpy as np
from enum import Enum
from .caffe_model import CaffeModel
from .onnx_model import OnnxModel
from .mlir_model import MLIRModel
#from .tensorflow_lite_model import TFLiteModel
#from .tensorflow_model import TFModel

class ModelFactory(object):
    def __init__(self):
        self.model_type = None
        self.model = None

    def load_model(self, model_type, model_file=None, weight_file=None, mlirfile=None):
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
        # elif model_type == 'tflite':
        #     self.model = TFLiteModel()
        #     self.model.load_model(model_file)
        # elif model_type == 'tensorflow':
        #     self.model = TFModel()
        #     self.model.load_model(model_file)
        else:
            raise RuntimeError("Model Type {} not support.".format(model_type))

    def inference(self, input):
        if self.model_type == 'mlir':
            outs = self.model.inference(input)
            return list(outs.values())[0]
        else:
            return self.model.inference(input)

    def get_all_tensor(self, input_data=None, npz_file=None):
        tensor_dict = self.model.get_all_tensor(input_data=input_data)
        if npz_file:
            np.savez(npz_file, **tensor_dict)
        return tensor_dict

    def get_op_info(self):
        return self.model.get_op_info()

    def get_input_name(self):
        input_names = [i for i in self.model.get_inputs()]
        if self.model_type == 'onnx':
            input_names = [i.name for i in input_names]
        return input_names

    def get_input_shape(self):
        return self.model.get_input_shape()

    def get_all_weights(self):
        return self.model.get_all_weight()