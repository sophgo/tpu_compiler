import pymlir
import abc
from enum import Enum
from .caffe_model import CaffeModel
from .onnx_model import OnnxModel
from .mlir_model import MLIRModel


ModelType = [
    'caffe',
    'mlir',
    'onnx',
]


class CVI_Model(object):
    def __init__(self):
        self.model_type = None
        self.model = None

    def load_model(self, model_type, model_file, weight_file=None, mlirfile=None):
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
            else:
                self.model = MLIRModel()
                self.model.load_model(mlirfile)
    def inference(self, input):
       return self.model.inference(input)

    def get_all_tensor(self, input_data, npz_file):
        self.model.get_all_tensor(input_data, npz_file)



