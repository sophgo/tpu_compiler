import inspect
from enum import Enum

import numpy as np

import pybind

base = 'tpu.'
class TPU_OpType(Enum):
    Weight_file = 'tpu.wieght_file'
    Input  = 'tpu.input'
    Load_Weight = 'tpu.load_weight'

    BatchNorm = 'tpu.batch_norm'
    Conv2d = 'tpu.conv2d'
    Crop = 'tpu.crop'
    ElementMul = 'tpu.eltwise_mul'
    PoolAvg2D = 'pool_avg_2d'
    Reshape = 'tpu.reshape'
    Scale = 'tpu.scale'
    Sigmoid = 'tpu.sigmoid'
   

def checkKey(dict, key):
    if not dict.has_key(key):
        raise AttributeError("No {} attr, please check".format(key))

def checkType(obj, type):
    if not isinstance(obj, type):
        raise AttributeError('{} is not {}'.format(obj, type))

def OpFactory(op_name, inputs, outputs, **kargs):
    assert(isinstance(op_name, str))
    assert(isinstance(inputs, list) and isinstance(outputs, list))
    checkKey(kargs, 'name')

    


class PyImporter():

    def __init__(self, input_shape, output_shape):
        """
            input_shape: List, put module input shape. ex: [1, 3, 224, 224]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert(isinstance(input_shape, list))
        assert(isinstance(output_shape, list))

        self.module = pybind.MLIRModule()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.boolType = self.module.make_type("i1")
        self.i32Type = self.module.make_type("i32")
        self.f32Type = self.module.make_type("f32")
        self.NoneType = self.module.make_none_type()
        self.indexType = self.module.make_index_type()
        self.func_ctx =None
        self.declare_func()

    def __del__(self):
        print('Close mlir builder context')
        self.func_ctx.__exit__(None, None, None)

    def buildOp(self, op_type, inputOperands, outputOperands, **kargs):
        """
            op_type: String
            inputOpreands: List[pybind.op]
            outputOpreands: List[pybind.op]
            kargs: Dict
        """
        return pybind.op(op_type, inputOperands, outputOperands, **kargs)
    
    def add_none_op(self):
        return pybind.op("tpu.none", [], [self.NoneType])
        
    def add_conv_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            inputOperands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
            
        checkKey(kargs, 'dilation_h')
        checkKey(kargs, 'dilation_w')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'padding')
        checkKey(kargs, 'group')
        checkKey(kargs, 'is_dw')
        checkKey(kargs, 'with_bias')
        checkKey(kargs, 'do_relu')
       
        conv_name = self.module.stringAttr(op_name)
        conv_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'padding': self.module.stringAttr(kargs['padding']),
            'dilation_h': self.module.integerAttr(self.i32Type,  kargs['dilation_h']),
            'dilation_w': self.module.integerAttr(self.i32Type, kargs['dilation_w']),
            'group': self.module.integerAttr(self.i32Type, kargs['group']),
            'is_dw': self.module.boolAttr(kargs['is_dw']),
            'with_bias': self.module.boolAttr(kargs['with_bias']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
          }
      
        dict_attr = self.module.dictAttr(**conv_param)

        return self.buildOp(TPU_OpType.Conv2d.value, inputOperands, [
                     tensor_output_type], name=conv_name, param=dict_attr)
    
    def add_crop_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            args:
                crop_offset: List[int, int, int, int]
                crop_shape : List[int, int, int, int] 
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        
        checkKey(kargs, 'crop_offset')
        checkKey(kargs, 'crop_shape')
        crop_offset = kargs['crop_offset']
        crop_shape = kargs['crop_shape']
        checkType(crop_offset, list)
        checkType(crop_shape , list)
        crop_name = self.module.stringAttr(op_name)
        crop_offset_attr = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in crop_offset])
        crop_shape_attr = self.module.arrayAttr(
            [self.module.integerAttr(self.i32Type, x) for x in crop_shape])

        return self.buildOp(TPU_OpType.Crop.value, inputOperands, [
            tensor_output_type], name=crop_name, crop_offset=crop_offset_attr, crop_shape=crop_shape_attr)

    def print_module(self):
        print(self.module)

    def declare_func(self):
        tensor_input_type = self.module.make_ranked_tensor_type(
            self.f32Type, self.input_shape)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, self.output_shape)
        self.func_ctx = self.module.function_context("tpu_func", [tensor_input_type],
                                                     [tensor_output_type])
        print('Open mlir builder context')
        self.func_ctx.__enter__()

if __name__ == "__main__":
    importer = PyImporter([1, 3, 224, 224], [1, 1000])
    conv_param = {
        'stride_h':  1,
        'stride_w':  1,
        'padding': "SAME",
        'dilation_h': 1,
        'dilation_w': 1,
        'group': 1,
        'is_dw': True,
        'with_bias': True,
        'do_relu': False,
    }
    a = importer.add_none_op()
    
    b = importer.add_conv_op("conv_1", [a], [1, 1000], **conv_param)
    crop_param = {
        'crop_offset' : [0, 0, 1, 1],
        'crop_shape' : [1, 1, 112, 112]
    }
    c = importer.add_crop_op("crop_2", [b], [1, 3, 112, 112], **crop_param)
    importer.print_module()
  
