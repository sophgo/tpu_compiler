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
    Eltwise_Mul = 'tpu.eltwise_mul'
    PoolAvg2D = 'tpu.pool_avg_2d'
    Reshape = 'tpu.reshape'
    Scale = 'tpu.scale'
    Sigmoid = 'tpu.sigmoid'
   

def checkKey(dict, key):
    if not dict.has_key(key):
        raise AttributeError("No {} attr, please check".format(key))

def checkType(obj, type):
    if not isinstance(obj, type):
        raise AttributeError('{} is not {}'.format(obj, type))

    

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
    
    def add_weight_file_op(self, name):
        filename = self.module.stringAttr(name)
        self.buildOp(TPU_OpType.Weight_file.value, [], [], filename=filename)

    def add_load_file_op(self, name, output_tensor_shape):
        tensor_output_type = self.module.make_ranked_tensor_type(
             self.f32Type, output_tensor_shape)
        load_name = self.module.stringAttr(name)
        self.buildOp(TPU_OpType.Load_Weight.value, [], [tensor_output_type], name=load_name)

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

    def add_batchnorm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'variance_epsilon')

        variance_epsilon = kargs['variance_epsilon']
        checkType(variance_epsilon, float)

        batchnorm_name = self.module.stringAttr(op_name)
        variance_epsilon_attr = self.module.floatAttr(variance_epsilon)

        return self.buildOp(TPU_OpType.BatchNorm.value, inputOperands, [
            tensor_output_type], name=batchnorm_name, variance_epsilon=variance_epsilon_attr)
    
    def add_scale_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        scale_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Scale.value, inputOperands, [
            tensor_output_type], name=scale_name)
    
    def add_eltwise_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_mul = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Eltwise_Mul.value, inputOperands, [
            tensor_output_type], name=eltwise_mul)
    
    def add_pool_avg_2d_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'kernel_h')
        checkKey(kargs, 'kernel_w')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'do_relu')
        
        pool_avg_2d_name = self.module.stringAttr(op_name)
        pool_avg_2d_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'kernel_h': self.module.integerAttr(self.i32Type, kargs['kernel_h']),
            'kernel_w': self.module.integerAttr(self.i32Type, kargs['kernel_w']),
            'padding_b': self.module.integerAttr(self.i32Type, kargs['padding_b']),
            'padding_l': self.module.integerAttr(self.i32Type, kargs['padding_l']),
            'padding_r': self.module.integerAttr(self.i32Type, kargs['padding_r']),
            'padding_t': self.module.integerAttr(self.i32Type, kargs['padding_t']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
        }
        dict_attr = self.module.dictAttr(**pool_avg_2d_param)

        return self.buildOp(TPU_OpType.PoolAvg2D.value, inputOperands, [
            tensor_output_type], name=pool_avg_2d_name, param=dict_attr)

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

    batchnorm_parm = {
        'variance_epsilon': 0.001
    }
    d = importer.add_batchnorm_op(
        "batchnorm_3", [c], [1, 3, 112, 112], **batchnorm_parm)
    e = importer.add_scale_op("scale_4", [d], [1,3,112,112])
    f = importer.add_eltwise_mul_op("eltwise_mul_5", [e, d], [1, 3, 112, 112])
    pool_avg_2d_param = {
        'stride_h':  1,
        'stride_w':  1,
        'kernel_h':  1,
        'kernel_w':  1,
        'padding_b': 1,
        'padding_r': 1,
        'padding_t': 1,
        'padding_l': 1,
        'do_relu': False,
    }
    g = importer.add_pool_avg_2d_op(
        "pool_avg_2d_6", [e], [1, 3, 112, 112], **pool_avg_2d_param)
    importer.print_module()
  
