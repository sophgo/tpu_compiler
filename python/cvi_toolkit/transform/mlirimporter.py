from enum import Enum
import re
import pybind
import numpy as np
import sys

IS_PYTHON3 = sys.version_info > (3,)
from ..utils.log_setting import setup_logger

logger = setup_logger('root')

class TPU_OpType(Enum):
    Weight_file = 'tpu.weight_file'
    Input  = 'tpu.input'
    Load_Weight = 'tpu.load_weight'

    BatchNorm = 'tpu.batch_norm'
    BroadcastMul = 'tpu.broadcast_mul'
    BroadcastAdd = 'tpu.broadcast_add'
    Concat = 'tpu.concat'
    Conv2d = 'tpu.conv_2d'
    Crop = 'tpu.crop'
    Clip = 'tpu.clip'
    Eltwise_Add = 'tpu.eltwise_add'
    Eltwise_Mul = 'tpu.eltwise_mul'
    FullyConnected = 'tpu.fully_connected'
    Permute = 'tpu.permute'
    PixelShuffle = 'tpu.pixelshuffle'
    PoolAvg2D = 'tpu.pool_avg_2d'
    PoolMax2D  = 'tpu.pool_max_2d'
    Scale = 'tpu.scale'
    Sigmoid = 'tpu.sigmoid'
    Softmax = 'tpu.softmax'
    Reshape = 'tpu.reshape'
    Relu = 'tpu.relu'
    Upsample = 'tpu.upsample'

def checkKey(dict, key):
    if key not in dict:
        raise AttributeError("No {} attr, please check".format(key))

def checkType(obj, type):
    if not isinstance(obj, type):
        raise AttributeError('{} is not {}'.format(obj, type))


class MLIRImporter(object):
    def __init__(self, inputs_shape, outputs_shape):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert(isinstance(inputs_shape, list))
        assert(isinstance(outputs_shape, list))

        self.module = pybind.MLIRModule()
        self.input_shape_list = list()
        self.output_shape_list = list()
        for input in inputs_shape:
            assert(isinstance(input, list))
            self.input_shape_list.append(input)
        for output in outputs_shape:
            assert(isinstance(output, list))
            self.output_shape_list.append(output)

        self.boolType = self.module.make_type("i1")
        self.i32Type = self.module.make_type("i32")
        self.f32Type = self.module.make_type("f32")
        self.NoneType = self.module.make_none_type()
        self.indexType = self.module.make_index_type()
        self.func_ctx =None

        quant_param = {
            'is_asymmetric': self.module.boolAttr(False),
            'is_perchannel': self.module.boolAttr(False),
            'mode': self.module.stringAttr("NONE"),
            'param_type': self.module.stringAttr("NONE"),
            'threshold_max': self.module.floatAttr(0),
            'threshold_min': self.module.floatAttr(0)
        }
        self.quant_param = self.module.dictAttr(**quant_param)
        self.declare_func()

    def __del__(self):
        logger.debug('Close mlir builder context')
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

    def add_quant_reg(self, opreands):
        none = self.add_none_op()
        # We assigne 4 reg for quantization
        for i in range(4):
            opreands.append(none)
        return opreands

    def add_input_op(self, name, index):
        name = self.module.stringAttr(name)
        assert (index < len(self.func_args))
        return pybind.op(TPU_OpType.Input.value, [self.func_args[index]], [self.tensor_inputs_type[index]], name=name, quant=self.quant_param)

    def add_weight_file_op(self, name):
        filename = self.module.stringAttr(name)
        # TODO: our mlir not support mem type now
        mem_ref = self.module.make_memref_type(self.f32Type, [10])
        self.weightop = self.buildOp(TPU_OpType.Weight_file.value, [], [mem_ref], filename=filename)

    def add_load_file_op(self, name, output_tensor_shape):
        tensor_output_type = self.module.make_ranked_tensor_type(
             self.f32Type, output_tensor_shape)
        load_name = self.module.stringAttr(name)
        return self.buildOp(TPU_OpType.Load_Weight.value, [self.weightop], [tensor_output_type], name=load_name)

    def add_broadcast_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        broadcast_mul_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOpernads = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.BroadcastMul.value, inputOperands, [
            tensor_output_type], name=broadcast_mul_name, axis=axis_attr, quant=self.quant_param)

    def add_broadcast_add_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        broadcast_add_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOpernads = self.add_quant_reg(inputOperands)

        return self.buildOp(TPU_OpType.BroadcastAdd.value, inputOperands, [
            tensor_output_type], name=broadcast_add_name, axis=axis_attr, quant=self.quant_param)

    def add_clip_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) == 1)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        name = self.module.stringAttr(op_name)

        checkKey(kargs, 'min')
        checkKey(kargs, 'max')

        clip_min = kargs['min']
        clip_max = kargs['max']

        checkType(clip_min, float)
        checkType(clip_max, float)

        attr_dict = {
            'min': self.module.floatAttr(clip_min),
            'max': self.module.floatAttr(clip_max),
        }

        inputOpernads = self.add_quant_reg(inputOperands)

        return self.buildOp(TPU_OpType.Clip.value, inputOperands, [
            tensor_output_type], name=name, quant=self.quant_param, **attr_dict)


    def add_concat_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'axis')
        concat_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOpernads = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Concat.value, inputOperands, [
            tensor_output_type], name=concat_name, axis=axis_attr, quant=self.quant_param)

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
        inputOpernads = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Conv2d.value, inputOperands, [
                     tensor_output_type], name=conv_name, param=dict_attr, quant=self.quant_param)

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

    def add_eltwise_add_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        inputOpernads = self.add_quant_reg(inputOperands)

        eltwise_add = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Eltwise_Add.value, inputOperands, [
            tensor_output_type], name=eltwise_add, quant=self.quant_param)

    def add_eltwise_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        inputOpernads = self.add_quant_reg(inputOperands)

        eltwise_mul = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Eltwise_Mul.value, inputOperands, [
            tensor_output_type], name=eltwise_mul, quant=self.quant_param)

    def add_fully_connected_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        inputOpernads = self.add_quant_reg(inputOperands)

        fully_connected_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.FullyConnected.value, inputOperands, [
            tensor_output_type], name=fully_connected_name, quant=self.quant_param)

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
        inputOpernads = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.PoolAvg2D.value, inputOperands, [
            tensor_output_type], name=pool_avg_2d_name, param=dict_attr, quant=self.quant_param)

    def add_pool_max_2d_op(self, op_name, inputOperands, output_tensor_shape, **kargs):

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

        pool_max_2d_name = self.module.stringAttr(op_name)
        pool_max_2d_param = {
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
        dict_attr = self.module.dictAttr(**pool_max_2d_param)

        return self.buildOp(TPU_OpType.PoolMax2D.value, inputOperands, [
            tensor_output_type], name=pool_max_2d_name, param=dict_attr, quant=self.quant_param)

    def add_permute_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)

        permute_name = self.module.stringAttr(op_name)
        attr_dict = {
            'order0': self.module.integerAttr(self.i32Type, kargs['order0']),
            'order1': self.module.integerAttr(self.i32Type, kargs['order1']),
            'order2': self.module.integerAttr(self.i32Type, kargs['order2']),
            'order3': self.module.integerAttr(self.i32Type, kargs['order3']),
        }
        return self.buildOp(TPU_OpType.Permute.value, inputOperands, [
            tensor_output_type], name=permute_name, quant=self.quant_param, **attr_dict)

    def add_pixelshuffle_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
          self.f32Type, output_tensor_shape)

        pixelshuffle_name = self.module.stringAttr(op_name)
        attr_dict = {
            'upscale_factor': self.module.integerAttr(self.i32Type, kargs['upscale_factor'])
        }
        return self.buildOp(TPU_OpType.PixelShuffle.value, inputOperands, [
            tensor_output_type], name=pixelshuffle_name, quant=self.quant_param, **attr_dict)

    def add_relu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)

        relu_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Relu.value, inputOperands, [
            tensor_output_type], name=relu_name, quant=self.quant_param)

    def add_reshape_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        reshape_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Reshape.value, inputOperands, [
            tensor_output_type], name=reshape_name)

    def add_sigmoid_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)

        sigmoid_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for sigmoid quant table
        for i in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Sigmoid.value, inputOperands, [
            tensor_output_type], name=sigmoid_name, quant=self.quant_param)

    def add_softmax_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)

        softmax_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Softmax.value, inputOperands, [
            tensor_output_type], name=softmax_name)

    def add_upsample_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'scale')

        upsample_name = self.module.stringAttr(op_name)
        upsample_param = {
            'scale': self.module.integerAttr(self.i32Type, kargs['scale'])
        }
        return self.buildOp(TPU_OpType.Upsample.value, inputOperands, [
            tensor_output_type], name=upsample_name, quant=self.quant_param, **upsample_param)

    def add_return_op(self, Operands):
        return pybind.ret(Operands)

    def print_module(self):
        mlir_format = str(self.module)
        lines = mlir_format.splitlines()

        reg = '%[0-9]+'
        shape = ''
        new_strings = list()
        for i in lines:
            filter = "\W*(std\.return)\W*"
            ret = re.match(filter, i)
            if ret:
                reg_filter = "%[0-9]+"
                regs = re.findall(reg_filter, i)
                shape_filter = "\<[0-9A-Za-z]+\>"
                shapes = re.findall(shape_filter, i)
                if len(regs) != len(shapes): raise RuntimeError("{} is error format, regs v.s shapes number not match.".format(i))
                regstr = str()
                shapestr = str()
                for idx, (r, s) in enumerate(zip(regs, shapes)):
                    if idx != 0:
                        regstr += ", "
                        shapestr += ", "
                    regstr += r
                    shapestr += "tensor{}".format(s)

                new_line = "    return {} : {}".format(regstr, shapestr)
                new_strings.append(new_line)
            else:
                new_strings.append(i)
        ret = '\n'.join(new_strings)

        print(ret, flush=True)
        return ret

    def declare_func(self):
        self.tensor_inputs_type = list()
        for input_shape in self.input_shape_list:
            self.tensor_inputs_type.append(self.module.make_ranked_tensor_type(
                self.f32Type, input_shape))

        self.tensor_outputs_type = list()
        for output_shape in self.output_shape_list:
            self.tensor_outputs_type.append(self.module.make_ranked_tensor_type(
                self.f32Type, output_shape))

        self.func_ctx = self.module.function_context("tpu_func", self.tensor_inputs_type,
                                                     self.tensor_outputs_type)
        logger.debug('Open mlir builder context')
        fun = self.func_ctx.__enter__()
        self.func_args = list()
        for i in range(len(self.input_shape_list)):
            self.func_args.append(fun.arg(i))

