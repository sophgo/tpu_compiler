from .mlirimporter import MLIRImporter, checkKey, TPU_MODE, TPU_TensorType
from .BaseConverter import BaseConverter
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum
from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
    get_shape_size, get_TF_SAME_Padding
from ..quantization.QuantizeArithmetic import getFilterQscale, getRShiftAndMultiplierFromQScaleArray, getRShiftAndMultiplierFromQScale, \
    getMultiplierI8FromQScaleAndRShift, getRShiftForFilter, quantizeFilterRShift

from ..utils.log_setting import setup_logger

# tflite gen by flatbuffer
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.AddOptions import AddOptions
from tflite.BuiltinOperator import BuiltinOperator
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.LeakyReluOptions import LeakyReluOptions
from tflite.Model import Model
from tflite.Padding import Padding
from tflite.Pool2DOptions import Pool2DOptions
from tflite.PadOptions import PadOptions
from tflite.QuantizationParameters import QuantizationParameters
from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
from tflite.Tensor import Tensor as TFL_TENSOR
from tflite.TensorType import TensorType as TFL_TENSORTYPE

import logging
import numpy as np
import operator

logger = setup_logger('root')

log_flag = False

def make_sure_type(tensor_id, tensor_type):
    if tensor_id != tensor_type:
        raise RuntimeError(
            "{} tensor_type not match {}".format(tensor_id, tensor_type))


def np_uint8_to_fp32(uint8_arr):
    fp32_arr = np.frombuffer(uint8_arr.tobytes(), dtype=np.float32)
    return fp32_arr

def np_uint8_to_int32(uint8_arr):
    int32_arr = np.frombuffer(uint8_arr.tobytes(), dtype=np.int32)
    return int32_arr

def get_tensor_shape(tensor):
    if not isinstance(tensor, TFL_TENSOR): raise RuntimeError("Tensor is wrong type")

    x = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
    if tensor.ShapeLength() == 4:
        """
            In tflite define is NHWC
            return List of NCHW
        """
        return [x[0], x[3], x[1], x[2]]
    elif tensor.ShapeLength() == 2 or tensor.ShapeLength() == 1:
        return x
    else:
        raise ValueError("TODO this case shape len is {}".format(tensor.ShapeLength()))

def get_op_type(type_id):
    op_type_name = str()
    for attr in dir(BuiltinOperator()):
        if type_id == getattr(BuiltinOperator(), attr):
            op_type_name = str(attr)
            return op_type_name
    raise RuntimeError("tflite not support {} type id.".format(type_id))

class TFLiteNode():
    def __init__(self, name, type_id, inputs, outputs, proto):
        self.name = str(name)
        self.op_type = get_op_type(type_id)
        self.inputs = inputs
        self.outputs = outputs
        self.proto = proto

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')

class TFLiteTensor():
    def __init__(self, name, value, shape, op_type):
        if isinstance(name, int):
            self.name = str(name)
        else:
            self.name = name
        self.tensor_data = value
        self.shape = shape
        self.op_type = op_type

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')



class TFLiteConverter(BaseConverter):
    def __init__(self, model_name, tflite_model_file, mlir_file_path):
        super().__init__()
        self.model_name = model_name

        # read tflite model
        tf_buf = open(tflite_model_file, 'rb').read()
        self.tflite_model = Model.GetRootAsModel(tf_buf, 0)
        logger.debug("Tensorflow lite version {}".format(self.tflite_model.Version()))
        self.model_subgraph_len = self.tflite_model.SubgraphsLength()
        if self.model_subgraph_len != 1:
            raise RuntimeError("TODO")

        # TODO: we set subgraph length is 1,
        #       todo if mode subgraph
        self.tflite_graph = self.tflite_model.Subgraphs(0)

        self.input_node_number = self.tflite_graph.InputsLength()
        self.output_node_number = self.tflite_graph.OutputsLength()

        self.input_nodes = [self.tflite_graph.Inputs(i) for i in range(self.input_node_number)]
        self.output_nodes = [self.tflite_graph.Outputs(i) for i in range(self.output_node_number)]
        logger.info("Model:\ninput {}\noutput {}".format(self.input_nodes, self.output_nodes))

        self.operands_number = self.tflite_graph.OperatorsLength()
        logger.info("Operands number {}".format(self.operands_number))
        self.nodes = [self.tflite_graph.Operators(i) for i in range(self.operands_number)]


        self.mlir_file_path = mlir_file_path

        self.converted_nodes = list()
        self.converted_tensors = list()


        self.valueMap = dict() # {op_name: (mlir op, shape)}
        self.CVI = None # mlcvir pybind
        self.init_importer()
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.tfliteop_factory = {
            "ADD": lambda node: self.convert_add_op(node),
            "AVERAGE_POOL_2D": lambda node: self.convert_avg_pool_op(node),
            "CONV_2D": lambda node: self.convert_conv_op(node),
            "CONCATENATION": lambda node: self.convert_concat_op(node),
            "DEPTHWISE_CONV_2D": lambda node: self.convert_depthwise_conv_op(node),
            "DEQUANTIZE": lambda node: self.convert_skip_op(node),
            "FULLY_CONNECTED": lambda node: self.convert_fc_op(node),
            "LEAKY_RELU": lambda node: self.convert_leaky_relu_op(node),
            "MAX_POOL_2D": lambda node: self.convert_maxpool_op(node),
            "MEAN": lambda node: self.convert_mean_op(node),
            "PAD": lambda node: self.convert_pad_op(node),
            "QUANTIZE": lambda node: self.convert_quant_op(node),
            "RESIZE_NEAREST_NEIGHBOR": lambda node: self.convert_resize_op(node),
            "RESHAPE": lambda node: self.convert_reshape_op(node),
            "SOFTMAX": lambda node: self.convert_softmax_op(node),
        }

    def init_importer(self):
        # Make MLIR Function
        # get input shape
        inputs = list()
        for input_id in self.input_nodes:
            input_shape, _ = self.get_tensor_shape_and_data(input_id)
            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output_id in self.output_nodes:
            output_shape, _ = self.get_tensor_shape_and_data(output_id)
            outputs.append(output_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs)

    def get_tflite_tensor_shape(self, tensor):
        """
            Get TFLite Tensor Shape
            Input: tflite.Tensor.Tensor
            return:
                  List, (NHWC) if len is 4
        """
        return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    def get_tflite_tensor_data(self, tensor):
        """
            Get TFLite Tensor Data
            Input: tflite.Tensor.Tensor
            return:
                1d np.ndarray with each element is one byte, if no data, return 0
        """
        return self.tflite_model.Buffers(tensor.Buffer()).DataAsNumpy()

    def get_tensor_shape_and_data(self, tensor_index, data_type=np.float32):
        """
            Get TFLite Tensor Shape and Data
            Input: tensor index
            optional: data_type, default is np.float32
            return:
                List, shape
                ndarray, data, if no data, return None
        """
        tensor = self.tflite_graph.Tensors(tensor_index)
        shape = self.get_tflite_tensor_shape(tensor)
        data = self.get_tflite_tensor_data(tensor)

        HAS_DATA = not isinstance(data, int)
        if HAS_DATA:
            data = np.frombuffer(data.tobytes(), dtype=data_type)
            data = data.reshape(tuple(shape))

        if len(shape) == 4:
            """
                In tflite define is OHWI
                return List of OIHW
            """
            return [shape[0], shape[3], shape[1], shape[2]], np.ascontiguousarray(np.transpose(data, (0, 3, 1, 2))) if HAS_DATA else None
        elif len(shape) == 2 or len(shape) == 1:
            return shape, data if HAS_DATA else None
        else:
            raise ValueError("TODO this case shape len is {}".format(len(shape)))

    def addTensor(self, op_name, tensor_data, tensor_shape, op_type):
        self.converted_tensors.append(TFLiteTensor(op_name, tensor_data, tensor_shape, op_type))

    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise KeyError("No {} tensor in model".format(op_name))
        else:
            return find_tensor[0]

    def getTensorAttr(self, tensor):
        if not isinstance(tensor, TFL_TENSOR):
            raise RuntimeError("Tensor is wrong type")

        tensor_name = tensor.Name().decode('utf-8')
        tensor_type = tensor.Type()
        tensor_shpae = tensor.ShapeAsNumpy()
        quant_attr = tensor.Quantization()
        if quant_attr == None:
            quant_max = 0
            quant_min = 0
            scale = 0
            zero_point = 0
            quant_dim = 0
        else:
            quant_max = quant_attr.MaxAsNumpy()
            quant_min = quant_attr.MinAsNumpy()
            scale = quant_attr.ScaleAsNumpy()
            zero_point = quant_attr.ZeroPointAsNumpy()
            quant_dim = quant_attr.QuantizedDimension()
        if(not isinstance(quant_max, int) and not isinstance(quant_max, int)):
            threshold = max(abs(quant_max[0]), abs(quant_min[0]))
        else:
            threshold = 0

        return {
            "name": tensor_name,
            "quant_max": quant_max,
            "quant_min": quant_min,
            "scale": scale,
            "zero_point": zero_point,
            "quant_dim": quant_dim,
            "threshold": threshold,
            "type": tensor_type,
            "shape": tensor_shpae,
        }

    def createLoadWeightOp(self, tensor_idx, tensor_name, data_type=np.float32):
        shape, data = self.get_tensor_shape_and_data(tensor_idx, data_type)
        self.addTensor(tensor_name, data, shape, None)
        weight_op = self.CVI.add_load_file_op(tensor_name, shape)
        return weight_op, shape

    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.output_tensor_file, **tensor_npz)



    def convert_node(self):
        """convert tflite node to TFLiteNode"""
        for tflite_op in self.nodes:
            op_type = self.tflite_model.OperatorCodes(tflite_op.OpcodeIndex()).BuiltinCode()
            inputs = [tflite_op.Inputs(i) for i in range(tflite_op.InputsLength())]

            if tflite_op.OutputsLength() != 1:
                continue
            output = tflite_op.Outputs(0)
            name = str(output)
            node = TFLiteNode(name, op_type, inputs, output, tflite_op)
            if log_flag:
                node.print_info()
            self.converted_nodes.append(node)

    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        # add input op
        for idx, input in enumerate(self.input_nodes):
            input_tensor_idx = input
            input_shape, _ = self.get_tensor_shape_and_data(input_tensor_idx)
            input_op = self.CVI.add_input_op(str(input_tensor_idx), idx)
            self.addOperand(input_tensor_idx, input_op, input_shape, None)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        # add node op
        for n in self.converted_nodes:
            if log_flag:
                n.print_info()
            self.tfliteop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)

        # add return op
        return_op = list()
        # Set output
        for output in self.output_nodes:
            op, _, _ = self.getOperand(output)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)
        print("Save mlir file: {}".format(self.mlir_file_path))

    def add_activation_op(self, name, op, shape, activation, threshold=0):
        if activation == ActivationFunctionType.RELU6:
            relu_op = self.CVI.add_relu_op(
                "{}_relu".format(name), [op], shape)
            clip_param = {
                "min": 0.0,
                "max": 6.0,
            }
            clip_op = self.CVI.add_clip_op(
                "{}_clip".format(name), [relu_op], shape, **clip_param)
            return clip_op
        elif activation == ActivationFunctionType.NONE:
            return op
        elif activation == ActivationFunctionType.RELU:
            relu_op = self.CVI.add_relu_op(
                "{}_relu".format(name), [op], shape)
            return relu_op
        else:
            raise RuntimeError("Not support {} activation".format(activation))


    def convert_add_op(self, node):
        assert(node.op_type == "ADD")
        op1, input_shape1, threshold_x1 = self.getOperand(str(node.inputs[0]))
        op2, input_shape2, threshold_x2 = self.getOperand(str(node.inputs[1]))
        if input_shape1 != input_shape2:
                raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2))

        operands = list()
        operands.append(op1)
        operands.append(op2)
        output_shape = input_shape1

        add_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(add_tensor)
        threshold_y = tensor_attr['threshold']
        add_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0,3,1,2]] # nhwc -> nchw

        if log_flag:
            logger.info("{} {} threshold_x1: {}".format(
                node.inputs[0], input_shape1, threshold_x1))
            logger.info("{} {} threshold_x2: {}".format(
                node.inputs[1], input_shape2, threshold_x2))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, tensor_shape, threshold_y))

        op_build_info = node.proto.BuiltinOptions()
        # Parse the Table of options.
        add_table = AddOptions()
        add_table.Init(op_build_info.Bytes, op_build_info.Pos)

        assert(tensor_shape[1:] == output_shape[1:])
        add_param = {
            'do_relu': add_table.FusedActivationFunction() == ActivationFunctionType.RELU
        }
        CVIMlirQscale_x1 = threshold_x1/threshold_y
        CVIMlirQscale_x2 = threshold_x2/threshold_y
        MaxQscale = max(CVIMlirQscale_x1, CVIMlirQscale_x2)
        rshift, _ = getRShiftAndMultiplierFromQScale(MaxQscale, qdm=False)
        CVIMlirMultipiler_x1 = getMultiplierI8FromQScaleAndRShift(
            CVIMlirQscale_x1, rshift)
        CVIMlirMultipiler_x2 = getMultiplierI8FromQScaleAndRShift(
            CVIMlirQscale_x2, rshift)

        rshift_name = "{}_rshift".format(add_name)
        rshift_data = np.array([rshift])
        rshift_shape = list(rshift_data.shape)
        self.addTensor(rshift_name, rshift_data, rshift_data, None)
        rshift_op = self.CVI.add_load_file_op(
            rshift_name, rshift_shape, tensor_type=TPU_TensorType.FP32, storage="UINT8")
        operands.append(rshift_op)

        multipiler_name = "{}_multipiler".format(add_name)
        multipiler_data = np.array(
            [CVIMlirMultipiler_x1, CVIMlirMultipiler_x2])
        multipiler_shape = list(multipiler_data.shape)
        self.addTensor(multipiler_name, multipiler_data, multipiler_data, None)
        multipiler_op = self.CVI.add_load_file_op(
            multipiler_name, multipiler_shape, tensor_type=TPU_TensorType.FP32, storage="UINT8")
        operands.append(multipiler_op)

        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': False,
            'mode': TPU_MODE.INT8,
            'param_type': "RSHIFT_AND_M_I8",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }
        # add quant info to param
        add_param.update(int8_quant_info)

        add_op = self.CVI.add_eltwise_add_op(
            add_name, operands, output_shape, **add_param)
        self.addOperand(node.name, add_op, output_shape,
                threshold_y)

    def convert_avg_pool_op(self, node):
        assert(node.op_type == "AVERAGE_POOL_2D")
        op, input_shape, _ = self.getOperand(str(node.inputs[0]))

        avg_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(avg_tensor)
        threshold = tensor_attr['threshold']
        avg_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw


        on = input_shape[0]
        oc = input_shape[1]
        op_build_info = node.proto.BuiltinOptions()
        pool_table = Pool2DOptions()
        pool_table.Init(op_build_info.Bytes, op_build_info.Pos)
        pool_avg_2d_param = {
            'stride_h':  pool_table.StrideH(),
            'stride_w':  pool_table.StrideW(),
            'kernel_h':  pool_table.FilterWidth(),
            'kernel_w':  pool_table.FilterWidth(),
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op(avg_name, [op], output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, None)


    def convert_conv_op(self, node):
        assert(node.op_type == "CONV_2D")

        op, shape, threshold_x = self.getOperand(str(node.inputs[0]))
        operands = [op]

        conv_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(conv_tensor)
        threshold_y = tensor_attr['threshold']
        conv_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0,3,1,2]] # nhwc -> nchw
        tensor_type = tensor_attr['type']
        make_sure_type(tensor_type, TFL_TENSORTYPE.INT8)

        # filter
        filter_tensor_idx = node.inputs[1]
        filter_tensor = self.tflite_graph.Tensors(filter_tensor_idx)
        filter_attr = self.getTensorAttr(filter_tensor)
        filter_type = filter_attr['type']
        filter_name = filter_attr['name']
        filter_shape = filter_attr['shape']
        make_sure_type(filter_type, TFL_TENSORTYPE.INT8)
        # get quant info
        filter_scale = filter_attr['scale']

        if len(filter_scale) != filter_shape[0]: # perchannel
            raise RuntimeError(
                "{} filter_scale size is not match filter_shape channel ({})".format(len(filter_scale), filter_shape))

        # get filter data
        filter_data = self.get_tflite_tensor_data(
            filter_tensor)
        filter_data = np.frombuffer(filter_data.tobytes(), dtype=np.int8)
        filter_data = filter_data.reshape(tuple(filter_shape))

        # ohwi -> oihw
        filter_data = np.ascontiguousarray(
            np.transpose(filter_data, (0, 3, 1, 2)))
        filter_shape = [filter_shape[i] for i in [0, 3, 1, 2]]

        self.addTensor(filter_name, filter_data, filter_shape, None)
        filter_op = self.CVI.add_load_file_op(
            filter_name, filter_shape, tensor_type=TPU_TensorType.FP32, storage="INT8")
        operands.append(filter_op)

        Scale_x = threshold_x/127.0
        Scale_y = threshold_y/127.0
        # calculate rshift, multipiler
        CVIMlirQscale = getFilterQscale(
            filter_scale, Scale_x, Scale_y)  # [channel]
        rshift_data, mutlipiler_data = getRShiftAndMultiplierFromQScaleArray(
            CVIMlirQscale, qdm=True)

        # bias
        do_bias = len(node.inputs) == 3
        if do_bias:
            bias_tensor_idx = node.inputs[2]
            bias_tensor = self.tflite_graph.Tensors(bias_tensor_idx)
            bias_attr = self.getTensorAttr(bias_tensor)
            bias_type = bias_attr['type']
            bias_name = bias_attr['name']
            bias_shape = bias_attr['shape']

            make_sure_type(bias_type, TFL_TENSORTYPE.INT32)  # bias is int32
            bias_scale = bias_attr['scale']

            if len(bias_scale) != bias_shape[0]:
                raise RuntimeError(
                    "{} scale_scale size is not match bias_shape channel ({})".format(len(bias_scale), bias_shape))

            # get bias data
            bias_data = self.get_tflite_tensor_data(
                bias_tensor)
            bias_data = np.frombuffer(bias_data.tobytes(), dtype=np.int32)
            bias_data = bias_data.reshape(tuple(bias_shape))

            # dequant
            bias_data = bias_data.astype(np.float32) * bias_scale

            # B = Sw * Sx * Q(B)
            # Q(B) = B / Sx * Sw
            # Q(B) type is Int32
            bias_data = bias_data / (Scale_x * filter_scale)

            self.addTensor(bias_name, bias_data, bias_shape, None)
            bias_op = self.CVI.add_load_file_op(
                bias_name, bias_shape, tensor_type=TPU_TensorType.FP32, storage="INT32")
            operands.append(bias_op)

        if log_flag:
            logger.info("{} {} threshold_x: {}".format(
                node.inputs[0], shape, threshold_x))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, tensor_shape, threshold_y))
            for idx, (qscale, r, m, b) in enumerate(zip(CVIMlirQscale, rshift_data, mutlipiler_data, bias_scale)):
                logger.info("idx: {} qscale: {} rshift: {} mutlipiler:{}, b:{}".format(idx, qscale, r, m, b))

        # add rshift op
        rshift_name = "{}_rshift".format(conv_name)
        rshift_shape = list(rshift_data.shape)
        self.addTensor(rshift_name, rshift_data, rshift_shape, None)
        rshift_op = self.CVI.add_load_file_op(
            rshift_name, rshift_shape, tensor_type=TPU_TensorType.FP32, storage="UINT32")
        operands.append(rshift_op)

        # add multipiler op
        multipiler_name = "{}_multipiler".format(conv_name)
        multipiler_shape = list(mutlipiler_data.shape)
        self.addTensor(multipiler_name, mutlipiler_data,
                       multipiler_shape, None)
        mutlipiler_op = self.CVI.add_load_file_op(
            multipiler_name, multipiler_shape, tensor_type=TPU_TensorType.FP32, storage="UINT32")
        operands.append(mutlipiler_op)


        op_build_info = node.proto.BuiltinOptions()
        # Parse the Table of options.
        conv_table = Conv2DOptions()
        conv_table.Init(op_build_info.Bytes, op_build_info.Pos)

        stride_h = conv_table.StrideH()
        stride_w = conv_table.StrideW()
        dilation_h = conv_table.DilationHFactor()
        dilation_w = conv_table.DilationWFactor()
        padding = conv_table.Padding()

        if padding == Padding.SAME:
            padding_along_h = get_TF_SAME_Padding(shape[2], filter_shape[2], stride_h)
            padding_along_w = get_TF_SAME_Padding(shape[3], filter_shape[3], stride_w)
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        do_relu = conv_table.FusedActivationFunction() == ActivationFunctionType.RELU
        conv_param = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'padding': "SAME" if padding == Padding.SAME else "VALID",
            'dilation_h': dilation_h,
            'dilation_w': dilation_w,
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': 1,  # Don't have group option?
            'is_dw': False,
            'with_bias': len(node.inputs) > 2,
            'do_relu': do_relu,
            'ins': [],
        }
        on = shape[0]
        oc = filter_shape[0] # feature map size
        oh = calcConv2DSpatial(
            shape[2],
            filter_shape[2],
            conv_param['stride_h'],
            padding_t,
            padding_b,
            conv_param['dilation_h'],
        )
        ow = calcConv2DSpatial(
            shape[3],
            filter_shape[3],
            conv_param['stride_w'],
            padding_l,
            padding_r,
            conv_param['dilation_w'],
        )

        output_shape = [on, oc, oh, ow]
        assert(tensor_shape[1:] == output_shape[1:])

        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': True,
            'mode': TPU_MODE.INT8,
            'param_type': "RSHIFT_AND_M_I32",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }

        # add quant info to param
        conv_param.update(int8_quant_info)

        # store with int8 mode
        conv_op = self.CVI.add_conv_op(
            conv_name, operands, output_shape, **conv_param)

        self.addOperand(node.name, conv_op, output_shape,
                        threshold_y)

    def convert_concat_op(self, node):
        assert(node.op_type == "CONCATENATION")
        op1, input_shape1, threshold_x1 = self.getOperand(str(node.inputs[0]))
        op2, input_shape2, threshold_x2 = self.getOperand(str(node.inputs[1]))
        assert(len(input_shape1) == 4 and len(input_shape2) == 4)

        concat_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(concat_tensor)
        threshold_y = tensor_attr['threshold']
        concat_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw

        if log_flag:
            logger.info("{} {} threshold_x1: {}".format(
                node.input[0], input_shape1, threshold_x1))
            logger.info("{} {} threshold_x2: {}".format(
                node.input[1], input_shape2, threshold_x2))
            logger.info("{} {} threshold_y: {}".format(
                node.output, tensor_shape, threshold_y))

        op_build_info = node.proto.BuiltinOptions()
        concat_table = ConcatenationOptions()
        concat_table.Init(op_build_info.Bytes, op_build_info.Pos)

        axis = concat_table.Axis()
        if(axis == -1 or axis == 3):
            axis = 1 # in tflite is nhwc, but in mlir is nchw

        output_shape = list()

        for idx, op_shape in enumerate([input_shape1, input_shape2]):
            if idx == 0:
                # copy rather than referece
                output_shape = list(op_shape)
            else:
                for dim, value in enumerate(op_shape):
                    if dim == axis:
                        output_shape[dim] += value
                    else:
                        if output_shape[dim] != value:
                            raise ValueError("axis is {}, {} v.s {} shape can not be concat".format(
                                axis, output_shape, op_shape))

        assert(tensor_shape[1:] == output_shape[1:])
        concat_op = self.CVI.add_concat_op(
            concat_name, [op1, op2], output_shape, axis=axis)
        self.addOperand(node.name, concat_op, output_shape,
                        None)

    def convert_depthwise_conv_op(self, node):
        assert(node.op_type == "DEPTHWISE_CONV_2D")

        op, shape, _ = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)

        # filter
        filter_tensor_idx = node.inputs[1]
        filter_name = "{}_add_weight".format(filter_tensor_idx)

        filter_tensor = self.tflite_graph.Tensors(filter_tensor_idx)
        filter_shape = self.get_tflite_tensor_shape(filter_tensor)
        filter_data = self.get_tflite_tensor_data(filter_tensor)

        filter_data = np.frombuffer(filter_data.tobytes(), dtype=np.float32)
        filter_data = filter_data.reshape(tuple(filter_shape))

        # origin shape is (ic/g, kh, kw, g)
        g = filter_shape[3]
        kh = filter_shape[1]
        kw = filter_shape[2]
        ic = shape[1]
        on = shape[0]
        oc = ic
        # tranpose to (g, oc/g, ic/g, kh, kw)
        filter_data = np.transpose(filter_data, (3, 0, 1, 2))  # (g, ic/g, kh, kw)

        filter_data = np.ascontiguousarray(
            filter_data.flatten().reshape(g, int(ic/g), int(oc/g), kh, kw))
        filter_shape = [g, int(ic/g), int(oc/g), kh, kw]
        self.addTensor(filter_name, filter_data, filter_shape, None)
        filter_op = self.CVI.add_load_file_op(filter_name, filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = len(node.inputs) == 3
        if do_bias:
            bias_tensor_idx = node.inputs[2]
            bias_name = "{}_add_bias".format(bias_tensor_idx)
            bias_op, bias_shape = self.createLoadWeightOp(bias_tensor_idx, bias_name)
            operands.append(bias_op)

        op_build_info = node.proto.BuiltinOptions()
        # Parse the Table of options.
        depthwise_conv_table = DepthwiseConv2DOptions()
        depthwise_conv_table.Init(op_build_info.Bytes, op_build_info.Pos)
        # Check if input is padding op
        padding_data = None
        try:
            input_tensor = self.getTensor(str(node.inputs[0]))
            # Only padding case we handle
            assert(input_tensor.op_type == "PAD")
            padding_data = input_tensor.tensor_data
        except KeyError as k:
            # Not padding op
            print(k)
            pass
        stride_h = depthwise_conv_table.StrideH()
        stride_w = depthwise_conv_table.StrideW()

        # padding data order is NHWC
        # if padding data is not np.ndarray (not from bottom layer)
        # and conv_table.Padding() is SAME, we need to calculate it.
        if depthwise_conv_table.Padding() == Padding.SAME:
            padding_along_h = get_TF_SAME_Padding(shape[2], filter_shape[3], stride_h)
            padding_along_w = get_TF_SAME_Padding(shape[3], filter_shape[4], stride_w)
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        depthwise_conv_param = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'padding': "SAME" if depthwise_conv_table.Padding() == Padding.SAME or isinstance(padding_data, np.ndarray) else "VALID",
            'dilation_h': depthwise_conv_table.DilationHFactor(),
            'dilation_w': depthwise_conv_table.DilationWFactor(),
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': filter_shape[0],
            'is_dw': True,
            'with_bias': len(node.inputs) > 2,
            'do_relu': False,
            'ins': [],
        }
        oh = calcConv2DSpatial(
            shape[2],
            filter_shape[3],
            depthwise_conv_param['stride_h'],
            padding_t,
            padding_b,
            depthwise_conv_param['dilation_h'],
        )
        ow = calcConv2DSpatial(
            shape[3],
            filter_shape[4],
            depthwise_conv_param['stride_w'],
            padding_l,
            padding_r,
            depthwise_conv_param['dilation_w'],
        )
        output_shape = [on, oc, oh, ow]
        depthwise_conv_op = self.CVI.add_conv_op("{}".format(
            node.name), operands, output_shape, **depthwise_conv_param)
        depthwise_conv_op = self.add_activation_op("{}".format(
            node.name), depthwise_conv_op, output_shape, depthwise_conv_table.FusedActivationFunction())
        self.addOperand(node.name, depthwise_conv_op, output_shape,
                        None)

    def convert_fc_op(self, node):
        assert(node.op_type == "FULLY_CONNECTED")
        op, shape, threshold_x = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)

        fc_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(fc_tensor)
        threshold_y = tensor_attr['threshold']
        fc_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        fc_scale = tensor_attr['scale']


        # filter
        filter_tensor_idx = node.inputs[1]
        filter_tensor = self.tflite_graph.Tensors(filter_tensor_idx)
        filter_attr = self.getTensorAttr(filter_tensor)
        filter_type = filter_attr['type']
        filter_name = filter_attr['name']
        filter_shape = filter_attr['shape']
        make_sure_type(filter_type, TFL_TENSORTYPE.INT8)
        # get quant info
        filter_scale = filter_attr['scale']
        filter_max = filter_attr['quant_max']

        max_filter_value = max(filter_max)
        # calculate rshift, fully connected use rshift only
        rshift = getRShiftForFilter(
            max_filter_value, threshold_x, threshold_y)

        if log_flag:
            logger.info("{} {} threshold_x: {}".format(
                node.inputs[0], shape, threshold_x))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, tensor_shape, threshold_y))
            logger.info("rshift {} ".format(rshift))


        # get filter data
        filter_data = self.get_tflite_tensor_data(
            filter_tensor)
        filter_data = np.frombuffer(filter_data.tobytes(), dtype=np.int8)
        filter_data = filter_data.reshape(tuple(filter_shape))

        filter_data = filter_data.astype(np.float32) * filter_scale[0]

        filter_data = quantizeFilterRShift(
            filter_data, threshold_y, threshold_x, rshift)

        self.addTensor(filter_name, filter_data, filter_shape, None)
        filter_op = self.CVI.add_load_file_op(filter_name, filter_shape, storage="INT8")
        operands.append(filter_op)
        # bias
        do_bias = len(node.inputs) == 3
        if do_bias:
            bias_tensor_idx = node.inputs[2]
            bias_tensor = self.tflite_graph.Tensors(bias_tensor_idx)
            bias_attr = self.getTensorAttr(bias_tensor)
            bias_type = bias_attr['type']
            bias_name = bias_attr['name']
            bias_shape = bias_attr['shape']
            bias_scale = bias_attr['scale']

            make_sure_type(bias_type, TFL_TENSORTYPE.INT32)  # bias is int32
            # get bias data
            bias_data = self.get_tflite_tensor_data(
                bias_tensor)
            bias_data = np.frombuffer(bias_data.tobytes(), dtype=np.int32)
            bias_data = bias_data.reshape(tuple(bias_shape))

            # dequant
            bias_data = bias_data.astype(np.float32) * bias_scale[0]

            # quant to int16
            bias_data = bias_data * (127 / threshold_y * (1 << rshift))
            bias_data = np.clip(np.floor(bias_data+0.5),
                                np.iinfo(np.int16).min, np.iinfo(np.int16).max)

            self.addTensor(bias_name, bias_data, bias_shape, None)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_shape,  storage="INT16")
            operands.append(bias_op)


        rshift_data = np.array([rshift])
        rshift_name = "{}_rshift".format(fc_name)
        rshift_shape = list(rshift_data.shape)
        self.addTensor(rshift_name, rshift_data, rshift_shape, None)
        rshift_op = self.CVI.add_load_file_op(
            rshift_name, rshift_shape, tensor_type=TPU_TensorType.FP32, storage="UINT8")
        operands.append(rshift_op)

        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': False,
            'mode': TPU_MODE.INT8,
            'param_type': "RSHIFT_ONLY",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }
        # add quant info to param

        M = shape[0]
        N = bias_shape[0]
        output_shape = [M, N]
        assert(output_shape, tensor_shape)
        fc_op = self.CVI.add_fully_connected_op(
            fc_name, operands, output_shape, **int8_quant_info)
        self.addOperand(node.name, fc_op, output_shape, threshold_y)

    def convert_maxpool_op(self, node):
        assert(node.op_type == "MAX_POOL_2D")

        op, shape, threshold_x = self.getOperand(str(node.inputs[0]))


        max_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(max_tensor)
        threshold = tensor_attr['threshold']
        max_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw

        threshold_y = threshold


        op_build_info = node.proto.BuiltinOptions()
        pool_table = Pool2DOptions()
        pool_table.Init(op_build_info.Bytes, op_build_info.Pos)

        pool_max_2d_param = {
            'stride_h': pool_table.StrideH(),
            'stride_w': pool_table.StrideW(),
            'kernel_h': pool_table.FilterHeight(),
            'kernel_w': pool_table.FilterWidth(),
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
        }
        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': False,
            'mode': TPU_MODE.INT8,
            'param_type': "NONE",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }
        # add quant info to param
        pool_max_2d_param.update(int8_quant_info)

        on = shape[0]
        oc = shape[1]
        oh = calcPool2DFloor(shape[2], pool_max_2d_param['kernel_h'], pool_max_2d_param['stride_h'],
                             pool_max_2d_param['padding_b'], pool_max_2d_param['padding_t'])
        ow = calcPool2DFloor(shape[3], pool_max_2d_param['kernel_w'], pool_max_2d_param['stride_w'],
                             pool_max_2d_param['padding_r'], pool_max_2d_param['padding_l'])
        output_shape = [int(on), int(oc), int(oh), int(ow)]

        assert(tensor_shape[1:] == output_shape[1:])
        pool_max_op = self.CVI.add_pool_max_2d_op(max_name, [op], output_shape, **pool_max_2d_param)
        self.addOperand(node.name, pool_max_op, output_shape, threshold_y)

    def convert_mean_op(self, node):
        assert(node.op_type == "MEAN")
        """
            use avg_pool
        """
        # first input is activate, second is tensor of axis
        assert(len(node.inputs) == 2)
        op, input_shape, threshold_x = self.getOperand(str(node.inputs[0]))
        operands = [op]
        mean_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(mean_tensor)
        mean_shape = tensor_attr['shape']
        threshold_y = tensor_attr['threshold']
        mean_name = tensor_attr['name']

        if log_flag:
            logger.info("{} {} threshold_x: {}".format(
                node.inputs[0], input_shape, threshold_x))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, mean_shape, threshold_y))
        on = input_shape[0]
        oc = input_shape[1]
        kernel_h, kernel_w = input_shape[2:]
        pool_avg_2d_param = {
            'stride_h':  1,
            'stride_w':  1,
            'kernel_h':  input_shape[2],
            'kernel_w':  input_shape[3],
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
            'count_include_pad': False,
        }
        CVIMlirQscale = (threshold_x / threshold_y) / ( kernel_h * kernel_w)
        rshift_i8, _ = getRShiftAndMultiplierFromQScale(CVIMlirQscale)
        multipiler = getMultiplierI8FromQScaleAndRShift(CVIMlirQscale, rshift_i8)

        rshift_name = "{}_rshift".format(mean_name)
        rshift_data = np.array([rshift_i8])
        rshift_shape = list(rshift_data.shape)
        self.addTensor(rshift_name, rshift_data, rshift_data, None)
        rshift_op = self.CVI.add_load_file_op(
            rshift_name, rshift_shape, tensor_type=TPU_TensorType.FP32, storage="UINT8")
        operands.append(rshift_op)

        multipiler_name = "{}_multipiler".format(mean_name)
        multipiler_data = np.array([multipiler])
        multipiler_shape = list(multipiler_data.shape)
        self.addTensor(multipiler_name, multipiler_data, multipiler_data, None)
        multipiler_op = self.CVI.add_load_file_op(
            multipiler_name, multipiler_shape, tensor_type=TPU_TensorType.FP32, storage="UINT32")
        operands.append(multipiler_op)

        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': False,
            'mode': TPU_MODE.INT8,
            'param_type': "RSHIFT_AND_M_I8",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }
        # add quant info to param
        pool_avg_2d_param.update(int8_quant_info)

        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op(
            mean_name, operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, threshold_y)

    def convert_leaky_relu_op(self, node):
        assert(node.op_type == "LEAKY_RELU")
        # first input is activate, second is tensor
        assert(len(node.inputs) == 1)
        op, input_shape, _ = self.getOperand(str(node.inputs[0]))

        leaky_relu_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(leaky_relu_tensor)
        threshold = tensor_attr['threshold']
        leaky_relu_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw

        # Parse the Table of options.
        op_build_info = node.proto.BuiltinOptions()
        leaky_relu_table = LeakyReluOptions()
        leaky_relu_table.Init(op_build_info.Bytes, op_build_info.Pos)
        negative_slope = leaky_relu_table.Alpha()
        param = {
            'negative_slope': negative_slope
        }
        output_shape = input_shape
        assert(tensor_shape[1:] == output_shape[1:])
        l_relus_op = self.CVI.add_leaky_relu_op(
            leaky_relu_name, [op], output_shape, **param)
        self.addOperand(node.name, l_relus_op, output_shape,
                        None)

    def convert_pad_op(self, node):
        assert(node.op_type == "PAD")
        # first input is activate, second is tensor
        assert(len(node.inputs) == 2)
        op, input_shape, threshold_x  = self.getOperand(str(node.inputs[0]))
        pad_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(pad_tensor)
        threshold_y = tensor_attr['threshold']
        pad_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw

        if log_flag:
            logger.info("{} {} threshold_x: {}".format(
                node.inputs[0], input_shape, threshold_x))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, tensor_shape, threshold_y))

        # Parse the Table of options.
        op_build_info = node.proto.BuiltinOptions()
        pad_table = PadOptions()
        pad_table.Init(op_build_info.Bytes, op_build_info.Pos)
        padding_attr_tensor_idx = node.inputs[1]
        _, padding_data = self.get_tensor_shape_and_data(
            padding_attr_tensor_idx, data_type=np.int32)


        padding_data = padding_data[[0, 3, 1, 2], :]  # ohwc -> ochw
        padding_data = padding_data.flatten('F')
        dims = len(input_shape)
        pads_param = {
            "pads": padding_data.tolist(),
            "const_val": 0,
        }

        output_shape = np.sum(
            [input_shape, padding_data[:dims], padding_data[dims:]], axis=0)
        output_shape = [int(i) for i in output_shape]
        int8_quant_info = {
            'is_asymmetric': False,
            'is_perchannel': False,
            'mode': TPU_MODE.INT8,
            'param_type': "NONE",
            'threshold_max': threshold_y,
            'threshold_min': 0
        }

        # add quant info to param
        pads_param.update(int8_quant_info)

        assert(tensor_shape[1:] == output_shape[1:])
        pads_op = self.CVI.add_pad_op(
            pad_name, [op], output_shape, **pads_param)
        self.addOperand(node.name, pads_op, output_shape,
                        threshold_y)


    def convert_quant_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])

        input_tensor = self.tflite_graph.Tensors(node.inputs[0])
        input_attr = self.getTensorAttr(input_tensor)
        output_tensor = self.tflite_graph.Tensors(node.outputs)
        output_attr = self.getTensorAttr(output_tensor)
        output_tensor_name = output_attr['name']

        input_type = input_attr['type']
        output_type = output_attr['type']
        output_shape = input_shape
        if input_type == TFL_TENSORTYPE.FLOAT32 and output_type == TFL_TENSORTYPE.INT8:
            # get quantization attr
            threshold = output_attr['threshold']
            quant_op = self.CVI.add_quant_op(output_tensor_name, [
                                 op], output_shape, "NONE", "INT8", threshold=threshold)
            self.addOperand(node.name, quant_op, output_shape, threshold)
        else:
            # skip this op
            raise RuntimeError("skip")
            self.addOperand(node.name, op, input_shape, None)

    def convert_reshape_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

        reshape_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(reshape_tensor)
        threshold = tensor_attr['threshold']
        reshape_name = tensor_attr['name']
        tensor_shape = tensor_attr['shape'].tolist()
        tensor_shape = [tensor_shape[i] for i in [0, 3, 1, 2]]  # nhwc -> nchw

        output_shape_idx = node.inputs[1]
        _, output_shape = self.get_tensor_shape_and_data(
            output_shape_idx, data_type=np.int32)

        if len(output_shape) == 3:
            # hwc -> chw
            output_shape = [output_shape[2], output_shape[0], output_shape[1]]
            output_shape.insert(0, input_shape[0])  # add batch size

        elif len(output_shape) == 4:
            # nhwc -> nchw
            output_shape = [output_shape[0], output_shape[2],
                            output_shape[3], output_shape[1]]

        if -1 in output_shape:
            total_tensor_size = get_shape_size(input_shape)
            remain_dim = output_shape.index(-1)
            tmp_size = 1
            for i in range(len(output_shape)):
                if output_shape[i] == 0:
                    output_shape[i] = self.batch_size
                if i != remain_dim:
                    tmp_size *= output_shape[i]
                remain_size = total_tensor_size / tmp_size
                if not remain_size.is_integer():
                    raise RuntimeError("{} not divide exactly by {}".format(
                        total_tensor_size, tmp_size))
                output_shape[remain_dim] = int(remain_size)

        reshape_op = self.CVI.add_reshape_op(reshape_name, operands, output_shape)
        self.addOperand(node.name, reshape_op,
                        output_shape, None)

    def convert_resize_op(self, node):
        assert(node.op_type == "RESIZE_NEAREST_NEIGHBOR")
        # first input is activate, second is tensor
        assert(len(node.inputs) == 2)
        op, input_shape, _ = self.getOperand(str(node.inputs[0]))
        resize_tensor = self.tflite_graph.Tensors(node.outputs)

        # get quantization attr
        resize_quant_attr = resize_tensor.Quantization()
        resize_quant_max = resize_quant_attr.MaxAsNumpy()
        resize_quant_min = resize_quant_attr.MinAsNumpy()
        threshold = max(abs(resize_quant_max[0]), abs(resize_quant_min[0]))

        # Parse the Table of options.
        op_build_info = node.proto.BuiltinOptions()
        resize_table = ResizeNearestNeighborOptions()
        resize_table.Init(op_build_info.Bytes, op_build_info.Pos)
        resizeding_attr_tensor_idx = node.inputs[1]
        _, resizeding_data = self.get_tensor_shape_and_data(
            resizeding_attr_tensor_idx, data_type=np.int32)

        operands = list()
        operands.append(op)
        ic = input_shape[1]
        ih = input_shape[2]
        iw = input_shape[3]
        on = int(input_shape[0])
        oc = int(input_shape[1])
        oh = int(resizeding_data[0])
        ow = int(resizeding_data[1])
        group = ic
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        # use deconv(depthwise)
        deconv_param = {
            'stride_h':  int(oh / ih),
            'stride_w':  int(ow / iw),
            'padding': "VALID",
            'dilation_h': 1,
            'dilation_w': 1,
            'padding_t': 0,
            'padding_b': 0,
            'padding_l': 0,
            'padding_r': 0,
            'group': ic,
            'is_dw': False,
            'with_bias': False,
            'do_relu': False,
            'ins': [],
        }

        # deconv weight all one
        weight_shape = [group, int(
            oc/group), int(ic/group), int(oh / ih), int(ow / iw)]
        tensor_data = np.full(weight_shape, 1)
        weight_name = "{}_add_weight".format(node.name)
        self.addTensor(weight_name, tensor_data, tensor_data.shape, None)
        weight_op = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
        operands.append(weight_op)

        deconv_op = self.CVI.add_deconv_op(node.name, operands, output_shape, **deconv_param)
        self.addOperand(node.name, deconv_op, output_shape, None)

    def convert_skip_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        self.addOperand(node.name, op, input_shape, None)

    def convert_softmax_op(self, node):
        assert(node.op_type == "SOFTMAX")
        # first input is activate
        assert(len(node.inputs) == 1)
        op, shape, threshold_x = self.getOperand(str(node.inputs[0]))

        softmax_tensor = self.tflite_graph.Tensors(node.outputs)
        tensor_attr = self.getTensorAttr(softmax_tensor)
        threshold_y = tensor_attr['threshold']
        tensor_type = tensor_attr['type']
        tensor_shape = tensor_attr['shape']
        softmax_name = tensor_attr['name']
        make_sure_type(tensor_type, TFL_TENSORTYPE.INT8)

        # in mlir, softmax is use fp32
        if log_flag:
            logger.info("{} {} threshold_x: {}".format(
                node.inputs[0], shape, threshold_x))
            logger.info("{} {} threshold_y: {}".format(
                node.outputs, tensor_shape, threshold_y))

        quant_op = self.CVI.add_quant_op("{}_dequant".format(softmax_name), [
            op], tensor_shape, "INT8", "NONE", threshold=threshold_x)
        self.addOperand("{}_dequant".format(node.name),
                        quant_op, tensor_shape, threshold_y)
        softmax_param = {
            'axis': len(shape) - 1,
        }
        softmax_op = self.CVI.add_softmax_op(
            softmax_name, [quant_op], shape, cpu_mode=True, **softmax_param)
        self.addOperand(node.name, softmax_op, shape, threshold_y)

    def run(self):
        self.convert_node()
        self.convert_graph()
        self.TensortoNpz()
