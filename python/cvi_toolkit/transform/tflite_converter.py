from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum
from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
    get_shape_size, get_TF_SAME_Padding

from ..utils.log_setting import setup_logger

# tflite gen by flatbuffer

from tflite.BuiltinOperator import BuiltinOperator
from tflite.Conv2DOptions import Conv2DOptions
from tflite.Model import Model
from tflite.Padding import Padding
from tflite.Pool2DOptions import Pool2DOptions
from tflite.Tensor import Tensor as TFL_TENSOR
from tflite.ActivationFunctionType import ActivationFunctionType


import logging
import numpy as np
import operator

logger = setup_logger('root')

log_flag = logger.level <= logging.INFO

op_type_id = {
    BuiltinOperator.ADD: "ADD",
    BuiltinOperator.CONV_2D: "CONV_2D",
    BuiltinOperator.FULLY_CONNECTED: "FULLY_CONNECTED",
    BuiltinOperator.MAX_POOL_2D: "MAX_POOL_2D",
    BuiltinOperator.MEAN: "MEAN",
    BuiltinOperator.PAD: "PAD",
    BuiltinOperator.SOFTMAX: "SOFTMAX",
}

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




class TFLiteNode():
    def __init__(self, name, type_id, inputs, outputs, proto):
        self.name = str(name)
        self.op_type = op_type_id.get(type_id)
        if self.op_type is None:
            raise RuntimeError("Not support optype id {}".format(type_id))
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
            "CONV_2D": lambda node: self.convert_conv_op(node),
            "FULLY_CONNECTED": lambda node: self.convert_fc_op(node),
            "MAX_POOL_2D": lambda node: self.convert_maxpool_op(node),
            "MEAN": lambda node: self.convert_mean_op(node),
            "PAD": lambda node: self.convert_pad_op(node),
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
            output_shape, _ = self.get_tensor_shape_and_data(input_id)
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
                In tflite define is NHWC
                return List of NCHW
            """
            return [shape[0], shape[3], shape[1], shape[2]], np.transpose(data, (0,3,1,2)) if HAS_DATA else None
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

    def createLoadWeightOp(self, tensor_idx, tensor_name):
        shape, data = self.get_tensor_shape_and_data(tensor_idx)
        self.addTensor(tensor_name, data, shape, None)
        weight_op = self.CVI.add_load_file_op(tensor_name, shape)
        return weight_op, shape

    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            # Skip "num_batches_tracked"
            if "num_batches_tracked" in i.name:
                continue
            else:
                tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.output_tensor_file, **tensor_npz)

    def convert_node(self):
        """convert tflite node to TFLiteNode"""
        for tflite_op in self.nodes:
            op_type = self.tflite_model.OperatorCodes(tflite_op.OpcodeIndex()).BuiltinCode()
            inputs = [tflite_op.Inputs(i) for i in range(tflite_op.InputsLength())]

            if tflite_op.OutputsLength() != 1:
                raise RuntimeError("node output len greater than 1")
            output = tflite_op.Outputs(0)
            name = str(output)
            node = TFLiteNode(name, op_type, inputs, output, tflite_op)

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
            self.addOperand(input_tensor_idx, input_op, input_shape, TensorType.ACTIVATION)

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
            op, _, _ = self.getOperand(output.name)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)

    def convert_add_op(self, node):
        assert(node.op_type == "ADD")
        op1, input_shape1, _ = self.getOperand(str(node.inputs[0]))
        op2, input_shape2, _ = self.getOperand(str(node.inputs[1]))
        if input_shape1 != input_shape2:
                raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2))

        operands = list()
        operands.append(op1)
        operands.append(op2)
        output_shape = input_shape1

        add_op = self.CVI.add_eltwise_add_op("{}_{}".format(node.name, node.op_type), operands, output_shape)
        self.addOperand(node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_pad_op(self, node):
        assert(node.op_type == "PAD")
        """
            Fix: our mlir don't have padding op,
            We fuse with next Conv2d
            In tensorflow official resnet case, it can be work
            other case is TODO
            by Sam
        """
        # first input is activate, second is tensor
        assert(len(node.inputs) == 2)
        op, shape, _ = self.getOperand(str(node.inputs[0]))

        padding_attr_tensor_idx = node.inputs[1]
        padding_attr_shape , padding_attr_data = self.get_tensor_shape_and_data(padding_attr_tensor_idx, data_type=np.int32)

        self.addOperand(node.name, op, shape, TensorType.ACTIVATION)
        # For Conv2d Get this data
        self.addTensor(node.name, padding_attr_data, shape, "PAD")

    def convert_conv_op(self, node):
        assert(node.op_type == "CONV_2D")

        op, shape, _ = self.getOperand(str(node.inputs[0]))
        # Check if input is padding op
        padding_data = None
        try:
            input_tensor = self.getTensor(str(node.inputs[0]))
            # Only padding case we handle
            assert(input_tensor.op_type == "PAD")
            padding_data = input_tensor.tensor_data
        except KeyError as k:
            # Not padding op
            pass
        operands = list()
        operands.append(op)

        # filter
        filter_tensor_idx = node.inputs[1]
        filter_name = "{}_add_weight".format(filter_tensor_idx)
        filter_op, filter_shape = self.createLoadWeightOp(filter_tensor_idx, filter_name)
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
        conv_table = Conv2DOptions()
        conv_table.Init(op_build_info.Bytes, op_build_info.Pos)

        conv_param = {
            'stride_h': conv_table.StrideH(),
            'stride_w': conv_table.StrideW(),
            'padding': "SAME" if conv_table.Padding() == Padding.SAME and isinstance(padding_data, np.ndarray) else "VALUE",
            'dilation_h': conv_table.DilationHFactor(),
            'dilation_w': conv_table.DilationWFactor(),
            'group': 1, # Don't have group option?
            'is_dw': False,
            'with_bias': len(node.inputs) > 2,
            'do_relu': False,
        }
        on = shape[0]
        oc = filter_shape[0] # feature map size
        # padding data order is NHWC
        # if padding data is not np.ndarray (not from bottom layer)
        # and conv_table.Padding() is SAME, we need to calculate it.
        if conv_table.Padding() == Padding.SAME:
            out_h = ceil(shape[2]/conv_param['stride_h'])
            padding_h = get_TF_SAME_Padding(
                shape[2], out_h, filter_shape[2], conv_param['stride_h'])
            out_w = ceil(shape[3]/conv_param['stride_w'])
            padding_w = get_TF_SAME_Padding(
                shape[3], out_w, filter_shape[3], conv_param['stride_w'])
        else:
            padding_h = 0
            padding_w = 0

        oh = calcConv2DSpatial(
            shape[2],
            filter_shape[2],
            conv_param['stride_h'],
            padding_data[1][0] if isinstance(
                padding_data, np.ndarray) else padding_h,
            conv_param['dilation_h'],
        )
        ow = calcConv2DSpatial(
            shape[3],
            filter_shape[3],
            conv_param['stride_w'],
            padding_data[2][0] if isinstance(padding_data, np.ndarray) else padding_w,
            conv_param['dilation_w'],
        )
        print(conv_param)
        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op("{}_{}".format(
            node.name, node.op_type), operands, output_shape, **conv_param)
        self.addOperand(node.name, conv_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_fc_op(self, node):

        assert(node.op_type == "FULLY_CONNECTED")

        op, shape, _ = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)

        # filter
        filter_tensor_idx = node.inputs[1]
        filter_name = "{}_add_weight".format(filter_tensor_idx)
        filter_op, filter_shape = self.createLoadWeightOp(
            filter_tensor_idx, filter_name)
        operands.append(filter_op)

        # bias

        bias_tensor_idx = node.inputs[2]
        bias_name = "{}_add_bias".format(bias_tensor_idx)
        bias_op, bias_shape = self.createLoadWeightOp(
            bias_tensor_idx, bias_name)
        operands.append(bias_op)

        M = shape[0]
        K = shape[1]
        N = bias_shape[0]
        output_shape = [M, N]
        fc_op = self.CVI.add_fully_connected_op("{}_{}".format(node.name, node.op_type), operands, output_shape)
        self.addOperand(node.name, fc_op, output_shape, TensorType.ACTIVATION)

    def convert_maxpool_op(self, node):
        assert(node.op_type == "MAX_POOL_2D")

        op, shape, _ = self.getOperand(str(node.inputs[0]))
        # Check if input is padding op
        padding_data = None
        try:
            input_tensor = self.getTensor(str(node.inputs[0]))
            # Only padding case we handle
            assert(input_tensor.op_type == "PAD")
            padding_data = input_tensor.tensor_data
        except KeyError as k:
            # Not padding op
            pass

        operands = list()
        operands.append(op)

        op_build_info = node.proto.BuiltinOptions()
        pool_table = Pool2DOptions()
        pool_table.Init(op_build_info.Bytes, op_build_info.Pos)

        pool_max_2d_param = {
            'stride_h': pool_table.StrideH(),
            'stride_w': pool_table.StrideW(),
            'kernel_h': pool_table.FilterWidth(),
            'kernel_w': pool_table.FilterHeight(),
            'padding_b': padding_data[1][0] if isinstance(padding_data, np.ndarray) else 0,
            'padding_r': padding_data[2][0] if isinstance(padding_data, np.ndarray) else 0,
            'padding_t': padding_data[1][1] if isinstance(padding_data, np.ndarray) else 0,
            'padding_l': padding_data[2][1] if isinstance(padding_data, np.ndarray) else 0,
            'do_relu': False,
        }

        operands = list()
        operands.append(op)
        on = shape[0]
        oc = shape[1]
        oh = calcPool2DFloor(shape[2], pool_max_2d_param['kernel_h'], pool_max_2d_param['stride_h'], pool_max_2d_param['padding_b'])
        ow = calcPool2DFloor(shape[3], pool_max_2d_param['kernel_w'], pool_max_2d_param['stride_w'], pool_max_2d_param['padding_r'])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op("{}_{}".format(node.name, node.op_type), operands, output_shape, **pool_max_2d_param)
        self.addOperand(node.name, pool_max_op, output_shape, TensorType.ACTIVATION)

    def convert_mean_op(self, node):
        assert(node.op_type == "MEAN")
        """
            Fix: our mlir don't have mean op,
            we use avg_pool workaround
            (Sam)
        """
        # first input is activate, second is tensor of axis
        assert(len(node.inputs) == 2)
        op, shape, _ = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)
        mean_tensor_idx = node.inputs[1]
        mean_shape, mean_attr_data = self.get_tensor_shape_and_data(
            mean_tensor_idx, data_type=np.int32)
        on = shape[0]
        oc = shape[1]
        pool_avg_2d_param = {
            'stride_h':  1,
            'stride_w':  1,
            'kernel_h':  shape[2],
            'kernel_w':  shape[3],
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(
            node.name, node.op_type), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_node()
        self.convert_graph()
        #self.TensortoNpz()


