from .mlirimporter import BaseConverterInterface, MLIRImporter, checkKey
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum
from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
                    get_shape_size

from ..utils.log_setting import setup_logger

# tflite gen by flatbuffer
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator

import logging
import numpy as np
import operator

logger = setup_logger('root')

log_flag = logger.level <= logging.INFO

class TensorType(Enum):
    ACTIVATION = 'ACTIVATION'
    TENSOR = 'TENSOR'

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
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')



class TFLiteConverter(BaseConverterInterface):
    def __init__(self, model_name, tflite_model_file, mlir_file_path):
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
            "PAD" : lambda node: self.convert_pad_op(node),
            "CONV_2D": lambda node: self.convert_conv_op(node),
        }

    def init_importer(self):
        # Make MLIR Function
        # get input shape
        inputs = list()
        for input_id in self.input_nodes:
            input_tensor = self.tflite_graph.Tensors(input_id)
            input_shape = [input_tensor.Shape(i) for i in range(input_tensor.ShapeLength())]
            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output_id in self.output_nodes:
            output_tensor = self.tflite_graph.Tensors(output_id)
            output_shape = [output_tensor.Shape(i) for i in range(output_tensor.ShapeLength())]
            outputs.append(output_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs)

    def get_tflite_tensor_shape(self, tensor_index):
        tensor = self.tflite_graph.Tensors(tensor_index)
        return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    def get_tflite_tensor_data(self, tensor_index):
        tensor = self.tflite_graph.Tensors(tensor_index)
        return self.tflite_model.Buffers(tensor.Buffer()).DataAsNumpy()

    def addOperand(self, op_name, op, shape, tensor_type):
        cprint("add opernand name: {}\nshape: {} \ntesnor_type {}".format(op_name, shape, tensor_type), "yellow")
        self.valueMap[op_name] = (op, shape, tensor_type)

    def getOperand(self, op_name):
        print(self.valueMap)
        return self.valueMap[op_name]

    def addTensor(self, op_name, tensor_data, tensor_shape):
        #cprint("add tensor, name: {}\ntensor data: {}".format(op_name, tensor_data), "yellow")
        self.converted_tensors.append(TFLiteTensor(op_name, tensor_data, tensor_shape))

    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise KeyError("No {} tensor in model".format(op_name))
        else:
            return find_tensor[0]

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
            input_shape = self.get_tflite_tensor_shape(input_tensor_idx)
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


    def convert_pad_op(self, node):
        assert(node.op_type == "PAD")
        # first input is activate, second is tensor
        assert(len(node.inputs) == 2)
        op, shape, _ = self.getOperand(node.inputs[0])

        padding_attr_tensor_idx = node.inputs[1]
        padding_attr_shape = self.get_tflite_tensor_shape(padding_attr_tensor_idx)
        padding_attr_data = self.get_tflite_tensor_data(padding_attr_tensor_idx)
        padding_attr_data = np.frombuffer(padding_attr_data.tobytes(), dtype=np.int32)
        padding_attr_data = padding_attr_data.reshape(tuple(padding_attr_shape))
        # Todo: add mlir with pad op
        self.addOperand(node.name, None, shape, TensorType.ACTIVATION)

    def convert_conv_op(self, node):
        assert(node.op_type == "CONV_2D")

        op, shape, _ = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)
        # filter
        filter_tensor_idx = node.inputs[1]
        filter_shape = self.get_tflite_tensor_shape(filter_tensor_idx)
        filter_data = self.get_tflite_tensor_data(filter_tensor_idx)
        filter_data = np.frombuffer(filter_data.tobytes(), dtype=np.float32)
        filter_name = "{}_add_weight".format(filter_tensor_idx)
        self.addTensor(filter_name, filter_data, filter_shape)
        filter_op = self.CVI.add_load_file_op(filter_name, filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = len(node.inputs) == 3
        if do_bias:
            bias_tensor_idx = node.inputs[2]
            bias_shape = self.get_tflite_tensor_shape(bias_tensor_idx)
            bias_data = self.get_tflite_tensor_data(bias_tensor_idx)
            bias_data = np.frombuffer(bias_data.tobytes(), dtype=np.float32)
            bias_name = "{}_add_bias".format(bias_tensor_idx)
            self.addTensor(bias_name, bias_data, bias_shape)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_shape)
            operands.append(bias_op)

        op_build_info = node.proto.BuiltinOptions()
        # Parse the Table of options.
        conv_table = tflite.Conv2DOptions()
        conv_table.Init(op_build_info.Bytes, op_build_info.Pos)


        conv_param = {
            'stride_h': conv_table.StrideH(),
            'stride_w': conv_table.StrideW(),
            'padding': "SAME" if conv_table.Padding() == tflite.Padding.SAME else "VALUE",
            'dilation_h': onnx_node.attrs['dilations'][0],
            'dilation_w': onnx_node.attrs['dilations'][1],
            'group': onnx_node.attrs['group'],
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
        }


        # with_bias = False
        # if (len(onnx_node.inputs) == 3):
        #     #with bias
        #     with_bias = True
        #     bias_name = onnx_node.inputs[2]
        #     bias_tensor = self.getTensor(bias_name)
        # conv_param = {
        #     'stride_h':  onnx_node.attrs['strides'][0],
        #     'stride_w':  onnx_node.attrs['strides'][1],
        #     'padding': "SAME" if onnx_node.attrs['pads'][0] > 0 else "VALID",
        #     'dilation_h': onnx_node.attrs['dilations'][0],
        #     'dilation_w': onnx_node.attrs['dilations'][1],
        #     'group': onnx_node.attrs['group'],
        #     'is_dw': False,
        #     'with_bias': len(onnx_node.inputs) > 2,
        #     'do_relu': False,
        # }

        # on = shape[0]
        # oc = filter_tensor.shape[0] # feature map size
        # oh = calcConv2DSpatial(
        #     shape[2],
        #     onnx_node.attrs['kernel_shape'][0],
        #     onnx_node.attrs['strides'][0],
        #     onnx_node.attrs['pads'][0],
        #     onnx_node.attrs['dilations'][0]
        # )
        # ow = calcConv2DSpatial(
        #     shape[3],
        #     onnx_node.attrs['kernel_shape'][1],
        #     onnx_node.attrs['strides'][1],
        #     onnx_node.attrs['pads'][1],
        #     onnx_node.attrs['dilations'][1]
        # )

        # if conv_param['group'] != 1:
        #     # filter shape s is in (g, oc/g, ic/g, kh, kw)
        #     g = conv_param['group']
        #     ic = shape[1]
        #     kh = onnx_node.attrs['kernel_shape'][0]
        #     kw = onnx_node.attrs['kernel_shape'][1]
        #     new_shape = [g, int(oc/g), int(ic/g), kh, kw]
        #     filter_op = self.CVI.add_load_file_op(filter_tensor.name, new_shape)
        # else:
        #     filter_op = self.CVI.add_load_file_op(filter_tensor.name, filter_shape)
        # operands.append(filter_op)

        # if with_bias:
        #     bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
        #     operands.append(bias_op)

        # output_shape = [on, oc, oh, ow]
        # conv_op = self.CVI.add_conv_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv_param)
        # self.addOperand(onnx_node.name, conv_op, output_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_node()
        self.convert_graph()
        #self.TensortoNpz()


