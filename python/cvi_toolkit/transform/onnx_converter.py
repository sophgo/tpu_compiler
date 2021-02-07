# ONNX Node define:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from onnx import numpy_helper, mapping
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number


import onnx
import logging
import numpy as np
import operator
import functools

from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
    get_shape_size, get_TF_SAME_Padding

from ..utils.log_setting import setup_logger

logger = setup_logger('root')

log_flag = logger.level <= logging.DEBUG


onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}

def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)

def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

class OnnxNode():
    def __init__(self, node):
        self.name = str(node.output[0])
        self.op_type = str(node.op_type)
        self.attrs = dict([(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr))) for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')
        cprint("    attrs: {}".format(self.attrs), 'white')
        for arg in self.attrs:
            cprint("        {}: {}".format(arg, self.attrs[arg]), 'green')

class OnnxTensor():
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')



class OnnxConverter(BaseConverter):
    def __init__(self, model_name, onnx_model, mlir_file_path,
                batch_size=1, preprocess_args=None):
        super().__init__()
        if isinstance(onnx_model, str):
            onnx_model = onnx.load(onnx_model)
        self.batch_size = batch_size
        self.model_name = model_name
        self.input_nodes = onnx_model.graph.input
        self.output_nodes = onnx_model.graph.output
        self.output_shapes = list()
        self.nodes = onnx_model.graph.node
        self.tensors = onnx_model.graph.initializer
        self.mlir_file_path = mlir_file_path

        self.remove_tensor_from_input_nodes()
        self.remove_tensor_from_output_nodes()

        self.converted_nodes = list()
        self.converted_tensors = list()
        self.preprocess_args = preprocess_args
        self.input_shapes = list()

        self.CVI = None
        self.output_weight_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.init_importer()

        self.onnxop_factory = {
            "Abs": lambda node: self.convert_abs_op(node),
            "Add": lambda node: self.convert_add_op(node),
            "AveragePool": lambda node: self.convert_avg_pool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Cast": lambda node: self.convert_cast_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "ConstantOfShape": lambda node: self.convert_constant_of_shape_op(node),
            "DepthToSpace": lambda node: self.convert_depth_to_space_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Equal": lambda node: self.convert_equal_op(node),
            "Exp" :lambda node: self.convert_activation_op(node),
            "Expand": lambda node: self.convert_expand_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_pool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_pool_op(node),
            "GRU": lambda node: self.convert_gru_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "InstanceNormalization": lambda node: self.convert_instancenorm_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "LSTM": lambda node: self.convert_lstm_op(node),
            "MatMul": lambda node: self.convert_matmul_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Max" : lambda node: self.convert_max_op(node),
            "Min" : lambda node: self.convert_min_op(node),
            "Mul" : lambda node: self.convert_mul_op(node),
            "Neg" : lambda node: self.convert_neg_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "Reciprocal": lambda node: self.convert_reciprocal_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "ReduceMean": lambda node: self.convert_reduce_mean_op(node),
            "ReduceMax": lambda node: self.convert_reduce_max_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid" :lambda node: self.convert_activation_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Sqrt": lambda node: self.convert_sqrt_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Sum": lambda node: self.convert_sum_op(node),
            "Tanh": lambda node: self.convert_activation_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Where": lambda node: self.convert_where_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "Upsample": lambda node: self.convert_upsample_op(node),

        }

    def __del__(self):
        del self.CVI

    def init_importer(self):
        # get input shape
        inputs = list()
        for input in self.input_nodes:
            input_shape = list()
            for i, dim in enumerate(input.type.tensor_type.shape.dim):
                # batch size
                # dim is zero, mean mutli batch
                if i == 0 and dim.dim_value == 0:
                    input_shape.append(self.batch_size)
                else:
                    input_shape.append(dim.dim_value)
            self.input_shapes.append(input_shape)
            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output in self.output_nodes:
            output_shape = list()
            for i, dim in enumerate(output.type.tensor_type.shape.dim):
                # i == 0 mean batch size
                # if dim is zero, mean mutli batch
                if i == 0 and dim.dim_value == 0:
                    output_shape.append(self.batch_size)
                else:
                    output_shape.append(dim.dim_value)
            outputs.append(output_shape)
            # keep mlir function output
            self.output_shapes.append(output_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs, "FP32", output_weight_file=self.output_weight_file)

    def remove_tensor_from_input_nodes(self):
        def find_name_in_tensor_list(name):
            for i in self.tensors:
                if name == i.name:
                    return True
            return False
        self.input_nodes = [x for x in self.input_nodes if not find_name_in_tensor_list(x.name)]

    def remove_tensor_from_output_nodes(self):
        def find_name_in_tensor_list(name):
            for i in self.input_nodes:
                if name == i.name:
                    return True
            return False
        self.output_nodes = [x for x in self.output_nodes if not find_name_in_tensor_list(x.name)]


    def addTensor(self, op_name, tensor_data, tensor_shape):
        #cprint("add tensor, name: {}\ntensor data: {}".format(op_name, tensor_data), "yellow")
        self.converted_tensors.append(OnnxTensor(op_name, tensor_data, tensor_shape))


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
                try:
                    tensor_npz[i.name] = i.tensor_data.astype(np.float32)
                except AttributeError as attr_err:
                    print("{} data type {} can not transform to float, skip it".format(i.name, type(i.tensor_data)))
                except:
                    raise
        np.savez(self.output_weight_file, **tensor_npz)

    @staticmethod
    def unsqueeze_shape(shape, axis):
        new_shape = [n for n in shape]
        for n in axis:
            new_shape.insert(n, 1)
        return new_shape

    @staticmethod
    def squeeze_shape(shape, axis):
        new_shape = []
        if len(axis) > 0:
            for i in range(len(shape)):
                if i not in axis:
                    new_shape.append(shape[i])
        else:
            new_shape = shape
        return new_shape


    def convert_node(self):
        """convert onnx node to OnnxNode"""
        for n in self.nodes:
            node = OnnxNode(n)
            if log_flag:
                node.print_info()
            self.converted_nodes.append(node)

    def convert_tensor(self):
        """convert onnx tensor to OnnxTensor"""
        for tensor in self.tensors:
            name = tensor.name
            shape = list(tensor.dims)
            data = numpy_helper.to_array(tensor).astype(np.float32)
            tensor = OnnxTensor(name, data, shape)
            #tensor.print_info()
            self.converted_tensors.append(tensor)
            self.addOperand(name, None, shape, TensorType.TENSOR)

    def convert_graph(self):
        """convert all to mlir"""

        # add input op
        for idx, input in enumerate(self.input_nodes):
            input_shape = list()
            for i, dim in enumerate(input.type.tensor_type.shape.dim):
                # batch size
                # dim is zero, mean mutli batch
                if i == 0 and dim.dim_value == 0:
                    input_shape.append(self.batch_size)
                else:
                    input_shape.append(dim.dim_value)

            if not self.preprocess_args:
                input_op = self.CVI.add_input_op(input.name, idx, **{})
            else:
                preprocess_hint = {
                    'mean': self.preprocess_args['perchannel_mean'],
                    'scale':  self.preprocess_args['perchannel_scale'],
                    'pixel_format': self.preprocess_args["pixel_format"],
                    'channel_order': self.preprocess_args["channel_order"],
                    'aligned': self.preprocess_args["aligned"],
                    'resize_dims': self.preprocess_args['resize_dims'],
                    'keep_aspect_ratio': self.preprocess_args['keep_aspect_ratio']
                }
                # add input op
                input_op = self.CVI.add_input_op(input.name, idx, **preprocess_hint)
            self.addOperand(input.name, input_op, input_shape, TensorType.ACTIVATION)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))
        # add node op
        for n in self.converted_nodes:
            if log_flag:
                n.print_info()
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)

        # add return op
        return_op = list()
        # Set output
        for idx, output in enumerate(self.output_nodes):
            op, shape, _ = self.getOperand(output.name)
            if shape != self.output_shapes[idx]:
                # reshape back
                assert(np.prod(shape) == np.prod(self.output_shapes[idx]))
                op = self.CVI.add_reshape_op("{}_reshaped".format(output.name),
                                                  [op], self.output_shapes[idx])
                self.addOperand(output.name, op, self.output_shapes[idx], TensorType.ACTIVATION)

            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)
        print("Save mlir file: {}".format(self.mlir_file_path))

    def convert_activation_op(self, onnx_node):
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        output_shape = input_shape
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            if onnx_node.op_type == "Sigmoid":
                tensor_data = 1.0 / (1.0 + np.exp(np.negative(tensor_data)))
            elif onnx_node.op_type == "Tanh":
                tensor_data = np.tanh(tensor_data)
            elif onnx_node.op_type == "Exp":
                tensor_data = np.exp(tensor_data)
            self.addTensor(onnx_node.name, tensor_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            if onnx_node.op_type == "Sigmoid":
                activation_op = self.CVI.add_sigmoid_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Tanh":
                activation_op = self.CVI.add_tanh_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Exp":
                activation_op = self.CVI.add_exp_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)

            self.addOperand(onnx_node.name, activation_op, output_shape, TensorType.ACTIVATION)

    def convert_abs_op(self, onnx_node):
        assert(onnx_node.op_type == "Abs")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.clip(tensor_data, 0, np.inf)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            output_shape = input_shape
            abs_op = self.CVI.add_abs_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, abs_op, output_shape, TensorType.ACTIVATION)

    def convert_add_op(self, onnx_node):

        assert(len(onnx_node.inputs) == 2)
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.ACTIVATION:
            # put activation first
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return

        operands = list()
        # broadcast add from constant
        # TODO: Our IR now only support channel broadcast
        # [n,c,h,w] broadcast mul [1,c,1,1]
        if tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            operands.append(op1)
            if len(input_shape2) == 1 or len(input_shape2) == 3: # [1] or [c, 1, 1] or [w]
                channel = input_shape1[1]
                width = -1 if len(input_shape1) != 4 else input_shape1[3]

                add_const_value = self.getTensor(onnx_node.inputs[1]).tensor_data

                if len(add_const_value.flatten()) == width:
                    """
                        broadcast constand value along width, use eltwise add
                    """
                    tensor_data = np.broadcast_to(
                        add_const_value, (input_shape1[0], input_shape1[1], input_shape1[2], input_shape1[3]))
                    output_shape = input_shape1
                    weight_name = "{}_add_weight".format(onnx_node.name)
                    self.addTensor(weight_name, tensor_data, tensor_data.shape)
                    op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
                    operands.append(op2)
                    add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                    self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)
                    return
                else:
                    # Use scale op, x * 1 + y
                    tensor_data = np.full(input_shape1[1], 1) # broadcast via channel
                    weight_name = "{}_add_weight".format(onnx_node.name)
                    self.addTensor(weight_name, tensor_data, tensor_data.shape)
                    op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
                    operands.append(op2)

                    if len(add_const_value.flatten()) == 1:
                        # only one constant
                        tensor_data = np.full(input_shape1[1], add_const_value[0]) # broadcast via channel
                    elif len(add_const_value.flatten()) == channel:
                        tensor_data = add_const_value.flatten()
                    else:
                        raise RuntimeError("could not broadcast input array from shape {} into shape {}".format(input_shape1, input_shape2))

                    bias_name = "{}_add_bias".format(onnx_node.name)
                    self.addTensor(bias_name, tensor_data, tensor_data.shape)
                    op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
                    operands.append(op3)

                    output_shape = input_shape1

                    scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                    self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)
                    return
            elif input_shape1 == input_shape2:
                # add const value with same shape, use eltwise add
                bias_name = "{}_add_bias".format(onnx_node.name)
                add_value = self.getTensor(onnx_node.inputs[1]).tensor_data
                self.addTensor(bias_name, add_value, add_value.shape)
                op3 = self.CVI.add_load_file_op(bias_name, add_value.shape)
                operands.append(op3)
                output_shape = input_shape1
                add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)

            elif input_shape1[:2] == input_shape2[:2] and input_shape2[2:] == [1, 1]:
                # broadcast add
                bias_name = "{}_add_bias".format(onnx_node.name)
                add_value = self.getTensor(onnx_node.inputs[1]).tensor_data
                self.addTensor(bias_name, add_value, add_value.shape)
                op3 = self.CVI.add_load_file_op(bias_name, add_value.shape)
                operands.append(op3)

                output_shape = input_shape1
                add_op = self.CVI.add_broadcast_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis = 1)
                self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)
                return
            else:
                raise RuntimeError("{} vs {} shape broadcast error".format(input_shape1, input_shape2))

        elif tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = tensor_data1 + tensor_data2
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        elif tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.ACTIVATION:
            if input_shape1 != input_shape2:
                if len(input_shape1) == 4 and input_shape2[-2:] == [1, 1] and input_shape1[1] == input_shape2[1]:
                    # broadcast add from activation
                    # [n,c,h,w] broadcast add [n,c,1,1]
                    # upsample [n,c,1,1] to [n,c,h,w], then eltwise add
                    scale_factor_h, scale_factor_w = input_shape1[2], input_shape1[3]
                    ic = int(input_shape1[1])
                    on = int(input_shape1[0])
                    oc = ic
                    oh = int(input_shape1[2])
                    ow = int(input_shape1[3])
                    output_shape = [int(on), int(oc), int(oh), int(ow)]
                    # use deconv(depthwise)
                    deconv_param = {
                        'stride_h':  scale_factor_h,
                        'stride_w':  scale_factor_w,
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
                    weight_shape = [ic, 1, 1, scale_factor_h, scale_factor_w]
                    deconv_tensor_data = np.full(weight_shape, 1)
                    weight_name = "{}_add_weight".format(onnx_node.name)
                    self.addTensor(
                        weight_name, deconv_tensor_data, deconv_tensor_data.shape)
                    weight_op = self.CVI.add_load_file_op(
                        weight_name, deconv_tensor_data.shape)

                    deconv_op = self.CVI.add_deconv_op("{}_{}".format(
                        onnx_node.name, onnx_node.op_type), [op2, weight_op], output_shape, **deconv_param)
                    self.addOperand(onnx_node.name, deconv_op,
                                    output_shape, TensorType.ACTIVATION)
                    op2 = deconv_op
                elif len(input_shape2) == 4 and input_shape1[-2:] == [1, 1] and input_shape1[1] == input_shape2[1]:
                    onnx_node.inputs[0] ,onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
                    return self.convert_add_op(onnx_node)
                else:
                    raise AttributeError("{} v.s. {} shape not same".format(
                        input_shape1, input_shape2))
            operands.append(op1)
            operands.append(op2)
            output_shape = input_shape1

            add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_avg_pool_op(self, onnx_node):
        assert(onnx_node.op_type == "AveragePool")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        on = input_shape[0]
        oc = input_shape[1]
        pads = onnx_node.attrs['pads'] if "pads" in onnx_node.attrs else [0, 0, 0, 0]

        strides = onnx_node.attrs['strides'] if "strides" in onnx_node.attrs else [1, 1]
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        if kernel_shape[0] == 1:
            count_include_pad = True

        pool_avg_2d_param = {
            'stride_h':  strides[0],
            'stride_w':  strides[1],
            'kernel_h':  kernel_shape[0],
            'kernel_w':  kernel_shape[1],
            'padding_t': pads[0],
            'padding_l': pads[1],
            'padding_b': pads[2],
            'padding_r': pads[3],
            'do_relu': False,
            'count_include_pad': count_include_pad,
        }
        oh = calcPool2DFloor(input_shape[2], pool_avg_2d_param['kernel_h'], pool_avg_2d_param['stride_h'], pool_avg_2d_param['padding_t'], pool_avg_2d_param['padding_b'])
        ow = calcPool2DFloor(input_shape[3], pool_avg_2d_param['kernel_w'], pool_avg_2d_param['stride_w'], pool_avg_2d_param['padding_l'], pool_avg_2d_param['padding_r'])
        output_shape = [int(on), int(oc), oh, ow]
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(onnx_node.name, pool_avg_op, output_shape, TensorType.ACTIVATION)

    def convert_batchnorm_op(self, onnx_node):
        assert(onnx_node.op_type == "BatchNormalization")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = onnx_node.attrs.get('epsilon', 0)
        # we fuse batchnorm and scale at here
        gamma_value = self.getTensor(onnx_node.inputs[1]).tensor_data
        beta_value = self.getTensor(onnx_node.inputs[2]).tensor_data
        mean_value = self.getTensor(onnx_node.inputs[3]).tensor_data
        var_value = self.getTensor(onnx_node.inputs[4]).tensor_data

        scale_name = "{}_0".format(onnx_node.name)
        scale_value = ((1.0 / np.sqrt(
                    var_value + epsilon)) * gamma_value)

        scale_op = self.CVI.add_load_file_op(scale_name, self.getTensor(onnx_node.inputs[1]).shape)
        # add new weight tensor
        self.addTensor(scale_name, scale_value, self.getTensor(onnx_node.inputs[1]).shape)

        offset_name =  "{}_1".format(onnx_node.name)
        offset_value = (-mean_value * scale_value) + beta_value
        offset_op = self.CVI.add_load_file_op(offset_name, self.getTensor(onnx_node.inputs[1]).shape)
        # add new bias tensor
        self.addTensor(offset_name, offset_value, self.getTensor(onnx_node.inputs[1]).shape)

        operands.append(scale_op)
        operands.append(offset_op)

        output_shape = input_shape
        scaleop = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, scaleop, output_shape, TensorType.ACTIVATION)

    def convert_cast_op(self, onnx_node):
        assert(onnx_node.op_type == "Cast")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            dtype = onnx_node.attrs.get('to')
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            if dtype == "int64":
                data = data.astype(np.int64)
            elif dtype == "int32":
                data = data.astype(np.int32)
            elif dtype == "float32":
                data = data.astype(np.float32)
            else:
                raise RuntimeError("{} dtype not support, please add".format(dtype))
            output_data = data
            output_shape = input_shape
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)

    def convert_constant_op(self, onnx_node):
        """
            Constant Op is tensor data at IR,
            we change it to load weight tensor, and store
        """
        assert(onnx_node.op_type == "Constant")
        onnx_tensor = onnx_node.attrs['value']
        np_tensor =  numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)

        if data_type in [np.float32, np.float64, np.int32, np.int64]:
            self.addTensor(onnx_node.name, np_tensor.astype(data_type), list(np_tensor.shape))
            self.addOperand(onnx_node.name, None, list(np_tensor.shape), TensorType.TENSOR)

        else:
            raise ValueError("Not Support {} type".format(data_type))

    def convert_constant_of_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "ConstantOfShape")
        tensor_shape = self.getTensor(onnx_node.inputs[0]).tensor_data.astype(np.int)
        onnx_tensor = onnx_node.attrs['value']
        tensor_value =  numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)

        if data_type in [np.float32, np.float64, np.int32, np.int64]:
            tensor_value = tensor_value.astype(data_type)
            constant_data = np.full(tuple(tensor_shape), tensor_value[0])
            # add new weight tensor
            self.addTensor(onnx_node.name, constant_data, list(constant_data.shape))
            self.addOperand(onnx_node.name, None, list(tensor_shape.shape), TensorType.TENSOR)
        else:
            raise ValueError("Not Support {} type".format(data_type))

    def convert_concat_op(self, onnx_node):
        assert(onnx_node.op_type == "Concat")
        if len(onnx_node.inputs) == 1:
            # convert concat op to reshape op if has only one input
            op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
            if tensor_type != TensorType.ACTIVATION:
                raise RuntimeError("Tensor can not concat with activation")
            operands = [op]
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type),
                                                  operands, input_shape)
            self.addOperand(onnx_node.name, reshape_op, input_shape, TensorType.ACTIVATION)
            return

        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        axis = onnx_node.attrs['axis']
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_datas = list()
            tensor_datas_shape = list()
            for i in onnx_node.inputs:
                t_d = self.getTensor(i).tensor_data
                # predict all shape, concat shape lens should all equal
                tensor_datas_shape.append(len(t_d.shape))
            # get frequency counts for unique values in an array
            x = np.array(tensor_datas_shape)
            y = np.bincount(x)
            ii = np.nonzero(y)[0]
            dim_size = np.vstack((ii,y[ii])).T[0][0] # [0][0] is most frequency shape

            for i in onnx_node.inputs:
                # FIXME: if tensor data is from weight, and it's shape just one
                #        match with other input shape, use squeeze
                t_d = self.getTensor(i).tensor_data
                if len(t_d.shape) == 1 and t_d[0] == -1:
                    if dim_size == len(t_d.shape):
                        pass # no need to expaned dims cuz all dim_size are same
                    else:
                        t_d = np.expand_dims(t_d, axis=0)
                tensor_datas.append(t_d)

             # handle input0 shape(1, 1), intput1 shape (1,)
            if np.array(tensor_datas[0]).size == np.array(tensor_datas[1]).size:
              shape = np.array(tensor_datas[1]).shape
              tensor_datas[0] = (np.array(tensor_datas[0]).reshape(shape)).tolist()

            n_t = np.concatenate(tuple(tensor_datas), axis=axis)
            self.addTensor(onnx_node.name, n_t, list(n_t.shape))
            self.addOperand(onnx_node.name, None, list(n_t.shape), TensorType.TENSOR)
        else:
            operands = list()
            in_shapes = list()

            for i in onnx_node.inputs:
                op, input_shape, tensor_type = self.getOperand(i)
                if tensor_type != TensorType.ACTIVATION:
                    raise RuntimeError("Tensor can not concat with activation")

                in_shapes.append(input_shape)
                operands.append(op)
            output_shape = list()

            for idx, op_shape in enumerate(in_shapes):
                if idx == 0:
                    # copy rather than referece
                    output_shape = list(op_shape)
                else:
                    for dim, value in enumerate(op_shape):
                        if dim == axis:
                            output_shape[dim] += value
                        else:
                            if output_shape[dim] != value:
                                raise ValueError("axis is {}, {} v.s {} shape can not be concat".format(axis, output_shape, op_shape))

            concat_op = self.CVI.add_concat_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis=axis)
            self.addOperand(onnx_node.name, concat_op, output_shape, TensorType.ACTIVATION)

    def convert_conv1d_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        dilations = onnx_node.attrs.get("dilations", [1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0])
        strides = onnx_node.attrs.get("strides",[1])
        conv_param = {
            'stride_h':  1,
            'stride_w':  strides[0],
            'padding': "SAME" if pads[0] > 0 else "VALID",
            'dilation_h': 1,
            'dilation_w': dilations[0],
            'padding_t': 0,
            'padding_b': 0,
            'padding_l': pads[0],
            'padding_r': pads[1],
            'group': group,
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
            'ins': [],
        }
        op, shape, _ = self.getOperand(onnx_node.inputs[0])
        # convert conv1d to conv2d
        if len(shape) == 3:
            shape.insert(2,1)
            reshape_input_op = self.CVI.add_reshape_op("{}_{}_input".format(
                    onnx_node.name, onnx_node.op_type), [op], shape)
            op = reshape_input_op
        operands = list()
        operands.append(op)
        filter_name = onnx_node.inputs[1]
        filter_tensor = self.getTensor(filter_name)
        if len(filter_tensor.shape) == 3:
            filter_tensor.shape.insert(2, 1)
            filter_tensor.tensor_data.reshape(filter_tensor.shape)
        filter_shape = filter_tensor.shape
        with_bias = False
        on = shape[0]
        oc = filter_shape[0] # feature map size
        oh = 1
        ow = calcConv2DSpatial(
            shape[3],
            onnx_node.attrs['kernel_shape'][0],
            strides[0],
            conv_param['padding_l'],
            conv_param['padding_r'],
            dilations[0]
        )
        filter_op = self.CVI.add_load_file_op(filter_tensor.name, filter_shape)
        operands.append(filter_op)
        if len(onnx_node.inputs) == 3:
            bias_name = onnx_node.inputs[2]
            bias_tensor = self.getTensor(bias_name)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)
        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv_param)
        new_output_shape = [on, oc, ow]
        reshape_back_op = self.CVI.add_reshape_op("{}_{}_back_dim".format(
                    onnx_node.name, onnx_node.op_type), [conv_op], new_output_shape)
        self.addOperand(onnx_node.name, reshape_back_op,
                                new_output_shape, TensorType.ACTIVATION)

    def convert_conv_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        if len(onnx_node.attrs['kernel_shape']) == 1:
            return self.convert_conv1d_op(onnx_node)
        if len(onnx_node.attrs['kernel_shape']) == 3:
            return self.convert_conv3d_op(onnx_node)

        op, shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        filter_name = onnx_node.inputs[1]
        filter_tensor = self.getTensor(filter_name)
        filter_shape = filter_tensor.shape
        with_bias = False
        if len(onnx_node.inputs) == 3:
            #with bias
            with_bias = True
            bias_name = onnx_node.inputs[2]
            bias_tensor = self.getTensor(bias_name)

        dilations = onnx_node.attrs.get("dilations", [1, 1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1])
        auto_pad = onnx_node.attrs.get("auto_pad", None)
        if auto_pad:
            pad_method = auto_pad.decode('utf-8')
            if pad_method == "SAME_UPPER":
                padding_along_h = get_TF_SAME_Padding(
                    shape[2], filter_shape[2], strides[0])
                padding_along_w = get_TF_SAME_Padding(
                    shape[3], filter_shape[3], strides[1])
                padding_t = padding_along_h // 2
                padding_l = padding_along_w // 2
                padding_b = padding_along_h - padding_t
                padding_r = padding_along_w - padding_l
                pads = [padding_t, padding_l, padding_b, padding_r]
            elif pad_method == "VALID":
                pass
            elif pad_method == "NOTSET":
                pass
            else:
                raise RuntimeError("Not support conv {} pad method".format(pad_method))
        conv_param = {
            'stride_h':  strides[0],
            'stride_w':  strides[1],
            'padding': "SAME" if pads[0] > 0 else "VALID",
            'dilation_h': dilations[0],
            'dilation_w': dilations[1],
            'padding_t': pads[0],
            'padding_b': pads[2],
            'padding_l': pads[1],
            'padding_r': pads[3],
            'group': group,
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
            'ins': [],
        }

        on = shape[0]
        oc = filter_tensor.shape[0] # feature map size
        oh = calcConv2DSpatial(
            shape[2],
            onnx_node.attrs['kernel_shape'][0],
            strides[0],
            conv_param['padding_t'],
            conv_param['padding_b'],
            dilations[0]
        )
        ow = calcConv2DSpatial(
            shape[3],
            onnx_node.attrs['kernel_shape'][1],
            strides[1],
            conv_param['padding_l'],
            conv_param['padding_r'],
            dilations[1]
        )

        if conv_param['group'] != 1:
            # filter shape s is in (g, oc/g, ic/g, kh, kw)
            g = conv_param['group']
            ic = shape[1]
            kh = onnx_node.attrs['kernel_shape'][0]
            kw = onnx_node.attrs['kernel_shape'][1]
            new_shape = [g, int(oc/g), int(ic/g), kh, kw]
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, new_shape)
            if g == oc:
                conv_param['is_dw'] = True

        else:
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, filter_shape)
        operands.append(filter_op)

        if with_bias:
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op, output_shape, TensorType.ACTIVATION)

    def convert_conv3d_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        dilations = onnx_node.attrs.get("dilations", [1,1,1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0,0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1,1])
        raise RuntimeError("TODO: Conv3d IR")
        conv_param = {
            'stride_h':  1,
            'stride_w':  strides[0],
            'padding': "SAME" if pads[0] > 0 else "VALID",
            'dilation_h': 1,
            'dilation_w': dilations[0],
            'padding_t': 0,
            'padding_b': 0,
            'padding_l': pads[0],
            'padding_r': pads[1],
            'group': group,
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
            'ins': [],
        }

    def convert_clip_op(self, onnx_node):
        assert(onnx_node.op_type == "Clip")
        clip_param = {
            'min':  onnx_node.attrs['min'],
            'max':  onnx_node.attrs['max'],
        }
        op, shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = shape
        # FIXME: Now not support clip quantize
        # Only support relu6 case
        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.clip(data, onnx_node.attrs['min'],onnx_node.attrs['max'])
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            if clip_param.get("min") == 0:
                # relu6
                relu_op = self.CVI.add_relu_op("{}_relu6_relu{}".format(onnx_node.name, onnx_node.op_type), [op], output_shape)
                clip_op = self.CVI.add_clip_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [relu_op], output_shape, **clip_param)
                self.addOperand(onnx_node.name, clip_op, output_shape, TensorType.ACTIVATION)
            else:
                raise RuntimeError("Not support clip min not zero case (min: {})".format(clip_param.get("min")))

    def convert_depth_to_space_op(self, onnx_node):
        assert(onnx_node.op_type == "DepthToSpace")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        upscale_factor = onnx_node.attrs['blocksize']
        mode = onnx_node.attrs.get("mode", "DCR")
        on = input_shape[0]
        oc = input_shape[1] / upscale_factor**2
        oh = upscale_factor * input_shape[2]
        ow = upscale_factor * input_shape[3]
        output_shape = [on, int(oc), oh, ow]
        operands = [op]
        attr={
            'upscale_factor': upscale_factor,
            'mode': mode,
        }
        pixel_shuffle_op = self.CVI.add_pixelshuffle_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
        self.addOperand(onnx_node.name, pixel_shuffle_op, output_shape, TensorType.ACTIVATION)

    def convert_div_op(self, onnx_node):
        assert(len(onnx_node.inputs) == 2)
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = data1 / data2
            if data1.dtype == data2.dtype:
                output_data = output_data.astype(data1.dtype)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        elif (len(input_shape2) ==1 and input_shape2[0] == 1) or \
            (len(input_shape2) == 0):
            # div(x) = input * (1/x) = scale(1/x) = input * (1/x) + 0
            operands = list()
            operands.append(op1)

            div_value = self.getTensor(onnx_node.inputs[1]).tensor_data
            tensor_data = np.full(input_shape1[1], 1 / (div_value * 1.0)) # broadcast via channel
            weight_name = "{}_div_weight".format(onnx_node.inputs[0])
            self.addTensor(weight_name, tensor_data, tensor_data.shape)
            op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
            operands.append(op2)

            tensor_data = np.full((input_shape1[1]), 0) # broadcast via channel
            bias_name = "{}_div_bias".format(onnx_node.inputs[0])
            self.addTensor(bias_name, tensor_data, tensor_data.shape)
            op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
            operands.append(op3)

            output_shape = input_shape1

            scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)

        else:
            raise RuntimeError("not implement yet, shape1 {}, shape2 {}",
                input_shape1, input_shape2)

    def convert_equal_op(self, onnx_node):
        assert(onnx_node.op_type == "Equal")
        assert(len(onnx_node.inputs) == 2)

        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])

        assert(input_shape0 == input_shape1)

        if tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR:
            # both are weight, do it offline
            tensor_data0 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data1 = self.getTensor(onnx_node.inputs[1]).tensor_data
            tensor_data = np.equal(tensor_data0, tensor_data1)

            self.addTensor(onnx_node.name, tensor_data, list(tensor_data.shape))
            self.addOperand(onnx_node.name, None, list(tensor_data.shape), TensorType.TENSOR)
        else:
            raise RuntimeError("not implement yet")

    def convert_expand_op(self, onnx_node):
        assert(onnx_node.op_type == "Expand")
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])
        operands = list()
        if tensor_type1 == TensorType.TENSOR:
            operands = list()
            operands.append(op0)
            tensor_data = self.getTensor(onnx_node.inputs[1]).tensor_data
            if len(tensor_data) == 6 and tensor_data[4] == input_shape0[4]:
                assert(input_shape0[3::2] == [1, 1])
                assert(np.any(tensor_data[:3] - input_shape0[:3]) == False)
                # input shape like 1, 128, 13, 1, 13, 1
                # new shape like 1. 128.  13.   2.  13.   2
                # input chould reshape (1*128*13, 1, 13, 1) -> (1*128*13, 1, 13, 1)
                # and seperate w-expand than h expand
                # 1. (1*128*13, 1, 13, 1) -> (1*128*13, 1, 13, 2)
                # 2. reshape from (1*128*13, 1, 13, 2) to (1*128*13, 1, 1, 13 * 2)
                # 3. (1*128*13, 1, 1, 13 * 2) -> (1*128*13, 1, 2, 13 * 2)

                # reshape input
                input_shape = [int(np.prod(tensor_data[:3]))] + input_shape0[3:]
                reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, "reshape"),
                    operands, input_shape)

                self.addOperand(onnx_node.name, reshape_op, input_shape, TensorType.ACTIVATION)

                operands = list()
                operands.append(reshape_op)

                attr={
                    'scale_h': 1,
                    'scale_w': int(tensor_data[5])
                }
                output_shape = input_shape
                output_shape[3] *= attr['scale_w']
                # 1, extend w
                upsample_op = self.CVI.add_upsample_op("{}_{}_w".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
                self.addOperand(onnx_node.name, upsample_op, output_shape, TensorType.ACTIVATION)
                # 2, reshape (h,w) to (1, hw)
                input_shape = output_shape[:2] + [1, np.prod(output_shape[2:])]
                reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, "reshape_w"),
                    [upsample_op], input_shape)

                self.addOperand(onnx_node.name, reshape_op, input_shape, TensorType.ACTIVATION)

                # 3, extend h
                operands = list()
                operands.append(reshape_op)

                attr={
                    'scale_h': int(tensor_data[3]),
                    'scale_w': 1
                }
                output_shape = input_shape
                output_shape[2] *= attr['scale_h']

            else:
                assert(len(tensor_data)==4)
                assert(input_shape0[2] == 1 and input_shape0[3] == 1)
                assert(input_shape0[0] == tensor_data[0] and input_shape0[1] == tensor_data[1])
                assert(tensor_data[2] == tensor_data[3])
                output_shape = list(tensor_data)
                attr={
                    'scale_h': int(tensor_data[2]),
                    'scale_w': int(tensor_data[2])
                }
            upsample_op = self.CVI.add_upsample_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
            self.addOperand(onnx_node.name, upsample_op, output_shape, TensorType.ACTIVATION)
        else:
            raise RuntimeError("not implement yet")

    def convert_flatten_op(self, onnx_node):
        assert(onnx_node.op_type == "Flatten")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        operands = list()
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            new_shape = (1, -1)
            output_data = np.reshape(tensor_data, new_shape)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            axis = onnx_node.attrs.get("axis", 1) # Default is 1
            if axis != 1:
                raise AttributeError("TODO: axis != 1 case")

            operands.append(op)
            reduce_shape = functools.reduce(operator.mul, input_shape[1:])
            output_shape = [input_shape[0], reduce_shape]
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, reshape_op, output_shape, TensorType.ACTIVATION)

    def convert_gather_op(self, onnx_node):
        """
            first input is tensor data, second input is constant
        """
        assert(onnx_node.op_type == "Gather")
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        axis = onnx_node.attrs.get('axis', 0)

        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            input_data =  self.getTensor(onnx_node.inputs[0]).tensor_data
            gather_indices = self.getTensor(onnx_node.inputs[1]).tensor_data
            new_data = np.take(input_data, gather_indices.astype(np.int64), axis=axis)
            self.addTensor(onnx_node.name, new_data, list(new_data.shape))
            self.addOperand(onnx_node.name, None, list(new_data.shape), TensorType.TENSOR)
        else:
            raise("TODO: Our Ir not support gather function")


    def convert_gemm_op(self, onnx_node):
        assert(onnx_node.op_type == "Gemm")
        #(M, K) * (K, N) => (M, N)
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])

        operands = list()
        operands.append(op)
        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_shape = weight_tensor.shape
        if onnx_node.attrs.get('transA', 1) == 0 and onnx_node.attrs.get('transB', 1) == 0:
            # mlir require second is transposed
            print("transpose b for mlir require", onnx_node.attrs, type(weight_shape))
            assert(len(weight_shape) == 2 and "shape should be 2 dim")
            weight_shape.reverse()
            weight_tensor_data = weight_tensor.tensor_data
            weight_tensor = np.ascontiguousarray(np.transpose(weight_tensor_data, (1, 0)))
            self.addTensor(weight_name, weight_tensor, weight_shape)

        weight_op = self.CVI.add_load_file_op(weight_name, weight_shape)
        operands.append(weight_op)

        bias_name = onnx_node.inputs[2]
        bias_tensor = self.getTensor(bias_name)
        bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
        operands.append(bias_op)

        M = input_shape[0]
        K = input_shape[1]
        N = bias_tensor.shape[0]
        output_shape = [M, N]
        fc_op = self.CVI.add_fully_connected_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, fc_op, output_shape, TensorType.ACTIVATION)

    def convert_global_pool_op(self, onnx_node):
        assert(onnx_node.op_type == "GlobalAveragePool" or onnx_node.op_type == "GlobalMaxPool")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        on = input_shape[0]
        oc = input_shape[1]

        pool_2d_param = {
            'stride_h':  1,
            'stride_w':  1,
            'kernel_h':  input_shape[2],
            'kernel_w':  input_shape[3],
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
            'count_include_pad': True,
        }
        output_shape = [int(on), int(oc), 1, 1]
        if onnx_node.op_type == "GlobalAveragePool":
            pool_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_2d_param)
        elif onnx_node.op_type == "GlobalMaxPool":
            pool_op = self.CVI.add_pool_max_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_2d_param)
        self.addOperand(onnx_node.name, pool_op, output_shape, TensorType.ACTIVATION)

    def convert_gru_op(self, onnx_node):
        assert(onnx_node.op_type == "GRU")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        seq_length, batch_size, _ = input_shape
        if batch_size > 1:
            raise RuntimeError("GRU does not support batch inference so far.")

        operands = list()
        operands.append(op)

        linear_before_reset = True if onnx_node.attrs.get("linear_before_reset", 1) == 1 else False
        bidirectional = True if onnx_node.attrs.get("direction", 'forward') == 'bidirectional' else False
        gru_param = {
            'linear_before_reset': bool(linear_before_reset),
            'bidirectional': bool(bidirectional),
        }

        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
        operands.append(weight_op)

        recurrence_name = onnx_node.inputs[2]
        recurrence_tensor = self.getTensor(recurrence_name)
        recurrence_op = self.CVI.add_load_file_op(recurrence_name, recurrence_tensor.shape)
        operands.append(recurrence_op)

        bias_name = onnx_node.inputs[3]
        if len(bias_name) != 0:
            bias_tensor = self.getTensor(bias_name)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        if len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("GRU does not test the case of specify the sequence_lens.")

        initial_h_name = onnx_node.inputs[5]
        if len(initial_h_name) != 0:
            _, _, tensor_type = self.getOperand(initial_h_name)

            if tensor_type == TensorType.TENSOR:
                initial_h_tensor = self.getTensor(initial_h_name)
                initial_h_op = self.CVI.add_load_file_op(initial_h_name, initial_h_tensor.shape)
                operands.append(initial_h_op)
                # initial_h shape = [num_directions, batch_size, hidden_size]
                # output shape = [seq_length, num_directions, batch_size, hidden_size]
                output_shape = [seq_length]
                output_shape.extend(initial_h_tensor.shape)
            else:
                raise RuntimeError("GRU only support initial_h from activation currently.")

        gru_op = self.CVI.add_gru_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **gru_param)
        self.addOperand(onnx_node.name, gru_op, output_shape, TensorType.ACTIVATION)

    def convert_leaky_relu_op(self, onnx_node):
        assert(onnx_node.op_type == "LeakyRelu")
        alpha = onnx_node.attrs.get("alpha", 0.01)
        leaky_relu_param = {
            'negative_slope': float(alpha),
        }
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = y = np.clip(tensor_data, 0, np.inf) + np.clip(tensor_data, -np.inf, 0) * alpha
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            output_shape = input_shape
            leaky_relu_op = self.CVI.add_leaky_relu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **leaky_relu_param)
            self.addOperand(onnx_node.name, leaky_relu_op, output_shape, TensorType.ACTIVATION)

    def convert_lrn_op(self, onnx_node):
        assert(onnx_node.op_type == "LRN")
        alpha = onnx_node.attrs.get('alpha', 0.0001)
        beta = onnx_node.attrs.get('beta', 0.75)
        bias = onnx_node.attrs.get('bias', 1.0)
        size = onnx_node.attrs['size']
        lrn_param = {
            'alpha': alpha, 'beta': beta, 'bias': bias, 'size': size,
        }
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            x = self.getTensor(onnx_node.inputs[0]).tensor_data
            square_sum = np.zeros(input_shape).astype(np.float32)
            for n, c, h, w in np.ndindex(input_shape):
                square_sum[n, c, h, w] = sum(x[n, max(0, c - int(math.floor((size - 1) / 2))):
                                               min(size, c + int(math.ceil((size - 1) / 2)) + 1), h, w] ** 2)
            output_data = x / ((bias + (alpha / size) * square_sum) ** beta)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            output_shape = input_shape
            lrn_op = self.CVI.add_lrn_op("{}_{}".format(
                onnx_node.name, onnx_node.op_type), operands, output_shape, **lrn_param)
            self.addOperand(onnx_node.name, lrn_op, output_shape, TensorType.ACTIVATION)

    def convert_lstm_op(self, onnx_node):
        assert(onnx_node.op_type == "LSTM")
        assert(len(onnx_node.inputs) == 7)
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        seq_length, batch_size, _ = input_shape
        hidden_size = onnx_node.attrs.get("hidden_size")
        if batch_size > 1:
            raise RuntimeError("LSTM does not support batch inference so far.")

        operands = list()
        operands.append(op)
        bidirectional = True if onnx_node.attrs.get("direction", 'forward') == 'bidirectional' else False
        num_direction = 1 if not bidirectional else 2
        lstm_param = {
            'bidirectional': bool(bidirectional),
        }

        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
        operands.append(weight_op)

        recurrence_name = onnx_node.inputs[2]
        recurrence_tensor = self.getTensor(recurrence_name)
        recurrence_op = self.CVI.add_load_file_op(recurrence_name, recurrence_tensor.shape)
        operands.append(recurrence_op)

        bias_name = onnx_node.inputs[3]
        if len(bias_name) != 0:
            bias_tensor = self.getTensor(bias_name)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        if len(onnx_node.inputs[4]) != 0:
            raise RuntimeError("LSTM does not test the case of specify the sequence_lens.")

        initial_h_name = onnx_node.inputs[5]
        if len(initial_h_name) != 0:
            _, _, tensor_type = self.getOperand(initial_h_name)

            if tensor_type == TensorType.TENSOR:
                initial_h_tensor = self.getTensor(initial_h_name)
                initial_h_op = self.CVI.add_load_file_op(initial_h_name, initial_h_tensor.shape)
                operands.append(initial_h_op)
                # initial_h shape = [num_directions, batch_size, hidden_size]
            else:
                raise RuntimeError("LSTM only support initial_h from activation currently.")

        initial_c_name = onnx_node.inputs[6]
        if len(initial_c_name) != 0:
            _, _, tensor_type = self.getOperand(initial_c_name)

            if tensor_type == TensorType.TENSOR:
                initial_c_tensor = self.getTensor(initial_c_name)
                initial_c_op = self.CVI.add_load_file_op(initial_c_name, initial_c_tensor.shape)
                if initial_h_name == initial_c_name :
                    operands.append(initial_h_op)
                else :
                    operands.append(initial_c_op)
            else:
                raise RuntimeError("LSTM only support initial_c from activation currently.")

        # output = all the intermediate output values of the hidden + output value of the cell
        output_shape = [seq_length + 1, num_direction, batch_size, hidden_size]

        lstm_op = self.CVI.add_lstm_op("{}_{}_cell_hidden".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **lstm_param)
        self.addOperand("{}_cell_hidden".format(onnx_node.name), lstm_op, output_shape, TensorType.ACTIVATION)

        attr = {"axis": 0, "offset": 0}
        output_shape = [seq_length, num_direction, batch_size, hidden_size]
        slice_op_hidden = self.CVI.add_slice_op("{}_{}".format(onnx_node.outputs[0], onnx_node.op_type), [lstm_op], output_shape, **attr)
        self.addOperand(onnx_node.outputs[0], slice_op_hidden, output_shape, TensorType.ACTIVATION)

        attr = {"axis": 0, "offset": seq_length}
        output_shape = [num_direction, batch_size, hidden_size]
        slice_op_cell = self.CVI.add_slice_op("{}_{}".format(onnx_node.outputs[2], onnx_node.op_type), [lstm_op], output_shape, **attr)
        self.addOperand(onnx_node.outputs[2], slice_op_cell, output_shape, TensorType.ACTIVATION)

    def convert_matmul_op(self, onnx_node):
        assert(onnx_node.op_type == "MatMul")
        # Use fully connectly op, set bias is zero
        #(M, K) * (K, N) => (M, N)
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])

        operands = list()
        operands.append(op)

        weight_name = "{}_add_weight".format(onnx_node.name)
        weight_tensor = self.getTensor(onnx_node.inputs[1]).tensor_data

        # batched matmul like (bs, K, M ) * (bs, M, N) = (bs, K, N)
        if len(weight_tensor.shape) == 3 and len(input_shape) == 3:
            # in onnx matmul data is put in (K,N), but mlir put in (N, K)
            weight_tensor = np.ascontiguousarray(np.transpose(weight_tensor, (0, 2, 1)))
            weight_shape = list(weight_tensor.shape)
            self.addTensor(weight_name, weight_tensor, weight_shape)
            weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
            operands.append(weight_op)

            bias_tensor = np.full(weight_tensor.shape[0], 0)
            bias_name = "{}_add_bias".format(onnx_node.name)
            self.addTensor(bias_name, bias_tensor, bias_tensor.shape)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

            B = input_shape[0]
            M = input_shape[1]
            K = input_shape[2]
            if input_shape[2] != weight_tensor.shape[2]:
                raise RuntimeError("{} vs {} can not matmul".format(input_shape, weight_tensor.shape))
            N = weight_tensor.shape[1]
            output_shape = [B, M, N]
            fc_op = self.CVI.add_fully_connected_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, fc_op, output_shape, TensorType.ACTIVATION)
        else:
            # in onnx matmul data is put in (K,N), but mlir put in (N, K)
            weight_tensor = np.ascontiguousarray(np.transpose(weight_tensor, (1, 0)))
            weight_shape = list(weight_tensor.shape)

            self.addTensor(weight_name, weight_tensor, weight_shape)
            weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
            operands.append(weight_op)

            bias_tensor = np.full(weight_tensor.shape[0], 0)
            bias_name = "{}_add_bias".format(onnx_node.name)
            self.addTensor(bias_name, bias_tensor, bias_tensor.shape)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

            M = input_shape[0]
            K = input_shape[1]
            if input_shape[1] != weight_tensor.shape[1]:
                raise RuntimeError("{} vs {} can not matmul".format(input_shape, weight_tensor.shape))
            N = weight_tensor.shape[0]
            output_shape = [M, N]

            fc_op = self.CVI.add_fully_connected_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, fc_op, output_shape, TensorType.ACTIVATION)

    def convert_maxpool_op(self, onnx_node):
        assert(onnx_node.op_type == "MaxPool")
        pads = onnx_node.attrs.get("pads",[0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1])

        pool_max_2d_param = {
            'stride_h': strides[0],
            'stride_w': strides[1],
            'kernel_h': onnx_node.attrs['kernel_shape'][0],
            'kernel_w': onnx_node.attrs['kernel_shape'][1],
            'padding_t': pads[0],
            'padding_l': pads[1],
            'padding_b': pads[2],
            'padding_r': pads[3],
            'do_relu': False,
        }

        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        on = input_shape[0]
        oc = input_shape[1]
        oh = calcPool2DFloor(input_shape[2], onnx_node.attrs['kernel_shape'][0], strides[0], pads[0], pads[2])
        ow = calcPool2DFloor(input_shape[3], onnx_node.attrs['kernel_shape'][1], strides[1], pads[1], pads[3])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_max_2d_param)
        self.addOperand(onnx_node.name, pool_max_op, output_shape, TensorType.ACTIVATION)

    def convert_max_op(self, onnx_node):
        assert(onnx_node.op_type == "Max")
        input_num = len(onnx_node.inputs)
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape
        if tensor_type == TensorType.TENSOR:
            for idx, _ in enumerate(onnx_node.inputs):
                if idx == 0: # first op skip
                    output_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                else:
                    _, _, tensor_type = self.getOperand(onnx_node.inputs[idx])
                    if tensor_type != TensorType.TENSOR:
                        raise RuntimeError("Wrong type")
                    tensor_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                    output_data = np.maximum(output_data, tensor_data)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        else:
            if input_num == 1:
                self.addOperand(onnx_node.name, op, output_shape, TensorType.ACTIVATION)
                return
            operands = list()
            operands.append(op)
            for index in range(1, input_num):
                op_i, input_shape_i, tensor_type_i = self.getOperand(onnx_node.inputs[index])
                operands.append(op_i)
                #broadcast not support now
                assert(input_shape_i == input_shape)
            max_op = self.CVI.add_eltwise_max_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, max_op, output_shape, TensorType.ACTIVATION)

    def convert_min_op(self, onnx_node):
        assert(onnx_node.op_type == "Min")
        input_num = len(onnx_node.inputs)
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape
        if tensor_type == TensorType.TENSOR:
            for idx, _ in enumerate(onnx_node.inputs):
                if idx == 0: # first op skip
                    output_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                else:
                    _, _, tensor_type = self.getOperand(onnx_node.inputs[idx])
                    if tensor_type != TensorType.TENSOR:
                        raise RuntimeError("Wrong type")
                    tensor_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                    output_data = np.minimum(output_data, tensor_data)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        else:
            if input_num == 1:
                self.addOperand(onnx_node.name, op, output_shape, TensorType.ACTIVATION)
                return
            operands = list()
            operands.append(op)
            for index in range(1, input_num):
                op_i, input_shape_i, tensor_type_i = self.getOperand(onnx_node.inputs[index])
                operands.append(op_i)
                #broadcast not support now
                assert(input_shape_i == input_shape)
            min_op = self.CVI.add_eltwise_min_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, min_op, output_shape, TensorType.ACTIVATION)

    def convert_mul_op(self, onnx_node):
        assert(onnx_node.op_type == "Mul")
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        operands = list()
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = tensor_data1 * tensor_data2
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
            return

        elif tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            # constant
            # x * constant + 0
            channel = input_shape1[1]
            mul_value = self.getTensor(onnx_node.inputs[1]).tensor_data

            if len(mul_value.flatten()) == 1:
                weight_data = np.full(channel, mul_value.flatten()[0]) # broadcast via channel
            elif len(mul_value.flatten()) == channel:
                weight_data = mul_value
            else:
                raise RuntimeError("could not broadcast input array from shape {} into shape {}".format(input_shape1, input_shape2))

            weight_name = "{}_mul_weight".format(onnx_node.inputs[0])
            weight_shape = list(weight_data.shape)
            self.addTensor(weight_name, weight_data, weight_shape)
            bias_data = np.full(channel, 0)
            weight_op = self.CVI.add_load_file_op(weight_name, weight_shape)
            bias_name = "{}_mul_bias".format(onnx_node.inputs[0])
            bias_shape = list(bias_data.shape)
            self.addTensor(bias_name, bias_data, bias_shape)
            bias_op = self.CVI.add_load_file_op(bias_name, bias_shape)
            operands.append(op1)
            operands.append(weight_op)
            operands.append(bias_op)
            output_shape = input_shape1
            scale_op = self.CVI.add_scale_op("{}_{}".format(
                onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, scale_op,
                            output_shape, TensorType.ACTIVATION)

        else:
            if input_shape1 == input_shape2:
                #eltwise mul
                output_shape = input_shape1
                mul_op = self.CVI.add_eltwise_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op1, op2], output_shape)
            else:
                # broadcast mul
                # TODO: only support broadcast mul channel axis now
                # [n, c, h, w] broadcast with [n, c]
                if np.prod(input_shape2) > np.prod(input_shape1):
                    # swap
                    op1, op2 = op2, op1
                    input_shape1, input_shape2 = input_shape2, input_shape1

                if len(input_shape1) == 3 and len(input_shape2) == 3 and \
                  input_shape2[2] == 1:
                    # hoist shape, [1, 507, 80] vs [1, 507, 1] could reshape to
                    # 1, 507, 80, 1 and 1, 507, 1, 1
                    input_shape1.append(1)
                    input_shape2.append(1)

                elif len(input_shape1) == 4:
                    if np.prod(input_shape2) == input_shape1[0] * input_shape1[1]:
                        # convert to scale op, e.g.
                        #   [4,3,28,28] x [1,3] => [4,3,28,28]
                        #   [4,3,28,28] x [4,3] => [4,3,28,28]
                        pass
                    elif (len(input_shape2) == 4 and input_shape1[2:] == input_shape2[2:] and
                            input_shape2[1] == 1 and (input_shape1[0] == input_shape2[0] or input_shape2[0] == 1)):
                        # bcast mul, e.g.
                        #   [4,3,28,28] x [4,1,28,28] => [4,3,28,28]
                        #   [4,3,28,28] x [1,1,28,28] => [4,3,28,28]
                        pass
                    else:
                        raise RuntimeError("{} vs {}  broadcast mul not support".format(
                            input_shape1, input_shape2))
                axis = 1
                output_shape = input_shape1
                mul_op = self.CVI.add_broadcast_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op1, op2], output_shape, axis=axis)

            self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)

    def convert_neg_op(self, onnx_node):
        assert(onnx_node.op_type == "Neg")
        # y = x * (-1) + 0
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.negative(tensor_data)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            # weight (-1)
            tensor_data = np.full(input_shape[1], -1) # broadcast via channel
            weight_name = "{}_add_weight".format(onnx_node.name)
            self.addTensor(weight_name, tensor_data, tensor_data.shape)
            op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
            operands.append(op2)
            # bias (0)
            bias_data = np.full(input_shape[1], 0)
            bias_name = "{}_add_bias".format(onnx_node.name)
            self.addTensor(bias_name, bias_data, bias_data.shape)
            op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
            operands.append(op3)

            output_shape = input_shape
            scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)

    def convert_pad_op(self, onnx_node):
        assert(onnx_node.op_type == "Pad")

        # get pad mode
        mode = onnx_node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode != "constant": raise RuntimeError("Todo support pad op mode {}".format(mode))

        # opset 11, value from second input
        if len(onnx_node.inputs) > 2:
            constant_value = self.getTensor(onnx_node.inputs[2]).tensor_data
        else:
            constant_value = onnx_node.attrs.get("value", 0.0)

        op, input_shape, input_type = self.getOperand(onnx_node.inputs[0])

        if len(onnx_node.inputs) > 1:
            # padding data from input
            _, _, pad_data_type = self.getOperand(onnx_node.inputs[1])
            if pad_data_type == TensorType.TENSOR:
                pads = list(self.getTensor(onnx_node.inputs[1]).tensor_data)
            else:
                raise RuntimeError("not support paddings data with runtime data")
        else:
            pads = onnx_node.attrs.get("pads")
            if pads == None:
                raise RuntimeError("No paddings value")
        if len(pads) != 2 * len(input_shape):
            raise RuntimeError("pads number is two times as same as input shape ({} v.s 2 * {})".format(len(pads), len(input_shape)))

        # fuesd if padding all zero
        if all(i == 0 for i in pads):
            print("All pad is zero ({}), Fuse padding op {}".format(pads, onnx_node.name))
            self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)
            return

        dims = len(input_shape)
        np_pads = tuple(zip(pads[:dims], pads[dims:]))
        pads_param = {
          "pads": pads,
          "const_val": constant_value,
        }

        output_shape = np.sum([input_shape, pads[:dims], pads[dims:]], axis=0)
        output_shape = [int(i) for i in output_shape]
        if input_type == TensorType.TENSOR :
            input_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            if mode == b'constant':
              output_data = np.pad(input_data, np_pads, 'constant', constant_values=constant_value)
            elif mode == b'reflect':
              output_data = np.pad(input_data, np_pads, 'reflect')
            else:
              output_data = np.pad(input_data, np_pads, 'edge')
            self.addTensor(onnx_node.name, output_data, list(output_shape))
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            pads_op = self.CVI.add_pad_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pads_param)
            self.addOperand(onnx_node.name, pads_op, output_shape, TensorType.ACTIVATION)

    def convert_prelu_op(self, onnx_node):
        assert(onnx_node.op_type == "PRelu")
        if len(onnx_node.inputs) != 2:
            raise ValueError("{} must equal to 2".format(onnx_node.op_type))
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        operands = list()
        operands.append(op1)
        output_shape = input_shape1
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = np.clip(tensor_data1, 0, np.inf) + \
                np.clip(tensor_data2, -np.inf, 0) * tensor_data2
            output_shape = output_data.shape
            self.addTensor(onnx_node.name, output_data, list(output_shape))
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        elif tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            slope = self.getTensor(onnx_node.inputs[1])
            slope_data = slope.tensor_data
            slope_name = "{}_slope_weight".format(onnx_node.name)
            slope_shape = slope.shape
            self.addTensor(slope_name, slope_data, slope_shape)
            slope_op = self.CVI.add_load_file_op(slope_name, slope_shape)
            operands.append(slope_op)
            prelu_op = self.CVI.add_prelu_op("{}_{}".format(
                onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, prelu_op,
                            output_shape, TensorType.ACTIVATION)
        else:
            operands.append(op2)
            prelu_op = self.CVI.add_prelu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, prelu_op,
                            output_shape, TensorType.ACTIVATION)

    def convert_reciprocal_op(self, onnx_node):
        assert(onnx_node.op_type == "Reciprocal")
        if len(onnx_node.inputs) != 1:
            raise ValueError("{} must only one input".format(onnx_node.op_type))
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])

        if tensor_type1 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = 1.0 / tensor_data1
            output_shape = output_data.shape
            self.addTensor(onnx_node.name, output_data, list(output_shape))
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op1)
            output_shape = input_shape1
            relu_op = self.CVI.add_reciprocal_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, relu_op, output_shape, TensorType.ACTIVATION)

    def convert_relu_op(self, onnx_node):
        assert(onnx_node.op_type == "Relu")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.clip(tensor_data, 0, np.inf)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            output_shape = input_shape
            relu_op = self.CVI.add_relu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, relu_op, output_shape, TensorType.ACTIVATION)

    def convert_reshape_op(self, onnx_node):
        assert(onnx_node.op_type == "Reshape")
        """
            first input is tensor data, second input is constant
        """
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        _, _, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        output_shape = list()
        operands = list()
        operands.append(op1)
        if tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            t = self.getTensor(onnx_node.inputs[1])
            output_shape = list(t.tensor_data.flatten())

            if -1 in output_shape:
                # At most one dimension of the new shape can be -1.
                # In this case, the value is inferred from the size of the tensor and the remaining dimensions
                # ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
                total_tensor_size = get_shape_size(input_shape1)
                remain_dim = output_shape.index(-1)
                tmp_size = 1
                for i in range(len(output_shape)):
                    if output_shape[i] == 0:
                        output_shape[i] = self.batch_size
                    if i != remain_dim:
                        tmp_size*=output_shape[i]
                remain_size  = total_tensor_size / tmp_size
                if not remain_size.is_integer():
                    raise RuntimeError("{} not divide exactly by {}".format(total_tensor_size, tmp_size))
                output_shape[remain_dim] = remain_size

            output_shape = [int(x) for x in output_shape]

            if len(output_shape) ==6:
                # Pixel Shuffle
                self.addOperand(onnx_node.name, op1, output_shape, TensorType.ACTIVATION)
                return
            if np.prod(input_shape1) != np.prod(output_shape):
                raise RuntimeError("can not reshape {} v.s. {}".format(input_shape1, output_shape))


            if output_shape == input_shape1:
                # same shape, fuse this op
                self.addOperand(onnx_node.name, op1, output_shape, TensorType.ACTIVATION)
                return
            else:
                reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                self.addOperand(onnx_node.name, reshape_op, output_shape, TensorType.ACTIVATION)
        elif tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            shape_data = self.getTensor(onnx_node.inputs[1]).tensor_data.astype(np.int)
            output_data = np.reshape(tensor_data, shape_data)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None,
                            output_shape, TensorType.TENSOR)
        else:
            raise RuntimeError("Second type must be {}".format(TensorType.TENSOR))

    def convert_resize_op(self, onnx_node):
        assert(onnx_node.op_type == "Resize")
        mode = onnx_node.attrs.get("mode", "nearest")

        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        if len(onnx_node.inputs) > 2:
            # onnx opset 11
            scale_factor = self.getTensor(onnx_node.inputs[2]).tensor_data
            if len(scale_factor) == 0:
                # size
                sizes = self.getTensor(onnx_node.inputs[3]).tensor_data
                use_size = True
            else:
                use_size = False

        else:
            # opset 10
            scale_factor = self.getTensor(onnx_node.inputs[1]).tensor_data
            use_size = False

        if mode == b'linear':
            coordinate_transformation_mode = \
                onnx_node.attrs.get(
                    "coordinate_transformation_mode", "half_pixel")
            if not use_size:
                sizes = input_shape * scale_factor
            on, oc, oh, ow = sizes
            attr = {
                'height': int(oh),
                'width': int(ow),
                'pad_beg': 0,
                'pad_end': 0,
                'shrink_factor': 0,
                'zoom_factor': 0,
                'coordinate_transformation_mode': coordinate_transformation_mode
            }

            output_shape = [int(i) for i in [on, oc, oh, ow]]

            interp_op = self.CVI.add_interp_op(
                "{}_{}".format(onnx_node.name, onnx_node.op_type), [op], output_shape, **attr)
            self.addOperand(onnx_node.name, interp_op,
                            output_shape, TensorType.ACTIVATION)
            return

        elif mode == b"nearest":
            operands = [op]
            ic = input_shape[1]
            ih = input_shape[2]
            iw = input_shape[3]
            on = int(input_shape[0])
            oc = int(input_shape[1])
            oh = int(sizes[2]) if use_size else int(
                input_shape[2] * scale_factor[2])
            ow = int(sizes[3]) if use_size else int(
                input_shape[3] * scale_factor[3])
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
            weight_name = "{}_add_weight".format(onnx_node.name)
            self.addTensor(weight_name, tensor_data, tensor_data.shape)
            weight_op = self.CVI.add_load_file_op(
                weight_name, tensor_data.shape)
            operands.append(weight_op)

            deconv_op = self.CVI.add_deconv_op("{}_{}".format(
                onnx_node.name, onnx_node.op_type), operands, output_shape, **deconv_param)
            self.addOperand(onnx_node.name, deconv_op,
                            output_shape, TensorType.ACTIVATION)

        else:
            raise RuntimeError("Unsupported mode {}".format(mode))

    def convert_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "Shape")
        _, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        data = np.array(input_shape)
        self.addTensor(onnx_node.name, data, list(data.shape))
        self.addOperand(onnx_node.name, None, list(
            data.shape), TensorType.TENSOR)

    def convert_slice_op(self, onnx_node):
        assert(onnx_node.op_type == "Slice")
        # check if upper op is pad op
        # if it is, pass to next layer
        try:
            pad_op_data = self.getTensor(
                "{}_pad".format(onnx_node.inputs[0])).tensor_data
            self.addTensor("{}_pad".format(onnx_node.name), pad_op_data, None)
        except KeyError:
            # not pad op, pass
            pass

        op, input_shape, tesnor_type = self.getOperand(onnx_node.inputs[0])
        # start
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[1])
        if _tesnor_type != TensorType.TENSOR:
            raise TypeError(
                "{} start type be tensor, not find".format(onnx_node.name))
        else:
            starts = self.getTensor(onnx_node.inputs[1]).tensor_data
        # ends
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[2])
        if _tesnor_type != TensorType.TENSOR:
            raise TypeError(
                "{} end type be tensor, not find".format(onnx_node.name))
        else:
            ends = self.getTensor(onnx_node.inputs[2]).tensor_data

        # axes
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[3])
        if _tesnor_type != TensorType.TENSOR:
            raise TypeError(
                "{} axes type be tensor, not find".format(onnx_node.name))
        else:
            axes = self.getTensor(onnx_node.inputs[3]).tensor_data

        steps = [1]
        assert(len(starts) == len(ends))
        assert(len(axes) == len(ends))

        if len(onnx_node.inputs) == 5:
            # steps
            _, _, _tesnor_type = self.getOperand(onnx_node.inputs[4])
            if _tesnor_type != TensorType.TENSOR:
                raise RuntimeError(
                    "{} steps type be tensor, not find".format(onnx_node.name))
            else:
                steps = self.getTensor(onnx_node.inputs[4]).tensor_data
                assert(len(steps) == 1)  # steps only has one value
                if steps[0] < 0:
                    raise RuntimeError("not support step < 0 case")
                if steps[0] > 1 and (len(axes) == 1 and axes[0] < 2):
                    raise RuntimeError("not support step > 1 and slice step with n/c")

                if steps[0] > 1:
                    # step eq as stride, leverage scale with stride
                    # only apply h/w, yolov5 case that step = [2] with axis = [2]
                    if len(ends) == 1 and ends[0] != np.iinfo(np.int64).max:
                        raise RuntimeError("not support end not set to max(np.iinfo(np.int64).max")
                    assert(tesnor_type != TensorType.TENSOR)
                    crop_shape = input_shape.copy()


                    _op = op
                    if starts[0] != 0:
                        # add slice to shift it
                        crop_offset = input_shape.copy()
                        idx = 0
                        for j in range(len(crop_shape)):
                            if j in axes:
                                ends[idx] = input_shape[j] if ends[idx] > input_shape[j] else ends[idx]
                                crop_shape[j] = ends[idx] - starts[idx]
                                crop_offset[j] = starts[idx]
                                idx += 1
                            else:
                                crop_shape[j] = input_shape[j]
                                crop_offset[j] = 0

                        crop_shape = [int(x) for x in crop_shape]
                        crop_offset = [int(x) for x in crop_offset]
                        crop_param = {
                            "crop_offset": list(crop_offset),
                            "crop_shape": list(crop_shape),
                        }

                        output_shape = crop_shape
                        crop_op = self.CVI.add_crop_op("{}_shift_{}".format(
                            onnx_node.name, onnx_node.op_type), [op], output_shape, **crop_param)
                        self.addOperand(onnx_node.name, crop_op,
                                        output_shape, TensorType.ACTIVATION)
                        _op = crop_op


                    crop_shape = input_shape.copy()

                    # lowering to scale with stride
                    on = crop_shape[0]
                    oc = crop_shape[1]
                    oc = crop_shape[1]
                    ic = oc

                    strides = [1, 1]
                    # 2 means rescale 4 dim index from 0,1,2,3 to only hw:0,1
                    strides[2 - axes[0]] = steps[0]

                    oh = calcPool2DFloor(crop_shape[2], 1, strides[0], 0, 0)
                    ow = calcPool2DFloor(crop_shape[3], 1, strides[1], 0, 0)

                    output_shape = [int(on), int(oc), int(oh), int(ow)]

                    conv_param = {
                        'stride_h':  strides[0],
                        'stride_w':  strides[1],
                        'padding': "VALID",
                        'dilation_h': 1,
                        'dilation_w': 1,
                        'padding_t': 0,
                        'padding_b': 0,
                        'padding_l': 0,
                        'padding_r': 0,
                        'group': ic,
                        'is_dw': True,
                        'with_bias': False,
                        'do_relu': False,
                        'ins': [],
                    }


                    weight_shape = [ic, 1, 1, 1, 1]
                    conv_tensor_data = np.full(weight_shape, 1)
                    weight_name = "{}_add_weight".format(onnx_node.name)
                    self.addTensor(weight_name, conv_tensor_data, conv_tensor_data.shape)
                    weight_op = self.CVI.add_load_file_op(weight_name, conv_tensor_data.shape)

                    operands = list()
                    operands.append(_op)
                    operands.append(weight_op)

                    conv_op = self.CVI.add_conv_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv_param)

                    self.addOperand(onnx_node.name, conv_op, output_shape, TensorType.ACTIVATION)

                    return

        if tesnor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            if len(axes) > 1:
                raise RuntimeError("Todo: Slice not support axes > 1 case")
            else:
                if steps[0] == -1:
                    output_data = tensor_data[starts[0]:ends[0]:steps[0]]
                else:
                    # slice
                    axis = int(axes[0])
                    start = int(starts[0])
                    end = ends[0] if ends[0] < np.iinfo(
                        np.int64).max else len(tensor_data) - 1
                    output_data = tensor_data.take(
                        indices=range(start, end), axis=axis)
                output_shape = list(output_data.shape)
                self.addTensor(onnx_node.name, output_data, output_shape)
                self.addOperand(onnx_node.name, None,
                                output_shape, TensorType.TENSOR)
            return
        else:
            crop_shape = input_shape.copy()
            crop_offset = input_shape.copy()
            idx = 0
            for j in range(len(crop_shape)):
                if j in axes:
                    ends[idx] = input_shape[j] if ends[idx] > input_shape[j] else ends[idx]
                    crop_shape[j] = ends[idx] - starts[idx]
                    crop_offset[j] = starts[idx]
                    idx += 1
                else:
                    crop_shape[j] = input_shape[j]
                    crop_offset[j] = 0

            crop_shape = [int(x) for x in crop_shape]
            crop_offset = [int(x) for x in crop_offset]
            crop_param = {
                "crop_offset": list(crop_offset),
                "crop_shape": list(crop_shape),
            }

            output_shape = crop_shape
            crop_op = self.CVI.add_crop_op("{}_{}".format(
                onnx_node.name, onnx_node.op_type), [op], output_shape, **crop_param)
            self.addOperand(onnx_node.name, crop_op,
                            output_shape, TensorType.ACTIVATION)

    def convert_softmax_op(self, onnx_node):
        assert(onnx_node.op_type == "Softmax")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape

        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.exp(data) / np.sum(np.exp(data), axis=(len(input_shape) - 1))
            output_shape = list(output_shape.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = [op]
            axis = len(input_shape) - 1
            for i in range(len(output_shape)):
                if output_shape[axis] == 1:
                    axis = axis -1
            softmax_param = {
                'axis': axis,
            }
            softmax_op = self.CVI.add_softmax_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **softmax_param)
            self.addOperand(onnx_node.name, softmax_op, output_shape, TensorType.ACTIVATION)

    def convert_skip_op(self, onnx_node):
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = data
            output_shape = input_shape
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)

    def convert_instancenorm_op(self, onnx_node):
        assert(onnx_node.op_type == "InstanceNormalization")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        epsilon = onnx_node.attrs.get('epsilon', 1e-5)

        # scale_value = self.getTensor(onnx_node.inputs[1]).tensor_data
        # bias_value = self.getTensor(onnx_node.inputs[2]).tensor_data

        scale_op = self.CVI.add_load_file_op(onnx_node.inputs[1], self.getTensor(onnx_node.inputs[1]).shape)
        bias_op = self.CVI.add_load_file_op(onnx_node.inputs[2], self.getTensor(onnx_node.inputs[2]).shape)

        operands.append(scale_op)
        operands.append(bias_op)

        output_shape = input_shape
        instancenorm_op = self.CVI.add_instancenorm_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, variance_epsilon=epsilon)
        self.addOperand(onnx_node.name, instancenorm_op, output_shape, TensorType.ACTIVATION)


    def convert_split_op(self, onnx_node):
        assert(onnx_node.op_type == "Split")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        split = onnx_node.attrs.get('split', [len(onnx_node.outputs)])
        if tensor_type == TensorType.TENSOR:

            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            if len(split) == 1:
                outputs = np.split(data, len(onnx_node.outputs))
                for i in onnx_node.outputs:
                    self.addTensor(str(i), outputs[i], list(outputs[i].shape))
                    self.addOperand(str(i), None, list(outputs[i].shape), TensorType.TENSOR)
            else:
                outputs = np.split(data, split)
                for i in onnx_node.outputs:
                    self.addTensor(str(i), outputs[i], list(outputs[i].shape))
                    self.addOperand(str(i), None, list(outputs[i].shape), TensorType.TENSOR)
        else:
            if len(input_shape) != 4 or axis != 1:
                raise RuntimeError("currently channel only, input must be 4")
            slice_num = len(split)
            offset = 0

            for i, name in zip(split, onnx_node.outputs):
                output_shape = [input_shape[0], i, input_shape[2], input_shape[3]]
                attr = {
                    "axis": 1,
                    "offset": offset
                 }
                slice_op = self.CVI.add_slice_op("{}_{}".format(
                    name, onnx_node.op_type), [op], output_shape, **attr)
                self.addOperand(name, slice_op,
                                output_shape, TensorType.ACTIVATION)
                offset = offset + i

    def convert_squeeze_op(self, onnx_node):
        assert(onnx_node.op_type == "Squeeze")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        checkKey(onnx_node.attrs, 'axes')
        axis_value_list = onnx_node.attrs['axes']
        if tensor_type == TensorType.ACTIVATION:
            new_shape = self.squeeze_shape(input_shape, axis_value_list)
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, new_shape)
            self.addOperand(onnx_node.name, reshape_op, new_shape, TensorType.ACTIVATION)
        else:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.squeeze(tensor_data, axis=axis_value_list[0])
            output_shape = list(tensor_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

    def convert_sqrt_op(self, onnx_node):
        assert(onnx_node.op_type == "Sqrt")
        _, _, tensor_type = self.getOperand(onnx_node.inputs[0])

        if tensor_type == TensorType.ACTIVATION:
            raise RuntimeError("Todo")
        else:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.sqrt(tensor_data)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None,
                            output_shape, TensorType.TENSOR)

    def convert_sub_op(self, onnx_node):
        assert(onnx_node.op_type == "Sub")
        # Y = X0 + X1*(-1)
        input_num = len(onnx_node.inputs)
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])


        output_shape = input_shape0
        if tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.ACTIVATION:
            if input_shape0 != input_shape1:
                # broadcast sub
                raise RuntimeError("Broadcast sub not support now")
            else:
                # eltwise sub
                # scale_op = X1 * (-1)
                operands1 = list()
                operands1.append(op1)
                tensor_data = np.full(input_shape1[1], -1) # broadcast via channel
                weight_name = "{}_add_weight".format(onnx_node.name)
                self.addTensor(weight_name, tensor_data, tensor_data.shape)
                op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
                operands1.append(op2)
                bias_data = np.full(input_shape1[1], 0)
                bias_name = "{}_add_bias".format(onnx_node.name)
                self.addTensor(bias_name, bias_data, bias_data.shape)
                op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
                operands1.append(op3)
                scale_op = self.CVI.add_scale_op("{}_scale_{}".format(onnx_node.name, onnx_node.op_type), operands1, output_shape)

                # add_op = X0 + scale_op
                operands0 = list()
                operands0.append(op0)
                operands0.append(scale_op)
                add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands0, output_shape)
                self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)
        elif tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.TENSOR:
            if np.prod(input_shape1) == 1:
                # constant
                # x * 1 + (-1) * constant
                constant_data = self.getTensor(onnx_node.inputs[1]).tensor_data
                weight_data = np.full(input_shape0[1], 1)
                weight_name = "{}_add_weight".format(onnx_node.name)
                weight_shape = list(weight_data.shape)
                self.addTensor(weight_name, weight_data, weight_shape)
                weight_op = self.CVI.add_load_file_op(
                    weight_name, weight_shape)

                bias_data = np.full(input_shape0[1], -1 * constant_data.flatten()[0])
                bias_name = "{}_add_bias".format(onnx_node.name)
                bias_shape = list(bias_data.shape)
                self.addTensor(bias_name, bias_data, bias_shape)
                bias_op = self.CVI.add_load_file_op(
                    bias_name, bias_shape)
                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op0, weight_op, bias_op], output_shape)
                self.addOperand(onnx_node.name, scale_op,
                                output_shape, TensorType.ACTIVATION)
            else:
                # broadcast with channel
                raise RuntimeError("TODO: broadcast sub with channel")

        elif tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR:
            tensor_data0 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data1 = self.getTensor(onnx_node.inputs[1]).tensor_data
            # sub
            output_data = tensor_data0 - tensor_data1
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

    def convert_sum_op(self, onnx_node):
        assert(onnx_node.op_type == "Sum")
        input_num = len(onnx_node.inputs)
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape
        if tensor_type == TensorType.TENSOR:
            for idx, _ in enumerate(onnx_node.inputs):
                if idx == 0: # first op skip
                    output_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                else:
                    _, _, tensor_type = self.getOperand(onnx_node.inputs[idx])
                    if tensor_type != TensorType.TENSOR:
                        raise RuntimeError("Wrong type")
                    tensor_data = self.getTensor(onnx_node.inputs[idx]).tensor_data
                    output_data = output_data + tensor_data
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)

        else:
            if input_num == 1:
                self.addOperand(onnx_node.name, op, output_shape, TensorType.ACTIVATION)
                return
            operands = list()
            operands.append(op)
            for index in range(1, input_num):
                op_i, input_shape_i, tensor_type_i = self.getOperand(onnx_node.inputs[index])
                operands.append(op_i)
                #broadcast not support now
                assert(input_shape_i == input_shape)
            sum_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, sum_op, output_shape, TensorType.ACTIVATION)

    def convert_transpose_op(self, onnx_node):
        assert(onnx_node.op_type == "Transpose")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        transpose_perm = onnx_node.attrs['perm']
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.transpose(tensor_data, transpose_perm)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            if transpose_perm == [0, 1, 4, 2, 5, 3]:
                # pixel shuffle
                if input_shape[2] != input_shape[3]:
                    raise ValueError("Pixel Shuffle Scale factor not same {} v.s.{}".format(input_shape[2], input_shape[3]))

                upscale_factor = input_shape[2]
                on = input_shape[0]
                oc = input_shape[1]
                oh = upscale_factor * input_shape[4]
                ow = upscale_factor * input_shape[5]
                output_shape = [on, oc, oh, ow]
                operands = [op]
                attr={
                    'upscale_factor': upscale_factor,
                    'mode': "CRD"
                }
                pixel_shuffle_op = self.CVI.add_pixelshuffle_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
                self.addOperand(onnx_node.name, pixel_shuffle_op, output_shape, TensorType.ACTIVATION)
            elif len(transpose_perm) == 4:
                # channel swap
                on = input_shape[transpose_perm[0]]
                oc = input_shape[transpose_perm[1]]
                oh = input_shape[transpose_perm[2]]
                ow = input_shape[transpose_perm[3]]
                output_shape = [on, oc, oh, ow]
                operands = [op]
                attr = {
                    'order0': transpose_perm[0],
                    'order1': transpose_perm[1],
                    'order2': transpose_perm[2],
                    'order3': transpose_perm[3],
                }

                is_0231 = transpose_perm == [0, 1, 3, 2]
                permute_0231_shape = list(output_shape)
                if is_0231:
                    # try to reshape it, current support
                    # 0231 : (N, C, H, W) -> (N, H, W, C)
                    # that could be reshaps as
                    # <n,c,h,w> -> <n*c, h, 1, w> -permute-> <n*c, 1, w, h>
                    # reshape
                    reshape = [np.prod(input_shape[:2]), input_shape[2], 1, input_shape[3]]
                    reshape_op = self.CVI.add_reshape_op("{}_{}_for_0231".format(
                        onnx_node.name, onnx_node.op_type), operands, reshape)

                    # permute 0231
                    attr = {
                        'order0': 0,
                        'order1': 2,
                        'order2': 3,
                        'order3': 1,
                    }
                    operands = [reshape_op]
                    output_shape = [reshape[0], 1, input_shape[3], input_shape[2]]
                    # reshape back

                permute_op = self.CVI.add_permute_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
                self.addOperand(onnx_node.name, permute_op, output_shape, TensorType.ACTIVATION)

                if is_0231:
                    reshape_op = self.CVI.add_reshape_op("{}_{}_back_for_0231".format(
                        onnx_node.name, onnx_node.op_type), [permute_op], permute_0231_shape)
                    self.addOperand(onnx_node.name, reshape_op, output_shape, TensorType.ACTIVATION)

            elif len(transpose_perm) == 3:
                """
                    Our tpu only support 4 dim transpose, we reshape 3dim to 4
                    and after transpose reshape back
                """
                input_shape.insert(0, 1)
                reshape_op = self.CVI.add_reshape_op("{}_{}_to_four_dim".format(
                    onnx_node.name, onnx_node.op_type), [op], input_shape)
                on = input_shape[0]
                oc = input_shape[transpose_perm[0]+1]
                oh = input_shape[transpose_perm[1]+1]
                ow = input_shape[transpose_perm[2]+1]
                output_shape = [on, oc, oh, ow]

                attr = {
                    'order0': 0,
                    'order1': transpose_perm[0]+1,
                    'order2': transpose_perm[1]+1,
                    'order3': transpose_perm[2]+1,
                }
                permute_op = self.CVI.add_permute_op("{}_{}".format(
                    onnx_node.name, onnx_node.op_type), [reshape_op], output_shape, **attr)
                output_shape = output_shape[1:]
                reshape_back_op = self.CVI.add_reshape_op("{}_{}_back_dim".format(
                    onnx_node.name, onnx_node.op_type), [permute_op], output_shape)
                self.addOperand(onnx_node.name, reshape_back_op,
                                output_shape, TensorType.ACTIVATION)
            elif len(transpose_perm) == 5:
                """
                    Our tpu only support 4 dim transpose, dim5 not support
                    if transpose_perm first element is 0 and input_shape first is 1(not batch)
                    we can skip this dim
                """
                if transpose_perm[0] == 0 and input_shape[0] == 1:
                    # reshape to dim 4
                    new_shape = input_shape[1:]
                    reshape_op = self.CVI.add_reshape_op("{}_{}_to_four_dim".format(
                        onnx_node.name, onnx_node.op_type), [op], new_shape)

                    new_transpose_term = [x - 1 for x in transpose_perm]
                    # skip first diim
                    new_transpose_term = new_transpose_term[1:]

                    # tranpose
                    on = new_shape[new_transpose_term[0]]
                    oc = new_shape[new_transpose_term[1]]
                    oh = new_shape[new_transpose_term[2]]
                    ow = new_shape[new_transpose_term[3]]
                    output_shape = [on, oc, oh, ow]

                    attr = {
                        'order0': new_transpose_term[0],
                        'order1': new_transpose_term[1],
                        'order2': new_transpose_term[2],
                        'order3': new_transpose_term[3],
                    }

                    permute_op = self.CVI.add_permute_op("{}_{}".format(
                        onnx_node.name, onnx_node.op_type), [reshape_op], output_shape, **attr)

                    output_shape.insert(0, 1)

                    reshape_back_op = self.CVI.add_reshape_op("{}_{}_back_dim".format(
                        onnx_node.name, onnx_node.op_type), [permute_op], output_shape)
                    self.addOperand(onnx_node.name, reshape_back_op,
                        output_shape, TensorType.ACTIVATION)
                else:
                    raise RuntimeError("transpose dim 5 is not support")
            else:
                raise RuntimeError("only support dim 4 transpose and pixel shuffle case")

    def convert_where_op(self, onnx_node):
        assert(onnx_node.op_type == "Where")

        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[2])

        if tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR and \
            tensor_type2 == TensorType.TENSOR:
            # both are weight, do it offline
            tensor_data0 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data1 = self.getTensor(onnx_node.inputs[1]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[2]).tensor_data
            tensor_data = np.where(tensor_data0, tensor_data1, tensor_data2)

            self.addTensor(onnx_node.name, tensor_data, list(tensor_data.shape))
            self.addOperand(onnx_node.name, None, list(tensor_data.shape), TensorType.TENSOR)
        else:
            raise RuntimeError("not support tensor_type x in activation")


    def convert_unsqueeze_op(self, onnx_node):
        """Unsqueeze """
        assert(onnx_node.op_type == "Unsqueeze")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        checkKey(onnx_node.attrs, 'axes')
        axis_value_list = onnx_node.attrs['axes']
        if tensor_type == TensorType.TENSOR:
            t = self.getTensor(onnx_node.inputs[0])
            new_t = t.tensor_data
            for a in axis_value_list:
                new_t = np.expand_dims(new_t, axis=a)
            self.addTensor(onnx_node.name, new_t, list(new_t.shape))
            self.addOperand(onnx_node.name, None, list(new_t.shape), TensorType.TENSOR)
        else:
            if len(axis_value_list) != 1:
                raise RuntimeError("now only support one axis")
            new_shape = self.unsqueeze_shape(input_shape, axis_value_list)
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op], new_shape)
            self.addOperand(onnx_node.name, reshape_op, new_shape, TensorType.ACTIVATION)

    def convert_upsample_op(self, onnx_node):
        assert(onnx_node.op_type == "Upsample")
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        if tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            scale_factor = self.getTensor(onnx_node.inputs[1]).tensor_data
            if len(scale_factor) != 4:
                raise RuntimeError("scale_factor length should be 4")
            if scale_factor[0] != 1 and scale_factor[1] != 1:
                raise RuntimeError("Not support n,c upsample")
            if scale_factor[2] != scale_factor[3]:
                raise RuntimeError("TODO&FIXME:Our IR need to fix it, support w and h upsample")

            operands = list()
            operands.append(op1)
            on = int(input_shape1[0])
            oc = int(input_shape1[1])
            oh = int(input_shape1[2] * scale_factor[2])
            ow = int(input_shape1[3] * scale_factor[3])
            attr={
                'scale_h': int(scale_factor[2]),
                'scale_w': int(scale_factor[2])
            }
            output_shape = [on, oc, oh, ow]
            upsample_op = self.CVI.add_upsample_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
            self.addOperand(onnx_node.name, upsample_op, output_shape, TensorType.ACTIVATION)

    def convert_reduce_mean_op(self, onnx_node):
        assert(onnx_node.op_type == "ReduceMean")
        checkKey(onnx_node.attrs, 'axes')
        checkKey(onnx_node.attrs, 'keepdims')
        axes = onnx_node.attrs['axes']
        keepdims = onnx_node.attrs['keepdims']
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape0

        axis = 0
        if len(axes) == 1:
            if axes[0] != 2 and axes[0] != 3:
                raise RuntimeError("{} axis not support, please add".format(axes[0]))

            # deep copy
            output_shape = list(input_shape0)
            output_shape.pop(axes[0])

            if keepdims != 0:
                output_shape.extend([1])

        elif len(axes) == 2 and axes[0] == 2 and axes[1] == 3:
            if keepdims == 0:
                output_shape = input_shape0[:-2]
            else:
                output_shape = input_shape0[:-2]
                output_shape.extend([1,1])
        else:
            raise RuntimeError("axes type not support for now")

        attr = {
            'axes': axes
        }
        print("output shape: ", output_shape)
        operands = list()
        operands.append(op0)
        reduce_mean_op = self.CVI.add_reduce_mean_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
        self.addOperand(onnx_node.name, reduce_mean_op, output_shape, TensorType.ACTIVATION)

    def convert_reduce_max_op(self, onnx_node):
        assert(onnx_node.op_type == "ReduceMax")
        checkKey(onnx_node.attrs, 'axes')
        axes = onnx_node.attrs['axes']
        keepdims = onnx_node.attrs['keepdims']
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape0
        print("reduce max, input: ", input_shape0, "axes: ", axes, "keepdims: ", keepdims)
        #
        if len(axes) == 1 and axes[0] == 3:
            # remove the last dimension
            if keepdims == 0:
                output_shape = input_shape0[:-1]
            else:
                output_shape = input_shape0[:-1]
                output_shape.extend([1])
        elif len(axes) == 1 and axes[0] == 1:
            if keepdims == 0:
                output_shape = (input_shape0[0:1])
                output_shape.extend(input_shape0[2:])
            else:
                output_shape = input_shape0[0:1]
                output_shape.extend([1,])
                output_shape.extend(input_shape0[2:])
        else:
            raise RuntimeError("axes type not support: ", axes)

        attr = {
            'axes': axes
        }

        operands = list()
        operands.append(op0)
        reduce_mean_op = self.CVI.add_reduce_max_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
        self.addOperand(onnx_node.name, reduce_mean_op, output_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_node()
        self.convert_tensor()
        self.convert_graph()
        self.TensortoNpz()


