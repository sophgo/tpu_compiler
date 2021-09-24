# ONNX Node define:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

import enum
from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from onnx import numpy_helper, mapping
from termcolor import colored, cprint
from math import floor, ceil
from fractions import gcd
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

class BaseNode():
    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')
        cprint("    attrs: {}".format(self.attrs), 'white')
        for arg in self.attrs:
            cprint("        {}: {}".format(arg, self.attrs[arg]), 'green')


class OnnxNode(BaseNode):
    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0]
        info["op_type"] = node.op_type
        info["attrs"] = [(attr.name, \
                          translate_onnx(attr.name, convert_onnx_attribute_proto(attr))) \
                          for attr in node.attribute]
        info["inputs"] = node.input
        info["outputs"] = node.output
        super().__init__(info)
        self.node_proto = node


class OnnxTensor():
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')

class Redundancy():
    def __init__(self):
        # -1: reconstructing op's input; 0,1,2,3...: the idx of eatch redundandent op's input
        self.op_list = {
            "Std_unbiased" :[{"ReduceMean": (-1,)}, {"Shape": (-1,)}, {"Constant": None}, {"Gather": (1,2)}, {"ReduceProd": (3,)},
                    {"Sub": (-1, 0)}, {"Mul": (5, 5)}, {"ReduceMean": (6,)}, {"Cast": (4,)}, {"Mul": (7, 8)}, {"Constant": None},
                    {"Sub": (8, 10)}, {"Div": (9, 11)}, {"Sqrt": (12,)},],
            "Std_biased": [{"ReduceMean": (-1,)}, {"Sub": (-1, 0)}, {"Mul": (1, 1)}, {"ReduceMean": (2,)}, {"Sqrt": (3,)},],
        }
        # get attr from which redundandent op with specify key (op_idx_in_op_list, (org_op_attr, new_op_attr))
        self.attr_idxes = {
            "Std_unbiased": [(0, ("axes", "dim")), (7, ("keepdims",)), ("default", {"unbiased": True})],
            "Std_biased": [(0, ("axes", "dim")), (3, ("keepdims",)), ("default", {"unbiased": False})],
            }

    def refine(self, converted_nodes):
        for op_tpye in self.op_list.keys():
            pattern = self.op_list[op_tpye]
            pattern_idx = 0
            redundancies = []
            re_op_inp = None
            for nidx, node in enumerate(converted_nodes):
                match_success = False
                if node.op_type == list(pattern[pattern_idx].keys())[0]:
                    op_input_idxes = list(pattern[pattern_idx].values())[0]
                    # for Constant
                    if op_input_idxes is None:
                         redundancies.append(nidx)
                         pattern_idx += 1
                         continue

                    for i in op_input_idxes:
                        if -1 == i or set(converted_nodes[redundancies[i]].outputs).intersection(set(node.inputs)):
                            redundancies.append(nidx)
                            match_success = True
                    _tmp = list(set(redundancies))
                    _tmp.sort(key=redundancies.index)
                    redundancies = _tmp

                    pattern_idx += 1
                    if match_success and len(pattern) == pattern_idx:
                        # get attr and form op
                        attrs = {}
                        attr_idxes = self.attr_idxes[op_tpye]
                        for i, k in attr_idxes:
                            if "default" == i:
                                attrs.update(k)
                                continue
                            attrs.update({k[-1]: converted_nodes[redundancies[i]].attrs[k[0]]})
                        info = {}
                        info["name"] = node.name
                        info["op_type"] = op_tpye.split("_")[0]

                        info["attrs"] = attrs
                        info["inputs"] = converted_nodes[redundancies[0]].inputs
                        info["outputs"] = node.outputs
                        new_node = BaseNode(info)
                        for i in redundancies:
                            converted_nodes[i] = None
                        converted_nodes[redundancies[-1]] = new_node
                        # reset wait for another pattern
                        pattern_idx = 0
                        redundancies.clear()

                if not match_success:
                    pattern_idx = 0
                    redundancies.clear()
            converted_nodes = [node for node in converted_nodes if node]
        return converted_nodes

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
        self.refine = Redundancy()

        self.onnxop_factory = {
            "Abs": lambda node: self.convert_abs_op(node),
            "Add": lambda node: self.convert_add_op(node),
            "ArgMax": lambda node: self.convert_argmax_op(node),
            "AveragePool": lambda node: self.convert_avg_pool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Cast": lambda node: self.convert_cast_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "ConvTranspose": lambda node: self.convert_conv_transpose_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "ConstantOfShape": lambda node: self.convert_constant_of_shape_op(node),
            "DepthToSpace": lambda node: self.convert_depth_to_space_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Equal": lambda node: self.convert_equal_op(node),
            "Elu" :lambda node: self.convert_activation_op(node),
            "Exp" :lambda node: self.convert_activation_op(node),
            "Expand": lambda node: self.convert_expand_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_pool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_pool_op(node),
            "GRU": lambda node: self.convert_gru_op(node),
            "HardSigmoid": lambda node: self.convert_hard_sigmoid_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "InstanceNormalization": lambda node: self.convert_instancenorm_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "Log": lambda node: self.convert_log_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "LSTM": lambda node: self.convert_lstm_op(node),
            "LayerNorm": lambda node: self.convert_layernorm_op(node),
            "MatMul": lambda node: self.convert_matmul_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Max" : lambda node: self.convert_max_op(node),
            "Min" : lambda node: self.convert_min_op(node),
            "Mul" : lambda node: self.convert_mul_op(node),
            "Neg" : lambda node: self.convert_neg_op(node),
            "Pad": lambda node: self.convert_pad_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "Pow": lambda node: self.convert_pow_op(node),
            "Reciprocal": lambda node: self.convert_reciprocal_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "ReduceL2": lambda node: self.convert_reduce_l2_op(node),
            "ReduceMean": lambda node: self.convert_reduce_mean_op(node),
            "ReduceMax": lambda node: self.convert_reduce_max_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid" :lambda node: self.convert_activation_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softplus" :lambda node: self.convert_activation_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Sqrt": lambda node: self.convert_sqrt_op(node),
            "Std": lambda node: self.convert_std_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Sum": lambda node: self.convert_sum_op(node),
            "Tanh": lambda node: self.convert_activation_op(node),
            "Tile": lambda node: self.convert_tile_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Where": lambda node: self.convert_where_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "YoloDetection": lambda node: self.convert_yolo_detection_op(node)
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
                if i == 0 and dim.dim_value <= 0 and self.batch_size != 0:
                    input_shape.append(self.batch_size)
                else:
                    input_shape.append(dim.dim_value)
            # if len(input_shape) != 4:
            #     input_shape.extend([1] * (4 - len(input_shape)))
            self.input_shapes.append(input_shape)

            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output in self.output_nodes:
            output_shape = list()
            for i, dim in enumerate(output.type.tensor_type.shape.dim):
                # i == 0 mean batch size
                # if dim is zero, mean mutli batch
                if i == 0 and dim.dim_value <= 0 and self.batch_size != 0:
                    output_shape.append(self.batch_size)
                else:
                    output_shape.append(dim.dim_value)
            outputs.append(output_shape)
            # keep mlir function output
            self.output_shapes.append(output_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs, "FP32", output_weight_file=self.output_weight_file)

    def getNode(self, op_name):
        for node in self.converted_nodes:
            if node and node.name == op_name:
                return node
        return None

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

    def add_none_op(self):
        return self.CVI.add_none_op()

    def check_need(self, name):
        for node in self.converted_nodes:
            for i in node.inputs:
                if i == name:
                    return True
        for o in self.output_nodes:
            if name == o.name:
                return True
        return False

    # support like: [3, 1, 2] + [3, 2, 2] ; [2, 3] + [3, 2, 3]
    # only support one direction bcast
    def bcast_shape(self, lshape, rshape):
        if len(rshape) > len(lshape):
            lshape, rshape = rshape, lshape
        if len(lshape) > len(rshape):
            rshape = [1] * (len(lshape) - len(rshape)) + rshape
        sub = np.prod(lshape) - np.prod(rshape)
        if sub == 0:
            return None
        for a, b in zip(lshape, rshape):
            if a != b and ((sub > 0 and b != 1) or (sub < 0 and a != 1)):
                return None
        oshape = list(lshape) if sub > 0 else list(rshape)
        return oshape

    def same_shape(self, lshape, rshape):
        if len(rshape) > len(lshape):
            lshape, rshape = rshape, lshape
        if len(lshape) > len(rshape):
            rshape = [1] * (len(lshape) - len(rshape)) + rshape
        for a,b in zip(lshape, rshape):
            if a != b:
                return None
        oshape = list(lshape)
        return oshape

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
        if axis == None:
            for dim in shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 0:
                new_shape.append(1)
        elif len(axis) > 0:
            for i in range(len(shape)):
                if i not in axis:
                    new_shape.append(shape[i])
        else:
            new_shape = shape
        return new_shape

    def get_ndim(self, org_shape, dims):
        expand_dims = dims - len(org_shape)

        if expand_dims == 0:
            return org_shape

        # expand from high dim
        return list(np.full(expand_dims, 1)) + list(org_shape)

    def add_reshape(self, to_shape, node_name, node_type, operands):
        src_reshape_op = self.CVI.add_reshape_op(node_name, operands, to_shape)
        self.addOperand(node_name, src_reshape_op, to_shape, TensorType.ACTIVATION)
        return [src_reshape_op]

    def add_extend_4dim(self, org_shape, node_name, node_type, operands):
        expand_dims = 4 - len(org_shape)

        if expand_dims == 0:
            return operands, org_shape

        # expand from high dim
        nchw_shape = list(np.full(expand_dims, 1)) + list(org_shape)

        name = "{}_{}_extend_to_4_dim".format(node_name, node_type)

        return self.add_reshape(nchw_shape, name, node_type, operands), nchw_shape


    def convert_node(self):
        """convert onnx node to OnnxNode"""
        for n in self.nodes:
            node = OnnxNode(n)
            if log_flag:
                node.print_info()
            self.converted_nodes.append(node)

    def refine_node(self):
        """reconstruct some node like hard_sigmoid"""
        # swap Costant_Div with Mul (Mul -> Constant Div to Constant Div -> Mul)
        # for pp model
        preNodes = dict()
        # for node in self.converted_nodes.values():
        for i, node in enumerate(self.converted_nodes):
            if node.op_type == "Mul" or node.op_type == "Constant":
                # mulOut2divIn.update{node.outputs[0]: (i, node.inputs)}
                # mul_list.append(node.outputs[0])
                preNodes.update({node.outputs[0]: i})
            elif node.op_type == "Div" and node.inputs[1] in preNodes.keys() \
                 and node.inputs[0] in preNodes.keys() \
                 and self.converted_nodes[preNodes[node.inputs[1]]].op_type == "Constant": \
                # swap input and output (notice name eq outputs[0])
                mul_idx = preNodes[node.inputs[0]]
                mul = self.converted_nodes[mul_idx]
                node.inputs[0] = mul.inputs[1]
                node.outputs[0] = mul.name

                mul.inputs[1] = node.outputs[0]
                mul.outputs[0] = node.name
                # follow the name eq outputs[0] rule
                node.name = node.outputs[0]
                mul.name = mul.outputs[0]

                self.converted_nodes[i] = mul
                self.converted_nodes[mul_idx] = node
                # self.converted_nodes.update({node.name: node, mul.name: mul})
                preNodes.clear()  # just need swap nearst connected Mul-Constant_div pattern

        # hard_sigmoid_pattern = ["Constant", "Add", "Clip", "Constant", "Div"]
        # for pp after swap is ["Constant", "Constant", "Add", "Clip", "Div"]
        consumed = list()      # idx
        reserve_out  = dict()  # {node_out: idx}
        base_pattern = ["Add", "Clip", "Div"]
        alpha, beta = 0., 0.
        hs_in, hs_out = "", ""
        for i, node in enumerate(self.converted_nodes):
            if node.op_type == "Constant":
                reserve_out.update({node.outputs[0]: i})
                continue
            if len(base_pattern) > 0 and node.op_type == base_pattern[0]:
                base_pattern.pop(0)
                if node.op_type == "Add":
                    betaIdx = reserve_out.get(node.inputs[1], None)
                    # if betaIdx is not None and self.converted_nodes[betaIdx].op_type == "Constant":
                    if betaIdx is not None:
                        beta = self.converted_nodes[betaIdx].attrs['value']
                        beta = numpy_helper.to_array(beta)
                        hs_in = node.inputs[0]
                        reserve_out.update({node.outputs[0]: i})
                        consumed.append(reserve_out.pop(node.inputs[1]))
                    else:
                        base_pattern =  ["Add", "Clip", "Div"]
                        reserve_out.clear()
                        consumed.clear()
                elif node.op_type == "Clip":
                    pre_out = reserve_out.get(node.inputs[0], None)
                    if pre_out is not None:
                        reserve_out.update({node.outputs[0]: i})
                        consumed.append(reserve_out.pop(node.inputs[0]))
                    else:
                        base_pattern =  ["Add", "Clip", "Div"]
                        reserve_out.clear()
                        consumed.clear()
                elif node.op_type == "Div":
                    pre_out = reserve_out.get(node.inputs[0], None)
                    alphaIdx = reserve_out.get(node.inputs[1], None)
                    if alphaIdx is not None and pre_out is not None:
                        alpha = self.converted_nodes[alphaIdx].attrs['value']
                        alpha = numpy_helper.to_array(alpha)
                        hs_out = node.outputs[0]
                        consumed.append(reserve_out.pop(node.inputs[0]))
                        consumed.append(reserve_out.pop(node.inputs[1]))
                    else:
                        base_pattern = ["Add", "Clip", "Div"]
                        reserve_out.clear()
                        consumed.clear()
            # reconstruct hsigmoid node
            if len(base_pattern) == 0:
                beta = 1.* beta / alpha
                alpha = 1. / alpha
                hs_node = onnx.helper.make_node('HardSigmoid',
                                                inputs=[hs_in],
                                                outputs=[hs_out],
                                                alpha=alpha,
                                                beta=beta
                                               )
                hs_node = OnnxNode(hs_node)
                self.converted_nodes[i] = hs_node

                # remove consumed nodes
                for j in consumed:
                    self.converted_nodes[j] = None
                # reset and wait for next pattern
                base_pattern = ["Add", "Clip", "Div"]
                reserve_out.clear()
                consumed.clear()
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # form layerNorm op from onnx
        # support default param.elementwise_affine
        consumed = list()      # idx
        ln_in, ln_out = [], []
        base_pattern =  ["ReduceMean", "Sub", "Pow", "ReduceMean",
                         "Add", "Sqrt", "Div", "Mul", "Add"]
        for i, node in enumerate(self.converted_nodes):
            if node.op_type == "Constant":
                continue
            if len(base_pattern) > 0 and node.op_type == base_pattern[0]:
                base_pattern.pop(0)
                consumed.append(i)
                if node.op_type == "ReduceMean" and len(ln_in) == 0:
                    ln_in.append(node.inputs[0])

                elif node.op_type == "Mul":
                    # second operand should be weight tensor
                    if not self.getNode(node.inputs[1]):
                        ln_in.append(node.inputs[1])
                    else:
                        base_pattern = ["ReduceMean", "Sub", "Pow", "ReduceMean",
                                        "Add", "Sqrt", "Div", "Mul", "Add"]
                        consumed.clear()
                        ln_in.clear()
                        ln_out.clear()

                elif node.op_type == "Add":
                    opd_b = self.getNode(node.inputs[1])
                    if not opd_b: # is weight tensor
                        b_tensor = self.getTensor(node.inputs[1])
                        if not b_tensor.shape:
                            eps = b_tensor.tensor_data
                        else:
                            ln_in.append(node.inputs[1])
                            ln_out.append(node.outputs[0])
                    elif opd_b.op_type == "Constant":
                        eps = numpy_helper.to_array(opd_b.attrs['value'])
            else:
                base_pattern = ["ReduceMean", "Sub", "Pow", "ReduceMean",
                                "Add", "Sqrt", "Div", "Mul", "Add"]
                consumed.clear()
                ln_in.clear()
                ln_out.clear()

            # reconstruct layerNorm node
            if len(base_pattern) == 0:
                # remove consumed nodes
                for j in consumed:
                    self.converted_nodes[j] = None
                info = {}
                info["name"] = ln_out[0]
                info["op_type"] = "LayerNorm"
                info["attrs"] = {"eps": eps}
                info["inputs"] = ln_in
                info["outputs"] = ln_out
                ln_node = BaseNode(info)
                self.converted_nodes[i] = ln_node
                # reset and wait for next pattern
                base_pattern = ["ReduceMean", "Sub", "Pow", "ReduceMean",
                                "Add", "Sqrt", "Div", "Mul", "Add"]
                consumed.clear()
                ln_in.clear()
                ln_out.clear()
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # merge matmul + bias
        i = 1
        while i < len(self.converted_nodes):
            node = self.converted_nodes[i]
            if node.op_type != "MatMul":
                i += 1
                continue
            # rhs should be weight, skip if it's activation
            filter_node = self.getNode(node.inputs[1])
            if filter_node and filter_node.op_type != 'Constant':
                i += 1
                continue
            if i + 1 >= len(self.converted_nodes):
                i += 1
                continue
            next_node = self.converted_nodes[i+1]
            if next_node.op_type != "Add":
                i += 1
                continue
            add_node = self.getNode(next_node.inputs[1])
            if add_node and add_node.op_type != 'Constant':
                i += 1
                continue

            info = {}
            info["name"] = next_node.name
            info["op_type"] = "MatMul"
            info["attrs"] = node.attrs
            info["inputs"] = [node.inputs[0], node.inputs[1], next_node.inputs[1]]
            info["outputs"] = [next_node.outputs[0]]
            matmul_node = BaseNode(info)
            self.converted_nodes[i] = None
            self.converted_nodes[i+1] = matmul_node
            i += 1
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # merge log(exp + add_1) * const => softplus
        i = 1
        while i < len(self.converted_nodes):
            node = self.converted_nodes[i]
            if node.op_type != "Exp":
                i += 1
                continue
            if i + 1 >= len(self.converted_nodes):
                i += 1
                continue
            const_node = self.converted_nodes[i+1]
            if const_node.op_type != "Constant":
                i += 1
                continue
            bias = numpy_helper.to_array(const_node.attrs['value'])
            if bias != 1.0:
                i += 1
                continue
            add_node = self.converted_nodes[i+2]
            if add_node.op_type != "Add":
                i += 1
                continue

            log_node = self.converted_nodes[i+3]
            if log_node.op_type != "Log":
                i += 1
                continue

            const_node = self.converted_nodes[i+4]
            if const_node.op_type != "Constant":
                i += 1
                continue
            scale = numpy_helper.to_array(const_node.attrs['value'])

            mul_node = self.converted_nodes[i+5]
            if mul_node.op_type != "Mul":
                i += 1
                continue

            info = {}
            info["name"] = mul_node.name
            info["op_type"] = 'Softplus'
            info["attrs"] = {'scale': scale}
            info["inputs"] = [node.inputs[0]]
            info["outputs"] = [mul_node.outputs[0]]
            sp_node = BaseNode(info)
            self.converted_nodes[i] = None
            self.converted_nodes[i+1] = None
            self.converted_nodes[i+2] = None
            self.converted_nodes[i+3] = None
            self.converted_nodes[i+4] = None
            self.converted_nodes[i+5] = sp_node
            i += 5
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # merge exp + const => exp
        i = 1
        while i < len(self.converted_nodes):
            node = self.converted_nodes[i]
            if node.op_type != "Exp":
                i += 1
                continue
            if i + 2 >= len(self.converted_nodes):
                i += 1
                continue
            const_node = self.converted_nodes[i+1]
            if const_node.op_type != "Constant":
                i += 1
                continue
            bias = numpy_helper.to_array(const_node.attrs['value'])
            add_node = self.converted_nodes[i+2]
            if add_node.op_type != "Add":
                i += 1
                continue
            info = {}
            info["name"] = add_node.name
            info["op_type"] = 'Exp'
            info["attrs"] = {'bias': bias}
            info["inputs"] = [node.inputs[0]]
            info["outputs"] = [add_node.outputs[0]]
            exp_node = BaseNode(info)
            self.converted_nodes[i] = None
            self.converted_nodes[i+1] = None
            self.converted_nodes[i+2] = exp_node
            i += 2
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # merge  a * sigmoid(x) + b => sigmoid
        i = 1
        while i < len(self.converted_nodes):
            node = self.converted_nodes[i]
            if node.op_type != "Sigmoid":
                i += 1
                continue
            if i + 4 >= len(self.converted_nodes):
                i += 1
                continue

            const_node = self.converted_nodes[i+1]
            if const_node.op_type != "Constant":
                i += 1
                continue
            scale = numpy_helper.to_array(const_node.attrs['value'])
            mul_node = self.converted_nodes[i+2]
            if mul_node.op_type != "Mul":
                i += 1
                continue

            const_node = self.converted_nodes[i+3]
            if const_node.op_type != "Constant":
                i += 1
                continue
            bias = numpy_helper.to_array(const_node.attrs['value'])
            add_node = self.converted_nodes[i+4]
            if add_node.op_type != "Add":
                i += 1
                continue
            info = {}
            info["name"] = add_node.name
            info["op_type"] = 'Sigmoid'
            info["attrs"] = {'scale': scale, 'bias': bias}
            info["inputs"] = [node.inputs[0]]
            info["outputs"] = [add_node.outputs[0]]
            sigmoid_node = BaseNode(info)
            self.converted_nodes[i] = None
            self.converted_nodes[i+1] = None
            self.converted_nodes[i+2] = None
            self.converted_nodes[i+3] = None
            self.converted_nodes[i+4] = sigmoid_node
            i += 4
        self.converted_nodes = [node for node in self.converted_nodes if node]

        # upsample to resize nearst
        for i, node in enumerate(self.converted_nodes):
            if node.op_type == "Upsample":
                info = {}
                info["name"] = node.name
                info["op_type"] = "Resize"
                mode = node.attrs.get('mode',b'nearest')
                info['attrs'] = {'mode':mode,'coordinate_transformation_mode':b'asymmetric'}
                info["inputs"] = node.inputs
                info["outputs"] = node.outputs
                resize_node = BaseNode(info)
                self.converted_nodes[i] = resize_node
        self.converted_nodes = self.refine.refine(self.converted_nodes)

    def convert_tensor(self):
        """convert onnx tensor to OnnxTensor"""
        for tensor in self.tensors:
            name = tensor.name
            shape = list(tensor.dims)
            data = numpy_helper.to_array(tensor).astype(np.float32)
            tensor = OnnxTensor(name, data, shape)
            # tensor.print_info()
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
                if i == 0 and dim.dim_value <= 0 and self.batch_size != 0:
                    input_shape.append(self.batch_size)
                else:
                    input_shape.append(dim.dim_value)
            image = (len(input_shape) == 4 and input_shape[1] <=4) or \
                    (len(input_shape) == 3) # gray
            if not self.preprocess_args or not image:
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
            input_shape = list(input_shape)
            # if len(input_shape) != 4:
            #     input_shape.extend([1] * (4 - len(input_shape)))
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
        logger.info("Save mlir file: {}".format(self.mlir_file_path))

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
                scale = onnx_node.attrs.get('scale', 1)
                bias = onnx_node.attrs.get('bias', 0)
                activation_op = self.CVI.add_sigmoid_op("{}_{}".format(onnx_node.name, onnx_node.op_type),
                                                       operands, output_shape, scale=scale, bias=bias)
            elif onnx_node.op_type == "Tanh":
                activation_op = self.CVI.add_tanh_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Elu":
                activation_op = self.CVI.add_elu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Exp":
                scale = onnx_node.attrs.get('scale', 1)
                bias = onnx_node.attrs.get('bias', 0)
                activation_op = self.CVI.add_exp_op("{}_{}".format(onnx_node.name, onnx_node.op_type),
                                                       operands, output_shape, scale=scale, bias=bias)
            elif onnx_node.op_type == "Sqrt":
                activation_op = self.CVI.add_sqrt_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Softplus":
                scale = onnx_node.attrs.get('scale', 1)
                bias = onnx_node.attrs.get('bias', 0)
                activation_op = self.CVI.add_softplus_op("{}_{}".format(onnx_node.name, onnx_node.op_type),
                                                       operands, output_shape, scale=scale, bias=bias)

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
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])

        if tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.ACTIVATION:
            # put activation first
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return

        if tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.TENSOR:
            weight_name = onnx_node.inputs[1]
            weight_data = self.getTensor(onnx_node.inputs[1]).tensor_data
            if input_shape0 == input_shape1:
                self.addTensor(weight_name, weight_data, input_shape1)
                op2 = self.CVI.add_load_file_op(weight_name, input_shape1)
                add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op0, op2], input_shape0)
                self.addOperand(onnx_node.name, add_op, input_shape0, TensorType.ACTIVATION)
                return
            elif len(input_shape0) > 1 and (np.prod(input_shape1) == 1 or np.prod(input_shape1) == input_shape0[1]):
                # Use scale op, x * 1 + y
                scale_data = np.full(input_shape0[1], 1) # broadcast via channel
                scale_name = "{}_scale".format(weight_name)
                self.addTensor(scale_name, scale_data, scale_data.shape)
                op2 = self.CVI.add_load_file_op(scale_name, scale_data.shape)

                if len(input_shape1) == 0:
                    bias_data = np.full(input_shape0[1], weight_data)
                elif len(input_shape1) == 1 and np.prod(input_shape1) == 1:
                    bias_data = np.full(input_shape0[1], weight_data[0])
                else:
                    bias_data = weight_data.flatten()
                bias_name = "{}_bias".format(weight_name)
                self.addTensor(bias_name, bias_data, bias_data.shape)
                op3 = self.CVI.add_load_file_op(bias_name, bias_data.shape)

                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op0,op2,op3], input_shape0)
                self.addOperand(onnx_node.name, scale_op, input_shape0, TensorType.ACTIVATION)
                return
            else:
                weight_name = onnx_node.inputs[1]
                weight_data = self.getTensor(onnx_node.inputs[1]).tensor_data
                output_shape = self.bcast_shape(input_shape0, input_shape1)
                if output_shape == input_shape0:
                    weight = np.zeros(output_shape, dtype=np.float32)
                    weight_data = weight + weight_data
                    weight_name = weight_name + "_broadcast"
                    input_shape1 = output_shape
                if input_shape0 == input_shape1:
                    self.addTensor(weight_name, weight_data, input_shape1)
                    op3 = self.CVI.add_load_file_op(weight_name, input_shape1)
                    add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op0, op3], input_shape0)
                    self.addOperand(onnx_node.name, add_op, input_shape0, TensorType.ACTIVATION)
                    return
        elif tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = tensor_data1 + tensor_data2
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
            return
        elif tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.ACTIVATION:
            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            output_shape = self.same_shape(input_shape0, input_shape1)
            if output_shape != None:
                add_op = self.CVI.add_eltwise_add_op(name, [op0, op1], output_shape)
                self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)
                return
            output_shape = self.bcast_shape(input_shape0, input_shape1)
            if output_shape != None:
                if np.prod(input_shape0) < np.prod(input_shape1):
                    op0, op1 = op1, op0
                add_op = self.CVI.add_broadcast_add_op(name, [op0, op1], output_shape)
                self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)
                return
        raise RuntimeError("Add {} {} not support now".format(input_shape0, input_shape1))

    def convert_argmax_op(self, onnx_node):
        assert(onnx_node.op_type == "ArgMax")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = list(input_shape)
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis = len(input_shape) + axis
        output_shape[axis] = 1
        attrs = {'axis': axis}
        argmax_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        abs_op = self.CVI.add_argmax_op(argmax_name, operands, output_shape, **attrs)
        self.addOperand(onnx_node.name, abs_op, output_shape, TensorType.ACTIVATION)

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
        onnx_name = onnx_node.name

        is1d = len(input_shape) == 3

        if is1d:
            # extend w
            strides = strides + [1]
            kernel_shape = kernel_shape + [1]
            pads = [pads[0], 0, pads[1], 0]
            input_shape = input_shape + [1]
            onnx_name = f"{onnx_node.name}_reshaped"

            # reshape to n,c,h,1 for friendly with optimizer
            reshape_input_op = self.CVI.add_reshape_op(
                    f"{onnx_node.name}_{onnx_node.op_type}_reshape",
                    [op], input_shape)
            operands[0] = reshape_input_op

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
        self.addOperand(onnx_name, pool_avg_op, output_shape, TensorType.ACTIVATION)
        if is1d:
            # reshape back
            shape = output_shape[:-1]
            reshape_input_op = self.CVI.add_reshape_op(f"{onnx_node.name}_{onnx_node.op_type}_reshaped",
                    [pool_avg_op], shape)
            onnx_name = onnx_node.name
            self.addOperand(onnx_name, reshape_input_op,
                    shape, TensorType.ACTIVATION)

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
            if tensor_type == TensorType.TENSOR:
                self.addTensor(onnx_node.name, self.getTensor(onnx_node.inputs[0]).tensor_data, input_shape)
                self.addOperand(onnx_node.name, None, input_shape, TensorType.TENSOR)
            else:
                self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)
            return

        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        axis = onnx_node.attrs['axis']
        if axis < 0:
            axis += len(input_shape1)
        assert(axis >=0 and axis < len(input_shape1))
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            max_dims = 0
            for i in onnx_node.inputs:
                data = self.getTensor(i).tensor_data
                max_dims = len(data.shape) if len(data.shape) > max_dims else max_dims
            arrays = list()
            for i in onnx_node.inputs:
                data = self.getTensor(i).tensor_data
                if len(data.shape) != max_dims:
                    data = np.expand_dims(data, axis)
                arrays.append(data)
            n_t = np.concatenate(arrays, axis=axis)
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
                        elif output_shape[dim] != value:
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
        op, shape_, _ = self.getOperand(onnx_node.inputs[0])
        shape = shape_[:]
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
            elif pad_method == "SAME_LOWER":
                 # the extra padding is added at the beginning for SAME_LOWER.
                 padding_along_h = get_TF_SAME_Padding(
                     shape[2], filter_shape[2], strides[0])
                 padding_along_w = get_TF_SAME_Padding(
                     shape[3], filter_shape[3], strides[1])
                 padding_b = padding_along_h // 2
                 padding_r = padding_along_w // 2
                 padding_t = padding_along_h - padding_b
                 padding_l = padding_along_w - padding_r
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

    def convert_conv_transpose_op(self, onnx_node):
        assert(onnx_node.op_type == "ConvTranspose")

        op, _shape, _ = self.getOperand(onnx_node.inputs[0])
        shape = list(_shape)
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

        # reshape to 4dim
        # for torch.nn.ConvTranspose1d case, just handle w
        # if input is <1, 384, 234>, it should reshape to <1, 384, 1, 234>
        op_name = onnx_node.name
        mlir_name = "{}_{}".format(op_name, onnx_node.op_type)

        is_shape_3 = len(_shape) == 3
        if is_shape_3:
            strides = onnx_node.attrs.get("strides", [])
            if len(strides) != 1:
                raise RuntimeError("ConvTranspose1d just give one stride")
            assert(len(filter_shape) == len(_shape))
            # reshape input
            n, c, w = _shape
            shape = [n, c, 1, w]
            name = "{}_to4".format(onnx_node.name)
            op = self.add_reshape(shape, name, onnx_node.op_type, [op])[0]
            operands[0] = op

            # reshape filter
            n, c, w = filter_shape
            filter_shape = [n, c, 1, w]

            op_name = onnx_node.name + "_reshape"
            mlir_name = "{}_{}".format(op_name, onnx_node.op_type)


        dilations = onnx_node.attrs.get("dilations", [1, 1])
        # If not present, the dilation defaults to 1 along each spatial axis
        dilations = self.get_ndim(dilations, 2)
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0,0,0])
        # If not present, the padding defaults to 0 along start and end of each spatial axis
        output_padding = onnx_node.attrs.get("output_padding",[0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1])
        # the stride defaults to 1 along each spatial axis
        strides = self.get_ndim(strides, 2)
        auto_pad = onnx_node.attrs.get("auto_pad", None)
        is_deconv = True

        if is_shape_3 and len(pads) == 2:
            _pads = np.zeros(4).astype(np.int)
            # only apply w
            for idx, _ in enumerate(pads):
                _pads[2 * idx + 1] = _
            pads = _pads

        if len(output_padding) < 4:
            # expand 0 to 4dim
            expand_dims = 4 - len(output_padding)
            # expand from high dim
            output_padding = list(np.full(expand_dims, 0)) + list(output_padding)

        conv_param = {
            'stride_h':  strides[0],
            'stride_w':  strides[1],
            'padding': "VALID",
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

        n, ic, ih, iw = shape

        on = shape[0]
        assert(ic == filter_shape[0])
        oc = filter_tensor.shape[1] * group # feature map size
        oh = (ih - 1) * strides[0] - pads[0] - pads[2] + dilations[0] * (filter_shape[2] - 1) + 1 + output_padding[-2]
        ow = (iw - 1) * strides[1] - pads[1] - pads[3] + dilations[1] * (filter_shape[3] - 1) + 1 + output_padding[-1]

        if conv_param['group'] != 1:
            g = conv_param['group']
            kh = onnx_node.attrs['kernel_shape'][0]
            kw = onnx_node.attrs['kernel_shape'][1]
            new_shape = [g, int(filter_shape[0]/g), filter_shape[1], kh, kw]
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, new_shape)
        else:
            # onnx weigh layout is <ic, oc, kh, kw> and we transpose it to <oc, ic, kh, kw>
            weight_tensor_data = filter_tensor.tensor_data.reshape(filter_shape)
            weight_tensor = np.ascontiguousarray(np.transpose(weight_tensor_data, (1,0,2,3)))
            filter_shape = weight_tensor.shape
            oc = weight_tensor.shape[0]
            self.addTensor(filter_tensor.name, weight_tensor, filter_shape)
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, filter_shape)

        operands.append(filter_op)

        if with_bias:
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        output_shape = [on, oc, oh, ow]
        deconv_op = self.CVI.add_deconv_op(mlir_name, operands, output_shape, **conv_param)
        self.addOperand(op_name, deconv_op, output_shape, TensorType.ACTIVATION)

        if is_shape_3:
            # no h
            output_shape = [on, oc, ow]
            self.add_reshape(output_shape, onnx_node.name, onnx_node.op_type, [deconv_op])

    def convert_conv3d_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        dilations = onnx_node.attrs.get("dilations", [1,1,1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0,0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1,1])
        conv3d_param = {
            'stride_d': strides[0],
            'stride_h':  strides[1],
            'stride_w':  strides[2],
            'padding': "SAME" if pads[0] > 0 else "VALID",
            'dilation_d': dilations[0],
            'dilation_h': dilations[1],
            'dilation_w': dilations[2],
            'padding_d0': pads[0],
            'padding_d1': pads[1],
            'padding_t': pads[2],
            'padding_b': pads[3],
            'padding_l': pads[4],
            'padding_r': pads[5],
            'group': group,
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
            'ins': [],
        }
        op, shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        filter_name = onnx_node.inputs[1]
        filter_tensor = self.getTensor(filter_name)
        filter_shape = filter_tensor.shape

        filter_op = self.CVI.add_load_file_op(filter_name, filter_shape)
        operands.append(filter_op)

        with_bias = False
        if len(onnx_node.inputs) == 3:
            #with bias
            with_bias = True
            bias_name = onnx_node.inputs[2]
            bias_tensor = self.getTensor(bias_name)

            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        on, ic, id, ih, iw = shape
        oc, ic, kd, kh, kw = filter_tensor.shape
        od = floor((id + 2 * pads[0] - dilations[0] * (kd - 1) - 1) / strides[0] + 1)
        oh = floor((ih + 2 * pads[1] - dilations[1] * (kh - 1) - 1) / strides[1] + 1)
        ow = floor((iw + 2 * pads[2] - dilations[2] * (kw - 1) - 1) / strides[2] + 1)
        output_shape = [on, oc, int(od), int(oh), int(ow)]

        conv3d_op = self.CVI.add_conv3d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv3d_param)
        self.addOperand(onnx_node.name, conv3d_op, output_shape, TensorType.ACTIVATION)

    def convert_clip_op(self, onnx_node):
        assert(onnx_node.op_type == "Clip")
        if len(onnx_node.inputs) == 3:
            min_val = self.getTensor(onnx_node.inputs[1])
            max_val = self.getTensor(onnx_node.inputs[2])
            clip_min = float(min_val.tensor_data)
            clip_max = float(max_val.tensor_data)
        else:
            clip_min = onnx_node.attrs.get('min', -np.inf)
            clip_max = onnx_node.attrs.get('max', np.inf)

        clip_param = {
            'min':  clip_min,
            'max':  clip_max,
        }
        op, shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = shape

        if abs(0 - clip_min) < 1e-5 and clip_max >= 1e5:
            print("we treat clip as relu cus min {} max {} are reasonable error".format(clip_min, clip_max))
            relu_op = self.CVI.add_relu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op], shape)
            self.addOperand(onnx_node.name, relu_op, shape, TensorType.ACTIVATION)
            return

        # FIXME: Now not support clip quantize
        # Only support relu6 case
        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.clip(data, clip_min, clip_max)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            if clip_min == 0:
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
            return
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
            return
        elif tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.ACTIVATION:
            # rewrite to x * (1/y)
            # get 1/y
            output_shape = list(input_shape2)
            name = "{}_reciprocal_{}".format(onnx_node.name, onnx_node.op_type)
            _op = self.CVI.add_reciprocal_op(name, [op2], output_shape)

            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            output_shape = self.same_shape(input_shape1, input_shape2)
            if output_shape != None:
                mul_op = self.CVI.add_eltwise_mul_op( name, [op1, _op], output_shape)
                self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)
                return
            output_shape = self.bcast_shape(input_shape1, input_shape2)
            if output_shape != None:
                if np.prod(input_shape1) < np.prod(input_shape2):
                    op1, _op = _op, op1
                mul_op = self.CVI.add_broadcast_mul_op(name, [op1, _op], output_shape)
                self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)
                return
        raise RuntimeError("div not support, shape1 {}, shape2 {}", input_shape1, input_shape2)

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
        op0, input_shape, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        _, _, tensor_type1 = self.getOperand(onnx_node.inputs[1])
        if tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[1]).tensor_data
            new_shape = list(tensor_data)
            zeros0 = np.zeros(input_shape)
            zeros1 = np.zeros(new_shape)
            zeros = zeros0 + zeros1
            output_shape = list(zeros.shape)
            num_dims = len(output_shape)
            if len(input_shape) != num_dims:
                input_shape2 = [1] * (num_dims - len(input_shape)) + input_shape
                name2 = "{}_reshape".format(onnx_node.inputs[0])
                op0 = self.CVI.add_reshape_op(name2, [op0], input_shape2)
                input_shape = list(input_shape2)
            first = 0
            if num_dims > 4:
                first = num_dims - 4
                assert(len(input_shape) == len(output_shape))
                assert(input_shape[:num_dims - 3] == output_shape[:num_dims - 3])
            last_shape = list(input_shape)
            last_op = op0
            for dim in range(first, num_dims):
                if input_shape[dim] == output_shape[dim]:
                    continue
                attr = {
                    'axis': dim - first,
                    'tiles': output_shape[dim]
                }
                last_shape[dim] = output_shape[dim]
                if last_shape == output_shape:
                    last_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
                else:
                    last_name = "{}_tile_{}".format(onnx_node.name, dim)
                last_op = self.CVI.add_tile_op(last_name, [last_op], last_shape, **attr)
            self.addOperand(onnx_node.name, last_op, last_shape, TensorType.ACTIVATION)
            return
        elif tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            shape_data = self.getTensor(onnx_node.inputs[1]).tensor_data
            new_shape = list(shape_data)
            zeros = np.zeros(new_shape, dtype = tensor_data.dtype)
            output_data = tensor_data + zeros
            self.addTensor(onnx_node.name, output_data, list(output_data.shape))
            self.addOperand(onnx_node.name, None, list(output_data.shape), TensorType.TENSOR)
            return
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
        elif tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            indices = self.getTensor(onnx_node.inputs[1]).tensor_data
            if indices.size == 1:
                offset = indices.flatten()[0]
                attr = {"axis": axis, "offset": offset}
                tmp = np.take(np.ones(input_shape1), np.array([offset]), axis=axis)
                output_shape = list(tmp.shape)
                print("out:", output_shape)
                slice_op_ = self.CVI.add_slice_op("{}_{}".format(onnx_node.outputs[0], onnx_node.op_type), [op1], output_shape, **attr)
                self.addOperand(onnx_node.outputs[0], slice_op_, output_shape, TensorType.ACTIVATION)
            else:
                logger.warning("indices:", indices)
                raise("TODO: Our Ir not support gather function")
        elif tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.ACTIVATION:
            op, shape, _ = self.getOperand(onnx_node.inputs[1])
            operands = list()
            operands.append(op)
            word_emb_name = onnx_node.inputs[0]
            word_emb_tensor = self.getTensor(word_emb_name)
            table_name = word_emb_name + onnx_node.name
            self.addTensor(table_name, word_emb_tensor.tensor_data, word_emb_tensor.shape)
            word_emb_op = self.CVI.add_load_file_op(table_name, word_emb_tensor.shape)
            operands.append(word_emb_op)
            _, embedding_dim = word_emb_tensor.shape
            output_shape = shape
            output_shape.append(embedding_dim)
            embedding_op = self.CVI.add_embedding_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, embedding_op, output_shape, TensorType.ACTIVATION)
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
        seq_length, batch_size, input_size = input_shape

        linear_before_reset = True if onnx_node.attrs.get(
            "linear_before_reset", 1) == 1 else False
        bidirectional = True if onnx_node.attrs.get(
            "direction", 'forward') == b'bidirectional' else False
        gru_param = {
            'linear_before_reset': bool(linear_before_reset),
            'bidirectional': bool(bidirectional),
        }
        num_dir = 2 if bidirectional else 1
        # fc x*weight+bias first
        operands = list()
        operands.append(op)

        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_shape = weight_tensor.shape
        assert(weight_shape[0] == num_dir)
        assert(weight_shape[2] == input_size)
        hidden_size = weight_shape[1]//3
        N = weight_shape[0] * weight_shape[1]
        K = weight_shape[2]
        weight_shape = [N, K]
        fc_weight = weight_name + "_FC"
        self.addTensor(fc_weight, weight_tensor.tensor_data, weight_shape)
        weight_op = self.CVI.add_load_file_op(fc_weight, weight_shape)
        operands.append(weight_op)

        bias_name = onnx_node.inputs[3]
        bias_tensor = self.getTensor(bias_name)
        [w_bias, r_bias] = np.split(bias_tensor.tensor_data, 2, axis=1)
        fc_bias = bias_name + "_FC"
        bias_shape = [N]
        self.addTensor(fc_bias, w_bias, bias_shape)
        bias_op = self.CVI.add_load_file_op(fc_bias, bias_shape)
        operands.append(bias_op)

        output_shape = [seq_length, batch_size, N]
        fc_op = self.CVI.add_fully_connected_op(
            "{}_FC".format(onnx_node.name), operands, output_shape)

        operands.clear()
        operands.append(fc_op)

        recurrence_name = onnx_node.inputs[2]
        recurrence_tensor = self.getTensor(recurrence_name)
        recurrence_op = self.CVI.add_load_file_op(
            recurrence_name, recurrence_tensor.shape)
        operands.append(recurrence_op)

        new_bias = bias_name + "_recurrence"
        bias_shape = [num_dir, 3 * hidden_size]
        self.addTensor(new_bias, r_bias, bias_shape)
        r_bias_op = self.CVI.add_load_file_op(new_bias, bias_shape)
        operands.append(r_bias_op)

        num_input = len(onnx_node.inputs)

        if num_input > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError(
                "GRU does not test the case of specify the sequence_lens.")

        if num_input > 5 and len(onnx_node.inputs[5]) != 0:
            initial_h_name = onnx_node.inputs[5]
            init_op, _, tensor_type = self.getOperand(initial_h_name)

            if tensor_type == TensorType.TENSOR:
                initial_h_tensor = self.getTensor(initial_h_name)
                initial_h_op = self.CVI.add_load_file_op(
                    initial_h_name, initial_h_tensor.shape)
                operands.append(initial_h_op)
            else:
                operands.append(init_op)

        out0 = onnx_node.outputs[0]
        out1 = onnx_node.outputs[1]
        need0 = len(out0) > 0 and self.check_need(out0)
        need1 = len(out1) > 0 and self.check_need(out1)
        if need0 and need1:
            # TODO: out1 = out0[-1]
            raise RuntimeError("GRU only support one output currently.")
        elif need0:
            output_shape = [seq_length, num_dir, batch_size, hidden_size]
            name = "{}_{}".format(out0, onnx_node.op_type)
            gru_op = self.CVI.add_gru_op(
                name, operands, output_shape, **gru_param)
            self.addOperand(out0, gru_op, output_shape,
                            TensorType.ACTIVATION)
        elif need1:
            output_shape = [num_dir, batch_size, hidden_size]
            name = "{}_{}".format(out1, onnx_node.op_type)
            gru_op = self.CVI.add_gru_op(
                name, operands, output_shape, **gru_param)
            self.addOperand(out1, gru_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_hard_sigmoid_op(self, onnx_node):
        operands = list()
        assert(onnx_node.op_type == "HardSigmoid")
        alpha = onnx_node.attrs.get("alpha", 1.0 / 6)
        beta = onnx_node.attrs.get("beta", 0.5)
        hard_sigmoid_param = {
            'alpha': alpha,
            'beta': beta,
        }
        op1, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])

        if tensor_type == tensor_type.TENSOR:
            raise RuntimeError("Hardsigmoid does not support TENSOR input so far.")

        operands.append(op1)

        # convert to clip(in * alpha + beta, 0, 1)
        tensor_data = np.full(input_shape[1], alpha)
        alpha_name = "{}_hardSdigmoid_alpha".format(onnx_node.name)
        self.addTensor(alpha_name, tensor_data, tensor_data.shape)
        op2 = self.CVI.add_load_file_op(alpha_name, tensor_data.shape)
        operands.append(op2)
        tensor_data = np.full(input_shape[1], beta)
        beta_name = "{}_hardSdigmoid_beta".format(onnx_node.name)
        self.addTensor(beta_name, tensor_data, tensor_data.shape)
        op3 = self.CVI.add_load_file_op(beta_name, tensor_data.shape)
        operands.append(op3)
        output_shape = input_shape
        scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)  # TensorType.ACTIVATION
        # relu6
        clip_param = {
            'min':  0.,
            'max':  1.,
        }
        relu_op = self.CVI.add_relu_op("{}_relu6_relu{}".format(onnx_node.name, onnx_node.op_type), [scale_op], output_shape)
        clip_op = self.CVI.add_clip_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [relu_op], output_shape, **clip_param)
        self.addOperand(onnx_node.name, clip_op, output_shape, TensorType.ACTIVATION)

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

    def convert_log_op(self, onnx_node):
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        if tensor_type == TensorType.TENSOR:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.log(tensor_data)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands = list()
            operands.append(op)
            output_shape = input_shape
            log_op = self.CVI.add_log_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, log_op, output_shape, TensorType.ACTIVATION)

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
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        seq_length, batch_size, input_size = input_shape

        bidirectional = True if onnx_node.attrs.get(
            "direction", 'forward') == b'bidirectional' else False
        lstm_param = {
            'bidirectional': bool(bidirectional),
        }
        num_dir = 2 if bidirectional else 1
        noneOp = self.add_none_op()
        # fc x*weight+bias first
        operands = list()
        operands.append(op)

        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_shape = weight_tensor.shape
        assert(weight_shape[0] == num_dir)
        assert(weight_shape[2] == input_size)
        hidden_size = weight_shape[1]//4
        N = weight_shape[0] * weight_shape[1]
        K = weight_shape[2]
        weight_shape = [N, K]
        fc_weight = weight_name + "_FC"
        self.addTensor(fc_weight, weight_tensor.tensor_data, weight_shape)
        weight_op = self.CVI.add_load_file_op(fc_weight, weight_shape)
        operands.append(weight_op)

        bias_name = onnx_node.inputs[3]
        bias_tensor = self.getTensor(bias_name)
        [w_bias, r_bias] = np.split(bias_tensor.tensor_data, 2, axis=1)
        fc_bias = bias_name + "_FC"
        bias_shape = [N]
        self.addTensor(fc_bias, w_bias, bias_shape)
        bias_op = self.CVI.add_load_file_op(fc_bias, bias_shape)
        operands.append(bias_op)

        output_shape = [seq_length, batch_size, N]
        fc_op = self.CVI.add_fully_connected_op(
            "{}_FC".format(onnx_node.name), operands, output_shape)

        operands.clear()
        operands.append(fc_op)

        recurrence_name = onnx_node.inputs[2]
        recurrence_tensor = self.getTensor(recurrence_name)
        recurrence_op = self.CVI.add_load_file_op(
            recurrence_name, recurrence_tensor.shape)
        operands.append(recurrence_op)

        new_bias = bias_name + "_recurrence"
        bias_shape = [num_dir, 4 * hidden_size]
        self.addTensor(new_bias, r_bias, bias_shape)
        r_bias_op = self.CVI.add_load_file_op(new_bias, bias_shape)
        operands.append(r_bias_op)

        num_input = len(onnx_node.inputs)

        if num_input > 4 and len(onnx_node.inputs[4]) != 0:
            raise RuntimeError(
                "GRU does not test the case of specify the sequence_lens.")

        initial_h_name = ""
        initial_h_op = None
        if num_input > 5 and len(onnx_node.inputs[5]) != 0:
            initial_h_name = onnx_node.inputs[5]
            init_op, _, tensor_type = self.getOperand(initial_h_name)

            if tensor_type == TensorType.TENSOR:
                initial_h_tensor = self.getTensor(initial_h_name)
                initial_h_op = self.CVI.add_load_file_op(
                    initial_h_name, initial_h_tensor.shape)
                operands.append(initial_h_op)
            else:
                operands.append(init_op)
        else:
            operands.append(noneOp)

        if num_input > 6 and len(onnx_node.inputs[6]) != 0:
            initial_c_name = onnx_node.inputs[6]
            init_op, _, tensor_type = self.getOperand(initial_c_name)

            if tensor_type == TensorType.TENSOR:
                initial_c_tensor = self.getTensor(initial_c_name)
                if initial_c_name == initial_h_name:
                    operands.append(initial_h_op)
                else:
                    initial_c_op = self.CVI.add_load_file_op(
                        initial_c_name, initial_c_tensor.shape)
                    operands.append(initial_c_op)
            else:
                operands.append(init_op)
        else:
            operands.append(noneOp)

        out0 = onnx_node.outputs[0] # all
        out1 = onnx_node.outputs[1] # h last
        out2 = onnx_node.outputs[2] # c last
        need0 = len(out0) > 0 and self.check_need(out0)
        need1 = len(out1) > 0 and self.check_need(out1)
        need2 = len(out2) > 0 and self.check_need(out2)
        if need1 or need2 or not need0:
            raise RuntimeError("LSTM only support first output currently.")
        output_shape = [seq_length, num_dir, batch_size, hidden_size]
        name = "{}_{}".format(out0, onnx_node.op_type)
        lstm_op = self.CVI.add_lstm_op(name, operands, output_shape, **lstm_param)
        self.addOperand(out0, lstm_op, output_shape, TensorType.ACTIVATION)


    def convert_matmul_op(self, onnx_node):
        assert(onnx_node.op_type == "MatMul")
        # Use fully connectly op, set bias is zero
        #(M, K) * (K, N) => (M, N)
        lhs_op, lhs_shape, _ = self.getOperand(onnx_node.inputs[0])
        rhs_op, rhs_shape, rhs_type = self.getOperand(onnx_node.inputs[1])
        K = rhs_shape[-2]
        N = rhs_shape[-1]
        output_shape = list(lhs_shape)
        output_shape[-1] = N
        fc_name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        # rhs is weight
        if rhs_type == TensorType.TENSOR:
            operands = list()
            weight_name = "{}_add_weight".format(onnx_node.name)
            weight_tensor = self.getTensor(onnx_node.inputs[1]).tensor_data
            weight_tensor_shape = weight_tensor.shape
            order = [ i for i in range(len(weight_tensor_shape))]
            order[-1],order[-2] = order[-2],order[-1]
            weight_tensor = np.ascontiguousarray(np.transpose(weight_tensor, order))

            operands.append(lhs_op)
            rhs_shape = weight_tensor.shape
            self.addTensor(weight_name, weight_tensor, rhs_shape)
            rhs_op = self.CVI.add_load_file_op(weight_name, rhs_shape)
            operands.append(rhs_op)

            if len(onnx_node.inputs) == 3:
                bias_tensor = self.getTensor(onnx_node.inputs[2]).tensor_data
                bias_op = self.CVI.add_load_file_op(onnx_node.inputs[2], bias_tensor.shape)
                operands.append(bias_op)

            fc_op = self.CVI.add_fully_connected_op(fc_name, operands, output_shape)
            self.addOperand(onnx_node.name, fc_op, output_shape, TensorType.ACTIVATION)
        else:
            matmul_op = self.CVI.add_matmul_op(fc_name, [lhs_op, rhs_op], output_shape)
            self.addOperand(onnx_node.name, matmul_op, output_shape, TensorType.ACTIVATION)

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
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.ACTIVATION:
            # put activation first
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_mul_op(onnx_node)
            return
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
            if np.prod(input_shape2) == 1 or np.prod(input_shape2) == channel:
                if len(mul_value.flatten()) == 1:
                    weight_data = np.full(channel, mul_value.flatten()[0]) # broadcast via channel
                else:
                    weight_data = mul_value
                weight_name = "{}_mul_weight".format(onnx_node.inputs[0])
                weight_shape = list(weight_data.shape)
                self.addTensor(weight_name, weight_data, weight_shape)
                weight_op = self.CVI.add_load_file_op(weight_name, weight_shape)
                output_shape = input_shape1
                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op1, weight_op], output_shape)
                self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)
                return
            output_shape = self.bcast_shape(input_shape1, input_shape2)
            weight_name = onnx_node.inputs[1]
            weight_data = mul_value
            if output_shape == input_shape1:
                weight = np.zeros(output_shape, dtype=np.float32)
                weight_data = weight + mul_value
                weight_name = onnx_node.inputs[1] + "_broadcast"
                input_shape2 = output_shape
            if input_shape1 == input_shape2:
                self.addTensor(weight_name, weight_data, input_shape2)
                op3 = self.CVI.add_load_file_op(weight_name, input_shape2)
                mul_op = self.CVI.add_eltwise_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op1, op3], input_shape1)
                self.addOperand(onnx_node.name, mul_op, input_shape1, TensorType.ACTIVATION)
                return
        else:
            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            output_shape = self.same_shape(input_shape1, input_shape2)
            if output_shape != None:
                mul_op = self.CVI.add_eltwise_mul_op(name, [op1, op2], output_shape)
                self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)
                return
            output_shape = self.bcast_shape(input_shape1, input_shape2)
            if output_shape != None:
                if np.prod(input_shape2) > np.prod(input_shape1):
                    op1, op2 = op2, op1
                mul_op = self.CVI.add_broadcast_mul_op(name, [op1, op2], output_shape)
                self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)
                return
        raise RuntimeError("Mul {} {} not support now".format(input_shape1, input_shape2))


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

    def convert_reflectionpad_op(self, onnx_node):
        #assert(onnx_node.op_type == "ReflectionPad")

        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        num_dims = len(input_shape)
        zeros = [0] * num_dims * 2

        if len(onnx_node.inputs) > 1:
            # padding data from input
            _, _, pad_data_type = self.getOperand(onnx_node.inputs[1])
            if pad_data_type == TensorType.TENSOR:
                pads = list(self.getTensor(onnx_node.inputs[1]).tensor_data)
            else:
                raise RuntimeError(
                    "not support paddings data with runtime data")
        else:
            pads = onnx_node.attrs.get("pads", zeros)
        if pads == zeros:
            self.addOperand(onnx_node.name, op, input_shape,
                            TensorType.ACTIVATION)
        lpad = pads[num_dims - 1]
        rpad = pads[num_dims * 2 - 1]
        pads[num_dims - 1] = 0
        pads[num_dims*2-1] = 0
        if pads != zeros:
            raise RuntimeError("only support last dim to do reflectionpad!")

        # 1d case
        pads_param = {
            "pads": [lpad, rpad],
        }
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1] + lpad + rpad

        new_op = self.CVI.add_reflectionpad_op("{}_{}".format(
            onnx_node.name, onnx_node.op_type), [op], output_shape, **pads_param)
        self.addOperand(onnx_node.name, new_op,
                        output_shape, TensorType.ACTIVATION)


    def convert_pad_op(self, onnx_node):
        assert(onnx_node.op_type == "Pad")

        # get pad mode
        mode = onnx_node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")

        if mode == "constant":
            pass
        elif mode == "edge":
            pass
        elif mode == 'reflect':
            return self.convert_reflectionpad_op(onnx_node)
        else:
            raise RuntimeError("Todo support pad op mode {}".format(mode))

        # opset 11, value from second input
        if len(onnx_node.inputs) > 2:
            constant_value = self.getTensor(onnx_node.inputs[2]).tensor_data
        else:
            constant_value = onnx_node.attrs.get("value", 0.0)

        op, _input_shape, input_type = self.getOperand(onnx_node.inputs[0])

        input_shape = list(_input_shape)

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
          "pad_mode": mode,
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
            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            if len(input_shape) == 3:
                reshape_input_shape = list(input_shape)
                reshape_input_shape.insert(0, 1)
                reshape_op = self.CVI.add_reshape_op(name + "_input_reshape", [op], reshape_input_shape)
                reshape_output_shape = list(output_shape)
                reshape_output_shape.insert(0, 1)
                pads.insert(0, 0)
                pads.insert(4, 0)
                pads_op = self.CVI.add_pad_op(name + "_4dim_pad", [reshape_op], reshape_output_shape,
                              pads=pads, const_val=constant_value)
                reshape_back_op = self.CVI.add_reshape_op(name, [pads_op], output_shape)
                self.addOperand(onnx_node.name, reshape_back_op, output_shape, TensorType.ACTIVATION)
            else:
                pads_op = self.CVI.add_pad_op(name, [op], output_shape, **pads_param)
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
            slope_shape = (1, output_shape[1], 1, 1)
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

    def convert_pow_op(self, onnx_node):
        assert(onnx_node.op_type == "Pow")
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])
        if tensor_type1 == TensorType.TENSOR and np.prod(input_shape1) == 1:
            data = self.getTensor(onnx_node.inputs[1]).tensor_data.astype(np.int32)
            if data > 4:
                raise RuntimeError("Power not support {} {}".format(input_shape0, input_shape1))
            mul_op = op0
            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            for i in range(data - 1):
                name_i = "{}_{}".format(name, i)
                mul_op = self.CVI.add_eltwise_mul_op(name_i, [mul_op, op0], input_shape0)
            self.addOperand(onnx_node.name, mul_op, input_shape0, TensorType.ACTIVATION)
            return
        raise RuntimeError("Power not support {} {}".format(input_shape0, input_shape1))

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
                        output_shape[i] = input_shape1[i]
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
                logger.info(self.CVI.print_module())
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
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            raise RuntimeError("Second type must be {}".format(TensorType.TENSOR))

    @staticmethod
    def half_pixel_scale(scale, pad):
        # for example, scale = 2, scale_list = [0.75, 0.25, 0.25, 0.75]
        count = int(scale)
        scale_list = [0.0] * count * 2
        for i in range(count):
            idx = pad - i
            if idx < 0:
                idx = idx + count
            distance = (0.5 + i) / scale + 0.5
            distance = distance - int(distance)
            scale_list[idx] = 1 - distance
            scale_list[count + idx] = distance
        return scale_list

    # when resize by linear half pixel, with integer scale_h and integer scale_w
    def resize_to_conv1(self, onnx_node, op, input_shape, output_shape, scale_h, scale_w):
        # pad edge
        pads_param = {
            "pads": [0, 0, 1, 1, 0, 0, 1, 1],
            "const_val": 0,
            "pad_mode": 'edge',
        }
        pad_shape = list(input_shape)
        for idx, v in enumerate(pad_shape):
            pad_shape[idx] = int(
                pads_param['pads'][idx] + v + pads_param['pads'][idx + 4])

        name = "{}_{}_pad_edge".format(
            onnx_node.name, onnx_node.op_type)
        pads_op = self.CVI.add_pad_op(
            name, [op], pad_shape, **pads_param)

        # conv to replace with resize
        input_shape = list(pad_shape)
        ic = input_shape[1]
        # w-direction
        ins_h = int(scale_h - 1)
        ins_w = int(scale_w - 1)
        stride_h = 1
        stride_w = 1
        kh = int(2 * scale_h)
        kw = int(2 * scale_w)
        pad_t = int(scale_h/2) - 1
        pad_b = int(scale_h) - pad_t - 2
        pad_l = int(scale_w/2) - 1
        pad_r = int(scale_w) - pad_l - 2
        conv_param = {
            'stride_h':  stride_h,
            'stride_w':  stride_w,
            'padding': "VALID",
            'dilation_h': 1,
            'dilation_w': 1,
            'padding_t': pad_t,
            'padding_b': pad_b,
            'padding_l': pad_l,
            'padding_r': pad_r,
            'group': ic,
            'is_dw': True,
            'with_bias': False,
            'do_relu': False,
            'ins': [ins_w, ins_h],
        }

        # weight_shape = [ic, 1, 1, kh, kw]
        factor_w = np.array(self.half_pixel_scale(
            scale_w, pad_l)).reshape(1, kw)
        factor_h = np.array(self.half_pixel_scale(
            scale_h, pad_t)).reshape(kh, 1)
        factor = np.dot(factor_h, factor_w).reshape(1, 1, 1, kh, kw)
        conv_tensor_data = np.tile(np.array(factor), (ic, 1, 1, 1, 1))
        weight_name = "{}_add_weight".format(onnx_node.name)
        self.addTensor(weight_name, conv_tensor_data,
                       conv_tensor_data.shape)
        weight_op = self.CVI.add_load_file_op(
            weight_name, conv_tensor_data.shape)

        operands = list()
        operands.append(pads_op)
        operands.append(weight_op)

        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        conv_op = self.CVI.add_conv_op(
            name, operands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op,
                        output_shape, TensorType.ACTIVATION)
        return

    # when resize by linear half pixel, with scale_h = 0.5 and scale_w = 0.5
    def resize_to_conv2(self, onnx_node, op, input_shape, output_shape):
        kh, kw, sh, sw = 2, 2, 2, 2
        ic = input_shape[1]
        conv_param = {
            'stride_h':  sh,
            'stride_w':  sw,
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
            'ins': [0, 0],
        }
        filter_shape = [ic, 1, 1, kh, kw]
        filter_data = np.array([0.25, 0.25, 0.25, 0.25]
                               * ic, np.float32).reshape(filter_shape)
        filter_name = "{}_conv_filter".format(onnx_node.name)
        self.addTensor(filter_name, filter_data, filter_shape)
        filter_op = self.CVI.add_load_file_op(filter_name, filter_shape)
        operands = list()
        operands.append(op)
        operands.append(filter_op)

        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        conv_op = self.CVI.add_conv_op(
            name, operands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op,
                        output_shape, TensorType.ACTIVATION)
        return

    # when resize by nearest, with integer scale_h and integer scale_w
    def resize_to_conv3(self, onnx_node, op, input_shape, output_shape, scale_h, scale_w):
        operands = [op]
        ic = input_shape[1]
        # use deconv(depthwise)
        deconv_param = {
            'stride_h':  int(scale_h),
            'stride_w':  int(scale_w),
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
        weight_shape = [ic, 1, 1, int(scale_h), int(scale_w)]
        tensor_data = np.full(weight_shape, 1, np.float32)
        weight_name = "{}_add_weight".format(onnx_node.name)
        self.addTensor(weight_name, tensor_data, tensor_data.shape)
        weight_op = self.CVI.add_load_file_op(
            weight_name, tensor_data.shape)
        operands.append(weight_op)

        deconv_op = self.CVI.add_deconv_op("{}_{}".format(
            onnx_node.name, onnx_node.op_type), operands, output_shape, **deconv_param)
        self.addOperand(onnx_node.name, deconv_op,
                        output_shape, TensorType.ACTIVATION)
        return

    def resize_to_interp(self, onnx_node, op, output_shape, coordinate_transformation_mode):
        oh, ow = output_shape[2], output_shape[3]
        attr = {
            'height': int(oh),
            'width': int(ow),
            'pad_beg': 0,
            'pad_end': 0,
            'shrink_factor': 0,
            'zoom_factor': 0,
            'coordinate_transformation_mode': coordinate_transformation_mode
        }

        interp_op = self.CVI.add_interp_op(
            "{}_{}".format(onnx_node.name, onnx_node.op_type), [op], output_shape, **attr)
        self.addOperand(onnx_node.name, interp_op,
                        output_shape, TensorType.ACTIVATION)
        return

    def convert_resize_op(self, onnx_node):
        assert(onnx_node.op_type == "Resize")
        mode = onnx_node.attrs.get("mode", "nearest")

        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        scale_factor = []
        sizes = []

        if len(onnx_node.inputs) > 2:
            # onnx opset 11
            scale_factor = self.getTensor(onnx_node.inputs[2]).tensor_data
            if len(scale_factor) == 0:
                sizes = self.getTensor(onnx_node.inputs[3]).tensor_data
                scale_factor = sizes / input_shape
            else:
                sizes = input_shape * scale_factor
        else:
            # opset 10
            scale_factor = self.getTensor(onnx_node.inputs[1]).tensor_data
            sizes = input_shape * scale_factor

        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            raise RuntimeError("Resize only support h/w")

        output_shape = [int(i) for i in sizes]
        scale_h = scale_factor[2]
        scale_w = scale_factor[3]
        if scale_h == 1.0 and scale_w == 1.0:
            self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)
            return

        coordinate_transformation_mode = onnx_node.attrs.get(
                "coordinate_transformation_mode", "half_pixel")
        if mode == b'linear':
            if output_shape[2] > 1 and output_shape[3] > 1 and coordinate_transformation_mode == b"pytorch_half_pixel":
                coordinate_transformation_mode = b"half_pixel"
            if coordinate_transformation_mode == b"half_pixel":
                if int(scale_h) == scale_h and int(scale_w) == scale_w:
                    self.resize_to_conv1(onnx_node, op, input_shape, output_shape, scale_h, scale_w)
                    return
                if scale_h == scale_w and scale_h == 0.5:
                    self.resize_to_conv2(onnx_node, op, input_shape, output_shape)
                    return
            # by cpu
            self.resize_to_interp(onnx_node, op, output_shape, coordinate_transformation_mode)
            return
        elif mode == b'nearest':
            if scale_h == int(scale_h) and scale_w == int(scale_w):
                self.resize_to_conv3(onnx_node, op, input_shape, output_shape, scale_h, scale_w)
                return
            mode = "nearest"
            if coordinate_transformation_mode != b"asymmetric":
                mode = "nearest_half_pixel"
            self.resize_to_interp(onnx_node, op, output_shape, mode)
            return
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
        starts = []
        ends = []
        axes = []
        # start
        if (len(onnx_node.inputs) > 1):
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
        else:
            starts = onnx_node.attrs.get('starts')
            ends = onnx_node.attrs.get('ends')
            axes = onnx_node.attrs.get('axes')

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
                if steps[0] > 1 and (len(axes) == 1 and axes[0] < 2):
                    raise RuntimeError("not support step > 1 and slice step with n/c")

                if steps[0] == -1:
                    tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
                    output_data = tensor_data[starts[0]:ends[0]:steps[0]]
                    output_shape = list(output_data.shape)
                    self.addTensor(onnx_node.name, output_data, output_shape)
                    self.addOperand(onnx_node.name, None,
                                    output_shape, TensorType.TENSOR)
                elif steps[0] > 1:
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
                    end = int(end)
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

            output_shape = list(crop_shape)
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
            axis = onnx_node.attrs.get('axis', -1)
            if axis < 0:
                axis += len(input_shape)
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

    def convert_layernorm_op(self, onnx_node):
        assert(onnx_node.op_type == "LayerNorm")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        # take input last dimension as default normal_shape
        normal_shape = [input_shape[-1]]
        operands = [op]
        noneOp = self.add_none_op()
        operands.append(noneOp)  # table
        operands.append(noneOp)  # mantissa
        eps = onnx_node.attrs.get('epsilon', 1e-5)
        attrs = {
            "eps": eps,
            "normal_shape": list(normal_shape)
        }
        num_input = len(onnx_node.inputs)
        if num_input == 3:
            weight = self.getTensor(onnx_node.inputs[1])
            weight_op = self.CVI.add_load_file_op(onnx_node.inputs[1], weight.shape)
            operands.append(weight_op)
            bias = self.getTensor(onnx_node.inputs[2])
            bias_op = self.CVI.add_load_file_op(onnx_node.inputs[2], bias.shape)
            operands.append(bias_op)
        elif num_input == 1:
            operands.append(noneOp)
            operands.append(noneOp)
        else:
            raise RuntimeError("num_input must be 1 or 3 (with scale and bias")
        layernorm_op = self.CVI.add_layernorm_op("{}_{}".format(onnx_node.name, "Add"), operands, input_shape, **attrs)
        self.addOperand(onnx_node.name, layernorm_op,input_shape, TensorType.ACTIVATION)

    def convert_split_op(self, onnx_node):
        assert(onnx_node.op_type == "Split")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        axis = onnx_node.attrs.get('axis', 0)
        num_output = len(onnx_node.outputs)
        if axis < 0:
            axis += len(input_shape)
        slice = input_shape[axis] // num_output
        split = onnx_node.attrs.get('split', [slice] * num_output)
        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            outputs = np.split(data, split)
            for i in onnx_node.outputs:
                self.addTensor(str(i), outputs[i], list(outputs[i].shape))
                self.addOperand(str(i), None, list(outputs[i].shape), TensorType.TENSOR)
        else:
            offset = 0
            for i, name in zip(split, onnx_node.outputs):
                output_shape = list(input_shape)
                output_shape[axis] = i
                attr = {
                    "axis": axis,
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
        input_num = len(onnx_node.inputs)
        # opset 13 take axes as input
        if input_num == 2 :
            axis_value_list = self.getTensor(onnx_node.inputs[1]).tensor_data
        else:
            axis_value_list = onnx_node.attrs.get('axes', None)
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
            self.convert_activation_op(onnx_node)
        else:
            tensor_data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.sqrt(tensor_data)
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None,
                            output_shape, TensorType.TENSOR)

    def convert_sub_op(self, onnx_node):
        assert(onnx_node.op_type == "Sub")
        # Y = X0 - X1
        input_num = len(onnx_node.inputs)
        assert(input_num == 2)
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[1])
        if tensor_type0 == TensorType.ACTIVATION and tensor_type1 == TensorType.ACTIVATION:
            name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
            output_shape = self.same_shape(input_shape0, input_shape1)
            if output_shape != None:
                sub_op = self.CVI.add_eltwise_sub_op(name, [op0, op1], output_shape)
                self.addOperand(onnx_node.name, sub_op, output_shape, TensorType.ACTIVATION)
                return
            output_shape = self.bcast_shape(input_shape0, input_shape1)
            if output_shape != None:
                assert(np.prod(input_shape0) > np.prod(input_shape1)) # support later
                sub_op = self.CVI.add_broadcast_sub_op(name, [op0, op1], output_shape)
                self.addOperand(onnx_node.name, sub_op, output_shape, TensorType.ACTIVATION)
                return
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
                bias_op = self.CVI.add_load_file_op(bias_name, bias_shape)
                output_shape = input_shape0
                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op0, weight_op, bias_op], output_shape)
                self.addOperand(onnx_node.name, scale_op,
                                output_shape, TensorType.ACTIVATION)
                return

        elif tensor_type0 == TensorType.TENSOR and tensor_type1 == TensorType.TENSOR:
            tensor_data0 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data1 = self.getTensor(onnx_node.inputs[1]).tensor_data
            # sub
            output_data = tensor_data0 - tensor_data1
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
            return
        raise RuntimeError("Sub {} {} not support now".format(input_shape0, input_shape1))

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

    def convert_tile_op(self, onnx_node):
        assert(onnx_node.op_type == "Tile")
        input_op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        _, tile_shape, tile_type = self.getOperand(onnx_node.inputs[1])
        assert(tensor_type == TensorType.ACTIVATION)
        assert(tile_type == TensorType.TENSOR)
        assert(1 == len(tile_shape))
        assert(tile_shape[0] == len(input_shape))
        tile_data = self.getTensor(onnx_node.inputs[1]).tensor_data
        if np.prod(tile_data) == 1:
            self.addOperand(onnx_node.name, input_op, input_shape, TensorType.ACTIVATION)
            return
        last_shape = list(input_shape)
        last_op = input_op
        last_i = 0
        last_name = ""
        for i in range(tile_shape[0]):
            last_i = tile_shape[0] - i - 1
            if tile_data[last_i] != 1:
                break
        for i in range(last_i+1):
            if tile_data[i] == 1:
                continue
            attr = {
                'axis': i,
                'tiles': int(tile_data[i])
            }
            last_name = onnx_node.name
            if i != last_i:
                last_name += "_{}".format(i)
            last_shape[i] = last_shape[i] * tile_data[i]
            last_op = self.CVI.add_tile_op("{}_{}".format(last_name, onnx_node.op_type), [last_op], last_shape, **attr)
        self.addOperand(last_name, last_op, last_shape, TensorType.ACTIVATION)

    def convert_transpose_op(self, onnx_node):
        assert(onnx_node.op_type == "Transpose")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        # default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
        perm_default = list(np.arange(len(input_shape))[::-1])
        transpose_perm = onnx_node.attrs.get('perm', perm_default)
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
            else:
                output_shape = list(input_shape)
                for i in range(len(transpose_perm)):
                    output_shape[i] = input_shape[transpose_perm[i]]
                attr = {
                    'order': transpose_perm,
                }
                permute_op = self.CVI.add_permute_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op], output_shape, **attr)
                self.addOperand(onnx_node.name, permute_op, output_shape, TensorType.ACTIVATION)

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
        input_num = len(onnx_node.inputs)
        # opset 13 take axes as input
        if input_num == 2 :
            axis_value_list = self.getTensor(onnx_node.inputs[1]).tensor_data
        else:
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
            tmp_data = np.expand_dims(np.zeros(input_shape), axis=axis_value_list)
            new_shape = list(tmp_data.shape)
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op], new_shape)
            self.addOperand(onnx_node.name, reshape_op, new_shape, TensorType.ACTIVATION)

    def convert_yolo_detection_op(self, onnx_node):
        assert(onnx_node.op_type == 'YoloDetection')
        _, input_shape, _ = self.getOperand(onnx_node.inputs[0])

        operands = list()
        for input in onnx_node.inputs:
            op, _, _ = self.getOperand(input)
            operands.append(op)

        nms_threshold = onnx_node.attrs['nms_threshold']
        obj_threshold = onnx_node.attrs['obj_threshold']

        net_input_h = onnx_node.attrs.get("net_input_h", 608)
        net_input_w = onnx_node.attrs.get("net_input_w", 608)
        keep_topk = onnx_node.attrs.get('keep_topk', 200)
        spp_net = onnx_node.attrs.get('spp_net', False)
        tiny = onnx_node.attrs.get('tiny', False)
        yolo_v4 = onnx_node.attrs.get('yolo_v4', False)
        class_num = onnx_node.attrs.get('num_classes', 80)

        anchors = ','.join([str(x) for x in onnx_node.attrs['anchors']])
        if not anchors:
            if tiny:
                anchors = "10,14,23,27,37,58,81,82,135,169,344,319"
            elif yolo_v4:
                anchors = "142,110,192,243,459,401,36,75,76,55,72,146,12,16,19,36,40,28"
            else:
                anchors = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326"

        param = {
            'net_input_h': net_input_h,
            "net_input_w": net_input_w,
            "nms_threshold": nms_threshold,
            "obj_threshold": obj_threshold,
            "keep_topk": keep_topk,
            "spp_net": spp_net,
            "tiny": tiny,
            "yolo_v4": yolo_v4,
            "class_num": class_num,
            "anchors": anchors
        }
        output_shape = [input_shape[0], 1, keep_topk, 6]
        new_op = self.CVI.add_yolo_detection_op(
            "{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **param)
        self.addOperand(onnx_node.name, new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_reduce_l2_op(self, onnx_node):
        assert(onnx_node.op_type == "ReduceL2")
        checkKey(onnx_node.attrs, 'axes')
        axes = onnx_node.attrs['axes']
        keepdims = onnx_node.attrs['keepdims']
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape0
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
        reduce_l2_op = self.CVI.add_reduce_l2_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
        self.addOperand(onnx_node.name, reduce_l2_op, output_shape, TensorType.ACTIVATION)

    def convert_reduce_mean_op(self, onnx_node):
        assert(onnx_node.op_type == "ReduceMean")
        checkKey(onnx_node.attrs, 'axes')
        axes = onnx_node.attrs['axes']
        keepdims = onnx_node.attrs.get('keepdims', 1)
        op0, input_shape0, tensor_type0 = self.getOperand(onnx_node.inputs[0])
        output_shape = input_shape0

        axis = 0
        if len(axes) == 1:
            if axes[0] == -1:
                axes[0] = len(input_shape0) - 1

            if axes[0] != 2 and axes[0] != 3:
                raise RuntimeError("{} axis not support, please add".format(axes[0]))

            output_shape = list(input_shape0)
            if keepdims == 0:
                output_shape.pop(axes[0])
            else:
                output_shape[axes[0]] = 1

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

    def convert_std_op(self, onnx_node):
        assert(onnx_node.op_type == "Std")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        num_dim = len(input_shape)
        # take input last dimension as default normal_shape
        operands = [op]
        dims = onnx_node.attrs.get('dim')  # dim to do std
        unbiased = onnx_node.attrs.get('unbiased', True)
        keepdim = onnx_node.attrs.get('keepdim', False)

        if type(dims) != list:
            dims = list(dims)
        start_dim = num_dim
        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] = dims[i] + num_dim
            if start_dim > dims[i]:
                start_dim = dims[i]
        for i in range(start_dim, num_dim):
            if i not in dims:
                raise RuntimeError("std not support dims:{}\n".format(dims))

        attrs = {
            "start_dim": start_dim,
            "unbiased": unbiased,
        }
        output_shape = list(input_shape)
        if keepdim:
            for i in range(start_dim, num_dim):
                output_shape[i] = 1
        else:
            output_shape = input_shape[:start_dim]
        std_op = self.CVI.add_std_op("{}_{}".format(
            onnx_node.name, "Std"), operands, output_shape, **attrs)
        self.addOperand(onnx_node.name, std_op,
                        output_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_tensor()
        self.convert_node()
        self.refine_node()
        self.convert_graph()
        self.TensortoNpz()
