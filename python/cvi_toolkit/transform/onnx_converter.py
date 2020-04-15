from .mlirimporter import BaseConverterInterface, MLIRImporter, checkKey
from onnx import numpy_helper, mapping
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum

import logging
import numpy as np

from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
                    get_shape_size

class TensorType(Enum):
    ACTIVATION = 'ACTIVATION'
    TENSOR = 'TENSOR'

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
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

class OnnxNode():
    def __init__(self, node):
        self.name = str(node.name)
        if self.name == '':
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



class OnnxConverter(BaseConverterInterface):
    def __init__(self, model_name, onnx_model, mlir_file_path):
        self.model_name = model_name
        self.input_nodes = onnx_model.graph.input
        self.output_nodes = onnx_model.graph.output
        self.nodes = onnx_model.graph.node
        self.tensors = onnx_model.graph.initializer
        self.mlir_file_path = mlir_file_path

        self.converted_nodes = list()
        self.converted_tensors = list()

        self.valueMap = dict() # {op_name: (mlir op, shape)}
        self.CVI = None
        self.init_importer()
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avg_pool_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Mul" : lambda node: self.convert_mul_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid" :lambda node: self.convert_sigmoid_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
            "Upsample": lambda node: self.convert_upsample_op(node),
        }

    def init_importer(self):
        # get input shape
        inputs = list()
        for input in self.input_nodes:
            input_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output in self.output_nodes:
            output_shape = list()
            for dim in output.type.tensor_type.shape.dim:
                output_shape.append(dim.dim_value)
            outputs.append(output_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs)

    def addOperand(self, op_name, op, shape, tensor_type):
        cprint("add opernand name: {}\nshape: {}".format(op_name, shape, tensor_type), "yellow")
        self.valueMap[op_name] = (op, shape, tensor_type)

    def getOperand(self, op_name):
        return self.valueMap[op_name]

    def addTensor(self, op_name, tensor_data, tensor_shape):
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
                tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.output_tensor_file, **tensor_npz)

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

    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        # add input op
        for idx, input in enumerate(self.input_nodes):
            input_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
            input_op = self.CVI.add_input_op(input.name, idx)
            self.addOperand(input.name, input_op, input_shape, TensorType.ACTIVATION)
        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))
        # add node op
        for n in self.converted_nodes:
            n.print_info()
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)

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

    def convert_add_op(self, onnx_node):
        assert(len(onnx_node.inputs) == 2)
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        # broadcast add from constant
        if tensor_type1 == TensorType.ACTIVATION and tensor_type2 == TensorType.TENSOR:
            if len(input_shape2) ==1 and input_shape2[0] == 1:
                # we use depthwise for quantize, it could eq x * 1 + y

                operands = list()
                operands.append(op1)

                tensor_data = np.full(input_shape1[1], 1) # broadcast via channel
                weight_name = "{}_add_weight".format(onnx_node.inputs[0])
                self.addTensor(weight_name, tensor_data, tensor_data.shape)
                op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
                operands.append(op2)

                add_value = self.getTensor(onnx_node.inputs[1]).tensor_data
                tensor_data = np.full((input_shape1[1]), add_value[0]) # broadcast via channel
                bias_name = "{}_add_bias".format(onnx_node.inputs[0])
                self.addTensor(bias_name, tensor_data, tensor_data.shape)
                op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
                operands.append(op3)

                output_shape = input_shape1

                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)

            else:
                raise RuntimeError("TODO other axis")
        else:
            # eltwise add
            if input_shape1 != input_shape2:
                raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2))
            operands = list()
            operands.append(op1)
            operands.append(op2)
            output_shape = input_shape1

            add_op = self.CVI.add_eltwise_add_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_div_op(self, onnx_node):
        assert(len(onnx_node.inputs) == 2)
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        if len(input_shape2) ==1 and input_shape2[0] == 1:
            # div(x) = input * (1/x) = scale(1/x) = input * (1/x) + 0
            operands = list()
            operands.append(op1)

            div_value = self.getTensor(onnx_node.inputs[1]).tensor_data
            tensor_data = np.full(input_shape1[1], 1 / (div_value * 1.0)) # broadcast via channel
            weight_name = "{}_div_weight".format(onnx_node.inputs[0])
            self.addTensor(weight_name, tensor_data, tensor_data.shape)
            op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
            operands.append(op2)

            # TODO: add Nonp op
            tensor_data = np.full((input_shape1[1]), 0) # broadcast via channel
            bias_name = "{}_div_bias".format(onnx_node.inputs[0])
            self.addTensor(bias_name, tensor_data, tensor_data.shape)
            op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
            operands.append(op3)

            output_shape = input_shape1

            scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)

        else:
            raise RuntimeError("not implement yet")

    def convert_batchnorm_op(self, onnx_node):
        assert(onnx_node.op_type == "BatchNormalization")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = onnx_node.attrs['epsilon']
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
            np_tensor = np_tensor.astype(np.float32).flatten()
            # add new weight tensor

            self.addTensor(onnx_node.name, np_tensor, np_tensor.shape)
            self.addOperand(onnx_node.name, None, list(np_tensor.shape), TensorType.TENSOR)

        else:
            raise ValueError("Not Support {} type".format(data_type))

    def convert_concat_op(self, onnx_node):
        assert(onnx_node.op_type == "Concat")
        if len(onnx_node.inputs) < 2:
            raise ValueError("{} must great than 2".format(onnx_node.op_type))
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])

        axis = onnx_node.attrs['axis']
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            t1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            t2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            n_t = np.concatenate((t1, t2), axis=axis)
            self.addTensor(onnx_node.name, n_t, list(n_t.shape))
            self.addOperand(onnx_node.name, None, list(n_t.shape), TensorType.TENSOR)
        else:
            operands = [op1, op2]
            output_shape = list()
            for idx, (s1, s2) in enumerate(zip(input_shape1, input_shape2)):
                if  idx== axis:
                    output_shape.append(s1+s2)
                else:
                    assert(s1 == s2)
                    output_shape.append(s1)

            concat_op = self.CVI.add_concat_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis=axis)
            self.addOperand(onnx_node.name, concat_op, output_shape, TensorType.ACTIVATION)

    def convert_conv_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        conv_param = {
            'stride_h':  onnx_node.attrs['strides'][0],
            'stride_w':  onnx_node.attrs['strides'][1],
            'padding': "SAME" if onnx_node.attrs['pads'][0] > 0 else "VALID",
            'dilation_h': onnx_node.attrs['dilations'][0],
            'dilation_w': onnx_node.attrs['dilations'][1],
            'group': onnx_node.attrs['group'],
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
        }
        op, shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        filter_name = onnx_node.inputs[1]
        filter_tensor = self.getTensor(filter_name)
        filter_shape = filter_tensor.shape
        with_bias = False
        if (len(onnx_node.inputs) == 3):
            #with bias
            with_bias = True
            bias_name = onnx_node.inputs[2]
            bias_tensor = self.getTensor(bias_name)


        on = shape[0]
        oc = filter_tensor.shape[0] # feature map size
        oh = calcConv2DSpatial(
            shape[2],
            onnx_node.attrs['kernel_shape'][0],
            onnx_node.attrs['strides'][0],
            onnx_node.attrs['pads'][0],
            onnx_node.attrs['dilations'][0]
        )
        ow = calcConv2DSpatial(
            shape[3],
            onnx_node.attrs['kernel_shape'][1],
            onnx_node.attrs['strides'][1],
            onnx_node.attrs['pads'][1],
            onnx_node.attrs['dilations'][1]
        )

        if conv_param['group'] != 1:
            # filter shape s is in (g, oc/g, ic/g, kh, kw)
            g = conv_param['group']
            ic = shape[1]
            kh = onnx_node.attrs['kernel_shape'][0]
            kw = onnx_node.attrs['kernel_shape'][1]
            new_shape = [g, int(oc/g), int(ic/g), kh, kw]
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, new_shape)
        else:
            filter_op = self.CVI.add_load_file_op(filter_tensor.name, filter_shape)
        operands.append(filter_op)

        if with_bias:
            bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
            operands.append(bias_op)

        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op, output_shape, TensorType.ACTIVATION)

    def convert_clip_op(self, onnx_node):
        assert(onnx_node.op_type == "Clip")
        clip_param = {
            'min':  onnx_node.attrs['min'],
            'max':  onnx_node.attrs['max'],
        }
        op, shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)

        output_shape = shape
        conv_op = self.CVI.add_clip_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **clip_param)
        self.addOperand(onnx_node.name, conv_op, output_shape, TensorType.TENSOR)


    def convert_flatten_op(self, onnx_node):
        assert(onnx_node.op_type == "Flatten")
        if onnx_node.attrs["axis"] != 1:
            raise AttributeError("TODO: axis != 1 case")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = [input_shape[0], input_shape[1]]
        reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, reshape_op, output_shape, TensorType.ACTIVATION)

    def convert_gather_op(self, onnx_node):
        """
            first input is tensor data, second input is constant
        """
        assert(onnx_node.op_type == "Gather")
        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[1])
        if 'axis' in onnx_node.attrs:
            axis = onnx_node.attrs['axis']
        else:
            axis = 0
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            input_data =  self.getTensor(onnx_node.inputs[0]).tensor_data
            gather_indices = self.getTensor(onnx_node.inputs[1]).tensor_data
        else:
            raise RuntimeError("TODO: our IR no gather define")

        new_shape = input_shape1
        if new_shape[axis] > len(gather_indices):
            new_shape[axis] = len(gather_indices)
        else:
            raise ValueError("Gather input shape dim {} ({}) must great than {} ({})".format(axis, input_shape, len(gather_indices), gather_indices))
        # TODO: our IR no Gather function, please add
        if tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            new_data = np.take(input_data, gather_indices.tolist())
            self.addTensor(onnx_node.name, new_data, list(new_data.shape))
            self.addOperand(onnx_node.name, None, new_shape, TensorType.TENSOR)

    def convert_gemm_op(self, onnx_node):
        assert(onnx_node.op_type == "Gemm")
        #(M, K) * (K, N) => (M, N)
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])

        operands = list()
        operands.append(op)
        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
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

    def convert_global_avg_pool_op(self, onnx_node):
        assert(onnx_node.op_type == "GlobalAveragePool")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        # print(input_shape)
        on = input_shape[0]
        oc = input_shape[1]

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
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(onnx_node.name, pool_avg_op, output_shape, TensorType.ACTIVATION)

    def convert_maxpool_op(self, onnx_node):
        assert(onnx_node.op_type == "MaxPool")
        pool_max_2d_param = {
            'stride_h': onnx_node.attrs['strides'][0],
            'stride_w': onnx_node.attrs['strides'][1],
            'kernel_h': onnx_node.attrs['kernel_shape'][0],
            'kernel_w': onnx_node.attrs['kernel_shape'][1],
            'padding_b': onnx_node.attrs['pads'][0],
            'padding_r': onnx_node.attrs['pads'][1],
            'padding_t': onnx_node.attrs['pads'][2],
            'padding_l': onnx_node.attrs['pads'][3],
            'do_relu': False,
        }

        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        on = input_shape[0]
        oc = input_shape[1]
        oh = calcPool2DFloor(input_shape[2], onnx_node.attrs['kernel_shape'][0], onnx_node.attrs['strides'][0], onnx_node.attrs['pads'][0])
        ow = calcPool2DFloor(input_shape[3], onnx_node.attrs['kernel_shape'][1], onnx_node.attrs['strides'][1], onnx_node.attrs['pads'][1])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_max_2d_param)
        self.addOperand(onnx_node.name, pool_max_op, output_shape, TensorType.ACTIVATION)

    def convert_relu_op(self, onnx_node):
        assert(onnx_node.op_type == "Relu")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        relu_op = self.CVI.add_relu_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, relu_op, output_shape, TensorType.ACTIVATION)

    def convert_mul_op(self, onnx_node):
        assert(onnx_node.op_type == "Mul")
        op1, input_shape1, _ = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, _ = self.getOperand(onnx_node.inputs[1])

        operands = list()
        operands.append(op1)
        operands.append(op2)
        if input_shape1 == input_shape2:
            #eltwise mul
            output_shape = input_shape1
            mul_op = self.CVI.add_eltwise_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        else:
            #broadcast mul
            # FixMe: we only support broadcast mul axis now
            if len(input_shape1) != 4 and (len(input_shape2) != 2 or len(input_shape2) != 4) :
                raise RuntimeError("{} vs {}  broadcast mul not support".format(input_shape1, input_shape2))
            axis = 1
            output_shape = input_shape1
            mul_op = self.CVI.add_broadcast_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis=axis)

        self.addOperand(onnx_node.name, mul_op, output_shape, TensorType.ACTIVATION)

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

            if output_shape == input_shape1:
                # same shape, fuse this op
                self.addOperand(onnx_node.name, op1, output_shape, TensorType.ACTIVATION)
                return
            else:
                reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                self.addOperand(onnx_node.name, reshape_op, output_shape, TensorType.ACTIVATION)
        else:
            raise RuntimeError("Second type must be {}".format(TensorType.TENSOR))

    def convert_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "Shape")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        data = np.array(input_shape)
        self.addTensor(onnx_node.name, data, list(data.shape))
        self.addOperand(onnx_node.name, None, list(data.shape), TensorType.TENSOR)

    def convert_sigmoid_op(self, onnx_node):
        assert(onnx_node.op_type == "Sigmoid")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        output_shape = input_shape
        sigmoid_op = self.CVI.add_sigmoid_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
        self.addOperand(onnx_node.name, sigmoid_op, output_shape, TensorType.ACTIVATION)

    def convert_squeeze_op(self, onnx_node):
        assert(onnx_node.op_type == "Squeeze")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        checkKey(onnx_node.attrs, 'axes')
        if tensor_type == TensorType.ACTIVATION:
            axis_value_list = onnx_node.attrs['axes']
            new_shape = self.squeeze_shape(input_shape, axis_value_list)
            reshape_op = self.CVI.add_reshape_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, new_shape)
            self.addOperand(onnx_node.name, reshape_op, new_shape, TensorType.ACTIVATION)
        else:
            raise RuntimeError("Todo, Squeeze input type is tensor")

    def convert_transpose_op(self, onnx_node):
        assert(onnx_node.op_type == "Transpose")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        transpose_perm = onnx_node.attrs['perm']
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
                'upscale_factor': upscale_factor
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
            permute_op = self.CVI.add_permute_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
            self.addOperand(onnx_node.name, permute_op, output_shape, TensorType.ACTIVATION)
        else:
            raise RuntimeError("TODO")

    def convert_unsqueeze_op(self, onnx_node):
        """Unsqueeze """
        assert(onnx_node.op_type == "Unsqueeze")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        checkKey(onnx_node.attrs, 'axes')
        if tensor_type == TensorType.TENSOR:
            t = self.getTensor(onnx_node.inputs[0])
            axis_value_list = onnx_node.attrs['axes']
            for a in axis_value_list:
                new_t = np.expand_dims(t.tensor_data, axis=a)
            self.addTensor(onnx_node.name, new_t, list(new_t.shape))
            self.addOperand(onnx_node.name, None, list(new_t.shape), TensorType.TENSOR)
        else:
            raise RuntimeError("Todo")

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
                'scale': int(scale_factor[2])
            }
            output_shape = [on, oc, oh, ow]
            upsample_op = self.CVI.add_upsample_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
            self.addOperand(onnx_node.name, upsample_op, output_shape, TensorType.ACTIVATION)


    def run(self):
        self.convert_node()
        self.convert_tensor()
        self.convert_graph()
        self.TensortoNpz()


