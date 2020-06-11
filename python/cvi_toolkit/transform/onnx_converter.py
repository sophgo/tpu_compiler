from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from onnx import numpy_helper, mapping
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number

import onnx
import logging
import numpy as np

from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
                    get_shape_size

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
    def __init__(self, model_name, onnx_model, mlir_file_path, batch_size=1):
        super().__init__()
        if isinstance(onnx_model, str):
            onnx_model = onnx.load(onnx_model)
        self.batch_size = batch_size
        self.model_name = model_name
        self.input_nodes = onnx_model.graph.input
        self.output_nodes = onnx_model.graph.output
        self.nodes = onnx_model.graph.node
        self.tensors = onnx_model.graph.initializer
        self.mlir_file_path = mlir_file_path

        self.remove_tensor_from_input_nodes()

        self.converted_nodes = list()
        self.converted_tensors = list()

        self.CVI = None
        self.init_importer()

        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "AveragePool": lambda node: self.convert_avg_pool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Clip": lambda node: self.convert_clip_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "ConstantOfShape": lambda node: self.convert_constant_of_shape_op(node),
            "DepthToSpace": lambda node: self.convert_depth_to_space_op(node),
            "Div": lambda node: self.convert_div_op(node),
            "Dropout": lambda node: self.convert_skip_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_pool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_pool_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "LeakyRelu": lambda node: self.convert_leaky_relu_op(node),
            "LRN": lambda node: self.convert_lrn_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Max" : lambda node: self.convert_max_op(node),
            "Min" : lambda node: self.convert_min_op(node),
            "Mul" : lambda node: self.convert_mul_op(node),
            "Neg" : lambda node: self.convert_neg_op(node),
            "PRelu": lambda node: self.convert_prelu_op(node),
            "Reciprocal": lambda node: self.convert_reciprocal_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Resize": lambda node: self.convert_resize_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid" :lambda node: self.convert_activation_op(node),
            "Slice": lambda node: self.convert_slice_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "Split": lambda node: self.convert_split_op(node),
            "Squeeze": lambda node: self.convert_squeeze_op(node),
            "Sub": lambda node: self.convert_sub_op(node),
            "Sum": lambda node: self.convert_sum_op(node),
            "Tanh": lambda node: self.convert_activation_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
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

        # init importer
        self.CVI = MLIRImporter(inputs, outputs)

    def remove_tensor_from_input_nodes(self):
        def find_name_in_tensor_list(name):
            for i in self.tensors:
                if name == i.name:
                    return True
            return False
        self.input_nodes = [x for x in self.input_nodes if not find_name_in_tensor_list(x.name)]


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
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

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
            input_op = self.CVI.add_input_op(input.name, idx)
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
        for output in self.output_nodes:
            op, _, _ = self.getOperand(output.name)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)

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
            self.addTensor(onnx_node.name, tensor_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            if onnx_node.op_type == "Sigmoid":
                activation_op = self.CVI.add_sigmoid_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            elif onnx_node.op_type == "Tanh":
                activation_op = self.CVI.add_tanh_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            self.addOperand(onnx_node.name, activation_op, output_shape, TensorType.ACTIVATION)

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
            # Use scale op, x * 1 + y
            if len(input_shape1) == 4 and (len(input_shape2) == 1 or len(input_shape2) == 3): # [1] or [c, 1, 1]
                channel = input_shape1[1]
                # only one constant
                operands.append(op1)
                tensor_data = np.full(input_shape1[1], 1) # broadcast via channel
                weight_name = "{}_add_weight".format(onnx_node.name)
                self.addTensor(weight_name, tensor_data, tensor_data.shape)
                op2 = self.CVI.add_load_file_op(weight_name, tensor_data.shape)
                operands.append(op2)

                add_value = self.getTensor(onnx_node.inputs[1]).tensor_data
                if len(add_value.flatten()) == 1:
                    tensor_data = np.full(input_shape1[1], add_value[0]) # broadcast via channel
                elif len(add_value.flatten()) == channel:
                    tensor_data = add_value.flatten()
                else:
                    raise RuntimeError("could not broadcast input array from shape {} into shape {}".format(input_shape1, input_shape2))

                bias_name = "{}_add_bias".format(onnx_node.name)
                self.addTensor(bias_name, tensor_data, tensor_data.shape)
                op3 = self.CVI.add_load_file_op(bias_name, tensor_data.shape)
                operands.append(op3)

                output_shape = input_shape1

                scale_op = self.CVI.add_scale_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
                self.addOperand(onnx_node.name, scale_op, output_shape, TensorType.ACTIVATION)
            else:
                raise RuntimeError("{} vs {} shape broadcast error".format(input_shape1, input_shape2))

        elif tensor_type1 == TensorType.TENSOR and tensor_type2 == TensorType.TENSOR:
            tensor_data1 = self.getTensor(onnx_node.inputs[0]).tensor_data
            tensor_data2 = self.getTensor(onnx_node.inputs[1]).tensor_data
            output_data = tensor_data1 + tensor_data2
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            # eltwise add
            if input_shape1 != input_shape2:
                raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2))
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
        }
        oh = calcPool2DFloor(input_shape[2], pool_avg_2d_param['kernel_h'], pool_avg_2d_param['stride_h'], pool_avg_2d_param['padding_t'], pool_avg_2d_param['padding_b'])
        ow = calcPool2DFloor(input_shape[2], pool_avg_2d_param['kernel_w'], pool_avg_2d_param['stride_w'], pool_avg_2d_param['padding_l'], pool_avg_2d_param['padding_r'])
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
            # if tensor is constant, make tensor shape dim is 1
            if len(np_tensor.shape) == 0:
                np_tensor = np_tensor.flatten()

            self.addTensor(onnx_node.name, np_tensor.astype(np.float32).flatten(), list(np_tensor.shape))
            self.addOperand(onnx_node.name, None, list(np_tensor.shape), TensorType.TENSOR)

        else:
            raise ValueError("Not Support {} type".format(data_type))

    def convert_constant_of_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "ConstantOfShape")
        op, input_shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        onnx_tensor = onnx_node.attrs['value']
        np_tensor =  numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)

        if data_type in [np.float32, np.float64, np.int32, np.int64]:
            np_tensor = np_tensor.astype(np.float32).flatten()
            constant_data = np.full(tuple(input_shape), np_tensor[0])
            # add new weight tensor
            self.addTensor(onnx_node.name, constant_data, input_shape)
            self.addOperand(onnx_node.name, None, input_shape, TensorType.TENSOR)
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
            for i in onnx_node.inputs:
                tensor_datas.append(self.getTensor(i).tensor_data)

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
                    output_shape = op_shape
                else:
                    for dim, value in enumerate(op_shape):
                        if dim == axis:
                            output_shape[dim] += value
                        else:
                            if output_shape[dim] != value:
                                raise ValueError("axis is {}, {} v.s {} shape can not be concat".format(axis, output_shape, op_shape))

            concat_op = self.CVI.add_concat_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis=axis)
            self.addOperand(onnx_node.name, concat_op, output_shape, TensorType.ACTIVATION)

    def convert_conv_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        dilations = onnx_node.attrs.get("dilations", [1, 1])
        group = onnx_node.attrs.get("group", 1)
        pads = onnx_node.attrs.get("pads",[0,0,0,0])
        strides = onnx_node.attrs.get("strides",[1,1])
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
        op, shape, tensor_type = self.getOperand(onnx_node.inputs[0])
        output_shape = shape
        operands = list()
        if tensor_type == TensorType.TENSOR:
            data = self.getTensor(onnx_node.inputs[0]).tensor_data
            output_data = np.clip(data, onnx_node.attrs['min'],onnx_node.attrs['max'])
            output_shape = list(output_data.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            operands.append(op)
            clip_op = self.CVI.add_clip_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **clip_param)
            self.addOperand(onnx_node.name, clip_op, output_shape, TensorType.TENSOR)

    def convert_depth_to_space_op(self, onnx_node):
        assert(onnx_node.op_type == "DepthToSpace")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        upscale_factor = onnx_node.attrs['blocksize']

        on = input_shape[0]
        oc = input_shape[1] / upscale_factor**2
        oh = upscale_factor * input_shape[2]
        ow = upscale_factor * input_shape[3]
        output_shape = [on, int(oc), oh, ow]
        operands = [op]
        attr={
            'upscale_factor': upscale_factor
        }
        pixel_shuffle_op = self.CVI.add_pixelshuffle_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **attr)
        self.addOperand(onnx_node.name, pixel_shuffle_op, output_shape, TensorType.ACTIVATION)

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
        }
        output_shape = [int(on), int(oc), 1, 1]
        if onnx_node.op_type == "GlobalAveragePool":
            pool_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_2d_param)
        elif onnx_node.op_type == "GlobalMaxPool":
            pool_op = self.CVI.add_pool_max_2d_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, **pool_2d_param)
        self.addOperand(onnx_node.name, pool_op, output_shape, TensorType.ACTIVATION)

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
                weight_data = np.full(channel, mul_value[0]) # broadcast via channel
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
            operands.append(op1)
            operands.append(op2)
            if input_shape1 == input_shape2:
                #eltwise mul
                output_shape = input_shape1
                mul_op = self.CVI.add_eltwise_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape)
            else:
                # broadcast mul
                # TODO: only support broadcast mul channel axis now
                # [n, c, h, w] broadcast with [n, c]
                if len(input_shape1) != 4 or np.prod(input_shape2) != input_shape1[0] * input_shape1[1] :
                    raise RuntimeError("{} vs {}  broadcast mul not support".format(input_shape1, input_shape2))
                axis = 1
                output_shape = input_shape1
                mul_op = self.CVI.add_broadcast_mul_op("{}_{}".format(onnx_node.name, onnx_node.op_type), operands, output_shape, axis=axis)

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
        if mode != b"nearest":
            raise RuntimeError("Unsupported mode {}".format(mode))

        op1, input_shape1, tensor_type1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2, tensor_type2 = self.getOperand(onnx_node.inputs[2])
        if tensor_type1 != TensorType.ACTIVATION or tensor_type2 != TensorType.TENSOR:
            raise RuntimeError("Unsupported tensor type")

        scale_factor = self.getTensor(onnx_node.inputs[2]).tensor_data
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


    def convert_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "Shape")
        op, input_shape, _ = self.getOperand(onnx_node.inputs[0])
        data = np.array(input_shape)
        self.addTensor(onnx_node.name, data, list(data.shape))
        self.addOperand(onnx_node.name, None, list(data.shape), TensorType.TENSOR)

    def convert_slice_op(self, onnx_node):
        assert(onnx_node.op_type == "Slice")
        op, input_shape, tesnor_type = self.getOperand(onnx_node.inputs[0])
        # start
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[1])
        if _tesnor_type != TensorType.TENSOR:
            raise RuntimeError("{} start type be tensor, not find".format(onnx_node.name))
        else:
            starts = self.getTensor(onnx_node.inputs[1]).tensor_data
        # ends
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[2])
        if _tesnor_type != TensorType.TENSOR:
            raise RuntimeError("{} end type be tensor, not find".format(onnx_node.name))
        else:
            ends = self.getTensor(onnx_node.inputs[2]).tensor_data
        # axes
        _, _, _tesnor_type = self.getOperand(onnx_node.inputs[3])
        if _tesnor_type != TensorType.TENSOR:
           raise RuntimeError("{} axes type be tensor, not find".format(onnx_node.name))
        else:
            axes = self.getTensor(onnx_node.inputs[3]).tensor_data

        if len(onnx_node.inputs) == 5:
            # steps
            _, _, _tesnor_type = self.getOperand(onnx_node.inputs[4])
            if _tesnor_type != TensorType.TENSOR:
                raise RuntimeError(
                    "{} steps type be tensor, not find".format(onnx_node.name))
            else:
                steps = self.getTensor(onnx_node.inputs[4]).tensor_data
                assert(len(steps.flatten()) == 1)  # steps only has one value
                if steps.flatten()[0] != 1:
                    raise RuntimeError("only support one steps slices")

        assert(len(starts) == len(ends))
        assert(len(axes) == len(ends))
        if tesnor_type == TensorType.TENSOR:
            raise RuntimeError("TODO")
        else:
            crop_shape = input_shape.copy()
            crop_offset = input_shape.copy()
            idx = 0
            for j in range(len(crop_shape)):
                if j in axes:
                    crop_shape[j] = ends[idx] - starts[idx]
                    crop_offset[j] = starts[idx]
                    idx +=1
                else:
                    crop_shape[j] = input_shape[j]
                    crop_offset[j] = 0

            crop_param = {
                "crop_offset": crop_offset,
                "crop_shape": crop_shape,
            }

            output_shape = crop_shape
            crop_op = self.CVI.add_crop_op("{}_{}".format(onnx_node.name, onnx_node.op_type), [op], output_shape, **crop_param)
            self.addOperand(onnx_node.name, crop_op, output_shape, TensorType.ACTIVATION)


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
            output_shape = list(output_shape.shape)
            self.addTensor(onnx_node.name, output_data, output_shape)
            self.addOperand(onnx_node.name, None, output_shape, TensorType.TENSOR)
        else:
            self.addOperand(onnx_node.name, op, input_shape, TensorType.ACTIVATION)

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

                bias_data = np.full(input_shape0[1], -1 * constant_data[0])
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
                output_shape = input_shape[1:]
                reshape_back_op = self.CVI.add_reshape_op("{}_{}_back_dim".format(
                    onnx_node.name, onnx_node.op_type), [permute_op], input_shape)
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
                    print(output_shape)
                    reshape_back_op = self.CVI.add_reshape_op("{}_{}_back_dim".format(
                        onnx_node.name, onnx_node.op_type), [permute_op], input_shape)
                    self.addOperand(onnx_node.name, reshape_back_op,
                        output_shape, TensorType.ACTIVATION)
                else:
                    raise RuntimeError("transpose dim 5 is not support")
            else:
                raise RuntimeError("only support dim 4 transpose and pixel shuffle case")

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


