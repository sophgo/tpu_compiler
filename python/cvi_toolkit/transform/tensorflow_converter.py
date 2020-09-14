from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum
from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
    get_shape_size, get_TF_SAME_Padding, turn_shape_nhwc_to_nchw, turn_data_hwio_to_oihw, \
    turn_shape_hwio_to_oihw
from ..utils.log_setting import setup_logger
from ..utils.tf_utils import *


import tensorflow as tf
IS_TF2 = tf.__version__.startswith("2.")
if not IS_TF2:
    raise ImportError("Tensorflow use 2.0 or more, now version is {}".format(
        tf.__version__))
from tensorflow.python.framework import tensor_util





import logging
import numpy as np
import operator
import functools

logger = setup_logger('root')
log_flag = logger.level <= logging.INFO

class TFNode():
    def __init__(self, node):
        self.name = node.node_def.name
        self.inputs = [tf_node_name(i.name) for i in node.inputs]
        self.outputs = [tf_node_name(i.name) for i in node.outputs]
        self.op_type = node.type
        self.shape = None
        self.has_tensor_data = False
        self.tensor_data = None
        self.attr = dict()
        if not self.outputs:
            print("{} has no output, skip it".format(self.name))
            return

        assert(self.name == self.outputs[0])

        ignored_attr = {"unknown_rank", "_class", "Tshape", "use_cudnn_on_gpu", "Index", "Tpaddings",
                        "TI", "Tparams", "Tindices", "Tlen", "Tdim", "Tin", "dynamic_size", "Tmultiples",
                        "Tblock_shape", "Tcrops", "index_type", "Taxis", "U", "maxval",
                        "Tout", "Tlabels", "Tindex", "element_shape", "Targmax", "Tperm", "Tcond",
                        "T_threshold", "element_dtype", "shape_type", "_lower_using_switch_merge",
                        "parallel_iterations", "_num_original_outputs", "output_types", "output_shapes",
                        "key_dtype", "value_dtype", "Tin", "Tout", "capacity", "component_types", "shapes",
                        "Toutput_types"}
        self.shape = list(node.outputs[0].shape)
        for a in node.node_def.attr:
            if a == "dtype":
                self.attr[a] = get_tf_node_attr(node, "dtype")
            elif a == "T":
                dtype = get_tf_node_attr(node, a)
            elif a in {"output_type", "output_dtype", "out_type", "Tidx", "out_idx"}:
                # Tidx is used by Range
                # out_idx is used by ListDiff
                self.attr[a] = get_tf_node_attr(node, a)
            elif a == "shape":
                shape = get_tf_shape_attr(node)
                if shape is not None:
                    self.shape = shape
            elif a == "output_shapes":
                # we should not need it since we pull the shapes above already
                pass
            elif a == "value":
                tensor = get_tf_node_attr(node, a)
                np_data = tensor_util.MakeNdarray(tensor)
                self.has_tensor_data = True
                self.tensor_data = np_data
            elif a == "DstT":
                self.attr["to"] = get_tf_node_attr(node, "DstT")
            elif a == "SrcT":
                continue
            elif a in ignored_attr:
                continue
            else:
                self.attr[a] = get_tf_node_attr(node, a)

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')
        cprint("    shape: {}".format(self.shape), 'white')
        cprint("    has_tensor_data: {}".format(self.has_tensor_data), 'white')
        cprint("    attr: {}".format(self.attr), 'green')


class TFTensor():
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')



class TFConverter(BaseConverter):
    def __init__(self, model_name, model_path, mlir_file_path, batch_size=1):
        super().__init__()
        self.model_name = model_name
        self.batch_size=batch_size

        # read tensorflow model
        self.tf_graph, self.inputs, self.outputs = from_saved_model(model_path)
        self.mlir_file_path = mlir_file_path

        self.converted_nodes = list()
        self.converted_tensors = list()

        self.CVI = None # mlcvir pybind

        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.tensorflowop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "AddV2": lambda node: self.convert_add_v2_op(node),
            "Activation": lambda node: self.convert_activation_op(node),
            "AvgPool": lambda node: self.convert_avg_pool_op(node),
            "AveragePooling2D": lambda node: self.convert_avg_pool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "BiasAdd": lambda node: self.convert_biasadd_op(node),
            "FusedBatchNormV3": lambda node: self.convert_batchnorm_v2_op(node),
            "Conv2D": lambda node: self.convert_conv_op(node),
            "Concatenate": lambda node: self.convert_concat_op(node),
            "ConcatV2": lambda node: self.convert_concat_op(node),
            "DepthwiseConv2D": lambda node: self.convert_depthwise_conv_op(node),
            "DepthwiseConv2dNative": lambda node: self.convert_depthwise_conv_op(node),
            "Dense": lambda node: self.convert_fc_op(node),
            "Dropout":  lambda node: self.convert_skip_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "GlobalAveragePooling2D": lambda node: self.convert_global_avg_pool_op(node),
            "Identity": lambda node: self.convert_skip_op(node),
            "MatMul": lambda node: self.convert_fc_v2_op(node),
            "MaxPooling2D": lambda node: self.convert_maxpool_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Mean": lambda node: self.convert_mean_op(node),
            "NoOp": lambda node: None,
            "Pad": lambda node: self.convert_pad_op(node),
            "Pack": lambda node: self.convert_skip_op(node),
            "Placeholder": lambda node: None,
            "ReLU": lambda node: self.convert_activation_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Relu6": lambda node:self.convert_relu6_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Shape": lambda node: self.convert_skip_op(node),
            "Softmax": lambda node: self.convert_softmax_op(node),
            "StridedSlice": lambda node: self.convert_skip_op(node),
            "ZeroPadding2D": lambda node: self.convert_pad_op(node),
        }

    def filter_input_without_placeholder(self):
        """
            In tensorflow const node value maybe in input nodes,
            filter it
        """
        new_inputs = list()
        for i_name in self.inputs:
            node_name = tf_node_name(i_name)
            find_input = [
                t for t in self.converted_nodes if t.name == node_name and t.op_type == "Placeholder"]
            if len(find_input) < 1:
                print("input tensor {} is not Placeholder ".format(node_name))
                continue
            else:
                new_inputs.append(i_name)
        self.inputs = new_inputs
        return

    def init_importer(self):
        # Make MLIR Function
        # get input shape
        self.mlir_inputs = list()
        for i_name in self.inputs:
            node_name = tf_node_name(i_name)
            find_input = [
                t for t in self.converted_nodes if t.name == node_name and t.op_type == "Placeholder"]
            if len(find_input) < 1:
                raise KeyError("input tensor {} not found ".format(node_name))
            else:
                input_node = find_input[0]
            i_shape = list(input_node.shape)
            if i_shape[0] == None or i_shape[0] == -1:
                i_shape[0] = self.batch_size
            self.mlir_inputs.append(turn_shape_nhwc_to_nchw(i_shape))

        # get output shape
        self.mlir_outputs = list()
        for o_name in self.outputs:
            node_name = tf_node_name(o_name)
            find_output = [
                t for t in self.converted_nodes if t.name == node_name and t.op_type == "Identity"]
            if len(find_output) < 1:
                raise KeyError("output tensor {} not found".format(node_name))
            else:
                output_node = find_output[0]
            o_shape = list(output_node.shape)
            if o_shape[0] == None or o_shape[0] == -1:
                o_shape[0] = self.batch_size
            if len(o_shape) == 4:
                o_shape = turn_shape_nhwc_to_nchw(o_shape)
            self.mlir_outputs.append(o_shape)

        # init importer
        self.CVI = MLIRImporter(self.mlir_inputs, self.mlir_outputs)

    def addTensor(self, op_name, tensor_data, tensor_shape, op_type):
        self.converted_tensors.append(TFTensor(op_name, tensor_data, tensor_shape))

    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise KeyError("No {} tensor in model".format(op_name))
        else:
            return find_tensor[0]

    def createLoadWeightOp(self, tensor_name, tensor_data, tensor_shape):
        self.addTensor(tensor_name, tensor_data, tensor_shape, None)
        weight_op = self.CVI.add_load_file_op(tensor_name, tensor_shape)
        return weight_op

    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.output_tensor_file, **tensor_npz)

    def convert_graph(self):
        """convert all to mlir"""

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.tf_graph, name='')
        with tf_session(graph=graph):
            node_list = graph.get_operations()
            for node in node_list:
                attr = dict()
                n = TFNode(node)
                # n.print_info()
                if n.op_type == "Const" and n.has_tensor_data:  # weight
                    self.addTensor(
                        n.name, turn_data_hwio_to_oihw(n.tensor_data), \
                        turn_shape_hwio_to_oihw(n.shape), None)
                else:
                    self.converted_nodes.append(n)

        self.filter_input_without_placeholder()
        self.init_importer()

        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        # add input op
        for idx, input in enumerate(self.inputs):
            name = tf_node_name(input)
            input_op = self.CVI.add_input_op(name, idx)
            input_shape = self.mlir_inputs[idx]
            self.addOperand(name, input_op, input_shape, TensorType.ACTIVATION)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        # add node op
        for n in self.converted_nodes:
            if log_flag:
                n.print_info()
            self.tensorflowop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)

        # add return op
        return_op = list()
        # Set output
        for output in self.outputs:
            name = tf_node_name(output)
            op, _, _ = self.getOperand(name)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)
        print("Save mlir file: {}".format(self.mlir_file_path))

    def convert_activation_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        if node.op_type == "Activation":
            if node.config['activation'] == "relu":
                activation_op = self.CVI.add_relu_op("{}".format(node.name), operands, output_shape)
            elif node.config['activation'] == "softmax":
                axis = len(input_shape) - 1

                for i in range(len(output_shape)):
                    if output_shape[axis] == 1:
                        axis = axis -1
                softmax_param = {
                    'axis': axis,
                }
                activation_op = self.CVI.add_softmax_op(node.name, operands, output_shape, **softmax_param)
            else:
                raise RuntimeError("No support {} activation".format(node.config['activation']))
        elif node.op_type == "ReLU":
            max_value = node.config['max_value']
            if int(max_value) == 6:
                # relu6
                relu_op = self.CVI.add_relu_op(
                    "{}_relu".format(node.name), operands, output_shape)
                clip_param = {
                    "min": 0.0,
                    "max": 6.0,
                }
                activation_op = self.CVI.add_clip_op(
                    "{}".format(node.name), [relu_op], output_shape, **clip_param)
            else:
                activation_op = self.CVI.add_relu_op("{}".format(node.name), operands, output_shape)
        else:
            raise RuntimeError("No support {} activation".format(node.op_type))
        self.addOperand(node.name, activation_op, output_shape, TensorType.ACTIVATION)

    def convert_add_op(self, node):
        assert(node.op_type == "Add")
        op1, input_shape1, _ = self.getOperand(node.inputs[0])
        op2, input_shape2, _ = self.getOperand(node.inputs[1])
        if input_shape1 != input_shape2:
            raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2))

        operands = list()
        operands.append(op1)
        operands.append(op2)
        output_shape = input_shape1

        add_op = self.CVI.add_eltwise_add_op("{}".format(node.name), operands, output_shape)
        self.addOperand(node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_add_v2_op(self, node):
        assert(node.op_type == "AddV2")
        op1, input_shape1, _ = self.getOperand(node.inputs[0])
        op2, input_shape2, _ = self.getOperand(node.inputs[1])
        if input_shape1 != input_shape2:
            raise AttributeError("{} v.s. {} shape not same".format(
                input_shape1, input_shape2))

        output_shape = input_shape1
        # check output if same with tf graph
        assert(output_shape[1:] != node.shape[1:])
        add_op = self.CVI.add_eltwise_add_op(node.name, [op1, op2], output_shape)
        self.addOperand(node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_avg_pool_op(self, node):
        assert(node.op_type == "AvgPool")

        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

        ksize = turn_shape_nhwc_to_nchw(node.attr.get("ksize"))
        ksize_h, ksize_w = ksize[2:]
        strides = turn_shape_nhwc_to_nchw(node.attr.get("strides"))
        stride_h, stride_w = strides[2:]

        padding_method = node.attr.get("padding").decode('utf-8')

        if padding_method == "SAME":
            padding_along_h = get_TF_SAME_Padding(
                input_shape[2], ksize_h, stride_h)
            padding_along_w = get_TF_SAME_Padding(
                input_shape[3], ksize_w, stride_w)
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        on = input_shape[0]
        oc = input_shape[1]
        oh = calcPool2DFloor(
            input_shape[2],
            ksize_h,
            stride_h,
            padding_t,
            padding_b,
        )
        ow = calcPool2DFloor(
            input_shape[3],
            ksize_w,
            stride_w,
            padding_l,
            padding_r,
        )

        pool_avg_2d_param = {
            'stride_h':  stride_h,
            'stride_w':  stride_w,
            'kernel_h':  ksize_h,
            'kernel_w':  ksize_w,
            'padding_t': padding_t,
            'padding_b': padding_b,
            'padding_l': padding_l,
            'padding_r': padding_r,
            'count_include_pad': False,
            'do_relu': False,
        }
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        assert(output_shape[1:] != node.shape[1:])
        pool_avg_op = self.CVI.add_pool_avg_2d_op(node.name, operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_batchnorm_op(self, node):
        assert(node.op_type == "BatchNormalization")
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = node.config['epsilon']

        weights = node.proto.get_weights()
        # we fuse batchnorm and scale at here
        if node.config.get('scale') == False:
            gamma_value = 1.0
        else:
            gamma_value = weights[0]
            weights = weights[1:]

        if node.config.get('center') == False:
            beta_value = 0.0
        else:
            beta_value = weights[0]
            weights = weights[1:]

        mean_value =  weights[0]
        var_value =  weights[1]

        scale_name = "{}_0".format(node.name)
        scale_value = ((1.0 / np.sqrt(
                    var_value + epsilon)) * gamma_value)

        scale_op = self.CVI.add_load_file_op(scale_name, scale_value.shape)
        # add new weight tensor
        self.addTensor(scale_name, scale_value, scale_value.shape, None)

        offset_name =  "{}_1".format(node.name)
        offset_value = (-mean_value * scale_value) + beta_value
        offset_op = self.CVI.add_load_file_op(offset_name, offset_value.shape)
        # add new bias tensor
        self.addTensor(offset_name, offset_value, offset_value.shape, None)

        operands.append(scale_op)
        operands.append(offset_op)

        output_shape = input_shape
        scaleop = self.CVI.add_scale_op("{}".format(node.name, node.op_type), operands, output_shape)
        self.addOperand(node.name, scaleop, output_shape, TensorType.ACTIVATION)

    def convert_batchnorm_v2_op(self, node):
        assert(node.op_type == "FusedBatchNormV3")
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = node.attr['epsilon']
        assert(len(node.inputs) == 5)
        gamma_value = self.getTensor(node.inputs[1]).tensor_data
        beta_value = self.getTensor(node.inputs[2]).tensor_data
        mean_value = self.getTensor(node.inputs[3]).tensor_data
        var_value = self.getTensor(node.inputs[4]).tensor_data


        scale_name = "{}_0".format(node.name)
        scale_value = ((1.0 / np.sqrt(
                    var_value + epsilon)) * gamma_value)

        scale_op = self.CVI.add_load_file_op(scale_name, scale_value.shape)
        # add new weight tensor
        self.addTensor(scale_name, scale_value, scale_value.shape, None)

        offset_name =  "{}_1".format(node.name)
        offset_value = (-mean_value * scale_value) + beta_value
        offset_op = self.CVI.add_load_file_op(offset_name, offset_value.shape)
        # add new bias tensor
        self.addTensor(offset_name, offset_value, offset_value.shape, None)

        operands.append(scale_op)
        operands.append(offset_op)
        output_shape = input_shape

        # check output if same with tf graph
        assert(output_shape[1:] != node.shape[1:])
        scaleop = self.CVI.add_scale_op("{}".format(node.name, node.op_type), operands, output_shape)
        self.addOperand(node.name, scaleop, output_shape, TensorType.ACTIVATION)

    def convert_biasadd_op(self, node):
        assert(node.op_type == "BiasAdd")
        op, input_shape, _ = self.getOperand(node.inputs[0])

        # filter
        bias_tensor = self.getTensor(node.inputs[1])

        bias_shape = bias_tensor.shape
        bias_data = bias_tensor.tensor_data
        if len(input_shape) == 4:
            bias_shape = list(bias_data.shape)
        elif len(input_shape) == 2:
            bias_shape = list(bias_data.shape)

        bias_op = self.createLoadWeightOp(
            node.inputs[1], bias_data, bias_shape)

        if input_shape[1] != bias_shape[0]:
            raise AttributeError("{} v.s. {} shape not same".format(
                input_shape, bias_shape))

        weight_data = np.full(input_shape[1], 1) # broadcast via channel
        weight_name = "{}_add_weight".format(node.name)
        self.addTensor(weight_name, weight_data, weight_data.shape, None)
        weight_op = self.CVI.add_load_file_op(weight_name, weight_data.shape)

        output_shape = input_shape
        # check output if same with tf graph
        # assert(output_shape[1:] == node.shape[1:])
        add_op = self.CVI.add_scale_op(
            node.name, [op, weight_op, bias_op], output_shape)
        self.addOperand(node.name, add_op, output_shape, TensorType.ACTIVATION)

    def convert_concat_op(self, node):
        """
            In tensorflow, case of dim 4, data format is NHWC,
            We handle all op in NCHW, if axis is 3, we change to 1
        """
        assert(node.op_type == "ConcatV2")
        concat_num = node.attr.get("N")

        in_shapes = list()
        operands = list()
        output_shape = list()
        for i in range(concat_num):
            op, input_shape, _ = self.getOperand(node.inputs[i])
            in_shapes.append(input_shape)
            operands.append(op)
        axis_tensor = self.getTensor(node.inputs[-1])
        axis = axis_tensor.tensor_data
        if len(in_shapes[0]) == 4 and axis != 3:
            raise RuntimeError("case of dim 4, data format is NHWC, we handle all op in NCHW, if axis is 3, we change to 1\n axis is {}".format(axis))
        elif len(in_shapes[0]) == 4 and axis == 3:
            logger.info("case of dim 4, data format is NHWC, we handle all op in NCHW, if axis is 3, we change to 1")
            axis = 1
        else:
            pass

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

        concat_op = self.CVI.add_concat_op(node.name, operands, output_shape, axis=axis)
        self.addOperand(node.name, concat_op, output_shape, TensorType.ACTIVATION)

    def convert_conv_op(self, node):
        assert(node.op_type == "Conv2D")
        op, input_shape, _ = self.getOperand(node.inputs[0])

        operands = list()
        operands.append(op)
        # filter
        filter_tensor = self.getTensor(node.inputs[1])
        filter_shape = filter_tensor.shape
        filter_op = self.CVI.add_load_file_op(
            node.inputs[1], filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = len(node.inputs) > 2
        if do_bias:
            bias_tensor = self.getTensor(node.inputs[2])
            bias_op = self.CVI.add_load_file_op(
                node.inputs[2], bias_tensor.shape)
            operands.append(bias_op)
        strides = turn_shape_nhwc_to_nchw(node.attr.get("strides"))
        stride_h, stride_w = strides[2:]

        dilations = turn_shape_nhwc_to_nchw(node.attr.get("dilations"))
        dilation_h, dilation_w = dilations[2:]

        padding_method = node.attr.get("padding").decode('utf-8')

        if padding_method == "SAME":
            padding_along_h = get_TF_SAME_Padding(
                input_shape[2], filter_shape[2], stride_h)
            padding_along_w = get_TF_SAME_Padding(
                input_shape[3], filter_shape[3], stride_w)
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        conv_param = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'padding': padding_method,
            'dilation_h': dilation_h,
            'dilation_w': dilation_w,
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': 1,  # Don't have group option?
            'is_dw': False,
            'with_bias': do_bias,
            'do_relu': False,
            'ins': [],
        }
        on, oc = input_shape[0], filter_shape[0]
        oh = calcConv2DSpatial(
            input_shape[2],
            filter_shape[2],
            stride_h,
            padding_t,
            padding_b,
            dilation_h,
        )
        ow = calcConv2DSpatial(
            input_shape[3],
            filter_shape[3],
            stride_w,
            padding_l,
            padding_r,
            dilation_w,
        )
        output_shape = [int(i) for i in [on, oc, oh, ow]]
        assert(output_shape[2:4] == node.shape[1:3] and output_shape[1] == node.shape[3]) # check output if same with tf graph
        conv_op = self.CVI.add_conv_op("{}".format(
            node.name), operands, output_shape, **conv_param)
        self.addOperand(node.name, conv_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_depthwise_conv_op(self, node):
        assert(node.op_type == "DepthwiseConv2dNative")
        op, input_shape, _ = self.getOperand(node.inputs[0])

        operands = list()
        operands.append(op)
        ic = input_shape[1]
        g = ic
        oc = ic

        # filter
        filter_tensor = self.getTensor(node.inputs[1])
        kh, kw = filter_tensor.shape[2:]
        filter_shape = [g, 1, 1, kh, kw]
        filter_op = self.CVI.add_load_file_op(
            node.inputs[1], filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = len(node.inputs) > 2
        if do_bias:
            bias_tensor = self.getTensor(node.inputs[2])
            bias_op = self.CVI.add_load_file_op(
                node.inputs[2], bias_tensor.shape)
            operands.append(bias_op)


        strides = turn_shape_nhwc_to_nchw(node.attr.get("strides"))
        stride_h, stride_w = strides[2:]

        dilations = turn_shape_nhwc_to_nchw(node.attr.get("dilations"))
        dilation_h, dilation_w = dilations[2:]

        padding_method = node.attr.get("padding").decode('utf-8')

        if padding_method == "SAME":
            padding_along_h = get_TF_SAME_Padding(input_shape[2], filter_shape[3], stride_h)
            padding_along_w = get_TF_SAME_Padding(input_shape[3], filter_shape[4], stride_w)
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
            'padding': padding_method,
            'dilation_h': dilation_h,
            'dilation_w': dilation_w,
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': g,
            'is_dw': True,
            'with_bias': do_bias,
            'do_relu': False,
            'ins': [],
        }
        on = input_shape[0]
        oh = calcConv2DSpatial(
            input_shape[2],
            filter_shape[3],
            stride_h,
            padding_t,
            padding_b,
            dilation_h,
        )
        ow = calcConv2DSpatial(
            input_shape[3],
            filter_shape[4],
            stride_w,
            padding_l,
            padding_r,
            dilation_w,
        )
        output_shape = [on, oc, oh, ow]
        # check output if same with tf graph
        assert(output_shape[1:] != node.shape[1:])
        depthwise_conv_op = self.CVI.add_conv_op(node.name, operands, output_shape, **depthwise_conv_param)
        self.addOperand(node.name, depthwise_conv_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_fc_op(self, node):
        assert(node.op_type == "Dense")
        op, shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

        # filter
        filter_data = node.proto.get_weights()[0]
        filter_data = np.ascontiguousarray(np.transpose(filter_data, (1, 0)))
        filter_shape = filter_data.shape
        filter_name = "{}_add_weight".format(node.name)
        filter_op = self.createLoadWeightOp(
            filter_name, filter_data, filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = node.config['use_bias']
        if do_bias:
            bias_data = node.proto.get_weights()[1]
            bias_shape = bias_data.shape
            bias_name = "{}_add_bias".format(node.name)
            bias_op = self.createLoadWeightOp(bias_name, bias_data, bias_shape)
            operands.append(bias_op)

        M = shape[0]
        K = shape[1]
        N = bias_shape[0]
        output_shape = [M, N]
        fc_op = self.CVI.add_fully_connected_op("{}_fc".format(node.name), operands, output_shape)
        self.addOperand("{}_fc".format(node.name), fc_op,
                        output_shape, TensorType.ACTIVATION)
        activation_operands = [fc_op]
        if node.config['activation'] == "softmax":
            softmax_param = {
                'axis': len(output_shape) - 1,
            }
            activation_op = self.CVI.add_softmax_op(
               node.name, activation_operands, output_shape, **softmax_param)
        elif node.config['activation'] == "relu":
            activation_op = self.CVI.add_relu_op(node.name, activation_operands, output_shape)
        else:
            raise RuntimeError(
                "TODO Activation is {}".format(node.config['activation']))

        self.addOperand(node.name, activation_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_fc_v2_op(self, node):
        assert(node.op_type == "MatMul")
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = [op]
        # filter
        filter_data = self.getTensor(node.inputs[1]).tensor_data
        filter_data = np.ascontiguousarray(np.transpose(filter_data, (1, 0)))
        filter_shape = filter_data.shape
        filter_name = node.inputs[1]
        filter_op = self.createLoadWeightOp(
            filter_name, filter_data, filter_shape)
        operands.append(filter_op)
        print(filter_shape)
        # bias
        do_bias = len(node.inputs) > 2
        if do_bias:
            bias_tensor = self.getTensor(node.inputs[2])
            bias_shape = bias_tensor.shape
            bias_op = self.CVI.add_load_file_op(
                node.inputs[2], bias_tensor.shape)
            operands.append(bias_op)

        M = input_shape[0]
        K = input_shape[1]
        N = filter_shape[0]
        output_shape = [M, N]
        fc_op = self.CVI.add_fully_connected_op(node.name, operands, output_shape)
        self.addOperand(node.name, fc_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_flatten_op(self, node):
        """
        In Tensorflow, flatten channels_last means [n,h,w,c] -> [n,hwc]
        But In milr, our order is [n,c,h,w] -> [n,chw]
        tranpose [nchw] -> [nhwc] than reshape to [n,hwc]
        if input dim less than 4, ignore
        """
        assert(node.op_type == "Flatten")
        op, input_shape, _ = self.getOperand(node.inputs[0])
        data_format = node.config.get('data_format')

        if data_format != "channels_last":
            raise RuntimeError("Not support {} data_format".format(data_format))
        if len(input_shape) > 4:
            raise RuntimeError("Todo, input dim is {} dim (only support <4 case)".format(len(input_shape)))
        elif len(input_shape) == 4:
            attr = {
                        'order0': 0,
                        'order1': 2,
                        'order2': 3,
                        'order3': 1,
            }
            permute_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
            op = self.CVI.add_permute_op("{}_transpose".format(
                node.name), [op], permute_shape, **attr)

        reduce_shape = functools.reduce(operator.mul, input_shape[1:])
        output_shape = [input_shape[0], reduce_shape]
        reshape_op = self.CVI.add_reshape_op(node.name, [op], output_shape)
        self.addOperand(node.name, reshape_op, output_shape, TensorType.ACTIVATION)

    def convert_global_avg_pool_op(self, node):
        assert(node.op_type == "GlobalAveragePooling2D")
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

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
            'count_include_pad':True,
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}".format(
            node.name), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_maxpool_op(self, node):
        assert(node.op_type == "MaxPool")
        op, shape, tensor_type = self.getOperand(node.inputs[0])

        operands = list()
        operands.append(op)
        strides = turn_shape_nhwc_to_nchw(node.attr.get("strides"))
        stride_h, stride_w = strides[2:]

        ksize = turn_shape_nhwc_to_nchw(node.attr.get("ksize"))
        ksize_h, ksize_w = ksize[2:]
        padding_method = node.attr.get("padding").decode('utf-8')
        if padding_method == "SAME":
            padding_along_h = get_TF_SAME_Padding(
                input_shape[2], filter_shape[2], stride_h)
            padding_along_w = get_TF_SAME_Padding(
                input_shape[3], filter_shape[3], stride_w)
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        pool_max_2d_param = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'kernel_h': ksize_h,
            'kernel_w': ksize_w,
            'padding_b': padding_b,
            'padding_r': padding_r,
            'padding_t': padding_t,
            'padding_l': padding_l,
            'do_relu': False,
        }

        on = shape[0]
        oc = shape[1]
        oh = calcPool2DFloor(shape[2], ksize_h, stride_h,
                             padding_b, padding_t)
        ow = calcPool2DFloor(shape[3], ksize_w, stride_w,
                             padding_r, padding_l)
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        assert(output_shape[1:] != node.shape[1:])
        pool_max_op = self.CVI.add_pool_max_2d_op("{}".format(node.name), [op], output_shape, **pool_max_2d_param)
        self.addOperand(node.name, pool_max_op, output_shape, TensorType.ACTIVATION)

    def convert_mean_op(self, node):
        assert(node.op_type == "Mean")
        op, input_shape, _ = self.getOperand(node.inputs[0])
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
            'count_include_pad':True,
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op(node.name, [op], output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_pad_op(self, node):
        assert(node.op_type == "Pad")
        assert(len(node.inputs) == 2)
        op, input_shape, _ = self.getOperand(node.inputs[0])
        padding_data = self.getTensor(node.inputs[1]).tensor_data

        padding_data = padding_data[[0, 3, 1, 2], :] # ohwc -> ochw
        padding_data = padding_data.flatten('F')
        dims = len(input_shape)
        pads_param = {
            "pads": padding_data.tolist(),
            "const_val": 0,
        }
        output_shape = np.sum(
            [input_shape, padding_data[:dims], padding_data[dims:]], axis=0)
        output_shape = [int(i) for i in output_shape]
        assert(output_shape[1:] != node.shape[1:])
        pads_op = self.CVI.add_pad_op(node.name, [op], output_shape, **pads_param)
        self.addOperand(node.name, pads_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_reshape_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        if len(node.shape) == 4:  # none batch size infomatiion [h, w, c]
            output_shape = [node.shape[i] for i in [0, 3, 1, 2]]
        else:
            output_shape = node.shape

        output_shape[0] = input_shape[0]
        if len(input_shape) == 4 and len(node.shape) == 2:
            # flatten
            attr = {
                        'order0': 0,
                        'order1': 2,
                        'order2': 3,
                        'order3': 1,
            }
            permute_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
            op = self.CVI.add_permute_op("{}_transpose".format(
                node.name), [op], permute_shape, **attr)
            reduce_shape = functools.reduce(operator.mul, input_shape[1:])
            output_shape = [input_shape[0], reduce_shape]
            reshape_op = self.CVI.add_reshape_op(node.name, [op], output_shape)
            self.addOperand(node.name, reshape_op, output_shape, TensorType.ACTIVATION)
            return
        reshape_op = self.CVI.add_reshape_op(node.name, operands, output_shape)
        self.addOperand(node.name, reshape_op, output_shape, TensorType.ACTIVATION)

    def convert_relu_op(self, node):
        op, output_shape, _ = self.getOperand(node.inputs[0])
        # relu
        relu_op = self.CVI.add_relu_op(
            node.name, [op], output_shape)

        self.addOperand(node.name, relu_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_relu6_op(self, node):
        op, output_shape, _ = self.getOperand(node.inputs[0])
        # relu6
        relu_op = self.CVI.add_relu_op("{}_relu".format(node.name), [op], output_shape)
        clip_param = {
            "min": 0.0,
            "max": 6.0,
        }
        assert(output_shape[1:] != node.shape[1:])
        activation_op = self.CVI.add_clip_op(node.name, [relu_op], output_shape, **clip_param)
        self.addOperand(node.name, activation_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_softmax_op(self, node):
        op, output_shape, _ = self.getOperand(node.inputs[0])
        axis = len(output_shape) - 1

        for i in range(len(output_shape)):
            if output_shape[axis] == 1:
                axis = axis - 1
        softmax_param = {
            'axis': axis,
        }
        softmax_op = self.CVI.add_softmax_op(
            node.name, [op], output_shape, **softmax_param)
        assert(output_shape[1:] == node.shape[1:])
        self.addOperand(node.name, softmax_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_skip_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        self.addOperand(node.name, op, input_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_graph()
        self.TensortoNpz()


