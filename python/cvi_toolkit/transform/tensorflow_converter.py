from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number
from enum import Enum
from .utils import calcConv2DSpatial, calcPool2DFloor, calcPool2DCeil, \
    get_shape_size, get_TF_SAME_Padding, turn_shape_nhwc_to_nchw, turn_data_hwio_to_oihw
from ..utils.log_setting import setup_logger

try:
    from packaging import version
    import tensorflow as tf
    # check Tf2.0
    IS_TF2 = version.parse("2.0.0") < version.parse(tf.__version__)
    if not IS_TF2:
        print("WANING, tf version is {}, we support TF2".format(
            version.parse(tf.__version__)))
except ImportError as error:
    tf = None



import logging
import numpy as np
import operator
import functools

logger = setup_logger('root')
log_flag = logger.level <= logging.INFO


class TFNode():
    def __init__(self, name, op_type, inputs, outputs, proto):
        self.name = str(name)
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.proto = proto
        self.config = proto.get_config()

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')
        cprint("    config: {}".format(self.config), 'green')


class TFTensor():
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



class TFConverter(BaseConverter):
    def __init__(self, model_name, model_path, mlir_file_path, batch_size=1):
        super().__init__()
        self.model_name = model_name
        self.batch_size=batch_size

        # read tensorflow model
        self.net = tf.keras.models.load_model(model_path, custom_objects = {'tf':tf})
        if not isinstance(self.net, tf.python.keras.engine.training.Model):
            raise RuntimeError("Not support tf type: {} now".format(type(self.net)))
        print(self.net.summary())

        self.layers = self.net.layers
        self.inputs = self.net.inputs
        self.outputs = self.net.outputs

        self.mlir_file_path = mlir_file_path

        self.converted_nodes = list()
        self.converted_tensors = list()

        self.CVI = None # mlcvir pybind
        self.init_importer()
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.tensorflowop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "Activation": lambda node: self.convert_activation_op(node),
            "AveragePooling2D": lambda node: self.convert_avg_pool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Conv2D": lambda node: self.convert_conv_op(node),
            "Concatenate": lambda node: self.convert_concat_op(node),
            "DepthwiseConv2D": lambda node: self.convert_depthwise_conv_op(node),
            "Dense": lambda node: self.convert_fc_op(node),
            "Dropout":  lambda node: self.convert_skip_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "GlobalAveragePooling2D": lambda node: self.convert_global_avg_pool_op(node),
            "InputLayer": lambda node: None,
            "MaxPooling2D": lambda node: self.convert_maxpool_op(node),
            "ReLU": lambda node: self.convert_activation_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "ZeroPadding2D": lambda node: self.convert_pad_op(node),
        }

    def init_importer(self):
        # Make MLIR Function
        # get input shape
        inputs = list()
        for i in self.inputs:
            i_shape = list(i.shape)
            if i_shape[0] == None:
                i_shape[0] = self.batch_size
            inputs.append(turn_shape_nhwc_to_nchw(i_shape))
        # get output shape
        outputs = list()
        for o in self.outputs:
            o_shape = list(o.shape)
            if o_shape[0] == None:
                o_shape[0] = self.batch_size
            if len(o_shape) == 4:
                o_shape = turn_shape_nhwc_to_nchw(o_shape)
            outputs.append(o_shape)

        # init importer
        self.CVI = MLIRImporter(inputs, outputs)

    def addTensor(self, op_name, tensor_data, tensor_shape, op_type):
        self.converted_tensors.append(TFTensor(op_name, tensor_data, tensor_shape, op_type))

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

    def convert_node(self):
        """convert tensorflow layer to TFNode"""
        for tf_op in self.layers:
            nodes = tf_op._inbound_nodes
            op_type = tf_op.__class__.__name__
            inputs = list()
            output = list()

            node = nodes[0] #FixMe: Hardcode here, assume only one node
            node_info = node.get_config()
            inbound_node = node_info['inbound_layers']
            if isinstance(inbound_node, list):
                for i in inbound_node:
                    inputs.append(i)
            else:
                inputs.append(inbound_node)
            output.append(node_info['outbound_layer'])

            name = tf_op.name
            node = TFNode(name, op_type, inputs, output, tf_op)
            self.converted_nodes.append(node)

    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        # add input op
        for idx, input in enumerate(self.inputs):
            input_shape = list(input.shape)
            if input_shape[0] == None:
                input_shape[0] = self.batch_size
            if len(input_shape) ==4:
                input_shape = turn_shape_nhwc_to_nchw(input_shape)
                # FixMe: Hardcore to strip ":""
            name = input.name.split(":")[0]
            input_op = self.CVI.add_input_op(name, idx)

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
            name = output.name.split(":")[0].split("/")[0]
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

    def convert_avg_pool_op(self, node):
        assert(node.op_type == "AveragePooling2D")

        op, shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

        pool_size = node.config.get("pool_size")
        padding = node.config.get("padding")
        strides = node.config.get("strides")
        data_format = node.config.get("data_format")
        if data_format != "channels_last": raise RuntimeError("Only support channel pool")

        if padding == "same":
            padding_along_h = get_TF_SAME_Padding(shape[2], pool_size[0], strides[0])
            padding_along_w = get_TF_SAME_Padding(shape[3], pool_size[1], strides[1])
            padding_t = padding_along_h // 2
            padding_l = padding_along_w // 2
            padding_b = padding_along_h - padding_t
            padding_r = padding_along_w - padding_l
        else:
            padding_t = 0
            padding_b = 0
            padding_l = 0
            padding_r = 0

        on = shape[0]
        oc = shape[1]
        oh = calcPool2DFloor(
            shape[2],
            pool_size[0],
            strides[0],
            padding_t,
            padding_b,
        )
        ow = calcPool2DFloor(
            shape[3],
            pool_size[1],
            strides[1],
            padding_l,
            padding_r,
        )

        pool_avg_2d_param = {
            'stride_h':  strides[0],
            'stride_w':  strides[1],
            'kernel_h':  pool_size[0],
            'kernel_w':  pool_size[1],
            'padding_t': padding_t,
            'padding_b': padding_b,
            'padding_l': padding_l,
            'padding_r': padding_r,
            'count_include_pad': False,
            'do_relu': False,
        }
        output_shape = [int(on), int(oc), int(oh), int(ow)]
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

    def convert_concat_op(self, node):
        """
            In tensorflow, case of dim 4, data format is NHWC,
            We handle all op in NCHW, if axis is 3, we change to 1
        """
        axis = node.config.get("axis")
        in_shapes = list()
        operands = list()
        output_shape = list()
        for i in node.inputs:
            op, input_shape, _ = self.getOperand(i)
            in_shapes.append(input_shape)
            operands.append(op)
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
        op, shape, tesnor_type = self.getOperand(node.inputs[0])

        operands = list()
        operands.append(op)
        # filter
        filter_data = node.proto.get_weights()[0]
        filter_data = turn_data_hwio_to_oihw(filter_data)
        filter_shape = filter_data.shape
        filter_name = "{}_add_weight".format(node.name)
        filter_op = self.createLoadWeightOp(filter_name, filter_data, filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = node.config['use_bias']
        if do_bias:
            bias_data = node.proto.get_weights()[1]
            bias_shape = bias_data.shape
            bias_name = "{}_add_bias".format(node.name)
            bias_op = self.createLoadWeightOp(bias_name, bias_data, bias_shape)
            operands.append(bias_op)

        stride_h = node.config['strides'][0]
        stride_w = node.config['strides'][1]
        # Check padding method
        if tesnor_type == TensorType.TENSOR:
            # get padding data from tensor
            padding_attr_data = self.getTensor(node.inputs[0]).tensor_data
            padding_t = padding_attr_data[0][0]
            padding_b = padding_attr_data[0][1]
            padding_l = padding_attr_data[1][0]
            padding_r = padding_attr_data[1][1]
        else:
            padding_attr_data = None
            if node.config['padding'] == "same":
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

        do_relu = node.config.get("activation", None) == "relu"
        conv_param = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'padding': "VALID",
            'dilation_h': node.config['dilation_rate'][0],
            'dilation_w': node.config['dilation_rate'][1],
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': 1,  # Don't have group option?
            'is_dw': False,
            'with_bias': node.config['use_bias'],
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
        conv_op = self.CVI.add_conv_op("{}".format(
            node.name), operands, output_shape, **conv_param)
        self.addOperand(node.name, conv_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_depthwise_conv_op(self, node):
        assert(node.op_type == "DepthwiseConv2D")
        op, shape, tesnor_type = self.getOperand(node.inputs[0])

        operands = list()
        operands.append(op)
        ic = shape[1]
        g = ic
        kh = node.config['kernel_size'][0]
        kw = node.config['kernel_size'][1]
        oc = ic

        # filter
        filter_data = node.proto.get_weights()[0]
        filter_data = turn_data_hwio_to_oihw(filter_data)
        filter_data = np.ascontiguousarray(
            filter_data.flatten().reshape(g, int(oc/g), int(ic/g), kh, kw))
        filter_shape = filter_data.shape
        filter_name = "{}_add_weight".format(node.name)

        filter_op = self.createLoadWeightOp(filter_name, filter_data, filter_shape)
        operands.append(filter_op)


        # bias
        do_bias = node.config['use_bias']
        if do_bias:
            bias_data = node.proto.get_weights()[1]
            bias_shape = bias_data.shape
            bias_name = "{}_add_bias".format(node.name)
            bias_op = self.createLoadWeightOp(bias_name, bias_data, bias_shape)
            operands.append(bias_op)

        stride_h = node.config['strides'][0]
        stride_w = node.config['strides'][1]
        # Check padding method
        if tesnor_type == TensorType.TENSOR:
            # get padding data from tensor
            padding_attr_data = self.getTensor(node.inputs[0]).tensor_data
            padding_t = padding_attr_data[0][0]
            padding_b = padding_attr_data[0][1]
            padding_l = padding_attr_data[1][0]
            padding_r = padding_attr_data[1][1]
        else:
            padding_attr_data = None
            if node.config['padding'] == "same":
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
            'padding': "SAME",
            'dilation_h': node.config['dilation_rate'][0],
            'dilation_w': node.config['dilation_rate'][1],
            'padding_t': int(padding_t),
            'padding_b': int(padding_b),
            'padding_l': int(padding_l),
            'padding_r': int(padding_r),
            'group': g,
            'is_dw': True,
            'with_bias': node.config['use_bias'],
            'do_relu': False,
            'ins': [],
        }
        on = shape[0]

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
        assert(node.op_type == "MaxPooling2D")
        op, shape, tensor_type = self.getOperand(node.inputs[0])

        if tensor_type == TensorType.TENSOR:
            # get padding data from tensor
            padding_attr_data = self.getTensor(node.inputs[0]).tensor_data
        else:
            padding_attr_data = None

        operands = list()
        operands.append(op)

        pool_max_2d_param = {
            'stride_h': node.config['strides'][0],
            'stride_w': node.config['strides'][1],
            'kernel_h': node.config['pool_size'][0],
            'kernel_w': node.config['pool_size'][1],
            'padding_b': padding_attr_data[0][0] if isinstance(padding_attr_data, np.ndarray) else 0,
            'padding_r': padding_attr_data[1][0] if isinstance(padding_attr_data, np.ndarray) else 0,
            'padding_t': padding_attr_data[0][1] if isinstance(padding_attr_data, np.ndarray) else 0,
            'padding_l': padding_attr_data[1][1] if isinstance(padding_attr_data, np.ndarray) else 0,
            'do_relu': False,
        }

        operands = list()
        operands.append(op)
        on = shape[0]
        oc = shape[1]
        oh = calcPool2DFloor(shape[2], pool_max_2d_param['kernel_h'], pool_max_2d_param['stride_h'],
                             pool_max_2d_param['padding_b'], pool_max_2d_param['padding_t'])
        ow = calcPool2DFloor(shape[3], pool_max_2d_param['kernel_w'], pool_max_2d_param['stride_w'],
                             pool_max_2d_param['padding_r'], pool_max_2d_param['padding_l'])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op("{}".format(node.name), operands, output_shape, **pool_max_2d_param)
        self.addOperand(node.name, pool_max_op, output_shape, TensorType.ACTIVATION)


    def convert_pad_op(self, node):
        assert(node.op_type == "ZeroPadding2D")
        """
            Fix: our mlir don't have padding op,
            We fuse with next Conv2d
            In tensorflow official resnet case, it can be work
            other case is TODO
            by Sam
        """
        # first input is activate, second is tensor
        op, shape, _ = self.getOperand(node.inputs[0])

        # Get padding data
        padding_attr_data = np.asarray(node.proto.get_config()['padding'])
        self.addOperand(node.name, op, shape, TensorType.TENSOR)
        # For Conv2d Get this data
        self.addTensor(node.name, padding_attr_data, shape, "PAD")

    def convert_reshape_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = list(node.config['target_shape']) # none batch size infomatiion [h, w, c]
        if len(output_shape) == 3:
            # hwc -> chw
            output_shape = [output_shape[2], output_shape[0], output_shape[1]]
        output_shape.insert(0, input_shape[0]) # add batch size
        reshape_op = self.CVI.add_reshape_op(node.name, operands, output_shape)
        self.addOperand(node.name, reshape_op, output_shape, TensorType.ACTIVATION)

    def convert_skip_op(self, node):
        op, input_shape, _ = self.getOperand(node.inputs[0])
        self.addOperand(node.name, op, input_shape, TensorType.ACTIVATION)

    def run(self):
        self.convert_node()
        self.convert_graph()
        self.TensortoNpz()


