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

logger = setup_logger('root')
log_flag = logger.level <= logging.INFO


class TFNode():
    def __init__(self, name, op_type, inputs, outputs, proto):
        self.name = str(name)
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.proto = proto

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')

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
        self.net = tf.keras.models.load_model(model_path)
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
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Conv2D": lambda node: self.convert_conv_op(node),
            "Dense": lambda node: self.convert_fc_op(node),
            "GlobalAveragePooling2D": lambda node: self.convert_global_avg_pool_op(node),
            "InputLayer": lambda node: None,
            "MaxPooling2D": lambda node: self.convert_maxpool_op(node),
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
    # weights = tf_op.get_weights()
            # print(tf_op.get_config())
            # for w in weights:
            #     print(w.shape)
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
            input_op = self.CVI.add_input_op(input.name, idx)
            name = input.name.split(":")[0] ## FixMe: Hardcore to strip ":""
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

    def convert_activation_op(self, node):
        assert(node.op_type == "Activation")
        config = node.proto.get_config()
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape

        if config['activation'] == "relu":
            activation_op = self.CVI.add_relu_op("{}".format(node.name), operands, output_shape)

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

    def convert_batchnorm_op(self, node):
        assert(node.op_type == "BatchNormalization")
        config = node.proto.get_config()
        op, input_shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = config['epsilon']

        # we fuse batchnorm and scale at here
        gamma_value = node.proto.get_weights()[0]
        beta_value = node.proto.get_weights()[1]
        mean_value = node.proto.get_weights()[2]
        var_value = node.proto.get_weights()[3]
        print(gamma_value.shape)
        print(beta_value.shape)
        print(mean_value.shape)
        print(var_value.shape)
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

    def convert_conv_op(self, node):
        assert(node.op_type == "Conv2D")
        config = node.proto.get_config()
        op, shape, tesnor_type = self.getOperand(node.inputs[0])
        if tesnor_type == TensorType.TENSOR:
            # get padding data from tensor
            padding_attr_data = self.getTensor(node.inputs[0]).tensor_data
        else:
            padding_attr_data = None
        print(padding_attr_data)
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
        do_bias = config['use_bias']
        if do_bias:
            bias_data = node.proto.get_weights()[1]
            bias_shape = bias_data.shape
            bias_name = "{}_add_bias".format(node.name)
            bias_op = self.createLoadWeightOp(bias_name, bias_data, bias_shape)
            operands.append(bias_op)


        print(config)
        conv_param = {
            'stride_h': config['strides'][0],
            'stride_w': config['strides'][1],
            'padding': "SAME" if config['padding'] == "same" or isinstance(padding_attr_data, np.ndarray) else "VALID",
            'dilation_h': config['dilation_rate'][0],
            'dilation_w': config['dilation_rate'][1],
            'group': 1,  # Don't have group option?
            'is_dw': False,
            'with_bias': config['use_bias'],
            'do_relu': False,
        }
        on = shape[0]
        oc = filter_shape[0] # feature map size
        # padding data order is NHWC
        # if padding data is not np.ndarray (not from bottom layer)
        # and conv_table.Padding() is SAME, we need to calculate it.
        if config['padding'] == "same":
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
            padding_attr_data[0][0] if isinstance(padding_attr_data, np.ndarray) else padding_h,
            conv_param['dilation_h'],
        )
        ow = calcConv2DSpatial(
            shape[3],
            filter_shape[3],
            conv_param['stride_w'],
            padding_attr_data[1][0] if isinstance(padding_attr_data, np.ndarray) else padding_w,
            conv_param['dilation_w'],
        )
        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op("{}".format(
            node.name), operands, output_shape, **conv_param)
        self.addOperand(node.name, conv_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_fc_op(self, node):
        assert(node.op_type == "Dense")
        config = node.proto.get_config()
        op, shape, _ = self.getOperand(node.inputs[0])
        operands = list()
        operands.append(op)

        # filter
        filter_data = node.proto.get_weights()[0]
        filter_data = np.transpose(filter_data, (1, 0))
        filter_shape = filter_data.shape
        filter_name = "{}_add_weight".format(node.name)
        filter_op = self.createLoadWeightOp(
            filter_name, filter_data, filter_shape)
        operands.append(filter_op)

        # bias
        do_bias = config['use_bias']
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
        fc_op = self.CVI.add_fully_connected_op("{}".format(node.name), operands, output_shape)
        self.addOperand(node.name, fc_op, output_shape, TensorType.ACTIVATION)

    def convert_global_avg_pool_op(self, node):
        assert(node.op_type == "GlobalAveragePooling2D")
        config = node.proto.get_config()
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
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}_{}".format(
            node.name, node.op_type), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_maxpool_op(self, node):
        assert(node.op_type == "MaxPooling2D")
        config = node.proto.get_config()
        op, shape, tensor_type = self.getOperand(node.inputs[0])

        if tensor_type == TensorType.TENSOR:
            # get padding data from tensor
            padding_attr_data = self.getTensor(node.inputs[0]).tensor_data
        else:
            padding_attr_data = None
        print(padding_attr_data)

        operands = list()
        operands.append(op)
        print(config)

        pool_max_2d_param = {
            'stride_h': config['strides'][0],
            'stride_w': config['strides'][1],
            'kernel_h': config['pool_size'][0],
            'kernel_w': config['pool_size'][1],
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
        oh = calcPool2DFloor(shape[2], pool_max_2d_param['kernel_h'], pool_max_2d_param['stride_h'], pool_max_2d_param['padding_b'])
        ow = calcPool2DFloor(shape[3], pool_max_2d_param['kernel_w'], pool_max_2d_param['stride_w'], pool_max_2d_param['padding_r'])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op("{}".format(node.name), operands, output_shape, **pool_max_2d_param)
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
        pool_avg_op = self.CVI.add_pool_avg_2d_op("{}".format(
            node.name), operands, output_shape, **pool_avg_2d_param)
        self.addOperand(node.name, pool_avg_op,
                        output_shape, TensorType.ACTIVATION)

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

    def convert_softmax_op(self, node):
        assert(node.op_type == "SOFTMAX")
        # first input is activate
        assert(len(node.inputs) == 1)
        op, shape, _ = self.getOperand(str(node.inputs[0]))
        operands = list()
        operands.append(op)
        self.addOperand(node.name, op, shape, TensorType.ACTIVATION)
        softmax_op = self.CVI.add_softmax_op("{}".format(
            node.name), operands, shape)

    def run(self):
        self.convert_node()
        self.convert_graph()
        self.TensortoNpz()


