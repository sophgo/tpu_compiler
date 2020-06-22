from .mlirimporter import MLIRImporter, checkKey
from .BaseConverter import BaseConverter, TensorType

import math
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from termcolor import colored, cprint


class CaffeTensor():
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape

    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')
        cprint("    size: {}".format(len(self.tensor_data)))


class CaffeConverter(BaseConverter):
    def __init__(self, model_name, prototxt, caffemodel, mlir_file_path, batch_size=1):
        super().__init__()
        self.model_name = model_name
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.batch_size = batch_size

        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        self.param = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), self.param)
        self.layers = self.param.layer
        self.inputs = self.net.inputs
        self.outputs = self.net.outputs
        self.blobs = self.net.blobs
        self.layer_dict = self.net.layer_dict

        self.mlir_file_path = mlir_file_path
        self.converted_tensors = list()

        self.input_shapes = list()
        for i in self.inputs:
            i_shape = list(self.blobs[i].shape)
            i_shape[0] = self.batch_size
            self.input_shapes.append(i_shape)
        # get output shape
        self.output_shapes = list()
        for o in self.outputs:
            o_shape = list(self.blobs[o].shape)
            o_shape[0] = self.batch_size
            self.output_shapes.append(o_shape)
        self.CVI = MLIRImporter(self.input_shapes, self.output_shapes)
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.caffeop_factory = {
            "BatchNorm": lambda layer: self.convert_batchnorm_op(layer),
            "Concat": lambda layer: self.convert_concat_op(layer),
            "Convolution": lambda layer: self.convert_convolution_op(layer),
            "ConvolutionDepthwise": lambda layer: self.convert_convolution_op(layer),
            "Crop": lambda layer: self.convert_crop_op(layer),
            "Deconvolution": lambda layer: self.convert_convolution_op(layer),
            "DetectionOutput": lambda layer: self.convert_detection_output_op(layer),
            "Dropout": lambda layer: self.convert_dropout_op(layer),
            "DummyData": lambda layer: self.convert_dummydata_op(layer),
            "Eltwise": lambda layer: self.convert_eltwise_op(layer),
            "Flatten": lambda layer: self.convert_flatten_op(layer),
            "InnerProduct": lambda layer: self.convert_inner_product_op(layer),
            "LRN": lambda layer: self.convert_lrn_op(layer),
            "Normalize": lambda layer: self.convert_normalize_op(layer),
            "Permute": lambda layer: self.convert_permute_op(layer),
            "Pooling": lambda layer: self.convert_pooling_op(layer),
            "Power": lambda layer: self.convert_power_op(layer),
            "PReLU": lambda layer: self.convert_prelu_op(layer),
            "PriorBox": lambda layer: self.convert_priorbox_op(layer),
            "ReLU": lambda layer: self.convert_relu_op(layer),
            "Reorg": lambda layer: self.convert_reorg_op(layer),
            "Reshape": lambda layer: self.convert_reshape_op(layer),
            "RetinaFaceDetection": lambda layer: self.convert_retinaface_detection_op(layer),
            "Scale": lambda layer: self.convert_scale_op(layer),
            "ShuffleChannel": lambda layer: self.convert_shufflechannel_op(layer),
            "Sigmoid": lambda layer: self.convert_sigmoid_op(layer),
            "Slice": lambda layer: self.convert_slice_op(layer),
            "Softmax": lambda layer: self.convert_softmax_op(layer),
            "Split": lambda layer: self.convert_split_op(layer),
            "TanH": lambda layer: self.convert_tanh_op(layer),
            "Upsample": lambda layer: self.convert_upsample_op(layer),
            "YoloDetection": lambda layer: self.convert_yolo_detection_op(layer),
        }

    def __del__(self):
        del self.CVI

    def addTensor(self, op_name, tensor_data, tensor_shape):
        self.converted_tensors.append(CaffeTensor(
            op_name, tensor_data, tensor_shape))

    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            tensor_npz[i.name] = i.tensor_data.astype(np.float32)
        np.savez(self.output_tensor_file, **tensor_npz)

    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise KeyError("No {} tensor in prototxt".format(op_name))
        else:
            return find_tensor[0]

    def noneOp(self):
        if not hasattr(self, 'none_op'):
            self.none_op = self.CVI.add_none_op()
        return self.none_op

    def blob_to_weight_op(self, layer, index, shape=None):
        name = layer.name + "_{}".format(index)
        blob = self.layer_dict[layer.name].blobs[index]
        value = blob.data
        weight_shape = shape if shape != None else list(blob.shape)
        self.addTensor(name, value, weight_shape)
        weight_op = self.CVI.add_load_file_op(name, weight_shape)
        return weight_op

    def convert_batchnorm_op(self, layer):
        assert(layer.type == "BatchNorm")
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        param = {
            'variance_epsilon': 1e-5
        }
        if layer.HasField('batch_norm_param') and layer.batch_norm_param.HasField('eps'):
            param['variance_epsilon'] = layer.batch_norm_param.eps

        blobs = self.layer_dict[layer.name].blobs
        for idx, blob in enumerate(blobs):
            blob_op = self.blob_to_weight_op(layer, idx)
            operands.append(blob_op)

        output_shape = input_shape
        new_op = self.CVI.add_batchnorm_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_concat_op(self, layer):
        assert(layer.type == 'Concat')
        input_num = len(layer.bottom)
        raise RuntimeError("{} will support later".format(layer.type))

    def convert_eltwise_op(self, layer):
        assert(layer.type == 'Eltwise')
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        p = layer.eltwise_param
        assert(len(p.coeff) == 0)
        operation = p.operation
        output_shape = input_shape
        if operation == 0:
            new_op = self.CVI.add_eltwise_mul_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        elif operation == 1:
            new_op = self.CVI.add_eltwise_add_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        elif operation == 2:
            new_op = self.CVI.add_eltwise_max_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        elif operation == 3:
            new_op = self.CVI.add_eltwise_min_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)

    @staticmethod
    def calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int(((_i_) + 2 * (_p_) - (_d_) * ((_k_)-1) - 1) / (_s_) + 1)

    @staticmethod
    def calcDeConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int((_s_) * (((_i_)) - 1) + (_d_) * ((_k_)-1) - 2 * (_p_) + 1)

    def convert_convolution_op(self, layer):
        assert(layer.type == "Convolution" or layer.type ==
               "ConvolutionDepthwise" or layer.type == 'Deconvolution')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        assert(len(input_shape) == 4)
        operands = list()
        operands.append(op)
        p = layer.convolution_param
        oc = p.num_output
        g = 1
        if layer.type == "ConvolutionDepthwise":
            g = oc
        else:
            g = p.group
        is_deconv = True if layer.type == 'Deconvolution' else False
        with_bias = p.bias_term
        kernel = [0, 0]
        if len(p.kernel_size) != 0:
            kernel[0] = p.kernel_size[1] if len(
                p.kernel_size) > 1 else p.kernel_size[0]
            kernel[1] = p.kernel_size[0]
        if p.HasField('kernel_h'):
            kernel[0] = p.kernel_h
        if p.HasField('kernel_w'):
            kernel[1] = p.kernel_w
        stride = [1, 1]
        if len(p.stride) != 0:
            stride[0] = p.stride[1] if len(p.stride) > 1 else p.stride[0]
            stride[1] = p.stride[0]
        if p.HasField('stride_h'):
            stride[0] = p.stride_h
        if p.HasField('stride_w'):
            stride[1] = p.stride_w
        padding = [0, 0]
        if len(p.pad) != 0:
            padding[0] = p.pad[1] if len(p.pad) > 1 else p.pad[0]
            padding[1] = p.pad[0]
        if p.HasField('pad_h'):
            padding[0] = p.pad_h
        if p.HasField('pad_w'):
            padding[1] = p.pad_w
        padding_str = 'SAME' if (padding[0] > 0 or padding[1] > 0) else 'VALID'
        dilation = [1, 1]
        if len(p.dilation) != 0:
            dilation[0] = p.dilation[1] if len(
                p.dilation) > 1 else p.dilation[0]
            dilation[1] = p.dilation[0]
        n = input_shape[0]
        ic = input_shape[1]
        ifmap = [input_shape[2], input_shape[3]]
        ofmap = [0, 0]
        if not is_deconv:
            ofmap[0] = self.calcConv2DSpatialOutput(
                ifmap[0], kernel[0], stride[0], padding[0], dilation[0])
            ofmap[1] = self.calcConv2DSpatialOutput(
                ifmap[1], kernel[1], stride[1], padding[1], dilation[1])
        else:
            ofmap[0] = self.calcDeConv2DSpatialOutput(ifmap[0], kernel[0], stride[0], padding[0],
                                                      dilation[0])
            ofmap[1] = self.calcDeConv2DSpatialOutput(ifmap[1], kernel[1], stride[1], padding[1],
                                                      dilation[1])
        is_dw = True if g == oc else False

        # filter op
        filter_shape = [g, oc / g, ic / g, kernel[0], kernel[1]
                        ] if g != 1 else [oc, ic, kernel[0], kernel[1]]
        filter_op = self.blob_to_weight_op(layer, 0, filter_shape)
        operands.append(filter_op)
        # bias op
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1)
            operands.append(bias_op)
        else:
            operands.append(self.noneOp())

        output_shape = [n, oc, ofmap[0], ofmap[1]]
        conv_param = {
            'dilation_h': dilation[0],
            'dilation_w': dilation[1],
            'stride_h': stride[0],
            'stride_w': stride[1],
            'padding': padding_str,
            'padding_t': padding[0],
            'padding_b': padding[0],
            'padding_l': padding[1],
            'padding_r': padding[1],
            'group': g,
            'is_dw': is_dw,
            'with_bias': with_bias,
            'do_relu': False
        }
        if not is_deconv:
            new_op = self.CVI.add_conv_op(
                layer.name, operands, output_shape, **conv_param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        else:
            new_op = self.CVI.add_deconv_op(
                layer.name, operands, output_shape, **conv_param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_inner_product_op(self, layer):
        assert(layer.type == 'InnerProduct')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        p = layer.inner_product_param
        with_bias = p.bias_term
        with_transpose = p.transpose
        assert(with_transpose == False)  # transpose not support now
        N = p.num_output
        M = input_shape[0]
        K = input_shape[1]
        reshape_first = False
        if len(input_shape) > 2:
            reshape_first = True
            for i in range(2, len(input_shape)):
                K *= input_shape[i]
        fc_op = op
        if reshape_first:
            fc_shape = [M, K]
            fc_operands = [op]
            fc_op = self.CVI.add_reshape_op(
                layer.name + '_reshape', fc_operands, fc_shape)
        operands.append(fc_op)
        # filter
        filter_op = self.blob_to_weight_op(layer, 0, [N, K])
        operands.append(filter_op)
        if with_bias:
            bias_op = self.blob_to_weight_op(layer, 1, [N])
            operands.append(bias_op)
        else:
            operands.append(self.noneOp())
        output_shape = [M, N]
        new_op = self.CVI.add_fully_connected_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_relu_op(self, layer):
        assert(layer.type == 'ReLU')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        negative_slope = 0.0
        if layer.HasField('batch_norm_param'):
            negative_slope = layer.batch_norm_param.negative_slope
        output_shape = input_shape
        if negative_slope == 0.0:
            new_op = self.CVI.add_relu_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        else:
            param = {
                'negative_slope': negative_slope
            }
            new_op = self.CVI.add_leaky_relu_op(
                layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_pooling_op(self, layer):
        assert(layer.type == 'Pooling')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.pooling_param
        pool_method = p.pool
        assert(pool_method == 0 or pool_method == 1)
        n = input_shape[0]
        c = input_shape[1]
        ifmap = [input_shape[2], input_shape[3]]
        kernel = [input_shape[2], input_shape[3]]
        is_global_pooling = p.global_pooling
        if not p.global_pooling:
            kernel[0] = p.kernel_h if p.HasField('kernel_h') else p.kernel_size
            kernel[1] = p.kernel_w if p.HasField('kernel_w') else p.kernel_size
        stride = [p.stride, p.stride]
        if p.HasField('stride_h'):
            stride[0] = p.stride_h
        if p.HasField('stride_w'):
            stride[1] = p.stride_w
        padding = [p.pad, p.pad]
        if p.HasField('pad_h'):
            paddding[0] = p.pad_h
        if p.HasField('pad_w'):
            padding[1] = p.pad_w
        ceil_mode = p.ceil_mode
        round_mode = p.round_mode
        if round_mode == 1:
            ceil_mode = False
        padding_tl = [padding[0], padding[1]]
        padding_br =  [padding[0], padding[1]]
        ofmap = [0, 0]
        for i in [0, 1]:
            if ceil_mode:
                ofmap[i] = math.ceil(
                    (ifmap[i] + 2.0 * padding[i] - kernel[i]) / stride[i]) + 1
            else:
                ofmap[i] = math.floor(
                    (ifmap[i] + 2.0 * padding[i] - kernel[i]) / stride[i]) + 1
            remain_pixel = (ifmap[i] + 2 * padding[i] - kernel[i]) % stride[i]
            if remain_pixel > 0:
                if ceil_mode:
                    padding_br[i] += (stride[i] - remain_pixel)
                else:
                    padding_br[i] -= remain_pixel
        if is_global_pooling:
            assert((padding[0] == 0) and (padding[1] == 0))
            assert((stride[0] == 1) and (stride[1] == 1))
            assert((ofmap[0] == 1) and (ofmap[1] == 1))
        pool_param = {
            'kernel_h': kernel[0],
            'kernel_w': kernel[1],
            'padding_t': padding_tl[0],
            'padding_b': padding_br[0],
            'padding_l': padding_tl[1],
            'padding_r': padding_br[1],
            'stride_h': stride[0],
            'stride_w': stride[1],
            'do_relu': False
        }
        output_shape = [n, c, ofmap[0], ofmap[1]]
        if pool_method == 0:  # MAX
            new_op = self.CVI.add_pool_max_2d_op(
                layer.name, operands, output_shape, **pool_param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
        elif pool_method == 1:  # AVE
            new_op = self.CVI.add_pool_avg_2d_op(
                layer.name, operands, output_shape, **pool_param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_scale_op(self, layer):
        assert(layer.type == 'Scale')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims == 4 or num_dims == 2)
        if len(layer.bottom) == 2:
            # add broadcast mul
            raise RuntimeError(
                "{} with two bottom will support later".format(layer.type))
        else:
            with_bias = False
            if layer.HasField('scale_param') and layer.scale_param.HasField('bias_term'):
                with_bias = layer.scale_param.bias_term
            # scale
            scale_op = self.blob_to_weight_op(layer, 0)
            operands.append(scale_op)
            # bias
            if with_bias:
                bias_op = self.blob_to_weight_op(layer, 1)
                operands.append(bias_op)
            else:
                operands.append(self.noneOp())
            output_shape = input_shape
            new_op = self.CVI.add_scale_op(
                layer.name, operands, output_shape)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)

    def convert_softmax_op(self, layer):
        assert(layer.type == 'Softmax')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        axis = 1
        if layer.HasField('softmax_param') and layer.softmax_param.HasField('axis'):
            axis = layer.softmax_param.axis
        param = {
            'axis': axis
        }
        output_shape = input_shape
        new_op = self.CVI.add_softmax_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        # add input op
        for idx, name in enumerate(self.inputs):
            input_shape = self.input_shapes[idx]
            input_op = self.CVI.add_input_op(name, idx)
            self.addOperand(name, input_op, input_shape, TensorType.ACTIVATION)

        for layer in self.layers:
            self.caffeop_factory.get(
                layer.type, lambda x: NoneAndRaise(x))(layer)

        # add return op
        return_op = list()
        # Set output
        for output in self.outputs:
            op, _, _ = self.getOperand(output)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open(self.mlir_file_path, "w") as f:
            f.write(mlir_txt)

    def run(self):
        self.convert_graph()
        self.TensortoNpz()
