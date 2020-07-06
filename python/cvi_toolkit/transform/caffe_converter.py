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
    def __init__(self, model_name, prototxt, caffemodel, mlir_file_path, batch_size=1, preprocess=None):
        super().__init__()
        self.model_name = model_name
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.batch_size = batch_size
        self.preprocess = preprocess

        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        self.param = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt).read(), self.param)
        self.layers = self.param.layer if len(
            self.param.layer) != 0 else self.param.layers
        self.inputs = self.net.inputs
        self.outputs = self.net.outputs
        self.blobs = self.net.blobs
        self.layer_dict = self.net.layer_dict

        self.mlir_file_path = mlir_file_path
        self.converted_tensors = list()
        self.input_shapes = list()
        self.output_shapes = list()
        self.CVI = None
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.caffeop_factory = {
            'BatchNorm': lambda layer: self.convert_batchnorm_op(layer),
            'Concat': lambda layer: self.convert_concat_op(layer),
            'Convolution': lambda layer: self.convert_convolution_op(layer),
            'ConvolutionDepthwise': lambda layer: self.convert_convolution_op(layer),
            'Crop': lambda layer: self.convert_crop_op(layer),
            'Deconvolution': lambda layer: self.convert_convolution_op(layer),
            'DetectionOutput': lambda layer: self.convert_detection_output_op(layer),
            'Dropout': lambda layer: self.convert_dropout_op(layer),
            'DummyData': lambda layer: self.convert_dummydata_op(layer),
            'Eltwise': lambda layer: self.convert_eltwise_op(layer),
            'Flatten': lambda layer: self.convert_flatten_op(layer),
            'InnerProduct': lambda layer: self.convert_inner_product_op(layer),
            'Input': lambda layer: self.convert_input_op(layer),
            'LRN': lambda layer: self.convert_lrn_op(layer),
            'Normalize': lambda layer: self.convert_normalize_op(layer),
            'Permute': lambda layer: self.convert_permute_op(layer),
            'Pooling': lambda layer: self.convert_pooling_op(layer),
            'Power': lambda layer: self.convert_power_op(layer),
            'PReLU': lambda layer: self.convert_prelu_op(layer),
            'PriorBox': lambda layer: self.convert_priorbox_op(layer),
            'ReLU': lambda layer: self.convert_relu_op(layer),
            'Reorg': lambda layer: self.convert_reorg_op(layer),
            'Reshape': lambda layer: self.convert_reshape_op(layer),
            'RetinaFaceDetection': lambda layer: self.convert_retinaface_detection_op(layer),
            'Scale': lambda layer: self.convert_scale_op(layer),
            'ShuffleChannel': lambda layer: self.convert_shufflechannel_op(layer),
            'Sigmoid': lambda layer: self.convert_sigmoid_op(layer),
            'Slice': lambda layer: self.convert_slice_op(layer),
            'Softmax': lambda layer: self.convert_softmax_op(layer),
            'Split': lambda layer: self.convert_split_op(layer),
            'TanH': lambda layer: self.convert_tanh_op(layer),
            'Upsample': lambda layer: self.convert_upsample_op(layer),
            'YoloDetection': lambda layer: self.convert_yolo_detection_op(layer),
        }
        # for caffe v1
        self.layer_type = {
            0: 'None', 35: 'Absval', 1: 'Accuracy', 30: 'Argmax', 2: 'Bnll',
            3: 'Concat', 37: 'ContrastiveLoss', 4: 'Convolution', 5: 'Data',
            39: 'Deconvolution', 6: 'Dropout', 32: 'DummyData', 7: 'EuclideanLoss',
            25: 'Eltwise', 38: 'Exp', 8: 'Flatten', 9: 'Hdf5Data', 10: 'Hdf5Output',
            28: 'HingeLoss', 11: 'Im2col', 12: 'ImageData', 13: 'InfogainLoss',
            14: 'InnerProduct', 15: 'LRN', 29: 'MemoryData', 16: 'MultinomialLogisticLoss',
            34: 'MVN', 17: 'Pooling', 26: 'Power', 18: 'ReLU', 19: 'Sigmoid',
            27: 'SigmoidCrossEntropyLoss', 36: 'Silence', 20: 'Softmax', 21: 'SoftmaxLoss',
            22: 'Split', 33: 'Slice', 23: 'Tanh', 24: 'WindowData', 31: 'Threshold',
        }
        self.init_importer()

    def __del__(self):
        del self.CVI

    def layerType(self, layer):
        if type(layer.type) == int:
            return self.layer_type.get(layer.type)
        else:
            return layer.type

    def init_importer(self):
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
            for layer in self.layers:
                if layer.name == o and self.layerType(layer) == 'DetectionOutput':
                    o_shape[2] = layer.detection_output_param.keep_top_k
                    break
            self.output_shapes.append(o_shape)
        self.CVI = MLIRImporter(self.input_shapes, self.output_shapes)

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
        blob_shape = list(blob.shape)
        value = np.array
        new_shape = list(blob_shape)
        if shape != None:
            new_shape = [int(i) for i in shape]
        if new_shape == blob_shape:
            value = blob.data
        else:
            value = blob.data.reshape(new_shape)
        self.addTensor(name, value, new_shape)
        weight_op = self.CVI.add_load_file_op(name, new_shape)
        return weight_op

    def convert_batchnorm_op(self, layer):
        assert(self.layerType(layer) == "BatchNorm")
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
        assert(self.layerType(layer) == 'Concat')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        input_num = len(layer.bottom)
        if input_num == 1:
            return self.addOperand(layer.top[0], op, input_shape, TensorType.ACTIVATION)
        axis = layer.concat_param.axis
        assert(axis < len(input_shape))
        concat_axis_dim = 0
        operands = list()
        for bottom in layer.bottom:
            bottom_op, shape, _ = self.getOperand(bottom)
            assert(len(shape) == len(input_shape))
            concat_axis_dim += shape[axis]
            operands.append(bottom_op)
        output_shape = list(input_shape)
        output_shape[axis] = concat_axis_dim
        param = {
            'axis': axis
        }
        new_op = self.CVI.add_concat_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    @ staticmethod
    def calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int(((_i_) + 2 * (_p_) - (_d_) * ((_k_)-1) - 1) / (_s_) + 1)

    @ staticmethod
    def calcDeConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_):
        return int((_s_) * (((_i_)) - 1) + (_d_) * ((_k_)-1) - 2 * (_p_) + 1)

    def convert_convolution_op(self, layer):
        assert(self.layerType(layer) == "Convolution" or self.layerType(layer) ==
               "ConvolutionDepthwise" or self.layerType(layer) == 'Deconvolution')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        assert(len(input_shape) == 4)
        operands = list()
        operands.append(op)
        p = layer.convolution_param
        oc = p.num_output
        g = 1
        if self.layerType(layer) == 'ConvolutionDepthwise':
            g = oc
        else:
            g = p.group
        is_deconv = True if self.layerType(layer) == 'Deconvolution' else False
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

    def convert_crop_op(self, layer):
        assert(self.layerType(layer) == 'Crop')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        crop_op, crop_shape, _ = self.getOperand(layer.bottom[1])
        p = layer.crop_param
        input_dim = len(input_shape)
        axis_index = p.axis
        start_axis = axis_index
        offset_size = len(p.offset)
        if offset_size > 1:
            assert(offset_size + axis_index <= input_dim)
        output_shape = list(input_shape)
        crop_offset = list(input_shape)
        for i in range(input_dim):
            offset = 0
            new_size = input_shape[i]
            if i >= start_axis:
                new_size = crop_shape[i]
                if offset_size == 1:
                    # If only one offset is given, all crops have the same offset.
                    offset = p.offset[0]
                elif offset_size > 1:
                    # For several offsets, the number of offsets must be equal to the
                    # number of dimensions to crop, that is dimensions after the axis.
                    offset = p.offset[i - start_axis]
            output_shape[i] = new_size
            crop_offset[i] = offset
        # TODO(charle.hu):if crop_op is dummy op, need to erase?
        operands = list()
        operands.append(op)
        param = {
            'crop_offset': crop_offset,
            'crop_shape': crop_shape
        }
        new_op = self.CVI.add_crop_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_detection_output_op(self, layer):
        assert(self.layerType(layer) == "DetectionOutput")
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.detection_output_param
        code_type = "CORNER"
        if p.code_type == 2:
            code_type = "CENTER_SIZE"
        elif p.code_type == 3:
            code_type = "CORNER_SIZE"
        param = {
            'num_classes': p.num_classes,
            'share_location': p.share_location,
            'background_label_id': p.background_label_id,
            'nms_threshold': p.nms_param.nms_threshold,
            'top_k': p.nms_param.top_k,
            'code_type': code_type,
            'keep_top_k': p.keep_top_k,
            'confidence_threshold': p.confidence_threshold
        }
        assert(1.0 == p.nms_param.eta)
        assert(False == p.variance_encoded_in_target)
        output_shape = [input_shape[0], 1, p.keep_top_k, 7]
        new_op = self.CVI.add_detection_output_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_dropout_op(self, layer):
        assert(self.layerType(layer) == 'Dropout')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        self.addOperand(layer.top[0], op, input_shape, TensorType.ACTIVATION)

    def convert_dummydata_op(self, layer):
        assert(self.layerType(layer) == 'DummyData')
        operands = list()
        p = layer.dummy_data_param
        assert(len(p.shape) > 0)
        output_shape = list(p.shape[0].dim)
        new_op = self.CVI.add_dummydata_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_eltwise_op(self, layer):
        assert(self.layerType(layer) == 'Eltwise')
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

    def convert_flatten_op(self, layer):
        assert(self.layerType(layer) == 'Flatten')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims > 1)
        output_shape = [input_shape[0], 1]
        for i in range(1, num_dims):
            output_shape[1] *= input_shape[i]
        new_op = self.CVI.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_inner_product_op(self, layer):
        assert(self.layerType(layer) == 'InnerProduct')
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

    def convert_input_op(self, layer):
        assert(self.layerType(layer) == 'Input')
        # do nothing

    def convert_lrn_op(self, layer):
        assert(self.layerType(layer) == 'LRN')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.lrn_param
        param = {
            'alpha': p.alpha,
            'beta': p.beta,
            'bias': p.k,
            'size': p.local_size,
        }
        output_shape = input_shape
        new_op = self.CVI.add_lrn_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_normalize_op(self, layer):
        assert(self.layerType(layer) == 'Normalize')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.norm_param
        param = {
            'across_spatial': p.across_spatial,
            'channel_shared': p.channel_shared,
        }
        assert(False == p.across_spatial)
        assert(len(input_shape) == 4)
        c = input_shape[1]
        # scale
        scale_shape = [1, c]
        scale_name = layer.name + "_0"
        blob = self.layer_dict[layer.name].blobs[0]
        scale_data = np.array
        if p.channel_shared:
            assert(blob.count == 1)
            value = blob.data.flatten()[0]
            scale_data = np.array([[value for i in range(c)]], dtype=float)
        else:
            assert(blob.count == c)
            scale_data = blob.data.reshape(scale_shape)
        self.addTensor(scale_name, scale_data, scale_shape)
        scale_op = self.CVI.add_load_file_op(scale_name, scale_shape)
        operands.append(scale_op)
        output_shape = input_shape
        new_op = self.CVI.add_normalize_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_permute_op(self, layer):
        assert(self.layerType(layer) == 'Permute')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.permute_param
        assert(len(p.order) == 4)
        output_shape = list(input_shape)
        for i in range(4):
            output_shape[i] = input_shape[p.order[i]]
        param = {
            'order0': p.order[0],
            'order1': p.order[1],
            'order2': p.order[2],
            'order3': p.order[3],
        }
        new_op = self.CVI.add_permute_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_pooling_op(self, layer):
        assert(self.layerType(layer) == 'Pooling')
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
            padding[0] = p.pad_h
        if p.HasField('pad_w'):
            padding[1] = p.pad_w
        ceil_mode = p.ceil_mode
        round_mode = p.round_mode
        if round_mode == 1:
            ceil_mode = False
        padding_tl = [padding[0], padding[1]]
        padding_br = [padding[0], padding[1]]
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
            'do_relu': False,
            'count_include_pad': True,
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

    def convert_power_op(self, layer):
        assert(self.layerType(layer) == 'Power')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        p = layer.power_param
        if p.shift == 0 and p.power == 1 and p.scale == 1:
            # do nothing
            return self.addOperand(layer.top[0], op, input_shape, TensorType.ACTIVATION)
        param = {
            'power': p.power,
            'scale': p.scale,
            'shift': p.shift,
        }
        output_shape = input_shape
        new_op = self.CVI.add_power_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_prelu_op(self, layer):
        assert(self.layerType(layer) == 'PReLU')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        c = input_shape[1]
        # negative_slope
        slope_op = self.blob_to_weight_op(layer, 0, [1, c, 1, 1])
        operands.append(slope_op)
        output_shape = input_shape
        new_op = self.CVI.add_prelu_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_priorbox_op(self, layer):
        assert(self.layerType(layer) == 'PriorBox')
        op0, input_shape0, _ = self.getOperand(layer.bottom[0])
        op1, input_shape1, _ = self.getOperand(layer.bottom[1])
        operands = list()
        operands.append(op0)
        operands.append(op1)
        assert(len(input_shape0) == 4)
        h = input_shape0[2]
        w = input_shape0[3]
        p = layer.prior_box_param
        min_size_size = len(p.min_size)
        max_size_size = len(p.max_size)
        min_size = p.min_size[0]
        max_size = 0.0
        if max_size_size == 1:
            max_size = p.max_size[0]
        aspect_ratio_size = len(p.aspect_ratio)
        assert(max_size_size <= 1 and min_size_size ==
               1 and aspect_ratio_size <= 2)
        param = {
            'min_size': min_size,
            'min_size_size': min_size_size,
            'max_size': max_size,
            'max_size_size': max_size_size,
            'aspect_ratio0': p.aspect_ratio[0],
            'aspect_ratios_size': aspect_ratio_size,
            'flip': p.flip,
            'clip': p.clip,
            'variance0': p.variance[0],
            'variance1': p.variance[1],
            'variance2': p.variance[2],
            'variance3': p.variance[3],
            'step': p.step,
            'offset': p.offset,
        }
        if aspect_ratio_size == 2:
            param['aspect_ratio1'] = p.aspect_ratio[1]

        aspect_ratios_ = list()
        aspect_ratios_.append(1.0)
        for i in range(aspect_ratio_size):
            ar = p.aspect_ratio[i]
            already_exist = False
            for j in range(len(aspect_ratios_)):
                if math.fabs(ar - aspect_ratios_[j]) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratios_.append(ar)
                if p.flip:
                    aspect_ratios_.append(1.0 / ar)
        num_priors = len(aspect_ratios_) * min_size_size + max_size_size
        output_shape = [1, 2, int(h * w * num_priors * 4)]
        new_op = self.CVI.add_priorbox_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_relu_op(self, layer):
        assert(self.layerType(layer) == 'ReLU')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4 or len(input_shape) == 2)
        negative_slope = layer.relu_param.negative_slope
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

    def convert_reorg_op(self, layer):
        assert(self.layerType(layer) == 'Reorg')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        stride = layer.reorg_param.stride
        output_shape = list(input_shape)
        output_shape[0] = input_shape[0]
        output_shape[1] = int(input_shape[1] * stride * stride)
        output_shape[2] = int(input_shape[2] / stride)
        output_shape[3] = int(input_shape[3] / stride)
        param = {
            'stride': stride
        }
        new_op = self.CVI.add_reorg_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_reshape_op(self, layer):
        assert(self.layerType(layer) == 'Reshape')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims == 4 or num_dims == 2)
        p = layer.reshape_param
        top_dims = list(p.shape.dim)
        output_shape = list()
        input_count = 1
        for dim in input_shape:
            input_count *= dim
        if num_dims == 4:
            num_axes = p.num_axes
            start_axis = p.axis if p.axis >= 0 else (
                num_dims + p.axis + 1)
            assert(start_axis >= 0)
            assert(start_axis <= num_dims)
            assert(num_axes >= -1)
            end_axis = num_dims if num_axes == -1 else (start_axis + num_axes)
            assert(end_axis <= num_dims)

            num_axes_replaced = end_axis - start_axis
            num_axes_retained = num_dims - num_axes_replaced
            num_new_axes = len(top_dims)

            copy_axes = list()
            inferred_axis = -1
            constant_count = 1
            for i in range(num_new_axes):
                dim = top_dims[i]
                if dim == 0:
                    copy_axes.append(i)
                elif dim == -1:
                    assert(inferred_axis == -1)
                    inferred_axis = i
                else:
                    constant_count *= dim
            output_shape = [0] * (num_axes_retained + num_new_axes)
            output_shape_index = 0
            for i in range(start_axis):
                output_shape[output_shape_index] = input_shape[i]
                output_shape_index += 1
            for i in range(num_new_axes):
                output_shape[output_shape_index] = top_dims[i]
                output_shape_index += 1
            for i in range(end_axis, num_dims):
                output_shape[output_shape_index] = input_shape[i]
                output_shape_index += 1
            assert(output_shape_index == len(output_shape))
            for i in range(len(copy_axes)):
                copy_axis_index = copy_axes[i]
                assert(num_dims > start_axis + copy_axis_index)
                output_shape[start_axis +
                             copy_axis_index] = input_shape[start_axis + copy_axis_index]

            if inferred_axis >= 0:
                # A -1 dim was specified; infer the correct dimension by computing the
                # product of the other dimensions.
                explicit_count = constant_count
                for i in range(start_axis):
                    explicit_count *= input_shape[i]
                for i in range(end_axis, num_dims):
                    explicit_count *= input_shape[i]
                for i in range(len(copy_axes)):
                    copy_axis_index = copy_axes[i]
                    explicit_count *= (output_shape[start_axis +
                                                    copy_axis_index])
                assert(0 == input_count % explicit_count)
                inferred_dim = input_count / explicit_count
                output_shape[start_axis + inferred_axis] = int(inferred_dim)
        else:
            # only support input shape size is 2 && output shape size is 3 case
            assert(len(top_dims) == 3)
            output_shape = [0, 0, 0]
            inference_dim = 0
            for i in range(len(top_dims)):
                dim = top_dims[i]
                if dim == 0:
                    output_shape[i] = int(input_shape[i])
                    input_count /= output_shape[i]
                elif dim == -1:
                    inference_dim = i
                else:
                    output_shape[i] = int(top_dims[i])
                    input_count /= top_dims[i]
            output_shape[inference_dim] = int(input_count)
        new_op = self.CVI.add_reshape_op(layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_retinaface_detection_op(self, layer):
        assert(self.layerType(layer) == 'RetinaFaceDetection')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        operands.append(op)
        p = layer.retinaface_detection_param
        nms_threshold = p.nms_threshold
        confidence_threshold = p.confidence_threshold
        keep_topk = p.keep_topk
        output_shape = [1, 1, keep_topk, 15]
        param = {
            'nms_threshold': nms_threshold,
            'confidence_threshold': confidence_threshold,
            'keep_topk': keep_topk,
        }
        new_op = self.CVI.add_retinaface_detection_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op, output_shape,
                        TensorType.ACTIVATION)

    def convert_scale_op(self, layer):
        assert(self.layerType(layer) == 'Scale')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        num_dims = len(input_shape)
        assert(num_dims == 4 or num_dims == 2)
        if len(layer.bottom) == 2:
            # add broadcast mul
            op1, _, _ = self.getOperand(layer.bottom[1])
            operands.append(op1)
            param = {
                'axis': 1
            }
            output_shape = input_shape
            new_op = self.CVI.add_broadcast_mul_op(
                layer.name, operands, output_shape, **param)
            self.addOperand(layer.top[0], new_op, output_shape,
                            TensorType.ACTIVATION)
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

    def convert_shufflechannel_op(self, layer):
        assert(self.layerType(layer) == 'ShuffleChannel')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        group = layer.shuffle_channel_param.group
        operands = list()
        operands.append(op)
        param = {
            'group': group
        }
        output_shape = input_shape
        new_op = self.CVI.add_shufflechannel_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_sigmoid_op(self, layer):
        assert(self.layerType(layer) == 'Sigmoid')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        new_op = self.CVI.add_sigmoid_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_slice_op(self, layer):
        assert(self.layerType(layer) == 'Slice')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        assert(len(input_shape) == 4)
        p = layer.slice_param
        axis = p.axis
        assert(axis == 1)  # only support channel slice
        bottom_slice_axis = input_shape[axis]
        top_size = len(layer.top)
        slice_num = len(p.slice_point)
        slices = list()
        if slice_num > 0:
            assert(slice_num == top_size - 1)
            assert(top_size < bottom_slice_axis)
            prev = 0
            for i in range(slice_num):
                assert(p.slice_point[i] > prev)
                slices.append(p.slice_point[i] - prev)
                prev = p.slice_point[i]
            slices.append(bottom_slice_axis - prev)
        else:
            assert(bottom_slice_axis % top_size == 0)
            for i in range(top_size):
                slices.append(bottom_slice_axis / top_size)
        offset = 0
        for i in range(top_size):
            output_shape = list(input_shape)
            output_shape[axis] = slices[i]
            param = {
                'axis': axis,
                'offset': offset,
            }
            new_op = self.CVI.add_slice_op("{}_{}".format(
                layer.name, i), operands, output_shape, **param)
            self.addOperand(layer.top[i], new_op,
                            output_shape, TensorType.ACTIVATION)
            offset += slices[i]

    def convert_softmax_op(self, layer):
        assert(self.layerType(layer) == 'Softmax')
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

    def convert_split_op(self, layer):
        assert(self.layerType(layer) == 'Split')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        # simply bypass, register top and bottom blobs to the same tensor
        for top in layer.top:
            self.addOperand(top, op, input_shape, TensorType.ACTIVATION)

    def convert_tanh_op(self, layer):
        assert(self.layerType(layer) == 'TanH')
        op, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        new_op = self.CVI.add_tanh_op(
            layer.name, operands, output_shape)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_upsample_op(self, layer):
        assert(self.layerType(layer) == 'Upsample')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        assert(len(input_shape) == 4)
        scale = layer.upsample_param.scale
        assert(scale == 2)
        output_shape = [input_shape[0], input_shape[1],
                        scale * input_shape[2], scale * input_shape[3]]
        param = {
            'scale': scale
        }
        new_op = self.CVI.add_upsample_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def convert_yolo_detection_op(self, layer):
        assert(self.layerType(layer) == 'YoloDetection')
        _, input_shape, _ = self.getOperand(layer.bottom[0])
        operands = list()
        for bottom in layer.bottom:
            op, _, _ = self.getOperand(bottom)
            operands.append(op)
        p = layer.yolo_detection_param
        param = {
            'net_input_h': p.net_input_h,
            "net_input_w": p.net_input_w,
            "nms_threshold": p.nms_threshold,
            "obj_threshold": p.obj_threshold,
            "keep_topk": p.keep_topk,
        }
        output_shape = [1, 1, p.keep_topk, 6]
        new_op = self.CVI.add_yolo_detection_op(
            layer.name, operands, output_shape, **param)
        self.addOperand(layer.top[0], new_op,
                        output_shape, TensorType.ACTIVATION)

    def do_pre_scale(self, input_name, op_name, scale):
        if scale == 1.0:
            return
        op, input_shape, _ = self.getOperand(input_name)
        operands = list()
        operands.append(op)
        # weight scale
        scale_name = op_name + "_0"
        c = input_shape[1]
        scale_shape = [c]
        scale_data = np.array([scale for i in range(c)], dtype=float)
        self.addTensor(scale_name, scale_data, scale_shape)
        scale_op = self.CVI.add_load_file_op(scale_name, scale_shape)
        operands.append(scale_op)
        # weiht bias
        bias_name = op_name + "_1"
        bias_shape = [c]
        bias_data = np.array([0.0 for i in range(c)], dtype=float)
        self.addTensor(bias_name, bias_data, bias_shape)
        bias_op = self.CVI.add_load_file_op(bias_name, bias_shape)
        operands.append(bias_op)
        output_shape = input_shape
        new_op = self.CVI.add_scale_op(op_name, operands, output_shape)
        self.addOperand(input_name, new_op, output_shape,
                        TensorType.ACTIVATION)

    def do_pre_mean(self, input_name, op_name, value, order):
        if len(value) == 0:
            return
        mean_value = [float(i) for i in value.strip().split(',')]
        assert(len(mean_value) == 3)
        op, input_shape, _ = self.getOperand(input_name)
        operands = list()
        operands.append(op)
        # weight mean
        mean_name = op_name + "_0"
        mean_shape = [input_shape[1]]
        mean_data = np.array(mean_value, dtype=float)
        if order != None:
            tmp = list(mean_value)
            for i, j in enumerate(order):
                mean_data[i] = tmp[j]
        self.addTensor(mean_name, mean_data, mean_shape)
        mean_op = self.CVI.add_load_file_op(mean_name, mean_shape)
        operands.append(mean_op)
        # weight variance
        variance_name = op_name + "_1"
        variance_shape = [input_shape[1]]
        variance_data = np.array([1.0, 1.0, 1.0], dtype=float)
        self.addTensor(variance_name, variance_data, variance_shape)
        variance_op = self.CVI.add_load_file_op(variance_name, variance_shape)
        operands.append(variance_op)
        # weight scale
        scale_name = op_name + "_2"
        scale_shape = [1]
        scale_data = np.array([1.0], dtype=float)
        self.addTensor(scale_name, scale_data, scale_shape)
        scale_op = self.CVI.add_load_file_op(scale_name, scale_shape)
        operands.append(scale_op)
        param = {
            'variance_epsilon': 0.0
        }
        output_shape = input_shape
        new_op = self.CVI.add_batchnorm_op(
            op_name, operands, output_shape, **param)
        self.addOperand(input_name, new_op, output_shape,
                        TensorType.ACTIVATION)

    def do_pre_swap_channel(self, input_name, op_name, order):
        op, input_shape, _ = self.getOperand(input_name)
        operands = list()
        operands.append(op)
        param = {
            'channel_order': order
        }
        output_shape = input_shape
        new_op = self.CVI.add_swap_channel_op(
            op_name, operands, output_shape, **param)
        self.addOperand(input_name, new_op, output_shape,
                        TensorType.ACTIVATION)

    def do_preprocess(self, input_name):
        if self.preprocess == None:
            return
        order = None
        if "swap_channel" in self.preprocess and len(self.preprocess["swap_channel"]) != 0:
            order = [int(i)
                     for i in self.preprocess["swap_channel"].strip().split(',')]
            assert(len(order) == 3 or len(order) == 0)
            need_order = False
            for i, data in enumerate(order):
                if i != data:
                    need_order = True
                    break
            if need_order == False:
                order = None
        if "raw_scale" in self.preprocess:
            self.do_pre_scale(input_name, "pre_raw_scale",
                              self.preprocess['raw_scale']/255.0)
        if "mean" in self.preprocess:
            self.do_pre_mean(input_name, "pre_mean",
                             self.preprocess['mean'], order)
        if "scale" in self.preprocess:
            self.do_pre_scale(input_name, "pre_scale",
                              self.preprocess['scale'])
        if order != None:
            self.do_pre_swap_channel(input_name, "pre_swap_channel", order)

    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        def NoneAndRaise(layer):
            raise RuntimeError(
                "{} Op not support now".format(self.layerType(layer)))

        # add input op
        for idx, name in enumerate(self.inputs):
            input_shape = self.input_shapes[idx]
            input_op = self.CVI.add_input_op(name, idx)
            self.addOperand(name, input_op, input_shape, TensorType.ACTIVATION)
            # only first input do preprocess
            if idx == 0:
                self.do_preprocess(name)

        for layer in self.layers:
            is_test_phase = True
            if len(layer.include) != 0:
                # only test phase convert
                is_test_phase = False
                for include in layer.include:
                    if include.HasField('phase') and include.phase == 1:
                        is_test_phase = True
                        break
            if is_test_phase:
                self.caffeop_factory.get(
                    self.layerType(layer), lambda x: NoneAndRaise(x))(layer)

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
