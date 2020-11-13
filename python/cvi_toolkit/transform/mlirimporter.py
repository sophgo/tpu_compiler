from enum import Enum
import re
import pybind
import numpy as np
import sys

IS_PYTHON3 = sys.version_info > (3,)
from ..utils.log_setting import setup_logger

logger = setup_logger('root')

class TPU_MODE(Enum):
    INT8 = 'INT8'
    FP32 = 'FP32'
    BF16 = 'BF16'

class TPU_TensorType(Enum):
    INT8 = 'INT8'
    FP32 = 'FP32'
    BF16 = 'BF16'
    INT16 = 'INT16'
    INT32 = 'INT32'

class TPU_OpType(Enum):
    Weight_file = 'tpu.weight_file'
    Input  = 'tpu.input'
    Interp  = 'tpu.interp'
    Load_Weight = 'tpu.load_weight'

    Abs = 'tpu.abs'
    BatchNorm = 'tpu.batch_norm'
    BroadcastMul = 'tpu.broadcast_mul'
    BroadcastAdd = 'tpu.broadcast_add'
    Concat = 'tpu.concat'
    Conv2d = 'tpu.conv_2d'
    Crop = 'tpu.crop'
    Clip = 'tpu.clip'
    CustomOp = 'tpu.custom_op'
    DeConv2d = 'tpu.deconv_2d'
    DetectionOutput = 'tpu.detectionoutput'
    DummyData = 'tpu.dummy'
    Eltwise_Add = 'tpu.eltwise_add'
    Eltwise_Max = 'tpu.eltwise_max'
    Eltwise_Min = 'tpu.eltwise_min'
    Eltwise_Mul = 'tpu.eltwise_mul'
    Exp = 'tpu.exp'
    FullyConnected = 'tpu.fully_connected'
    FrcnDetection = 'tpu.frcn_detection'
    GRU = 'tpu.gru'
    LeakyRelu = 'tpu.leaky_relu'
    LrnOne = 'tpu.lrn_one'
    LrnTwo = 'tpu.lrn_two'
    LrnThree = 'tpu.lrn_three'
    Lrn = 'tpu.lrn'
    LSTM = 'tpu.lstm'
    Normalize = 'tpu.normalize'
    Mish = 'tpu.mish'
    Pad = 'tpu.pad'
    Permute = 'tpu.permute'
    PixelShuffle = 'tpu.pixelshuffle'
    PoolAvg2D = 'tpu.pool_avg_2d'
    PoolMax2D  = 'tpu.pool_max_2d'
    PoolMask = 'tpu.pool_mask'
    Power = 'tpu.power'
    Preprocess = 'tpu.preprocess'
    PriorBox = 'tpu.priorbox'
    PRelu = 'tpu.prelu'
    Proposal = 'tpu.proposal'
    Quant = 'tpu.quant'
    Reciprocal = 'tpu.reciprocal'
    Reshape = 'tpu.reshape'
    Relu = 'tpu.relu'
    Reorg = 'tpu.reorg'
    RetinaFaceDetection = 'tpu.retinaface_detection'
    ROIPooling = 'tpu.roi_pooling'
    Scale = 'tpu.scale'
    ShuffelChannel = 'tpu.shuffle_channel'
    Sigmoid = 'tpu.sigmoid'
    Slice = 'tpu.slice'
    Softmax = 'tpu.softmax'
    SwapChannel = 'tpu.swap_channel'
    Tanh = 'tpu.tanh'
    Tile = 'tpu.tile'
    Upsample = 'tpu.upsample'
    YoloDetection = 'tpu.yolo_detection'
    ReduceMean = 'tpu.reduce_mean'
    ReduceMax = 'tpu.reduce_max'
    MatMul = 'tpu.matmul'
    BroadcastSub = 'tpu.broadcast_sub'
    Square = 'tpu.square'
    QuadraticSum = 'tpu.quadratic_sum'


def checkKey(dict, key):
    if key not in dict:
        raise AttributeError("No {} attr, please check".format(key))

def checkType(obj, type):
    if not isinstance(obj, type):
        raise AttributeError('{} is not {}'.format(obj, type))

def checkAttrType(attr):
    if type(attr) == int:
        return 'int'
    elif type(attr) == float:
        return 'float'
    elif type(attr) == str:
        return 'str'
    elif type(attr) == bool:
        return 'bool'
    elif type(attr) == list:
        if type(attr) == int:
            return 'int_arr'
        elif type(attr) == float:
            return 'float_arr'
        elif type(attr) == str:
            return 'str_arr'
        elif type(attr) == bool:
            return 'bool_arr'
    raise AttributeError("unsupported attributes type")

class MLIRImporter(object):
    def __init__(self, inputs_shape, outputs_shape, input_type="FP32"):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert(isinstance(inputs_shape, list))
        assert(isinstance(outputs_shape, list))

        self.module = pybind.MLIRModule()
        self.input_shape_list = list()
        self.output_shape_list = list()
        for input in inputs_shape:
            assert(isinstance(input, list))
            self.input_shape_list.append(input)
        for output in outputs_shape:
            assert(isinstance(output, list))
            self.output_shape_list.append(output)

        self.boolType = self.module.make_type("i1")
        self.i8Type = self.module.make_type("i8")
        self.i32Type = self.module.make_type("i32")
        self.bf16Type = self.module.make_type("bf16")
        self.f32Type = self.module.make_type("f32")
        self.NoneType = self.module.make_none_type()
        self.indexType = self.module.make_index_type()
        self.func_ctx =None

        quant_param = {
            'is_asymmetric': self.module.boolAttr(False),
            'is_perchannel': self.module.boolAttr(False),
            'mode': self.module.stringAttr("NONE"),
            'param_type': self.module.stringAttr("NONE"),
            'threshold_max': self.module.floatAttr(0),
            'threshold_min': self.module.floatAttr(0),
            'zero_point': self.module.integerAttr(self.i32Type, 0),
        }
        self.quant_param = self.module.dictAttr(**quant_param)
        self.input_type = input_type
        self.declare_func(input_type=input_type)

    def __del__(self):
        logger.debug('Close mlir builder context')
        self.func_ctx.__exit__(None, None, None)

    def _create_int8_quant_attr(self, is_asymmetric=False, is_perchannel=False, mode=TPU_MODE.INT8.value,
                           param_type="NONE", threshold_max=0, threshold_min=0, zero_point=0):
        quant_param = {
            'is_asymmetric': self.module.boolAttr(is_asymmetric),
            'is_perchannel': self.module.boolAttr(is_perchannel),
            'mode': self.module.stringAttr(mode),
            'param_type': self.module.stringAttr(param_type),
            'threshold_max': self.module.floatAttr(threshold_max),
            'threshold_min': self.module.floatAttr(threshold_min),
            'zero_point': self.module.integerAttr(self.i32Type, zero_point)
        }
        return quant_param

    def check_int8_param(self, **kargs):
        checkKey(kargs, 'is_asymmetric')
        checkKey(kargs, 'is_perchannel')
        checkKey(kargs, 'param_type')
        checkKey(kargs, 'threshold_max')
        checkKey(kargs, 'threshold_min')
        checkKey(kargs, 'zero_point')

    def create_int8_quant_attr(self, **kargs):
        self.check_int8_param(**kargs)
        param = self._create_int8_quant_attr(
            is_asymmetric=kargs['is_asymmetric'],
            is_perchannel=kargs['is_perchannel'],
            param_type=kargs['param_type'],
            threshold_max=kargs['threshold_max'],
            threshold_min=kargs['threshold_min'],
            zero_point=kargs['zero_point']
        )

        return self.module.dictAttr(**param)

    def get_input_type(self, input_op):
        _type = str(input_op.type())
        _type = _type.split('x')[-1].split('>')[0]
        if _type == "f32":
            return self.f32Type
        elif _type == "i8":
            return self.i8Type
        else:
            raise RuntimeError("No support {}".format(_type))

    def buildOp(self, op_type, inputOperands, outputOperands, **kargs):
        """
            op_type: String
            inputOpreands: List[pybind.op]
            outputOpreands: List[pybind.op]
            kargs: Dict
        """
        return pybind.op(op_type, inputOperands, outputOperands, **kargs)

    def add_none_op(self):
        return pybind.op("tpu.none", [], [self.NoneType])

    def add_quant_reg(self, opreands):
        none = self.add_none_op()
        # We assigne 4 reg for quantization
        for _ in range(4):
            opreands.append(none)
        return opreands

    def add_input_op(self, name, index, **kargs):
        name = self.module.stringAttr(name)
        assert (index < len(self.func_args))

        if kargs:
            color_order = "bgr" if kargs['color_order'].tolist() == [0,1,2] else "rgb"

            preprocess_param = {
                'mean': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['mean']]),
                'std':  self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['std']]),
                'input_scale': self.module.floatAttr(kargs['scale']),
                'raw_scale': self.module.floatAttr(kargs['raw_scale']),
                'color_order': self.module.stringAttr(color_order)
            }
        else:
            # use default preprocess param
            preprocess_param = {
                'mean': self.module.arrayAttr([self.module.floatAttr(x) for x in [0,0,0]]),
                'std':  self.module.arrayAttr([self.module.floatAttr(x) for x in [1,1,1]]),
                'input_scale': self.module.floatAttr(1.0),
                'raw_scale': self.module.floatAttr(255.0),
                'color_order': self.module.stringAttr("bgr")
            }
        preprocess_param_attr = self.module.dictAttr(**preprocess_param)

        quant_param = {
            'is_asymmetric': self.module.boolAttr(False),
            'is_perchannel': self.module.boolAttr(False),
            'mode': self.module.stringAttr("NONE"),
            'param_type': self.module.stringAttr("NONE"),
            'threshold_max': self.module.floatAttr(0),
            'threshold_min': self.module.floatAttr(0),
            'zero_point': self.module.integerAttr(self.i32Type, 0),
        }
        if self.input_type == "UINT8":
            quant_param['mode'] = self.module.stringAttr("INT8")
        quant_param_attr = self.module.dictAttr(**quant_param)
        return pybind.op(TPU_OpType.Input.value, [self.func_args[index]], [self.tensor_inputs_type[index]],
                name=name, quant=quant_param_attr, preprocess=preprocess_param_attr)

    def add_weight_file_op(self, name):
        filename = self.module.stringAttr(name)
        # TODO: our mlir not support mem type now
        mem_ref = self.module.make_memref_type(self.f32Type, [10])
        self.weightop = self.buildOp(TPU_OpType.Weight_file.value, [], [mem_ref], filename=filename)

    def add_load_file_op(self, name, output_tensor_shape, tensor_type=TPU_TensorType.FP32, storage="NONE"):
        storage = self.module.stringAttr(storage)
        if tensor_type == TPU_TensorType.FP32:
            tensor_output_type = self.module.make_ranked_tensor_type(
                 self.f32Type, output_tensor_shape)
        elif tensor_type == TPU_TensorType.INT32:
            tensor_output_type = self.module.make_ranked_tensor_type(
                self.i32Type, output_tensor_shape)
        elif tensor_type == TPU_TensorType.INT8:
            tensor_output_type = self.module.make_ranked_tensor_type(
                self.i8Type, output_tensor_shape)
        else:
            raise RuntimeError("No support type {}".format(tensor_type))
        load_name = self.module.stringAttr(name)
        return self.buildOp(TPU_OpType.Load_Weight.value, [self.weightop], [tensor_output_type], name=load_name, storage=storage)

    def add_broadcast_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        broadcast_mul_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.BroadcastMul.value, inputOperands, [
            tensor_output_type], name=broadcast_mul_name, axis=axis_attr, quant=self.quant_param)

    def add_broadcast_add_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        broadcast_add_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.BroadcastAdd.value, inputOperands, [
            tensor_output_type], name=broadcast_add_name, axis=axis_attr, quant=self.quant_param)

    def add_interp_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        mlir_attrs = {}
        for key in kargs:
            checkType(kargs[key], int)
            mlir_attrs[key] = self.module.integerAttr(self.i32Type, kargs[key])

        name = self.module.stringAttr(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Interp.value, inputOperands, [
            tensor_output_type], name=name, **mlir_attrs,
            quant=self.quant_param)

    def add_abs_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        abs_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Abs.value, inputOperands, [
            tensor_output_type], name=abs_name, quant=self.quant_param)

    def add_batchnorm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'variance_epsilon')

        variance_epsilon = kargs['variance_epsilon']
        checkType(variance_epsilon, float)

        batchnorm_name = self.module.stringAttr(op_name)
        variance_epsilon_attr = self.module.floatAttr(variance_epsilon)

        none = self.add_none_op()
        for _ in range(5 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.BatchNorm.value, inputOperands, [
            tensor_output_type], name=batchnorm_name, variance_epsilon=variance_epsilon_attr)

    def add_clip_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) == 1)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        name = self.module.stringAttr(op_name)

        checkKey(kargs, 'min')
        checkKey(kargs, 'max')

        clip_min = kargs['min']
        clip_max = kargs['max']

        checkType(clip_min, float)
        checkType(clip_max, float)

        attr_dict = {
            'min': self.module.floatAttr(clip_min),
            'max': self.module.floatAttr(clip_max),
        }
        inputOperands = self.add_quant_reg(inputOperands)

        return self.buildOp(TPU_OpType.Clip.value, inputOperands, [
            tensor_output_type], name=name, quant=self.quant_param, **attr_dict)


    def add_concat_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'axis')
        concat_name = self.module.stringAttr(op_name)

        axis_attr = self.module.integerAttr(self.i32Type, kargs['axis'])
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Concat.value, inputOperands, [
            tensor_output_type], name=concat_name, axis=axis_attr, quant=self.quant_param)

    def add_conv_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):
        """
            inputOperands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'dilation_h')
        checkKey(kargs, 'dilation_w')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'padding')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'group')
        checkKey(kargs, 'is_dw')
        checkKey(kargs, 'with_bias')
        checkKey(kargs, 'do_relu')
        checkKey(kargs, 'ins')

        conv_name = self.module.stringAttr(op_name)
        conv_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'padding': self.module.stringAttr(kargs['padding']),
            'dilation_h': self.module.integerAttr(self.i32Type,  kargs['dilation_h']),
            'dilation_w': self.module.integerAttr(self.i32Type, kargs['dilation_w']),
            'padding_t': self.module.integerAttr(self.i32Type, kargs['padding_t']),
            'padding_b': self.module.integerAttr(self.i32Type, kargs['padding_b']),
            'padding_l': self.module.integerAttr(self.i32Type, kargs['padding_l']),
            'padding_r': self.module.integerAttr(self.i32Type, kargs['padding_r']),
            'group': self.module.integerAttr(self.i32Type, kargs['group']),
            'is_dw': self.module.boolAttr(kargs['is_dw']),
            'with_bias': self.module.boolAttr(kargs['with_bias']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
            'ins': self.module.arrayAttr(
                [self.module.integerAttr(self.i32Type, x) for x in kargs['ins']]),
            'pad_value': self.module.integerAttr(self.i32Type, pad_value),
          }

        dict_attr = self.module.dictAttr(**conv_param)
        none = self.add_none_op()

        if mode == TPU_MODE.INT8:
            if len(inputOperands) < 4:
                raise RuntimeError(
                    "{} input, need more than 4 input operands".format(len(inputOperands)))

            quant_param = self.create_int8_quant_attr(**kargs)

            # input, weight, (bias), rshift, multipiler
            rshift, multipiler = inputOperands[-2:]
            inputOperands = inputOperands[:-2]
            for _ in range(5 - len(inputOperands)):
                inputOperands.append(none)
            inputOperands.append(rshift)
            inputOperands.append(multipiler)

        elif mode == TPU_MODE.FP32:
            if len(inputOperands) < 2:
                raise RuntimeError(
                    "{} input, need more than 2 input operands".format(len(inputOperands)))
            for _ in range(7 - len(inputOperands)):
                inputOperands.append(none)
            quant_param = self.quant_param

        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        ## quant param
        return self.buildOp(TPU_OpType.Conv2d.value, inputOperands, [
                     tensor_output_type], name=conv_name, param=dict_attr, quant=quant_param)

    def add_custom_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            args:
                operation_name: string
                do_quant: bool
                tpu: bool
                threshold_overwrite: string, 'none', 'backward' or 'forward'
                param: dictionary
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'operation_name')
        checkKey(kargs, 'do_quant')
        checkKey(kargs, 'tpu')
        checkKey(kargs, 'threshold_overwrite')
        checkKey(kargs, 'param')
        checkType(kargs['param'], dict)
        if kargs['threshold_overwrite'] not in ['none', 'backward', 'forward']:
            raise AttributeError("invalid value of parameter threshold_overwrite: {}"
                  .format(kargs['threshold_overwrite']))

        name = self.module.stringAttr(op_name)
        operation_name = self.module.stringAttr(kargs['operation_name'])
        do_quant = self.module.boolAttr(kargs['do_quant'])
        tpu = self.module.boolAttr(kargs['tpu'])
        threshold_overwrite = self.module.stringAttr(kargs['threshold_overwrite'])

        op_param = {}
        for key, val in kargs['param'].items():
            attr_type = checkAttrType(val)
            if attr_type == 'int':
                op_param[key] = self.module.integerAttr(self.i32Type, val)
            elif attr_type == 'float':
                op_param[key] = self.module.floatAttr(val)
            elif attr_type == 'str':
                op_param[key] = self.module.stringAttr(val)
            elif attr_type == 'bool':
                op_param[key] = self.module.boolAttr(val)
            elif attr_type == 'int_arr':
                arr = [self.module.integerAttr(self.i32Type, x) for x in val]
                op_param[key] = self.module.arrayAttr(arr)
            elif attr_type == 'float_arr':
                arr = [self.module.floatAttr(x) for x in val]
                op_param[key] = self.module.arrayAttr(arr)
            elif attr_type == 'str_arr':
                arr = [self.module.stringAttr(x) for x in val]
                op_param[key] = self.module.arrayAttr(arr)
            elif attr_type == 'bool_arr':
                arr = [self.module.boolAttr(x) for x in val]
                op_param[key] = self.module.arrayAttr(arr)
        param = self.module.dictAttr(**op_param)

        return self.buildOp(TPU_OpType.CustomOp.value, inputOperands, [
            tensor_output_type], name=name, operation_name=operation_name, quant=self.quant_param,
            do_quant=do_quant, param=param, tpu=tpu, threshold_overwrite = threshold_overwrite)

    def add_crop_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            args:
                crop_offset: List[int, int, int, int]
                crop_shape : List[int, int, int, int]
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'crop_offset')
        checkKey(kargs, 'crop_shape')

        crop_offset = kargs['crop_offset']
        crop_shape = kargs['crop_shape']
        checkType(crop_offset, list)
        checkType(crop_shape , list)

        crop_name = self.module.stringAttr(op_name)
        crop_offset_attr = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in crop_offset])
        crop_shape_attr = self.module.arrayAttr(
            [self.module.integerAttr(self.i32Type, x) for x in crop_shape])

        return self.buildOp(TPU_OpType.Crop.value, inputOperands, [
            tensor_output_type], name=crop_name, crop_offset=crop_offset_attr, quant=self.quant_param, crop_shape=crop_shape_attr)

    def add_detection_output_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'num_classes')
        checkKey(kargs, 'share_location')
        checkKey(kargs, 'background_label_id')
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'top_k')
        checkKey(kargs, 'code_type')
        checkKey(kargs, 'keep_top_k')
        checkKey(kargs, 'confidence_threshold')
        name_attr = self.module.stringAttr(op_name)
        param = {
            'num_classes': self.module.integerAttr(self.i32Type, kargs['num_classes']),
            'share_location': self.module.boolAttr(kargs['share_location']),
            'background_label_id': self.module.integerAttr(self.i32Type, kargs['background_label_id']),
            'nms_threshold': self.module.floatAttr(kargs['nms_threshold']),
            'top_k': self.module.integerAttr(self.i32Type, kargs['top_k']),
            'code_type': self.module.stringAttr(kargs['code_type']),
            'keep_top_k': self.module.integerAttr(self.i32Type, kargs['keep_top_k']),
            'confidence_threshold': self.module.floatAttr(kargs['confidence_threshold']),
        }
        return self.buildOp(TPU_OpType.DetectionOutput.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_deconv_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):
        """
            inputOperands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'dilation_h')
        checkKey(kargs, 'dilation_w')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'padding')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'group')
        checkKey(kargs, 'is_dw')
        checkKey(kargs, 'with_bias')
        checkKey(kargs, 'do_relu')
        checkKey(kargs, 'ins')

        deconv_name = self.module.stringAttr(op_name)
        deconv_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'padding': self.module.stringAttr(kargs['padding']),
            'dilation_h': self.module.integerAttr(self.i32Type,  kargs['dilation_h']),
            'dilation_w': self.module.integerAttr(self.i32Type, kargs['dilation_w']),
            'padding_t': self.module.integerAttr(self.i32Type, kargs['padding_t']),
            'padding_b': self.module.integerAttr(self.i32Type, kargs['padding_b']),
            'padding_l': self.module.integerAttr(self.i32Type, kargs['padding_l']),
            'padding_r': self.module.integerAttr(self.i32Type, kargs['padding_r']),
            'group': self.module.integerAttr(self.i32Type, kargs['group']),
            'is_dw': self.module.boolAttr(kargs['is_dw']),
            'with_bias': self.module.boolAttr(kargs['with_bias']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
            'ins': self.module.arrayAttr(
                [self.module.integerAttr(self.i32Type, x) for x in kargs['ins']]),
            'pad_value': self.module.integerAttr(self.i32Type, pad_value),
        }

        dict_attr = self.module.dictAttr(**deconv_param)
        none = self.add_none_op()
        for _ in range(7 - len(inputOperands)):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.DeConv2d.value, inputOperands, [
            tensor_output_type], name=deconv_name, param=dict_attr, quant=self.quant_param)

    def add_dummydata_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        assert(len(inputOperands) == 0)
        name_attr = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.DummyData.value, inputOperands, [
            tensor_output_type], name=name_attr)

    def add_eltwise_add_op(self, op_name, inputOperands, output_tensor_shape,  mode=TPU_MODE.FP32, do_relu=False, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")
        do_relu = self.module.boolAttr(do_relu)
        eltwise_add = self.module.stringAttr(op_name)
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)

            # input, weight, (bias), rshift, multipiler
            rshift, multipiler = inputOperands[-2:]
            inputOperands = inputOperands[:-2]
            none = self.add_none_op()
            for _ in range(4 - len(inputOperands)):
                inputOperands.append(none)
            inputOperands.append(rshift)
            inputOperands.append(multipiler)

        elif mode == TPU_MODE.FP32:
            inputOperands = self.add_quant_reg(inputOperands)
            quant_param = self.quant_param
        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        return self.buildOp(TPU_OpType.Eltwise_Add.value, inputOperands, [
            tensor_output_type], name=eltwise_add, quant=quant_param, do_relu=do_relu)

    def add_eltwise_max_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_max = self.module.stringAttr(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Max.value, inputOperands, [
            tensor_output_type], name=eltwise_max, quant=self.quant_param)

    def add_eltwise_min_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_min = self.module.stringAttr(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Min.value, inputOperands, [
            tensor_output_type], name=eltwise_min, quant=self.quant_param)

    def add_eltwise_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")


        eltwise_mul = self.module.stringAttr(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Mul.value, inputOperands, [
            tensor_output_type], name=eltwise_mul, quant=self.quant_param)

    def add_exp_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        op_name = self.module.stringAttr(op_name)
        none = self.add_none_op()

        # We assigne 4 reg for lut quant table
        for _ in range(2):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.Exp.value, inputOperands, [
            tensor_output_type], name=op_name, quant=self.quant_param)

    def add_fully_connected_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        if len(inputOperands) == 2:
            none = self.add_none_op()
            inputOperands.append(none)
            # No bias
        if mode == TPU_MODE.INT8:
            assert(kargs['param_type'] == "RSHIFT_ONLY")
            quant_param = self.create_int8_quant_attr(**kargs)
            rshift_op = inputOperands[-1]
            inputOperands = inputOperands[:-1]
            none = self.add_none_op()

            for _ in range(5 - len(inputOperands)):
                inputOperands.append(none)
            inputOperands.append(rshift_op)
            inputOperands.append(none)

        else:
            quant_param = self.quant_param
            inputOperands = self.add_quant_reg(inputOperands)
        fully_connected_name = self.module.stringAttr(op_name)

        return self.buildOp(TPU_OpType.FullyConnected.value, inputOperands, [
            tensor_output_type], name=fully_connected_name, quant=quant_param)

    def add_frcn_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'class_num')
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'obj_threshold')
        checkKey(kargs, 'keep_topk')

        name_attr=self.module.stringAttr(op_name)
        param = {
            'class_num': self.module.integerAttr(self.i32Type, kargs['class_num']),
            'nms_threshold': self.module.floatAttr(kargs['nms_threshold']),
            'obj_threshold': self.module.floatAttr(kargs['obj_threshold']),
            'keep_topk': self.module.integerAttr(self.i32Type, kargs['keep_topk'])
        }
        return self.buildOp(TPU_OpType.FrcnDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_gru_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        if len(inputOperands) < 5:
            raise ArithmeticError("input operand must great than 5. x, w, r, b, initial_h")

        gru_param = {
            'linear_before_reset': self.module.boolAttr(kargs['linear_before_reset']),
            'bidirectional': self.module.boolAttr(kargs['bidirectional'])
        }

        gru_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        for _ in range(4):#add 4 redundant input
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.GRU.value, inputOperands, [
            tensor_output_type], name=gru_name, quant=self.quant_param, **gru_param)

    def add_leaky_relu_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)

        checkKey(kargs, 'negative_slope')

        leaky_relu_param = {
            'negative_slope': self.module.floatAttr(kargs['negative_slope'])
        }

        leaky_relu_name = self.module.stringAttr(op_name)

        none = self.add_none_op()
        # quant_pos_scale, quant_pos_zeropoint, quant_neg_scale, quant_neg_zeropoint
        # quant_pos_rshift, quant_pos_multiplier, quant_neg_rshift, quant_neg_multiplier
        if mode == TPU_MODE.INT8:
            if len(inputOperands) < 4:
                raise RuntimeError(
                    "{} input, need more than 4 input operands".format(len(inputOperands)))

            quant_param = self.create_int8_quant_attr(**kargs)

            # [input, quant_pos_rshift, quant_pos_multiplier, quant_neg_rshift, quant_neg_multiplier]
            activation_op = inputOperands[0]
            pos_rshift, pos_multipiler = inputOperands[1:3]
            neg_rshift, neg_multipiler = inputOperands[3:5]
            inputOperands = [activation_op, none, none,
                             none, none, pos_rshift, pos_multipiler, neg_rshift, neg_multipiler]

        elif mode == TPU_MODE.FP32:
            if len(inputOperands) < 1:
                raise RuntimeError(
                    "{} input, need more than 1 input operands".format(len(inputOperands)))
            for _ in range( 9 - len(inputOperands)):
                inputOperands.append(none)
            quant_param = self.quant_param

        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        return self.buildOp(TPU_OpType.LeakyRelu.value, inputOperands, [
            tensor_output_type], name=leaky_relu_name, quant=quant_param, **leaky_relu_param)

    def add_lrn_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(self.f32Type, output_tensor_shape)

        checkKey(kargs, 'alpha')
        checkKey(kargs, 'beta')
        checkKey(kargs, 'bias')
        checkKey(kargs, 'size')

        lrn_param = {
            'alpha': self.module.floatAttr(kargs['alpha']),
            'beta': self.module.floatAttr(kargs['beta']),
            'k': self.module.floatAttr(kargs['bias']),
            'local_size': self.module.integerAttr(self.i32Type, kargs['size']),
        }

        lrn_name_1 = self.module.stringAttr("{}_one".format(op_name))
        lrn_name_2 = self.module.stringAttr("{}_two".format(op_name))
        lrn_name_3 = self.module.stringAttr("{}_three".format(op_name))
        lrn_name_main = self.module.stringAttr("{}".format(op_name))

        input_op = inputOperands[0]
        # lrn one
        lrn_one_op = self.buildOp(TPU_OpType.LrnOne.value, inputOperands, [
            tensor_output_type], name=lrn_name_1, quant=self.quant_param, **lrn_param)
        # lrn two
        operands2 = list()
        operands2.append(lrn_one_op)
        lrn_two_op = self.buildOp(TPU_OpType.LrnTwo.value, operands2, [
            tensor_output_type], name=lrn_name_2, quant=self.quant_param, **lrn_param)
        # lrn three
        operands3 = list()
        operands3.append(lrn_two_op)
        lrn_three_op = self.buildOp(TPU_OpType.LrnThree.value, operands3, [
            tensor_output_type], name=lrn_name_3, quant=self.quant_param, **lrn_param)
        # lrn
        none = self.add_none_op()
        operands = list()
        operands.append(input_op)
        operands.append(none)
        operands.append(none)
        operands.append(lrn_three_op)
        lrn_param['sum_rshift'] = self.module.integerAttr(self.i32Type, 0)
        lrn_param['lrn_rshift'] = self.module.integerAttr(self.i32Type, 0)
        lrn_param['quant_data0'] = self.module.integerAttr(self.i32Type, 0)
        lrn_param['quant_data1'] = self.module.integerAttr(self.i32Type, 0)
        lrn_param['norm_region'] = self.module.integerAttr(self.i32Type, 0)
        return self.buildOp(TPU_OpType.Lrn.value, operands, [
            tensor_output_type], name=lrn_name_main, quant=self.quant_param, **lrn_param)

    def add_lstm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        if len(inputOperands) < 6:
            raise ArithmeticError("input operand must great than 6. x, w, r, b, initial_h, initial_c")

        lstm_param = {
            'bidirectional': self.module.boolAttr(kargs['bidirectional'])
        }

        lstm_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        for _ in range(4):#add 4 redundant input
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.LSTM.value, inputOperands, [
            tensor_output_type], name=lstm_name, quant=self.quant_param, **lstm_param)

    def add_normalize_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'across_spatial')
        checkKey(kargs, 'channel_shared')
        name_attr = self.module.stringAttr(op_name)
        param = {
            'across_spatial': self.module.boolAttr(kargs['across_spatial']),
            'channel_shared': self.module.boolAttr(kargs['channel_shared']),
        }
        return self.buildOp(TPU_OpType.Normalize.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_mish_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        mish_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for mish quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Mish.value, inputOperands, [
            tensor_output_type], name=mish_name, quant=self.quant_param)

    def add_pad_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, ** kargs):
        """
            args:
                pads : List[int, int, int, int]
                const_val : int
        """

        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'pads')
        checkKey(kargs, 'const_val')

        pads = kargs['pads']
        const_val = kargs['const_val']
        checkType(pads, list)

        pad_name = self.module.stringAttr(op_name)
        pads_attr = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in pads])
        const_val_attr = self.module.floatAttr(const_val)
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)
        elif mode == TPU_MODE.FP32:
            quant_param = self.quant_param
        else:
            raise RuntimeError("No support quant mode {}".format(mode))

        return self.buildOp(TPU_OpType.Pad.value, inputOperands, [
            tensor_output_type], name=pad_name, quant=quant_param, pads=pads_attr, const_val=const_val_attr)

    def add_pool_mask_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'scale')
        pool_mask_name = self.module.stringAttr(op_name)
        pool_mask_param = {
            'scale': self.module.integerAttr(self.i32Type, kargs['scale'])
        }
        return self.buildOp(TPU_OpType.PoolMask.value, inputOperands, [
            tensor_output_type], name=pool_mask_name, **pool_mask_param)

    def add_pool_avg_2d_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'kernel_h')
        checkKey(kargs, 'kernel_w')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'do_relu')
        checkKey(kargs, 'count_include_pad')

        pool_avg_2d_name = self.module.stringAttr(op_name)
        pool_avg_2d_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'kernel_h': self.module.integerAttr(self.i32Type, kargs['kernel_h']),
            'kernel_w': self.module.integerAttr(self.i32Type, kargs['kernel_w']),
            'padding_b': self.module.integerAttr(self.i32Type, kargs['padding_b']),
            'padding_l': self.module.integerAttr(self.i32Type, kargs['padding_l']),
            'padding_r': self.module.integerAttr(self.i32Type, kargs['padding_r']),
            'padding_t': self.module.integerAttr(self.i32Type, kargs['padding_t']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
            'count_include_pad': self.module.boolAttr(kargs['count_include_pad']),
        }
        dict_attr = self.module.dictAttr(**pool_avg_2d_param)
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)

            # input, weight, (bias), rshift, multipiler
            rshift, multipiler = inputOperands[-2:]
            inputOperands = inputOperands[:-2]
            none = self.add_none_op()
            for _ in range(3 - len(inputOperands)):
                inputOperands.append(none)
            inputOperands.append(rshift)
            inputOperands.append(multipiler)
        elif mode == TPU_MODE.FP32:
            inputOperands = self.add_quant_reg(inputOperands)
            quant_param = self.quant_param
        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        return self.buildOp(TPU_OpType.PoolAvg2D.value, inputOperands, [
            tensor_output_type], name=pool_avg_2d_name, param=dict_attr, quant=quant_param)

    def add_pool_max_2d_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):

        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'kernel_h')
        checkKey(kargs, 'kernel_w')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'do_relu')

        pool_max_2d_name = self.module.stringAttr(op_name)
        pool_max_2d_param = {
            'stride_h': self.module.integerAttr(self.i32Type, kargs['stride_h']),
            'stride_w': self.module.integerAttr(self.i32Type, kargs['stride_w']),
            'kernel_h': self.module.integerAttr(self.i32Type, kargs['kernel_h']),
            'kernel_w': self.module.integerAttr(self.i32Type, kargs['kernel_w']),
            'padding_b': self.module.integerAttr(self.i32Type, kargs['padding_b']),
            'padding_l': self.module.integerAttr(self.i32Type, kargs['padding_l']),
            'padding_r': self.module.integerAttr(self.i32Type, kargs['padding_r']),
            'padding_t': self.module.integerAttr(self.i32Type, kargs['padding_t']),
            'do_relu': self.module.boolAttr(kargs['do_relu']),
            'count_include_pad': self.module.boolAttr(False), # max pool has no count_include_pad method
        }
        dict_attr = self.module.dictAttr(**pool_max_2d_param)
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)
        elif mode == TPU_MODE.FP32:
            quant_param = self.quant_param
        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")


        return self.buildOp(TPU_OpType.PoolMax2D.value, inputOperands, [
            tensor_output_type], name=pool_max_2d_name, param=dict_attr, quant=quant_param)

    def add_power_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'power')
        checkKey(kargs, 'scale')
        checkKey(kargs, 'shift')

        name_attr = self.module.stringAttr(op_name)
        param = {
            'power': self.module.floatAttr(kargs['power']),
            'scale': self.module.floatAttr(kargs['scale']),
            'shift': self.module.floatAttr(kargs['shift']),
        }
        return self.buildOp(TPU_OpType.Power.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_priorbox_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'min_size')
        checkKey(kargs, 'max_size')
        checkKey(kargs, 'aspect_ratios')
        checkKey(kargs, 'variance')
        checkKey(kargs, 'clip')
        checkKey(kargs, 'step_h')
        checkKey(kargs, 'step_w')
        checkKey(kargs, 'img_h')
        checkKey(kargs, 'img_w')
        checkKey(kargs, 'offset')
        checkKey(kargs, 'num_priors')
        checkKey(kargs, 'use_default_aspect_ratio')

        name_attr = self.module.stringAttr(op_name)
        param = {
            'min_size': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['min_size']]),
            'max_size': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['max_size']]),
            'aspect_ratios': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['aspect_ratios']]),
            'variance': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['variance']]),
            'clip': self.module.boolAttr(kargs['clip']),
            'step_h': self.module.floatAttr(kargs['step_h']),
            'step_w': self.module.floatAttr(kargs['step_w']),
            'img_h': self.module.integerAttr(self.i32Type, kargs['img_h']),
            'img_w': self.module.integerAttr(self.i32Type, kargs['img_w']),
            'offset': self.module.floatAttr(kargs['offset']),
            'num_priors': self.module.integerAttr(self.i32Type, kargs['num_priors']),
            'use_default_aspect_ratio': self.module.boolAttr(kargs['use_default_aspect_ratio']),
        }
        return self.buildOp(TPU_OpType.PriorBox.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_permute_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)
        checkKey(kargs, 'order0')
        checkKey(kargs, 'order1')
        checkKey(kargs, 'order2')
        checkKey(kargs, 'order3')

        permute_name = self.module.stringAttr(op_name)
        attr_dict = {
            'order0': self.module.integerAttr(self.i32Type, kargs['order0']),
            'order1': self.module.integerAttr(self.i32Type, kargs['order1']),
            'order2': self.module.integerAttr(self.i32Type, kargs['order2']),
            'order3': self.module.integerAttr(self.i32Type, kargs['order3']),
        }
        return self.buildOp(TPU_OpType.Permute.value, inputOperands, [
            tensor_output_type], name=permute_name, quant=self.quant_param, **attr_dict)

    def add_pixelshuffle_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
          self.f32Type, output_tensor_shape)

        checkKey(kargs, 'upscale_factor')
        checkKey(kargs, 'mode')

        pixelshuffle_name = self.module.stringAttr(op_name)
        attr_dict = {
            'upscale_factor': self.module.integerAttr(self.i32Type, kargs['upscale_factor']),
            'mode': self.module.stringAttr(kargs['mode'])
        }
        return self.buildOp(TPU_OpType.PixelShuffle.value, inputOperands, [
            tensor_output_type], name=pixelshuffle_name, quant=self.quant_param, **attr_dict)


    def add_preprocess_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        # preprocess not follow input type
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'mean')
        checkKey(kargs, 'std')
        checkKey(kargs, 'scale')
        checkKey(kargs, 'raw_scale')
        checkKey(kargs, 'color_order')
        checkKey(kargs, 'transpose_order')
        checkKey(kargs, 'crop_offset')
        checkKey(kargs, 'pads')
        checkKey(kargs, 'pad_const_val')

        preprocess_name = self.module.stringAttr(op_name)


        attrs = {
            'mean': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['mean']]),
            'std': self.module.arrayAttr([self.module.floatAttr(x) for x in kargs['std']]),
            'scale': self.module.floatAttr(kargs['scale']),
            'raw_scale': self.module.floatAttr(kargs['raw_scale']),
            'color_order': self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['color_order']]),
            'transpose_order': self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['transpose_order']]),
            'crop_offset': self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['crop_offset']]),
            'pads': self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['pads']]),
            'pad_const_val': self.module.integerAttr(self.i32Type, 0),
        }

        return self.buildOp(TPU_OpType.Preprocess.value, inputOperands, [
            tensor_output_type], name=preprocess_name, quant=self.quant_param, **attrs)

    def add_prelu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
        self.f32Type, output_tensor_shape)

        prelu_name = self.module.stringAttr(op_name)

        none = self.add_none_op()
        # quant_pos_scale, quant_pos_zeropoint, quant_neg_scale, quant_neg_zeropoint
        # quant_pos_rshift, quant_pos_multiplier, quant_neg_rshift, quant_neg_multiplier
        for _ in range( 10 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.PRelu.value, inputOperands, [
            tensor_output_type], name=prelu_name, quant=self.quant_param)


    def add_proposal_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'net_input_h')
        checkKey(kargs, 'net_input_w')
        checkKey(kargs, 'feat_stride')
        checkKey(kargs, 'anchor_base_size')
        checkKey(kargs, 'rpn_obj_threshold')
        checkKey(kargs, 'rpn_nms_threshold')
        checkKey(kargs, 'rpn_nms_post_top_n')

        proposal_name = self.module.stringAttr(op_name)
        attr_dict = {
            'net_input_h': self.module.integerAttr(self.i32Type, kargs['net_input_h']),
            'net_input_w': self.module.integerAttr(self.i32Type, kargs['net_input_w']),
            'feat_stride': self.module.integerAttr(self.i32Type, kargs['feat_stride']),
            'anchor_base_size': self.module.integerAttr(self.i32Type, kargs['anchor_base_size']),
            'rpn_obj_threshold': self.module.floatAttr(kargs['rpn_obj_threshold']),
            'rpn_nms_threshold': self.module.floatAttr(kargs['rpn_nms_threshold']),
            'rpn_nms_post_top_n': self.module.integerAttr(self.i32Type, kargs['rpn_nms_post_top_n']),
        }
        return self.buildOp(TPU_OpType.Proposal.value, inputOperands, [
            tensor_output_type], name=proposal_name, quant=self.quant_param, **attr_dict)

    def add_quant_op(self, op_name, inputOperands, output_tensor_shape, from_type, to_type, zero_point=0, **kargs):
        if to_type == "NONE":
            tensor_output_type = self.module.make_ranked_tensor_type(
                self.f32Type, output_tensor_shape)
        elif to_type == "INT8" or to_type == "UINT8":
            tensor_output_type = self.module.make_ranked_tensor_type(
                self.i8Type, output_tensor_shape)
        else:
            raise RuntimeError("No support {} to_type".format(to_type))

        checkKey(kargs, 'threshold')
        quant_name = self.module.stringAttr(op_name)
        attr_dict = {
            'from': self.module.stringAttr(from_type),
            'to': self.module.stringAttr(to_type),
            'threshold': self.module.floatAttr(kargs['threshold']),
            'zero_point': self.module.integerAttr(self.i32Type, zero_point),
        }
        return self.buildOp(TPU_OpType.Quant.value, inputOperands, [
            tensor_output_type], name=quant_name, **attr_dict)


    def add_reciprocal_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        reciprocal_name = self.module.stringAttr(op_name)

        # table and table_mantissa all are none
        none = self.add_none_op()
        for _ in range( 3 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.Reciprocal.value, inputOperands, [
            tensor_output_type], name=reciprocal_name, quant=self.quant_param)

    def add_relu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        relu_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Relu.value, inputOperands, [
            tensor_output_type], name=relu_name, quant=self.quant_param)

    def add_reorg_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'stride')
        name_attr = self.module.stringAttr(op_name)

        param = {
            'stride': self.module.integerAttr(self.i32Type, kargs['stride']),
        }
        return self.buildOp(TPU_OpType.Reorg.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_retinaface_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'confidence_threshold')
        checkKey(kargs, 'keep_topk')
        name_attr = self.module.stringAttr(op_name)
        param = {
            'nms_threshold':self.module.floatAttr(kargs['nms_threshold']),
            'confidence_threshold':self.module.floatAttr(kargs['confidence_threshold']),
            'keep_topk':self.module.integerAttr(self.i32Type, kargs['keep_topk']),
        }
        return self.buildOp(TPU_OpType.RetinaFaceDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_reshape_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        reshape_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Reshape.value, inputOperands, [
            tensor_output_type], name=reshape_name)

    def add_roipooling_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'pooled_h')
        checkKey(kargs, 'pooled_w')
        checkKey(kargs, 'spatial_scale')

        roipooling_name = self.module.stringAttr(op_name)
        attr_dict = {
            'pooled_h': self.module.integerAttr(self.i32Type, kargs['pooled_h']),
            'pooled_w': self.module.integerAttr(self.i32Type, kargs['pooled_w']),
            'spatial_scale': self.module.floatAttr(kargs['spatial_scale'])
        }
        return self.buildOp(TPU_OpType.ROIPooling.value, inputOperands, [
            tensor_output_type], name=roipooling_name, quant=self.quant_param, **attr_dict)

    def add_scale_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        scale_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Scale.value, inputOperands, [
            tensor_output_type], name=scale_name)

    def add_shufflechannel_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'group')
        attr_dict = {
            'group': self.module.integerAttr(self.i32Type, kargs['group']),
        }
        sc_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.ShuffelChannel.value, inputOperands, [
            tensor_output_type], name=sc_name, quant=self.quant_param, **attr_dict)

    def add_sigmoid_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        sigmoid_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for sigmoid quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Sigmoid.value, inputOperands, [
            tensor_output_type], name=sigmoid_name, quant=self.quant_param)

    def add_slice_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        attr_dict = {
            'axis': self.module.integerAttr(self.i32Type, kargs['axis']),
            'offset': self.module.integerAttr(self.i32Type, kargs['offset']),
        }

        slice_name = self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Slice.value, inputOperands, [
            tensor_output_type], name=slice_name, quant=self.quant_param, **attr_dict)

    def add_softmax_op(self, op_name, inputOperands, output_tensor_shape, cpu_mode=False,**kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        softmax_name = self.module.stringAttr(op_name)
        softmax_param = {
            'axis': self.module.integerAttr(self.i32Type, kargs['axis'])
        }
        none = self.add_none_op()
        if cpu_mode:
            op_name = TPU_OpType.Softmax.value + "_cpu"
        else:
            op_name = TPU_OpType.Softmax.value
            for _ in range(4):#add 4 redundant input
                inputOperands.append(none)
        return self.buildOp(op_name, inputOperands, [
            tensor_output_type], name=softmax_name, quant=self.quant_param, **softmax_param)

    def add_swap_channel_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        name_attr = self.module.stringAttr(op_name)
        checkKey(kargs, 'channel_order')
        order = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['channel_order']])
        param = {
            'channel_order': order,
        }
        return self.buildOp(TPU_OpType.SwapChannel.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_tanh_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        tanh_name = self.module.stringAttr(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for tanh quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Tanh.value, inputOperands, [
            tensor_output_type], name=tanh_name, quant=self.quant_param)

    def add_tile_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)

        checkKey(kargs, 'axis')
        checkKey(kargs, 'tiles')

        tile_name = self.module.stringAttr(op_name)
        resp = [1,1,1,1]
        resp[kargs['axis']] = kargs['tiles']
        resp = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in resp])
        tile_param = {
            'resp': resp
        }

        none = self.add_none_op()
        for _ in range(4):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.Tile.value, inputOperands, [
            tensor_output_type], name=tile_name, quant=self.quant_param, **tile_param)

    def add_upsample_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'scale_h')
        checkKey(kargs, 'scale_w')

        upsample_name = self.module.stringAttr(op_name)
        upsample_param = {
            'scale_h': self.module.integerAttr(self.i32Type, kargs['scale_h']),
            'scale_w': self.module.integerAttr(self.i32Type, kargs['scale_w'])
        }
        if len(inputOperands) < 2:
            none = self.add_none_op()
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Upsample.value, inputOperands, [
            tensor_output_type], name=upsample_name, quant=self.quant_param, **upsample_param)

    def add_reduce_mean_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'axes')

        reduce_name = self.module.stringAttr(op_name)
        axes = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['axes']])
        reduce_param = {
            'axes': axes
        }
        none = self.add_none_op()
        for _ in range( 5 - len(inputOperands)):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.ReduceMean.value, inputOperands, [
                tensor_output_type], name=reduce_name, quant=self.quant_param, **reduce_param)

    def add_reduce_max_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = self.module.make_ranked_tensor_type(
            self.get_input_type(inputOperands[0]), output_tensor_shape)
        checkKey(kargs, 'axes')

        reduce_name = self.module.stringAttr(op_name)
        axes = self.module.arrayAttr([self.module.integerAttr(self.i32Type, x) for x in kargs['axes']])
        reduce_param = {
            'axes': axes
        }

        none = self.add_none_op()
        for _ in range( 5 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.ReduceMax.value, inputOperands, [
                tensor_output_type], name=reduce_name, quant=self.quant_param, **reduce_param)

    def add_yolo_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type=self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        checkKey(kargs, 'net_input_h')
        checkKey(kargs, 'net_input_w')
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'obj_threshold')
        checkKey(kargs, 'keep_topk')
        checkKey(kargs, 'spp_net')
        checkKey(kargs, 'tiny')
        checkKey(kargs, 'yolo_v4')
        checkKey(kargs, 'class_num')
        checkKey(kargs, 'anchors')

        name_attr=self.module.stringAttr(op_name)
        param = {
            'net_input_h': self.module.integerAttr(self.i32Type, kargs['net_input_h']),
            'net_input_w': self.module.integerAttr(self.i32Type, kargs['net_input_w']),
            'nms_threshold': self.module.floatAttr(kargs['nms_threshold']),
            'obj_threshold': self.module.floatAttr(kargs['obj_threshold']),
            'keep_topk': self.module.integerAttr(self.i32Type, kargs['keep_topk']),
            'spp_net': self.module.boolAttr(kargs['spp_net']),
            'tiny': self.module.boolAttr(kargs['tiny']),
            'yolo_v4': self.module.boolAttr(kargs['yolo_v4']),
            'class_num': self.module.integerAttr(self.i32Type, kargs['class_num']),
            'anchors': self.module.stringAttr(kargs['anchors'])
        }
        return self.buildOp(TPU_OpType.YoloDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_matmul_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type=self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        name_attr=self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.MatMul.value, inputOperands, [tensor_output_type],
            name=name_attr, quant=self.quant_param)

    def add_square_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type=self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        name_attr=self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.Square.value, inputOperands, [tensor_output_type],
            name=name_attr, quant=self.quant_param)

    def add_quadratic_sum_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type=self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        name_attr=self.module.stringAttr(op_name)
        param = {}
        if 'high_precision' in kargs:
            param['high_precision'] = self.module.boolAttr(kargs['high_precision'])
        if 'axis' in kargs:
            param['axis'] = self.module.integerAttr(self.i32Type, kargs['axis'])
        return self.buildOp(TPU_OpType.QuadraticSum.value, inputOperands, [tensor_output_type],
            name=name_attr, quant=self.quant_param, **param)

    def add_broadcast_sub_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type=self.module.make_ranked_tensor_type(
            self.f32Type, output_tensor_shape)
        none = self.add_none_op()
        for _ in range(6 - len(inputOperands)):
            inputOperands.append(none)
        name_attr=self.module.stringAttr(op_name)
        return self.buildOp(TPU_OpType.BroadcastSub.value, inputOperands, [tensor_output_type],
            name=name_attr, quant=self.quant_param)

    def add_return_op(self, Operands):
        return pybind.ret(Operands)

    def print_module(self):
        mlir_format = str(self.module)
        lines = mlir_format.splitlines()

        new_strings = list()
        for i in lines:
            filter = r"\W*(std\.return)\W*"
            ret = re.match(filter, i)
            if ret:
                reg_filter = "%[0-9]+"
                regs = re.findall(reg_filter, i)
                shape_filter = r"\<[0-9A-Za-z]+\>"
                shapes = re.findall(shape_filter, i)
                if len(regs) != len(shapes): raise RuntimeError("{} is error format, regs v.s shapes number not match.".format(i))
                regstr = str()
                shapestr = str()
                for idx, (r, s) in enumerate(zip(regs, shapes)):
                    if idx != 0:
                        regstr += ", "
                        shapestr += ", "
                    regstr += r
                    shapestr += "tensor{}".format(s)

                new_line = "    return {} : {}".format(regstr, shapestr)
                new_strings.append(new_line)
            else:
                new_strings.append(i)
        ret = '\n'.join(new_strings)

        return ret

    def declare_func(self, input_type:str="FP32"):
        self.tensor_inputs_type = list()
        if input_type == "FP32":
            for input_shape in self.input_shape_list:
                self.tensor_inputs_type.append(self.module.make_ranked_tensor_type(
                    self.f32Type, input_shape))
        elif input_type == "UINT8":
            for input_shape in self.input_shape_list:
                self.tensor_inputs_type.append(self.module.make_ranked_tensor_type(
                    self.i8Type, input_shape))
        self.tensor_outputs_type = list()
        for output_shape in self.output_shape_list:
            self.tensor_outputs_type.append(self.module.make_ranked_tensor_type(
                self.f32Type, output_shape))

        self.func_ctx = self.module.function_context("tpu_func", self.tensor_inputs_type,
                                                     self.tensor_outputs_type)
        logger.debug('Open mlir builder context')
        fun = self.func_ctx.__enter__()
        self.func_args = list()
        for i in range(len(self.input_shape_list)):
            self.func_args.append(fun.arg(i))
