from ..utils.log_setting import setup_logger
from enum import Enum
import re
from mlir.ir import *
import numpy as np
import sys

IS_PYTHON3 = sys.version_info > (3,)

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
    Input = 'tpu.input'
    Interp = 'tpu.interp'
    Load_Weight = 'tpu.load_weight'

    Abs = 'tpu.abs'
    ArgMax = 'tpu.argmax'
    BatchNorm = 'tpu.batch_norm'
    BroadcastMul = 'tpu.broadcast_mul'
    BroadcastAdd = 'tpu.broadcast_add'
    Concat = 'tpu.concat'
    Conv2d = 'tpu.conv_2d'
    Conv3d = 'tpu.conv_3d'
    Crop = 'tpu.crop'
    Csc = 'tpu.csc'
    Clip = 'tpu.clip'
    CustomOp = 'tpu.custom_op'
    DeConv2d = 'tpu.deconv_2d'
    DetectionOutput = 'tpu.detectionoutput'
    DummyData = 'tpu.dummy'
    Eltwise_Add = 'tpu.eltwise_add'
    Eltwise_Max = 'tpu.eltwise_max'
    Eltwise_Min = 'tpu.eltwise_min'
    Eltwise_Mul = 'tpu.eltwise_mul'
    Equal = 'tpu.equal'
    Elu = 'tpu.elu'
    Exp = 'tpu.exp'
    Embedding = 'tpu.embedding'
    FullyConnected = 'tpu.fully_connected'
    FrcnDetection = 'tpu.frcn_detection'
    GRU = 'tpu.gru'
    InstanceNorm = 'tpu.instance_norm'
    LeakyRelu = 'tpu.leaky_relu'
    LrnOne = 'tpu.lrn_one'
    LrnTwo = 'tpu.lrn_two'
    LrnThree = 'tpu.lrn_three'
    Lrn = 'tpu.lrn'
    LSTM = 'tpu.lstm'
    LayerNorm = "tpu.layer_norm"
    Normalize = 'tpu.normalize'
    Mish = 'tpu.mish'
    Pad = 'tpu.pad'
    Permute = 'tpu.permute'
    PixelShuffle = 'tpu.pixelshuffle'
    PoolAvg2D = 'tpu.pool_avg_2d'
    PoolMax2D = 'tpu.pool_max_2d'
    PoolMask = 'tpu.pool_mask'
    Power = 'tpu.power'
    Preprocess = 'tpu.preprocess'
    PriorBox = 'tpu.priorbox'
    PRelu = 'tpu.prelu'
    Proposal = 'tpu.proposal'
    Quant = 'tpu.quant'
    ReQuant = 'tpu.requant'
    Reciprocal = 'tpu.reciprocal'
    Reshape = 'tpu.reshape'
    Relu = 'tpu.relu'
    Reorg = 'tpu.reorg'
    RetinaFaceDetection = 'tpu.retinaface_detection'
    Reverse = 'tpu.reverse'
    ReflectionPad = 'tpu.reflection_pad'
    ROIPooling = 'tpu.roi_pooling'
    Scale = 'tpu.scale'
    ShuffelChannel = 'tpu.shuffle_channel'
    Sigmoid = 'tpu.sigmoid'
    Slice = 'tpu.slice'
    SoftPlus = 'tpu.softplus'
    Softmax = 'tpu.softmax'
    Sqrt = 'tpu.sqrt'
    SwapChannel = 'tpu.swap_channel'
    Tanh = 'tpu.tanh'
    Tile = 'tpu.tile'
    Upsample = 'tpu.upsample'
    YoloDetection = 'tpu.yolo_detection'
    ReduceL2 = 'tpu.reduce_l2'
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
    def __init__(self, inputs_shape, outputs_shape, input_type="FP32", output_weight_file=None):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List, put module output shape. ex: [1, 1000]
        """
        assert(isinstance(inputs_shape, list))
        assert(isinstance(outputs_shape, list))
        if output_weight_file is None:
            raise RuntimeError("output weight file value is None")
        self.output_weight_file = output_weight_file
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.input_shape_list = list()
        self.output_shape_list = list()

        for input in inputs_shape:
            assert(isinstance(input, list))
            self.input_shape_list.append(input)
        for output in outputs_shape:
            assert(isinstance(output, list))
            self.output_shape_list.append(output)
            self.u8Type = IntegerType.get_unsigned(8)
            self.i8Type = IntegerType.get_signless(8)
            self.i32Type = IntegerType.get_signless(32)
            self.f32Type = F32Type.get()

            quant_param = {
                'is_asymmetric': BoolAttr.get(False),
                'is_perchannel': BoolAttr.get(False),
                'mode': StringAttr.get("NONE"),
                'param_type': StringAttr.get("NONE"),
                'threshold_max': FloatAttr.get_f32(0),
                'threshold_min': FloatAttr.get_f32(0),
                'zero_point': IntegerAttr.get(self.i32Type, 0),
            }
            self.quant_param = DictAttr.get(quant_param)
            self.input_type = input_type

        self.declare_func(input_type=input_type)

    def __del__(self):
        # logger.debug('Close mlir builder context')
        self.loc.__exit__(None, None, None)
        self.ctx.__exit__(None, None, None)

    def _create_int8_quant_attr(self, is_asymmetric=False, is_perchannel=False, mode=TPU_MODE.INT8.value,
                                param_type="NONE", threshold_max=0, threshold_min=0, zero_point=0):
        quant_param = {
            'is_asymmetric': BoolAttr.get(is_asymmetric),
            'is_perchannel': BoolAttr.get(is_perchannel),
            'mode': StringAttr.get(mode),
            'param_type':  StringAttr.get(param_type),
            'threshold_max': FloatAttr.get_f32(threshold_max),
            'threshold_min': FloatAttr.get_f32(threshold_min),
            'zero_point': IntegerAttr.get(self.i32Type, zero_point)
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

        return DictAttr.get(param)

    def get_input_type(self, input_op):
        _type = str(input_op.type)
        _type = _type.split('x')[-1].split('>')[0]
        if _type == "f32":
            return self.f32Type
        elif _type == "i8":
            return self.i8Type
        elif _type == "ui8":
            return self.u8Type
        else:
            raise RuntimeError("No support {}".format(_type))

    def buildOp(self, op_type, inputOperands, output_types, **kargs):
        """
            op_type: String
            inputOpreands: List[pybind.op]
            output_types: List[pybind.op]
            kargs: Dict
        """
        op = Operation.create(op_type, results=output_types,
                              operands=inputOperands, attributes=kargs)
        self.insert_point.insert(op)
        return op.results[0]

    def add_none_op(self):
        op = Operation.create("tpu.none", results=[
                              NoneType.get()], operands=[])
        self.insert_point.insert(op)
        return op.results[0]

    def add_quant_reg(self, opreands):
        none = self.add_none_op()
        # We assigne 4 reg for quantization
        for _ in range(4):
            opreands.append(none)
        return opreands

    def add_input_op(self, name, index, **kargs):
        assert (index < len(self.func_args))
        shape = [self.tensor_inputs_type[index].get_dim_size(x) \
                 for x in range(self.tensor_inputs_type[index].rank)]
        print("shape:", shape)
        mean = kargs.get('mean', [0, 0, 0])
        scale = kargs.get('scale', [1, 1, 1])
        pixel_format = kargs.get('pixel_format', 'BGR_PLANAR')
        channel_order = kargs.get('channel_order', 'bgr')
        keep_aspect_ratio = kargs.get('keep_aspect_ratio', False)
        resize_dims = kargs.get('resize_dims', shape[-2:])

        preprocess_param = {
            'mean': ArrayAttr.get([FloatAttr.get_f32(x) for x in mean]),
            'scale':  ArrayAttr.get([FloatAttr.get_f32(x) for x in scale]),
            'keep_aspect_ratio': BoolAttr.get(keep_aspect_ratio),
            'resize_dims': ArrayAttr.get([IntegerAttr.get(self.i32Type, x) for x in resize_dims]),
            'channel_order': StringAttr.get(channel_order),
            'pixel_format': StringAttr.get(pixel_format),
            'aligned': BoolAttr.get(False)
        }

        quant_param = {
            'is_asymmetric': BoolAttr.get(False),
            'is_perchannel': BoolAttr.get(False),
            'mode': StringAttr.get("NONE"),
            'param_type': StringAttr.get("NONE"),
            'threshold_max': FloatAttr.get_f32(0),
            'threshold_min': FloatAttr.get_f32(0),
            'zero_point': IntegerAttr.get(self.i32Type, 0),
        }

        attributes = {
            "name": StringAttr.get(name),
            "quant": DictAttr.get(quant_param),
        }
        if len(kargs) > 0:
            attributes["preprocess"] = DictAttr.get(preprocess_param)

        op = Operation.create(TPU_OpType.Input.value,
                              results=[self.tensor_inputs_type[index]],
                              operands=[self.func_args[index]],
                              attributes=attributes)
        self.insert_point.insert(op)
        return op.results[0]

    def add_load_file_op(self, name, output_tensor_shape,
                         tensor_type=TPU_TensorType.FP32,
                         storage="NONE"):
        if tensor_type == TPU_TensorType.FP32:
            tensor_output_type = RankedTensorType.get(
                output_tensor_shape, self.f32Type)
        elif tensor_type == TPU_TensorType.INT32:
            tensor_output_type = RankedTensorType.get(
                output_tensor_shape, self.i32Type)
        elif tensor_type == TPU_TensorType.INT8:
            tensor_output_type = RankedTensorType.get(
                output_tensor_shape, self.i8Type)
        else:
            raise RuntimeError("No support type {}".format(tensor_type))
        attributes = {
            "name": StringAttr.get(name),
            "storage": StringAttr.get(storage),
        }
        op = Operation.create(TPU_OpType.Load_Weight.value, results=[
                              tensor_output_type], operands=[self.weight_op], attributes=attributes)
        self.insert_point.insert(op)
        return op.results[0]

    def add_argmax_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) == 1)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        argmax_name = StringAttr.get(op_name)
        checkKey(kargs, 'axis')
        axis_attr = IntegerAttr.get(self.i32Type, kargs['axis'])
        # inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.ArgMax.value, inputOperands, [
            tensor_output_type], name=argmax_name, axis=axis_attr,
            quant=self.quant_param)

    def add_broadcast_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        broadcast_mul_name = StringAttr.get(op_name)

        axis_attr = IntegerAttr.get(self.i32Type, kargs['axis'])
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.BroadcastMul.value, inputOperands, [
            tensor_output_type], name=broadcast_mul_name, axis=axis_attr, quant=self.quant_param)

    def add_broadcast_add_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        broadcast_add_name = StringAttr.get(op_name)

        axis_attr = IntegerAttr.get(self.i32Type, kargs['axis'])
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.BroadcastAdd.value, inputOperands, [
            tensor_output_type], name=broadcast_add_name, axis=axis_attr, quant=self.quant_param)

    def add_interp_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        mlir_attrs = {}
        checkKey(kargs, 'height')
        checkKey(kargs, 'width')
        checkKey(kargs, 'pad_beg')
        checkKey(kargs, 'pad_end')
        checkKey(kargs, 'shrink_factor')
        checkKey(kargs, 'zoom_factor')
        checkKey(kargs, 'coordinate_transformation_mode')

        for key, value in kargs.items():
            if key == "coordinate_transformation_mode":
                mlir_attrs[key] = StringAttr.get(value)
            else:
                mlir_attrs[key] = IntegerAttr.get(self.i32Type, value)

        name = StringAttr.get(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Interp.value, inputOperands, [
            tensor_output_type], name=name, **mlir_attrs,
            quant=self.quant_param)

    def _add_op(self, op_name, inputOperands, output_tensor_shape, mlir_opname, ops=0, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        none = self.add_none_op()
        # ops for tpu used
        for _ in range(ops):
            inputOperands.append(none)

        name = StringAttr.get(op_name)
        return self.buildOp(mlir_opname, inputOperands, [
            tensor_output_type], name=name, quant=self.quant_param, **kargs)

    def add_acos_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tg_extra_ops = 2
        return self._add_op(op_name, inputOperands, output_tensor_shape,
                TPU_OpType.Acos.value, tg_extra_ops, **kargs)

    def add_abs_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        abs_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Abs.value, inputOperands, [
            tensor_output_type], name=abs_name, quant=self.quant_param)

    def add_batchnorm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'variance_epsilon')

        variance_epsilon = kargs['variance_epsilon']
        checkType(variance_epsilon, float)

        batchnorm_name = StringAttr.get(op_name)
        variance_epsilon_attr = FloatAttr.get_f32(variance_epsilon)

        none = self.add_none_op()
        for _ in range(5 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.BatchNorm.value, inputOperands, [
            tensor_output_type], name=batchnorm_name, variance_epsilon=variance_epsilon_attr)

    def add_clip_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) == 1)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        name = StringAttr.get(op_name)

        checkKey(kargs, 'min')
        checkKey(kargs, 'max')

        clip_min = kargs['min']
        clip_max = kargs['max']

        checkType(clip_min, float)
        checkType(clip_max, float)

        attr_dict = {
            'min': FloatAttr.get_f32(clip_min),
            'max': FloatAttr.get_f32(clip_max),
        }
        inputOperands = self.add_quant_reg(inputOperands)

        return self.buildOp(TPU_OpType.Clip.value, inputOperands, [
            tensor_output_type], name=name, quant=self.quant_param, **attr_dict)

    def add_concat_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        assert(len(inputOperands) >= 2)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'axis')
        concat_name = StringAttr.get(op_name)
        none = self.add_none_op()
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)
            inputOperands = self.add_quant_reg(inputOperands)
        elif mode == TPU_MODE.FP32:
            quant_param = self.quant_param
            inputOperands = self.add_quant_reg(inputOperands)
        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        axis_attr = IntegerAttr.get(self.i32Type, kargs['axis'])

        return self.buildOp(TPU_OpType.Concat.value, inputOperands, [
            tensor_output_type], name=concat_name, axis=axis_attr, quant=quant_param)

    def add_conv_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):
        """
            inputOperands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        # get_input_type
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
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

        conv_name = StringAttr.get(op_name)
        conv_param = {
            'stride_h':  IntegerAttr.get(self.i32Type, kargs['stride_h']),
            'stride_w':  IntegerAttr.get(self.i32Type, kargs['stride_w']),
            'padding': StringAttr.get(kargs['padding']),
            'dilation_h':  IntegerAttr.get(self.i32Type,  kargs['dilation_h']),
            'dilation_w':  IntegerAttr.get(self.i32Type, kargs['dilation_w']),
            'padding_t':  IntegerAttr.get(self.i32Type, kargs['padding_t']),
            'padding_b':  IntegerAttr.get(self.i32Type, kargs['padding_b']),
            'padding_l':  IntegerAttr.get(self.i32Type, kargs['padding_l']),
            'padding_r':  IntegerAttr.get(self.i32Type, kargs['padding_r']),
            'group':  IntegerAttr.get(self.i32Type, kargs['group']),
            'is_dw': BoolAttr.get(kargs['is_dw']),
            'with_bias': BoolAttr.get(kargs['with_bias']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            'ins': ArrayAttr.get(
                [IntegerAttr.get(self.i32Type, x) for x in kargs['ins']]),
            'pad_value':  IntegerAttr.get(self.i32Type, pad_value),
        }

        dict_attr = DictAttr.get(conv_param)
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

        # quant param
        return self.buildOp(TPU_OpType.Conv2d.value, inputOperands, [
            tensor_output_type], name=conv_name, param=dict_attr, quant=quant_param)

    def add_conv3d_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        conv_name = StringAttr.get(op_name)
        conv3d_param = {
            'stride_d':  IntegerAttr.get(self.i32Type, kargs['stride_d']),
            'stride_h':  IntegerAttr.get(self.i32Type, kargs['stride_h']),
            'stride_w':  IntegerAttr.get(self.i32Type, kargs['stride_w']),
            'padding': StringAttr.get(kargs['padding']),
            'dilation_d':  IntegerAttr.get(self.i32Type,  kargs['dilation_d']),
            'dilation_h':  IntegerAttr.get(self.i32Type,  kargs['dilation_h']),
            'dilation_w':  IntegerAttr.get(self.i32Type, kargs['dilation_w']),
            'padding_d0':  IntegerAttr.get(self.i32Type, kargs['padding_d0']),
            'padding_d1':  IntegerAttr.get(self.i32Type, kargs['padding_d1']),
            'padding_t':  IntegerAttr.get(self.i32Type, kargs['padding_t']),
            'padding_b':  IntegerAttr.get(self.i32Type, kargs['padding_b']),
            'padding_l':  IntegerAttr.get(self.i32Type, kargs['padding_l']),
            'padding_r':  IntegerAttr.get(self.i32Type, kargs['padding_r']),
            'group':  IntegerAttr.get(self.i32Type, kargs['group']),
            'is_dw': BoolAttr.get(kargs['is_dw']),
            'with_bias': BoolAttr.get(kargs['with_bias']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            'ins': ArrayAttr.get(
                [IntegerAttr.get(self.i32Type, x) for x in kargs['ins']])
        }

        dict_attr = DictAttr.get(conv3d_param)
        none = self.add_none_op()
        for _ in range(7 - len(inputOperands)):
            inputOperands.append(none)
        quant_param = self.quant_param
        return self.buildOp(TPU_OpType.Conv3d.value, inputOperands, [
            tensor_output_type], name=conv_name, param=dict_attr, quant=quant_param)

    def add_csc_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        assert(len(inputOperands) == 1)
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        name = StringAttr.get(op_name)

        checkKey(kargs, 'pixel_format')
        checkKey(kargs, 'aligned')

        attr_dict = {
            'pixel_format': StringAttr.get(kargs['pixel_format']),
            'aligned': BoolAttr.get(kargs['aligned'])
        }
        return self.buildOp(TPU_OpType.Csc.value, inputOperands, [
            tensor_output_type], name=name, quant=self.quant_param, **attr_dict)

    def add_custom_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            args:
                operation_name: string
                do_quant: bool
                tpu: bool
                threshold_overwrite: string, 'none', 'backward' or 'forward'
                param: dictionary
        """
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'operation_name')
        checkKey(kargs, 'do_quant')
        checkKey(kargs, 'tpu')
        checkKey(kargs, 'threshold_overwrite')
        checkKey(kargs, 'param')
        checkType(kargs['param'], dict)
        if kargs['threshold_overwrite'] not in ['none', 'backward', 'forward']:
            raise AttributeError("invalid value of parameter threshold_overwrite: {}"
                                 .format(kargs['threshold_overwrite']))

        name = StringAttr.get(op_name)
        operation_name = StringAttr.get(kargs['operation_name'])
        do_quant = BoolAttr.get(kargs['do_quant'])
        tpu = BoolAttr.get(kargs['tpu'])
        threshold_overwrite = StringAttr.get(kargs['threshold_overwrite'])

        op_param = {}
        for key, val in kargs['param'].items():
            attr_type = checkAttrType(val)
            if attr_type == 'int':
                op_param[key] = IntegerAttr.get(self.i32Type, val)
            elif attr_type == 'float':
                op_param[key] = FloatAttr.get_f32(val)
            elif attr_type == 'str':
                op_param[key] = StringAttr.get(val)
            elif attr_type == 'bool':
                op_param[key] = BoolAttr.get(val)
            elif attr_type == 'int_arr':
                arr = [IntegerAttr.get(self.i32Type, x) for x in val]
                op_param[key] = ArrayAttr.get(arr)
            elif attr_type == 'float_arr':
                arr = [FloatAttr.get_f32(x) for x in val]
                op_param[key] = ArrayAttr.get(arr)
            elif attr_type == 'str_arr':
                arr = [StringAttr.get(x) for x in val]
                op_param[key] = ArrayAttr.get(arr)
            elif attr_type == 'bool_arr':
                arr = [BoolAttr.get(x) for x in val]
                op_param[key] = ArrayAttr.get(arr)
        param = DictAttr.get(op_param)

        return self.buildOp(TPU_OpType.CustomOp.value, inputOperands, [
            tensor_output_type], name=name, operation_name=operation_name, quant=self.quant_param,
            do_quant=do_quant, param=param, tpu=tpu, threshold_overwrite=threshold_overwrite)

    def add_crop_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        """
            args:
                crop_offset: List[int, int, int, int]
                crop_shape : List[int, int, int, int]
        """
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'crop_offset')
        checkKey(kargs, 'crop_shape')

        crop_offset = kargs['crop_offset']
        crop_shape = kargs['crop_shape']
        checkType(crop_offset, list)
        checkType(crop_shape, list)

        crop_name = StringAttr.get(op_name)
        crop_offset_attr = ArrayAttr.get(
            [IntegerAttr.get(self.i32Type, x) for x in crop_offset])
        crop_shape_attr = ArrayAttr.get(
            [IntegerAttr.get(self.i32Type, x) for x in crop_shape])

        return self.buildOp(TPU_OpType.Crop.value, inputOperands, [
            tensor_output_type], name=crop_name, crop_offset=crop_offset_attr, quant=self.quant_param, crop_shape=crop_shape_attr)

    def add_detection_output_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'num_classes')
        checkKey(kargs, 'share_location')
        checkKey(kargs, 'background_label_id')
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'top_k')
        checkKey(kargs, 'code_type')
        checkKey(kargs, 'keep_top_k')
        checkKey(kargs, 'confidence_threshold')
        name_attr = StringAttr.get(op_name)
        param = {
            'num_classes':  IntegerAttr.get(self.i32Type, kargs['num_classes']),
            'share_location': BoolAttr.get(kargs['share_location']),
            'background_label_id':  IntegerAttr.get(self.i32Type, kargs['background_label_id']),
            'nms_threshold': FloatAttr.get_f32(kargs['nms_threshold']),
            'top_k':  IntegerAttr.get(self.i32Type, kargs['top_k']),
            'code_type': StringAttr.get(kargs['code_type']),
            'keep_top_k':  IntegerAttr.get(self.i32Type, kargs['keep_top_k']),
            'confidence_threshold': FloatAttr.get_f32(kargs['confidence_threshold']),
        }
        return self.buildOp(TPU_OpType.DetectionOutput.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_deconv_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):
        """
            inputOperands: List[pybind.op]
            output_tensorshape: List[int] output tensor type
            attrs: Dict, about op attrs
        """
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

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

        deconv_name = StringAttr.get(op_name)
        deconv_param = {
            'stride_h':  IntegerAttr.get(self.i32Type, kargs['stride_h']),
            'stride_w':  IntegerAttr.get(self.i32Type, kargs['stride_w']),
            'padding': StringAttr.get(kargs['padding']),
            'dilation_h':  IntegerAttr.get(self.i32Type,  kargs['dilation_h']),
            'dilation_w':  IntegerAttr.get(self.i32Type, kargs['dilation_w']),
            'padding_t':  IntegerAttr.get(self.i32Type, kargs['padding_t']),
            'padding_b':  IntegerAttr.get(self.i32Type, kargs['padding_b']),
            'padding_l':  IntegerAttr.get(self.i32Type, kargs['padding_l']),
            'padding_r':  IntegerAttr.get(self.i32Type, kargs['padding_r']),
            'group':  IntegerAttr.get(self.i32Type, kargs['group']),
            'is_dw': BoolAttr.get(kargs['is_dw']),
            'with_bias': BoolAttr.get(kargs['with_bias']),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            'ins': ArrayAttr.get(
                [IntegerAttr.get(self.i32Type, x) for x in kargs['ins']]),
            'pad_value':  IntegerAttr.get(self.i32Type, pad_value),
        }

        dict_attr = DictAttr.get(deconv_param)
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
        return self.buildOp(TPU_OpType.DeConv2d.value, inputOperands, [
            tensor_output_type], name=deconv_name, param=dict_attr, quant=quant_param)

    def add_dummydata_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        assert(len(inputOperands) == 0)
        name_attr = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.DummyData.value, inputOperands, [
            tensor_output_type], name=name_attr)

    def add_eltwise_add_op(self, op_name, inputOperands, output_tensor_shape,  mode=TPU_MODE.FP32, do_relu=False, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")
        do_relu = BoolAttr.get(do_relu)
        eltwise_add = StringAttr.get(op_name)
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

        coeff = None
        if "coeff" in kargs:
            coeff = ArrayAttr.get([FloatAttr.get_f32(x) for x in kargs['coeff']])
        else:
            coeff = ArrayAttr.get([FloatAttr.get_f32(x) for x in [1.0]*len(inputOperands)])

        param = {
            'coeff':  coeff
        }
        return self.buildOp(TPU_OpType.Eltwise_Add.value, inputOperands, [
            tensor_output_type], name=eltwise_add, quant=quant_param, do_relu=do_relu, **param)

    def add_eltwise_sub_op(self, op_name, inputOperands, output_tensor_shape,
                           mode=TPU_MODE.FP32, do_relu=False, **kargs):
        return self.add_eltwise_add_op(op_name, inputOperands, output_tensor_shape,
                                       mode, do_relu, coeff=[1, -1], **kargs)

    def add_eltwise_max_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        if "coeff" in kargs:
            assert(0 and "eltwise max not support coeff")
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_max = StringAttr.get(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Max.value, inputOperands, [
            tensor_output_type], name=eltwise_max, quant=self.quant_param)

    def add_eltwise_min_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        if "coeff" in kargs:
            assert(0 and "eltwise min not support coeff")
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_min = StringAttr.get(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Min.value, inputOperands, [
            tensor_output_type], name=eltwise_min, quant=self.quant_param)

    def add_eltwise_mul_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        if "coeff" in kargs:
            assert(0 and "eltwise mul not support coeff")
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        eltwise_mul = StringAttr.get(op_name)
        inputOperands = self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.Eltwise_Mul.value, inputOperands, [
            tensor_output_type], name=eltwise_mul, quant=self.quant_param)

    def add_equal_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tg_extra_ops = 4 # reserver for quant op
        return self._add_op(op_name, inputOperands, output_tensor_shape,
                TPU_OpType.Equal.value, tg_extra_ops, **kargs)

    def add_elu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        sigmoid_name = StringAttr.get(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for sigmoid quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Elu.value, inputOperands, [
            tensor_output_type], name=sigmoid_name, quant=self.quant_param)

    def add_exp_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        op_name = StringAttr.get(op_name)
        none = self.add_none_op()

        scale = kargs.get('scale', 1.0)
        bias = kargs.get('bias', 0)
        for _ in range(2):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.Exp.value, inputOperands, [
                                     tensor_output_type], name=op_name, quant=self.quant_param,
                                     scale=FloatAttr.get_f32(scale), bias=FloatAttr.get_f32(bias))

    def add_embedding_op(self, op_name, inputOperands, output_shape):
        tensor_output_type = RankedTensorType.get(
            tuple(output_shape), self.f32Type)

        op_name = StringAttr.get(op_name)

        return self.buildOp(TPU_OpType.Embedding.value, inputOperands, [tensor_output_type],
                            name=op_name, quant=self.quant_param)


    def add_fully_connected_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        if len(inputOperands) < 2:
            raise ArithmeticError("input operand must great than 2")

        if len(inputOperands) == 2:
            none = self.add_none_op()
            inputOperands.append(none)
            # No bias
        if mode == TPU_MODE.INT8:
            assert(kargs['param_type'] == "RSHIFT_AND_M_I32")
            quant_param = self.create_int8_quant_attr(**kargs)
            rshift_op = inputOperands[-2]
            multiplier_op = inputOperands[-1]
            inputOperands = inputOperands[:-1]
            none = self.add_none_op()

            for _ in range(5 - len(inputOperands)):
                inputOperands.append(none)
            inputOperands.append(rshift_op)
            inputOperands.append(multiplier_op)

        else:
            quant_param = self.quant_param
            inputOperands = self.add_quant_reg(inputOperands)
        fully_connected_name = StringAttr.get(op_name)

        return self.buildOp(TPU_OpType.FullyConnected.value, inputOperands, [
            tensor_output_type], name=fully_connected_name, quant=quant_param)

    def add_frcn_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'class_num')
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'obj_threshold')
        checkKey(kargs, 'keep_topk')

        name_attr = StringAttr.get(op_name)
        param = {
            'class_num':  IntegerAttr.get(self.i32Type, kargs['class_num']),
            'nms_threshold': FloatAttr.get_f32(kargs['nms_threshold']),
            'obj_threshold': FloatAttr.get_f32(kargs['obj_threshold']),
            'keep_topk':  IntegerAttr.get(self.i32Type, kargs['keep_topk'])
        }
        return self.buildOp(TPU_OpType.FrcnDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_gru_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        gru_param = {
            'linear_before_reset': BoolAttr.get(kargs['linear_before_reset']),
            'bidirectional': BoolAttr.get(kargs['bidirectional'])
        }

        gru_name = StringAttr.get(op_name)
        if len(inputOperands) < 8:
            none = self.add_none_op()
            for _ in range(8 - len(inputOperands)):  # add 4 redundant input
                inputOperands.append(none)

        return self.buildOp(TPU_OpType.GRU.value, inputOperands, [
            tensor_output_type], name=gru_name, quant=self.quant_param, **gru_param)

    def add_instancenorm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tg_extra_ops = 0
        checkKey(kargs, 'variance_epsilon')
        attr_dict = {
            'variance_epsilon': FloatAttr.get_f32(kargs['variance_epsilon']),
        }
        return self._add_op(op_name, inputOperands, output_tensor_shape,
                            TPU_OpType.InstanceNorm.value, tg_extra_ops, **attr_dict)


    def add_leaky_relu_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'negative_slope')

        leaky_relu_param = {
            'negative_slope': FloatAttr.get_f32(kargs['negative_slope'])
        }

        leaky_relu_name = StringAttr.get(op_name)

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
            for _ in range(9 - len(inputOperands)):
                inputOperands.append(none)
            quant_param = self.quant_param

        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        return self.buildOp(TPU_OpType.LeakyRelu.value, inputOperands, [
            tensor_output_type], name=leaky_relu_name, quant=quant_param, **leaky_relu_param)

    def add_lrn_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'alpha')
        checkKey(kargs, 'beta')
        checkKey(kargs, 'bias')
        checkKey(kargs, 'size')

        lrn_param = {
            'alpha': FloatAttr.get_f32(kargs['alpha']),
            'beta': FloatAttr.get_f32(kargs['beta']),
            'k': FloatAttr.get_f32(kargs['bias']),
            'local_size': IntegerAttr.get(self.i32Type, kargs['size']),
            'sum_rshift': IntegerAttr.get(self.i32Type, 0),
            'lrn_rshift': IntegerAttr.get(self.i32Type, 0),
            'quant_data0': IntegerAttr.get(self.i32Type, 0),
            'quant_data1': IntegerAttr.get(self.i32Type, 0),
            'norm_region': IntegerAttr.get(self.i32Type, 0)
        }

        lrn_name = StringAttr.get("{}".format(op_name))
        input_op = inputOperands[0]
        none = self.add_none_op()
        operands = list()
        operands.append(input_op)
        operands.append(none)
        operands.append(none)
        operands.append(none)
        return self.buildOp(TPU_OpType.Lrn.value, operands, [
            tensor_output_type], name=lrn_name, quant=self.quant_param, **lrn_param)

    def add_lstm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        lstm_param = {
            'bidirectional': BoolAttr.get(kargs['bidirectional'])
        }
        lstm_name = StringAttr.get(op_name)
        none = self.add_none_op()
        for _ in range(10 - len(inputOperands)):  # add 4 redundant input
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.LSTM.value, inputOperands, [
            tensor_output_type], name=lstm_name, quant=self.quant_param, **lstm_param)

    def add_layernorm_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        op_name = StringAttr.get(op_name)
        param = {
            "eps":FloatAttr.get_f32(kargs['eps']),
            "normalized_shape": ArrayAttr.get([IntegerAttr.get(self.i32Type, x) for x in kargs['normal_shape']])
        }
        return self.buildOp(TPU_OpType.LayerNorm.value, inputOperands, [tensor_output_type],
                name=op_name, quant=self.quant_param, **param)

    def add_normalize_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'across_spatial')
        checkKey(kargs, 'channel_shared')
        name_attr = StringAttr.get(op_name)
        param = {
            'across_spatial': BoolAttr.get(kargs['across_spatial']),
            'channel_shared': BoolAttr.get(kargs['channel_shared']),
        }
        return self.buildOp(TPU_OpType.Normalize.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_mish_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        mish_name = StringAttr.get(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for mish quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Mish.value, inputOperands, [
            tensor_output_type], name=mish_name, quant=self.quant_param)

    def add_reflectionpad_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, ** kargs):

        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'pads')

        pads = kargs['pads']

        op_name = StringAttr.get(op_name)
        none = self.add_none_op()
        inputOperands.append(none)
        inputOperands.append(none)
        param = {
            "pads": ArrayAttr.get([IntegerAttr.get(self.i32Type, x) for x in kargs['pads']])
        }
        return self.buildOp(TPU_OpType.ReflectionPad.value, inputOperands, [
            tensor_output_type], name=op_name, quant=self.quant_param, **param)


    def add_pad_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, ** kargs):
        """
            args:
                pads : List[int, int, int, int]
                const_val : int
        """

        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'pads')
        checkKey(kargs, 'const_val')

        pads = kargs['pads']
        const_val = kargs['const_val']
        pad_mode = kargs.get('pad_mode', 'constant')

        checkType(pads, list)

        pad_name = StringAttr.get(op_name)
        pads_attr = ArrayAttr.get(
            [IntegerAttr.get(self.i32Type, x) for x in pads])
        const_val_attr = FloatAttr.get_f32(const_val)
        pad_mode_attr = StringAttr.get(pad_mode)

        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)
        elif mode == TPU_MODE.FP32:
            quant_param = self.quant_param
        else:
            raise RuntimeError("No support quant mode {}".format(mode))

        return self.buildOp(TPU_OpType.Pad.value, inputOperands, [
            tensor_output_type], name=pad_name, quant=quant_param, pads=pads_attr, const_val=const_val_attr, mode=pad_mode_attr)

    def add_pool_mask_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'scale')
        pool_mask_name = StringAttr.get(op_name)
        pool_mask_param = {
            'scale':  IntegerAttr.get(self.i32Type, kargs['scale'])
        }
        return self.buildOp(TPU_OpType.PoolMask.value, inputOperands, [
            tensor_output_type], name=pool_mask_name, quant=self.quant_param, **pool_mask_param)

    def add_pool_avg_2d_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
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

        pool_avg_2d_name = StringAttr.get(op_name)
        pool_avg_2d_param = {
            'stride_h': IntegerAttr.get(self.i32Type, kargs['stride_h']),
            'stride_w': IntegerAttr.get(self.i32Type, kargs['stride_w']),
            'kernel_h': IntegerAttr.get(self.i32Type, kargs['kernel_h']),
            'kernel_w': IntegerAttr.get(self.i32Type, kargs['kernel_w']),
            'padding_b': IntegerAttr.get(self.i32Type, kargs['padding_b']),
            'padding_l': IntegerAttr.get(self.i32Type, kargs['padding_l']),
            'padding_r': IntegerAttr.get(self.i32Type, kargs['padding_r']),
            'padding_t': IntegerAttr.get(self.i32Type, kargs['padding_t']),
            'pad_value': IntegerAttr.get(self.i32Type, pad_value),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            'count_include_pad': BoolAttr.get(kargs['count_include_pad']),
        }
        dict_attr = DictAttr.get(pool_avg_2d_param)
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

    def add_pool_max_2d_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.FP32, pad_value=0, **kargs):

        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'kernel_h')
        checkKey(kargs, 'kernel_w')
        checkKey(kargs, 'padding_b')
        checkKey(kargs, 'padding_l')
        checkKey(kargs, 'padding_r')
        checkKey(kargs, 'padding_t')
        checkKey(kargs, 'stride_h')
        checkKey(kargs, 'stride_w')
        checkKey(kargs, 'do_relu')

        pool_max_2d_name = StringAttr.get(op_name)
        pool_max_2d_param = {
            'stride_h': IntegerAttr.get(self.i32Type, kargs['stride_h']),
            'stride_w': IntegerAttr.get(self.i32Type, kargs['stride_w']),
            'kernel_h': IntegerAttr.get(self.i32Type, kargs['kernel_h']),
            'kernel_w': IntegerAttr.get(self.i32Type, kargs['kernel_w']),
            'padding_b': IntegerAttr.get(self.i32Type, kargs['padding_b']),
            'padding_l': IntegerAttr.get(self.i32Type, kargs['padding_l']),
            'padding_r': IntegerAttr.get(self.i32Type, kargs['padding_r']),
            'padding_t': IntegerAttr.get(self.i32Type, kargs['padding_t']),
            'pad_value': IntegerAttr.get(self.i32Type, pad_value),
            'do_relu': BoolAttr.get(kargs['do_relu']),
            # max pool has no count_include_pad method
            'count_include_pad': BoolAttr.get(False),
        }
        dict_attr = DictAttr.get(pool_max_2d_param)
        if mode == TPU_MODE.INT8:
            quant_param = self.create_int8_quant_attr(**kargs)
        elif mode == TPU_MODE.FP32:
            quant_param = self.quant_param
        elif mode == TPU_MODE.BF16:
            raise RuntimeError("Not support BF16")

        return self.buildOp(TPU_OpType.PoolMax2D.value, inputOperands, [
            tensor_output_type], name=pool_max_2d_name, param=dict_attr, quant=quant_param)

    def add_power_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'power')
        checkKey(kargs, 'scale')
        checkKey(kargs, 'shift')

        name_attr = StringAttr.get(op_name)
        param = {
            'power': FloatAttr.get_f32(kargs['power']),
            'scale': FloatAttr.get_f32(kargs['scale']),
            'shift': FloatAttr.get_f32(kargs['shift']),
        }
        return self.buildOp(TPU_OpType.Power.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_priorbox_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

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

        name_attr = StringAttr.get(op_name)
        param = {
            'min_size': ArrayAttr.get([FloatAttr.get_f32(x) for x in kargs['min_size']]),
            'max_size': ArrayAttr.get([FloatAttr.get_f32(x) for x in kargs['max_size']]),
            'aspect_ratios': ArrayAttr.get([FloatAttr.get_f32(x) for x in kargs['aspect_ratios']]),
            'variance': ArrayAttr.get([FloatAttr.get_f32(x) for x in kargs['variance']]),
            'clip': BoolAttr.get(kargs['clip']),
            'step_h': FloatAttr.get_f32(kargs['step_h']),
            'step_w': FloatAttr.get_f32(kargs['step_w']),
            'img_h':  IntegerAttr.get(self.i32Type, kargs['img_h']),
            'img_w':  IntegerAttr.get(self.i32Type, kargs['img_w']),
            'offset': FloatAttr.get_f32(kargs['offset']),
            'num_priors':  IntegerAttr.get(self.i32Type, kargs['num_priors']),
            'use_default_aspect_ratio': BoolAttr.get(kargs['use_default_aspect_ratio']),
        }
        return self.buildOp(TPU_OpType.PriorBox.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_permute_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'order')

        permute_name = StringAttr.get(op_name)
        attr_dict = {
            'order':  ArrayAttr.get([IntegerAttr.get(self.i32Type, x) for x in kargs['order']]),
        }
        return self.buildOp(TPU_OpType.Permute.value, inputOperands, [
            tensor_output_type], name=permute_name, quant=self.quant_param, **attr_dict)

    def add_pixelshuffle_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'upscale_factor')
        checkKey(kargs, 'mode')

        pixelshuffle_name = StringAttr.get(op_name)
        attr_dict = {
            'upscale_factor':  IntegerAttr.get(self.i32Type, kargs['upscale_factor']),
            'mode': StringAttr.get(kargs['mode'])
        }
        return self.buildOp(TPU_OpType.PixelShuffle.value, inputOperands, [
            tensor_output_type], name=pixelshuffle_name, quant=self.quant_param, **attr_dict)

    def add_prelu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        prelu_name = StringAttr.get(op_name)

        none = self.add_none_op()
        # quant_pos_scale, quant_pos_zeropoint, quant_neg_scale, quant_neg_zeropoint
        # quant_pos_rshift, quant_pos_multiplier, quant_neg_rshift, quant_neg_multiplier
        for _ in range(10 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.PRelu.value, inputOperands, [
            tensor_output_type], name=prelu_name, quant=self.quant_param)

    def add_proposal_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'net_input_h')
        checkKey(kargs, 'net_input_w')
        checkKey(kargs, 'feat_stride')
        checkKey(kargs, 'anchor_base_size')
        checkKey(kargs, 'rpn_obj_threshold')
        checkKey(kargs, 'rpn_nms_threshold')
        checkKey(kargs, 'rpn_nms_post_top_n')

        proposal_name = StringAttr.get(op_name)
        attr_dict = {
            'net_input_h':  IntegerAttr.get(self.i32Type, kargs['net_input_h']),
            'net_input_w':  IntegerAttr.get(self.i32Type, kargs['net_input_w']),
            'feat_stride':  IntegerAttr.get(self.i32Type, kargs['feat_stride']),
            'anchor_base_size':  IntegerAttr.get(self.i32Type, kargs['anchor_base_size']),
            'rpn_obj_threshold': FloatAttr.get_f32(kargs['rpn_obj_threshold']),
            'rpn_nms_threshold': FloatAttr.get_f32(kargs['rpn_nms_threshold']),
            'rpn_nms_post_top_n':  IntegerAttr.get(self.i32Type, kargs['rpn_nms_post_top_n']),
        }
        return self.buildOp(TPU_OpType.Proposal.value, inputOperands, [
            tensor_output_type], name=proposal_name, quant=self.quant_param, **attr_dict)

    def add_quant_op(self, op_name, inputOperands, output_tensor_shape, from_type, to_type, zero_point=0, **kargs):
        if to_type == "NONE":
            tensor_output_type = RankedTensorType.get(
                tuple(output_tensor_shape), self.f32Type)
        elif to_type == "INT8" or to_type == "UINT8":
            tensor_output_type = RankedTensorType.get(
                tuple(output_tensor_shape), self.i8Type)
        else:
            raise RuntimeError("No support {} to_type".format(to_type))

        checkKey(kargs, 'scale')
        quant_name = StringAttr.get(op_name)
        attr_dict = {
            'from': StringAttr.get(from_type),
            'to': StringAttr.get(to_type),
            'scale': FloatAttr.get_f32(kargs['scale']),
            'zero_point':  IntegerAttr.get(self.i32Type, zero_point),
        }
        return self.buildOp(TPU_OpType.Quant.value, inputOperands, [
            tensor_output_type], name=quant_name, **attr_dict)

    def add_requant_op(self, op_name, inputOperands, output_tensor_shape, mode=TPU_MODE.INT8.value, **kargs):
        if mode != TPU_MODE.INT8.value:
            raise RuntimeError("Only support asymmetric mode")
        # Only in i8 case
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.i8Type)

        checkKey(kargs, 'zero_point')
        checkKey(kargs, 'qscale')
        zero_point = kargs['zero_point']
        qscale = kargs['qscale']
        quant_name = StringAttr.get(op_name)
        attr_dict = {
            'zero_point': IntegerAttr.get(self.i32Type, zero_point),
            'qscale': FloatAttr.get_f32(qscale)
        }
        return self.buildOp(TPU_OpType.ReQuant.value, inputOperands, [
            tensor_output_type], name=quant_name, **attr_dict)

    def add_reciprocal_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        reciprocal_name = StringAttr.get(op_name)

        # table and table_mantissa all are none
        none = self.add_none_op()
        for _ in range(3 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.Reciprocal.value, inputOperands, [
            tensor_output_type], name=reciprocal_name, quant=self.quant_param)

    def add_relu_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        relu_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Relu.value, inputOperands, [
            tensor_output_type], name=relu_name, quant=self.quant_param)

    def add_reorg_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'stride')
        name_attr = StringAttr.get(op_name)

        param = {
            'stride':  IntegerAttr.get(self.i32Type, kargs['stride']),
        }
        return self.buildOp(TPU_OpType.Reorg.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_retinaface_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'nms_threshold')
        checkKey(kargs, 'confidence_threshold')
        checkKey(kargs, 'keep_topk')
        name_attr = StringAttr.get(op_name)
        param = {
            'nms_threshold': FloatAttr.get_f32(kargs['nms_threshold']),
            'confidence_threshold': FloatAttr.get_f32(kargs['confidence_threshold']),
            'keep_topk': IntegerAttr.get(self.i32Type, kargs['keep_topk']),
        }
        return self.buildOp(TPU_OpType.RetinaFaceDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_reshape_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        reshape_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Reshape.value, inputOperands, [
            tensor_output_type], name=reshape_name)

    def add_reverse_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'axis')
        nameAttr = StringAttr.get(op_name)
        attr_dict = {
            'axis': IntegerAttr.get(self.i32Type, kargs['axis'])
        }
        return self.buildOp(TPU_OpType.Reverse.value, inputOperands, [
            tensor_output_type], name=nameAttr, quant=self.quant_param, **attr_dict)

    def add_roipooling_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'pooled_h')
        checkKey(kargs, 'pooled_w')
        checkKey(kargs, 'spatial_scale')

        roipooling_name = StringAttr.get(op_name)
        attr_dict = {
            'pooled_h':  IntegerAttr.get(self.i32Type, kargs['pooled_h']),
            'pooled_w':  IntegerAttr.get(self.i32Type, kargs['pooled_w']),
            'spatial_scale': FloatAttr.get_f32(kargs['spatial_scale'])
        }
        return self.buildOp(TPU_OpType.ROIPooling.value, inputOperands, [
            tensor_output_type], name=roipooling_name, quant=self.quant_param, **attr_dict)

    def add_scale_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        if len(inputOperands) < 3:
            none = self.add_none_op()
            inputOperands.append(none)

        scale_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Scale.value, inputOperands, [
            tensor_output_type], name=scale_name)

    def add_shufflechannel_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'group')
        attr_dict = {
            'group':  IntegerAttr.get(self.i32Type, kargs['group']),
        }
        sc_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.ShuffelChannel.value, inputOperands, [
            tensor_output_type], name=sc_name, quant=self.quant_param, **attr_dict)

    def add_sigmoid_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        sigmoid_name = StringAttr.get(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for sigmoid quant table
        scale = kargs.get('scale', 1.0)
        bias = kargs.get('bias', 0)
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Sigmoid.value, inputOperands, [
                                      tensor_output_type], name=sigmoid_name, quant=self.quant_param,
                                      scale=FloatAttr.get_f32(scale), bias=FloatAttr.get_f32(bias))

    def add_slice_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        attr_dict = {
            'axis':  IntegerAttr.get(self.i32Type, kargs['axis']),
            'offset':  IntegerAttr.get(self.i32Type, kargs['offset']),
        }

        slice_name = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Slice.value, inputOperands, [
            tensor_output_type], name=slice_name, quant=self.quant_param, **attr_dict)

    def add_sqrt_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        sqrt_name = StringAttr.get(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for sqrt quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Sqrt.value, inputOperands, [
            tensor_output_type], name=sqrt_name, quant=self.quant_param)

    def add_softplus_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        op_name = StringAttr.get(op_name)
        none = self.add_none_op()

        scale = kargs.get('scale', 1.0)
        bias = kargs.get('bias', 0)
        for _ in range(2):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.SoftPlus.value, inputOperands, [
            tensor_output_type], name=op_name, quant=self.quant_param,
            scale=FloatAttr.get_f32(scale), bias=FloatAttr.get_f32(bias))

    def add_softmax_op(self, op_name, inputOperands, output_tensor_shape, cpu_mode=False, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        softmax_name = StringAttr.get(op_name)
        softmax_param = {
            'axis':  IntegerAttr.get(self.i32Type, kargs['axis'])
        }
        none = self.add_none_op()
        if cpu_mode:
            op_name = TPU_OpType.Softmax.value + "_cpu"
        else:
            op_name = TPU_OpType.Softmax.value
            for _ in range(4):  # add 4 redundant input
                inputOperands.append(none)
        return self.buildOp(op_name, inputOperands, [
            tensor_output_type], name=softmax_name, quant=self.quant_param, **softmax_param)

    def add_swap_channel_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        name_attr = StringAttr.get(op_name)
        checkKey(kargs, 'channel_order')
        order = ArrayAttr.get([IntegerAttr.get(self.i32Type, x)
                               for x in kargs['channel_order']])
        param = {
            'channel_order': order,
        }
        return self.buildOp(TPU_OpType.SwapChannel.value, inputOperands, [
            tensor_output_type], name=name_attr, quant=self.quant_param, **param)

    def add_tanh_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        tanh_name = StringAttr.get(op_name)
        none = self.add_none_op()
        # We assigne 4 reg for tanh quant table
        for _ in range(2):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Tanh.value, inputOperands, [
            tensor_output_type], name=tanh_name, quant=self.quant_param)

    def add_tile_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))

        checkKey(kargs, 'axis')
        checkKey(kargs, 'tiles')

        tile_name = StringAttr.get(op_name)
        tile_param = {
            'axis':   IntegerAttr.get(self.i32Type, kargs['axis']),
            'tiles':  IntegerAttr.get(self.i32Type, kargs['tiles'])
        }

        return self.buildOp(TPU_OpType.Tile.value, inputOperands, [
            tensor_output_type], name=tile_name, quant=self.quant_param, **tile_param)

    def add_upsample_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'scale_h')
        checkKey(kargs, 'scale_w')

        upsample_name = StringAttr.get(op_name)
        upsample_param = {
            'scale_h':  IntegerAttr.get(self.i32Type, kargs['scale_h']),
            'scale_w':  IntegerAttr.get(self.i32Type, kargs['scale_w'])
        }
        if len(inputOperands) < 2:
            none = self.add_none_op()
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.Upsample.value, inputOperands, [
            tensor_output_type], name=upsample_name, quant=self.quant_param, **upsample_param)

    def add_reduce_l2_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'axes')

        reduce_name = StringAttr.get(op_name)
        axes = ArrayAttr.get([IntegerAttr.get(self.i32Type, x)
                              for x in kargs['axes']])
        reduce_param = {
            'axes': axes
        }
        none = self.add_none_op()
        for _ in range(5 - len(inputOperands)):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.ReduceL2.value, inputOperands, [
            tensor_output_type], name=reduce_name, quant=self.quant_param, **reduce_param)

    def add_reduce_mean_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'axes')

        reduce_name = StringAttr.get(op_name)
        axes = ArrayAttr.get([IntegerAttr.get(self.i32Type, x)
                              for x in kargs['axes']])
        reduce_param = {
            'axes': axes
        }
        none = self.add_none_op()
        for _ in range(5 - len(inputOperands)):
            inputOperands.append(none)
        return self.buildOp(TPU_OpType.ReduceMean.value, inputOperands, [
            tensor_output_type], name=reduce_name, quant=self.quant_param, **reduce_param)

    def add_reduce_max_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        checkKey(kargs, 'axes')

        reduce_name = StringAttr.get(op_name)
        axes = ArrayAttr.get([IntegerAttr.get(self.i32Type, x)
                              for x in kargs['axes']])
        reduce_param = {
            'axes': axes
        }

        none = self.add_none_op()
        for _ in range(5 - len(inputOperands)):
            inputOperands.append(none)

        return self.buildOp(TPU_OpType.ReduceMax.value, inputOperands, [
            tensor_output_type], name=reduce_name, quant=self.quant_param, **reduce_param)

    def add_yolo_detection_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
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

        name_attr = StringAttr.get(op_name)
        param = {
            'net_input_h':  IntegerAttr.get(self.i32Type, kargs['net_input_h']),
            'net_input_w':  IntegerAttr.get(self.i32Type, kargs['net_input_w']),
            'nms_threshold': FloatAttr.get_f32(kargs['nms_threshold']),
            'obj_threshold': FloatAttr.get_f32(kargs['obj_threshold']),
            'keep_topk':  IntegerAttr.get(self.i32Type, kargs['keep_topk']),
            'spp_net': BoolAttr.get(kargs['spp_net']),
            'tiny': BoolAttr.get(kargs['tiny']),
            'yolo_v4': BoolAttr.get(kargs['yolo_v4']),
            'class_num':  IntegerAttr.get(self.i32Type, kargs['class_num']),
            'anchors': StringAttr.get(kargs['anchors'])
        }
        return self.buildOp(TPU_OpType.YoloDetection.value, inputOperands, [
            tensor_output_type], name=name_attr, **param)

    def add_matmul_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        name_attr = StringAttr.get(op_name)
        self.add_quant_reg(inputOperands)
        return self.buildOp(TPU_OpType.MatMul.value, inputOperands, [tensor_output_type],
                            name=name_attr, quant=self.quant_param)

    def add_square_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        name_attr = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.Square.value, inputOperands, [tensor_output_type],
                            name=name_attr, quant=self.quant_param)

    def add_quadratic_sum_op(self, op_name, inputOperands, output_tensor_shape, **kargs):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        name_attr = StringAttr.get(op_name)
        param = {}
        if 'high_precision' in kargs:
            param['high_precision'] = BoolAttr.get(kargs['high_precision'])
        if 'axis' in kargs:
            param['axis'] = IntegerAttr.get(self.i32Type, kargs['axis'])
        return self.buildOp(TPU_OpType.QuadraticSum.value, inputOperands, [tensor_output_type],
                            name=name_attr, quant=self.quant_param, **param)

    def add_broadcast_sub_op(self, op_name, inputOperands, output_tensor_shape):
        tensor_output_type = RankedTensorType.get(
            tuple(output_tensor_shape), self.get_input_type(inputOperands[0]))
        none = self.add_none_op()
        for _ in range(6 - len(inputOperands)):
            inputOperands.append(none)
        name_attr = StringAttr.get(op_name)
        return self.buildOp(TPU_OpType.BroadcastSub.value, inputOperands, [tensor_output_type],
                            name=name_attr, quant=self.quant_param)

    def add_return_op(self, Operands):
        return_op = Operation.create(
            "std.return", operands=Operands, results=[])
        self.insert_point.insert(return_op)
        return

    def print_module(self):
        mlir_format = str(self.mlir_module)
        return mlir_format

    def declare_func(self, input_type: str = "FP32"):
        self.tensor_inputs_type = list()
        self.tensor_outputs_type = list()
        if input_type == "FP32":
            for input_shape in self.input_shape_list:
                self.tensor_inputs_type.append(
                    RankedTensorType.get(input_shape, self.f32Type))
        elif input_type == "UINT8":
            for input_shape in self.input_shape_list:
                self.tensor_inputs_type.append(
                    RankedTensorType.get(input_shape, self.u8Type))

        for output_shape in self.output_shape_list:
            self.tensor_outputs_type.append(
                RankedTensorType.get(output_shape, self.f32Type))

        input_args_type = str()
        output_tensor_type = str()
        for idx, input_tensor in enumerate(self.tensor_inputs_type):
            arg = "%args{}: ".format(idx)
            input_args_type += arg + input_tensor.__str__()
            if input_tensor is not self.tensor_inputs_type[-1]:
                input_args_type += ", "

        for output_shape in self.tensor_outputs_type:
            output_tensor_type += output_shape.__str__()
            if output_shape is not self.tensor_outputs_type[-1]:
                output_tensor_type += ", "
        if len(self.tensor_outputs_type) > 1:
            output_tensor_type = "({})".format(output_tensor_type)

        tpu_func = """
            func @tpu_func({input_args_type}) -> {output_tensor_type} {{
                %0 = \"tpu.weight_file\"() {{filename = \"{weight_file}\"}} : () -> memref<10xf32>
            }}
        """.format(input_args_type=input_args_type,
                   output_tensor_type=output_tensor_type,
                   weight_file=self.output_weight_file)
        self.mlir_module = Module.parse(tpu_func, self.ctx)
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.weight_op = self.entry_block.operations[0].operation.results[0]

        self.func_args = list()
        for i in self.entry_block.arguments:
            self.func_args.append(i)
