#!/usr/bin/python
import argparse
import os
import re
import random
import flatbuffers
from . import cpu_op
from collections import OrderedDict


cpu_ops = (
    'softmax',
    'quant',
    'retinaface_detection',
    'preprocess',
    'transpose',
    'yolo_detection'
)


def softmaxArgsSerialize(attributes):
    builder = flatbuffers.Builder(0)
    cpu_op.SoftmaxStart(builder)
    cpu_op.SoftmaxAddAxis(builder, int(attributes['axis']))
    args = cpu_op.SoftmaxEnd(builder)
    builder.Finish(args)
    return bytearray(builder.Output())


def quantArgsSerialize(attributes):
    builder = flatbuffers.Builder(0)
    strFrom = builder.CreateString(attributes['from'])
    strTo = builder.CreateString(attributes['to'])
    cpu_op.QuantizationStart(builder)
    cpu_op.QuantizationAddFrom_(builder, strFrom)
    cpu_op.QuantizationAddTo(builder, strTo)
    cpu_op.QuantizationAddThreshold(builder, float(attributes['threshold']))
    args = cpu_op.QuantizationEnd(builder)
    builder.Finish(args)
    return bytearray(builder.Output())


def retinafaceArgsSerialize(attributes):
    builder = flatbuffers.Builder(0)
    cpu_op.RetinaFaceDetectionStart(builder)
    cpu_op.RetinaFaceDetectionAddNmsThreshold(builder, float(attributes['nms_threshold']))
    cpu_op.RetinaFaceDetectionAddConfidenceThreshold(builder, float(attributes['confidence_threshold']))
    cpu_op.RetinaFaceDetectionAddKeepTopk(builder, int(attributes['keep_topk']))
    args = cpu_op.RetinaFaceDetectionEnd(builder)
    builder.Finish(args)
    return bytearray(builder.Output())


def preprocessArgsSerialize(attributes):
    builder = flatbuffers.Builder(0)
    print(attributes)
    mean = None
    color_order = None
    if 'mean' in attributes:
        cpu_op.PreprocessStartMeanVector(builder, len(attributes['mean']))
        for m in reversed(attributes['mean']):
            builder.PrependFloat32(m)
        mean = builder.EndVector(len(attributes['mean']))
    if 'color_order' in attributes:
        cpu_op.PreprocessStartColorOrderVector(builder, len(attributes['color_order']))
        for o in reversed(attributes['color_order']):
            builder.PrependInt32(o)
        color_order = builder.EndVector(len(attributes['color_order']))
    cpu_op.PreprocessStart(builder)
    if mean:
        cpu_op.PreprocessAddMean(builder, mean)
    if color_order:
        cpu_op.PreprocessAddColorOrder(builder, color_order)
    if 'scale' in attributes:
        cpu_op.PreprocessAddScale(builder, attributes['scale'])
    if 'raw_scale' in attributes:
        cpu_op.PreprocessAddRawScale(builder, attributes['raw_scale'])
    args = cpu_op.PreprocessEnd(builder)
    builder.Finish(args)
    return bytearray(builder.Output())


def yoloArgsSerialize(attributes):
    builder = flatbuffers.Builder(0)
    cpu_op.YoloDetectionStart(builder)
    cpu_op.YoloDetectionAddNetInputH(builder, float(attributes['net_input_h']))
    cpu_op.YoloDetectionAddNetInputW(builder, float(attributes['net_input_w']))
    cpu_op.YoloDetectionAddNmsThreshold(builder, float(attributes['nms_threshold']))
    cpu_op.YoloDetectionAddObjThreshold(builder, float(attributes['obj_threshold']))
    cpu_op.YoloDetectionAddKeepTopk(builder, int(attributes['keep_topk']))
    args = cpu_op.YoloDetectionEnd(builder)
    builder.Finish(args)
    return bytearray(builder.Output())


class Tensor:
    def __init__(self, id, attributes, shape, is_weight, op_type):
        self.id = id
        self.shape, self.dtype = self.__parse_shape(shape)
        self.name = attributes['name'] if 'name' in attributes else ''
        self.offset = attributes['offset'] if 'offset' in attributes else - 1
        if 'gaddr' in attributes:
            self.offset = attributes['gaddr']
        self.is_weight = is_weight
        self.op_type = op_type
        self.overwrote = True if 'fuse_next' in attributes else False
        if not self.overwrote:
            if ('tl_store_flag' in attributes) and attributes['tl_store_flag'] == 'false':
                self.overwrote = True

    def __parse_shape(self, shape):
        array = shape.split('x')
        shape = [int(x) for x in array[:-1]]
        if len(shape) > 4:
            accumulate = 1
            for x in shape[4:]:
                accumulate *= x
            shape[:] = [shape[0], shape[1], shape[2], accumulate]
        elif len(shape) == 3:
            shape[:] = [shape[0], shape[1], shape[2], 1]
        elif len(shape) == 2:
            shape[:] = [shape[0], shape[1], 1, 1]
        elif len(shape) == 1:
            shape[:] = [shape[0], 1, 1, 1]
        dtype = array[-1]
        if dtype == 'bf16':
            dtype = 'bf16'
        elif dtype == 'i16':
            dtype = 'int16'
        elif dtype == 'i8':
            dtype = 'int8'
        else:
            dtype = 'fp32'
        return shape, dtype

    def __str__(self):
        return '{}: {} {} [{}] {} {} {}'.format(self.id, self.name, self.offset,
                                                ','.join([str(x) for x in self.shape]),
                                                self.dtype, 'weight' if self.is_weight else 'neuron', self.op_type)


class Op:
    def __init__(self, type, attributes, inputs, weights, output):
        self.type = type
        self.inputs = inputs
        self.weights = weights
        self.output = output
        self.attributes = attributes
        self.attr_serial = self.__serialize_attrs(type, attributes)

    def __serialize_attrs(self, type, attributes):
        if type == 'softmax':
            return softmaxArgsSerialize(attributes)
        elif type == 'quant':
            return quantArgsSerialize(attributes)
        elif type == 'retinaface_detection':
            return retinafaceArgsSerialize(attributes)
        elif type == 'preprocess':
            return preprocessArgsSerialize(attributes)
        elif type == 'yolo_detection':
            return yoloArgsSerialize(attributes)
        else:
            return bytearray()

    def __str__(self):
        if len(self.weights) == 0:
            return "{}: ({}) -> {}".format(self.type,
                                           ','.join(self.inputs), self.output)
        else:
            return "{}: ({}|{}) -> {}".format(self.type,
                                              ','.join(self.inputs),
                                              ','.join(self.weights), self.output)


class Function:
    tpu_func_idx = 0

    def __init__(self, ops):
        self.cpu_function = False
        self.cpu_attr_serial = None
        if len(ops) == 1 and ops[0].type in cpu_ops:
            self.cpu_function = True
            self.cpu_attr_serial = ops[0].attr_serial
            self.name = ops[0].type
        else:
            self.name = self.__tpu_func_name()
        self.ops = ops
        self.consumers = set()
        self.producers = set()
        for op in ops:
            for input in op.inputs:
                self.consumers.add(input)
            self.producers.add(op.output)
        self.inputs = self.consumers - self.producers
        self.outputs = self.producers - self.consumers

    def __tpu_func_name(self):
        chars = '0123456789abcdef'
        tag = 'tpu_{}'.format(Function.tpu_func_idx)
        for i in range(8):
            tag += chars[random.randint(0, len(chars) - 1)]
        Function.tpu_func_idx += 1
        return tag

    def add_outputs(self, other):
        intersection = other.consumers & self.producers
        self.outputs |= intersection

    def __str__(self):
        string = 'func({}) -> ({}) {}'.format(','.join(list(self.inputs)),
                                              ','.join(list(self.outputs)), " {\n")
        for op in self.ops:
            string += "  {}\n".format(op)
        string += '}\n'
        return string


class MlirParser:
    def __init__(self, file):
        self.functions = []
        self.tensor_map = OrderedDict()
        self.weights = set()
        self.name = ''
        self.__none_args = set()
        self.__has_input = False
        self.ops = self.__parse(file)
        self.inputs, self.outputs = self.__find_inputs_outputs()
        self.__split_functions()
        tensor = self.tensor_map[list(self.inputs)[0]]
        self.batch = tensor.shape[0]
        self.__omit_tensors()

    def __parse_normal_val(self, value):
        if value.find(':') != -1:
            v, t = [x.strip("'\" ") for x in value.split(':')]
            if t == 'f32':
                return float(v)
            elif t == 'i32' or t == 'i64':
                return int(v)
            else:
                return v
        return value

    def __parse_array_val(self, value):
        values = [x.strip("'\" ") for x in value.split(',')]
        ret = []
        for v in values:
            ret.append(self.__parse_normal_val(v))
        return ret

    def __parse_map_val(self, value):
        values = [x.strip("'\" ") for x in value.split(',')]
        ret = {}
        for v in values:
            k, v = [x.strip("'\" ") for x in v.split('=')]
            ret[k] = self.__parse_normal_val(v)
        return ret

    def __parse_attributes(self, attributes):
        def split_attr(attr):
            arr = []
            ele = ''
            end_char = ','
            for c in attr:
                if c == '{':
                    ele += c
                    end_char = '}'
                elif c == '}':
                    ele += c
                    end_char = ','
                elif c == '[':
                    ele += c
                    end_char = ']'
                elif c == ']':
                    ele += c
                    end_char = ','
                elif c == end_char:
                    arr.append(ele)
                    ele = ''
                else:
                    ele += c
            if ele:
                arr.append(ele)
            return arr

        _dict = {}
        attrs = [x.strip() for x in split_attr(attributes)]
        for attr in attrs:
            key, value = [x.strip("'\" ") for x in re.match(
                '^(.*)\s*=\s*(.*)', attr).groups()]
            if value.startswith('['):
                value = re.match('\[(.*)\]', value).groups()[0]
                value = self.__parse_array_val(value)
            elif value.startswith('{'):
                value = re.match('\{(.*)\}', value).groups()[0]
                value = self.__parse_map_val(value)
            else:
                value = self.__parse_normal_val(value)
            _dict[key] = value
        return _dict

    def __parse_op(self, line):
        _output, _type = re.match('(%\d+?)\s*=\s*"tpu\.(.*?)"', line).groups()
        if _type == 'none':
            self.__none_args.add(_output)
            return None
        _output, _type, _args, _attributes = re.match(
            '(%\d+?)\s*=\s*"tpu\.(.*?)"\((.*)\)\s*{(.*)}', line).groups()
        _attributes = self.__parse_attributes(_attributes)

        if _type == 'weight_file':
            self.name = re.match(
                '(.*)_\d+_[a-z0-9]+', _attributes['filename']).groups()[0]
            return None

        _shape = re.match('.*?->\s*tensor<(.*?)>', line).groups()[0]
        _is_weight = (_type == 'load_weight')

        #if _type == 'tg_int8_input' or _type == 'tg_bf16_input' or _type == 'input':
        #    self.__has_input = True
        #    self.tensor_map[_output] = Tensor(
        #        _output, _attributes, _shape, _is_weight, _type)
        #    return None
        #if not self.__has_input:
        #    return None

        self.tensor_map[_output] = Tensor(
            _output, _attributes, _shape, _is_weight, _type)
        if _is_weight:
            self.weights.add(_output)
            return None
        if _type == 'input':
            return None

        args = [x.strip() for x in _args.split(',')]
        weights = []
        inputs = []
        for arg in args:
            if arg in self.weights:
                weights.append(arg)
            elif arg not in self.__none_args:
                inputs.append(arg)
        return Op(_type, _attributes, inputs, weights, _output)

    def __omit_tensors(self):
        omits = ['tg_int8_slice']
        remove = [k for k, v in self.tensor_map.items() if v.op_type in omits]
        for k in remove: del self.tensor_map[k]

    def __remove_input_op(self, ops):
        for i in range(1, len(ops)):
            if ops[i].type not in ['tg_int8_input', 'tg_bf16_input']:
                continue
            target = None
            for z in reversed(range(0, i)):
                if ops[z].output == ops[i].inputs[0]:
                    target = ops[z]
                    break
            self.__fuse_op(target, ops[i])
            for j in range(i+1, len(ops)):
                for k in range(len(ops[j].inputs)):
                    if ops[j].inputs[k] == ops[i].inputs[0]:
                        ops[j].inputs[k] = ops[i].output
        for i in reversed(range(len(ops))):
            if ops[i].type in ['tg_int8_input', 'tg_bf16_input']:
                del ops[i]

    def __remove_reshape_op(self, ops):
        for i in range(1, len(ops)):
            if ops[i].type != 'reshape':
                continue
            target = None
            for z in reversed(range(0, i)):
                if ops[z].output == ops[i].inputs[0]:
                    target = ops[z]
                    break
            self.__fuse_op(target, ops[i], False)
            for j in range(i+1, len(ops)):
                for k in range(len(ops[j].inputs)):
                    if ops[j].inputs[k] == ops[i].inputs[0]:
                        ops[j].inputs[k] = ops[i].output
        for i in reversed(range(len(ops))):
            if ops[i].type == 'reshape':
                del ops[i]

    def __fuse_op(self, dst, src, replace_name = True):
        _dst_tensor = self.tensor_map[dst.output]
        _src_tensor = self.tensor_map[src.output]
        del self.tensor_map[dst.output]
        del self.tensor_map[src.output]
        dst.output = src.output
        _dst_tensor.id = _src_tensor.id
        _dst_tensor.shape = _src_tensor.shape
        if _src_tensor.offset != -1:
            _dst_tensor.offset = _src_tensor.offset
        if replace_name:
            _dst_tensor.name = _src_tensor.name
        self.tensor_map[_dst_tensor.id] = _dst_tensor

    def __move_cpu_op_to_close_consumers(self):
        for i in reversed(range(len(self.ops))):
            if self.ops[i].type not in cpu_ops:
                continue
            if self.ops[i].output in self.outputs:
                self.ops.insert(len(self.ops)-1, self.ops[i])
                del self.ops[i]
                continue
            for j in range(i+1, len(self.ops)):
                if self.ops[i].output in self.ops[j].inputs:
                    self.ops.insert(j, self.ops[i])
                    del self.ops[i]
                    break

    def __parse_func_args(self, line):
        args = re.match('.+?\((.*?)\)', line).groups()[0]
        args = [x.strip() for x in args.split(',')]
        for arg in args:
            name, shape = re.match('(.+?):\s?tensor<(.+?)>', arg).groups()
            self.tensor_map[name] = Tensor(name, {'name': name}, shape, False, 'arg')

    def __parse(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        lines[:] = [x.strip() for x in lines]
        start = False
        ops = []
        for line in lines:
            if re.match('^return', line):
                break
            if re.match('^func', line):
                #self.__parse_func_args(line)
                start = True
            elif start:
                op = self.__parse_op(line)
                if op:
                    ops.append(op)
        self.__remove_input_op(ops)
        self.__remove_reshape_op(ops)
        return ops

    def __find_inputs_outputs(self):
        consumers = set()
        producers = set()
        for op in self.ops:
            for input in op.inputs:
                consumers.add(input)
            producers.add(op.output)
        inputs = consumers - producers
        outputs = producers - consumers
        return inputs, outputs

    def __split_functions(self):
        groups = list()
        groups.append([])
        idx = 0
        self.__move_cpu_op_to_close_consumers()
        for op in self.ops:
            if op.type in cpu_ops:
                if len(groups[idx]) == 0:
                    groups[idx].append(op)
                else:
                    groups.append([op])
                    idx += 1
                groups.append([])
                idx += 1
            else:
                groups[idx].append(op)
        groups[:] = [group for group in groups if len(group) > 0]
        self.functions = [Function(group) for group in groups]
        for i in reversed(range(len(self.functions) - 1)):
            self.functions[i].add_outputs(self.functions[i+1])


def check_file_existence(x):
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError('{0} does not exist.'.format(x))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing mlir file.')
    parser.add_argument('--mlir', required=True,
                        help='mlir model file', type=check_file_existence)
    args = parser.parse_args()
    mlir = MlirParser(args.mlir)
    for f in mlir.functions:
        print(f)
    print('\n')
    for op in mlir.ops:
        print(op)
    print('\n')
    for _, tensor in mlir.tensor_map.items():
        print(tensor)
    print('\n')
    print(mlir.inputs)
    print(mlir.outputs)
    print(mlir.batch)
    print(mlir.name)
