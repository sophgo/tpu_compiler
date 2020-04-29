#!/usr/bin/python3
import argparse
import os
import re
import random
import flatbuffers
import functools
from . import cpu_op
from collections import OrderedDict



# 'not is_cpu_quant' which means quant for mix precision belowing to TPU OP
def is_cpu_quant(op, cpu_ops):
    r = op.type in cpu_ops

    # NONE imply FP32
    if op.type == "quant" and hasattr(op, 'attributes'):
        param = op.attributes['param']
        attr_from = param.get('from', "NOT_SET")
        attr_to = param.get('to', "NOT_SET")
        if attr_from == "NONE" or attr_to == "NONE":
            # quant/dequant FP32
            pass
        else:
            # mix precision, pass to tpu
            r = False

    return r

class Tensor:
    def __init__(self, id, attributes, shape, is_weight, op_type):
        self.id = id
        self.shape, self.dtype, self.size = self.__parse_shape(shape)
        self.name = attributes['name'] if 'name' in attributes else ''
        self.offset = attributes['offset'] if 'offset' in attributes else - 1
        if 'gaddr' in attributes:
            self.offset = attributes['gaddr']
        self.is_weight = is_weight
        self.op_type = op_type
        self.overwrote = True if 'fuse_next' in attributes else False
        if not self.overwrote:
            if ('buffer_reused' in attributes) and attributes['buffer_reused'] == True:
                self.overwrote = True
            if ('tl_store_flag' in attributes) and attributes['tl_store_flag'] == False:
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
        size = functools.reduce(lambda x, y: x * y, shape)
        dtype = array[-1]
        if dtype == 'bf16':
            dtype = 'bf16'
            size *= 2
        elif dtype == 'i16':
            dtype = 'int16'
            size *= 2
        elif dtype == 'i8':
            dtype = 'int8'
        else:
            dtype = 'fp32'
            size *= 4
        return shape, dtype, size

    def __str__(self):
        return '{}: {} {} [{}] {} {} {}'.format(self.id, self.name, self.offset,
                                                ','.join([str(x) for x in self.shape]),
                                                self.dtype, 'weight' if self.is_weight else 'neuron', self.op_type)


class Op:
    cpu_ops = []
    def __init__(self, type, attributes, inputs, weights, output):
        self.inputs = inputs
        self.weights = weights
        self.output = output
        self.attributes = attributes
        self.type = type
        self.packed_attr = self.__pack_attributes(attributes)
        if self.type == 'generic_cpu':
            self.type = self.attributes['operation_name'].split('.')[-1]
            Op.cpu_ops.append(self.type)

    def __pack_attributes(self, attributes):
        if self.type != "generic_cpu":
            return bytearray()
        builder = flatbuffers.Builder(0)
        param = attributes['param']
        packed_attrs = []
        for k, val in param.items():
            key = builder.CreateString(k)
            if type(val) == bool:
                cpu_op.BoolAttrStart(builder)
                cpu_op.BoolAttrAddKey(builder, key)
                cpu_op.BoolAttrAddValue(builder, val)
                attr = cpu_op.BoolAttrEnd(builder)
                cpu_op.AttributeStart(builder)
                cpu_op.AttributeAddBoolAttr(builder, attr)
                packed_attrs.append(cpu_op.AttributeEnd(builder))
            elif type(val) == int:
                cpu_op.IntAttrStart(builder)
                cpu_op.IntAttrAddKey(builder, key)
                cpu_op.IntAttrAddValue(builder, val)
                attr = cpu_op.IntAttrEnd(builder)
                cpu_op.AttributeStart(builder)
                cpu_op.AttributeAddIntAttr(builder, attr)
                packed_attrs.append(cpu_op.AttributeEnd(builder))
            elif type(val) == float:
                cpu_op.FloatAttrStart(builder)
                cpu_op.FloatAttrAddKey(builder, key)
                cpu_op.FloatAttrAddValue(builder, val)
                attr = cpu_op.FloatAttrEnd(builder)
                cpu_op.AttributeStart(builder)
                cpu_op.AttributeAddFloatAttr(builder, attr)
                packed_attrs.append(cpu_op.AttributeEnd(builder))
            elif type(val) == str:
                v = builder.CreateString(val)
                cpu_op.StrAttrStart(builder)
                cpu_op.StrAttrAddKey(builder, key)
                cpu_op.StrAttrAddValue(builder, v)
                attr = cpu_op.StrAttrEnd(builder)
                cpu_op.AttributeStart(builder)
                cpu_op.AttributeAddStrAttr(builder, attr)
                packed_attrs.append(cpu_op.AttributeEnd(builder))
            elif type(val) == list:
                if type(val[0]) == int:
                    cpu_op.IntArrayAttrStartValueVector(builder, len(val))
                    for v in reversed(val):
                        builder.PrependInt32(v)
                    value = builder.EndVector(len(val))
                    cpu_op.IntArrayAttrStart(builder)
                    cpu_op.IntArrayAttrAddKey(builder, key)
                    cpu_op.IntArrayAttrAddValue(builder, value)
                    attr = cpu_op.IntArrayAttrEnd(builder)
                    cpu_op.AttributeStart(builder)
                    cpu_op.AttributeAddIntArrayAttr(builder, attr)
                    packed_attrs.append(cpu_op.AttributeEnd(builder))
                elif type(val[0]) == float:
                    cpu_op.FloatArrayAttrStartValueVector(builder, len(val))
                    for v in reversed(val):
                        builder.PrependFloat32(v)
                    value = builder.EndVector(len(val))
                    cpu_op.FloatArrayAttrStart(builder)
                    cpu_op.FloatArrayAttrAddKey(builder, key)
                    cpu_op.FloatArrayAttrAddValue(builder, value)
                    attr = cpu_op.FloatArrayAttrEnd(builder)
                    cpu_op.AttributeStart(builder)
                    cpu_op.AttributeAddFloatArrayAttr(builder, attr)
                    packed_attrs.append(cpu_op.AttributeEnd(builder))
                else:
                    raise Exception("unsupported {} list".format(type(val[0])))
            else:
                raise Exception("unsupported type {}".format(type(val)))
        cpu_op.ParameterStartAttributesVector(builder, len(packed_attrs))
        for attr in reversed(packed_attrs):
            builder.PrependUOffsetTRelative(attr)
        attrs = builder.EndVector(len(packed_attrs))
        cpu_op.ParameterStart(builder)
        cpu_op.ParameterAddAttributes(builder, attrs)
        args = cpu_op.ParameterEnd(builder)
        builder.Finish(args)
        return bytearray(builder.Output())

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
        self.packed_attr = None
        if len(ops) == 1 and is_cpu_quant(ops[0], Op.cpu_ops):
            self.cpu_function = True
            self.packed_attr = ops[0].packed_attr
            self.name = ops[0].type
        else:
            self.name = self.__tpu_func_name()
        self.ops = ops
        self.consumers = set()
        self.producers = set()
        for op in ops:
            for input in op.inputs:
                self.consumers.add(input)
                if self.cpu_function and op.weights:
                    for weight in op.weights:
                        self.consumers.add(weight)
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
        #for op in self.ops:
        #    print(op)
        #print('\n')
        self.inputs, self.outputs = self.__find_inputs_outputs()
        self.__split_functions()
        #for f in self.functions:
        #    print(f)
        #print("\n")
        #print('self.intputs', self.inputs)
        #print('self.outputs', self.outputs)
        self.neuron_size = self.__calc_neuron_size()
        tensor = self.tensor_map[list(self.inputs)[0]]
        self.batch = tensor.shape[0]
        self.__omit_tensors()
        self.__sort_tensor_map()

    def __calc_neuron_size(self):
        neuron_size = 0
        for _, tensor in self.tensor_map.items():
            if tensor.is_weight:
                continue
            if tensor.offset != -1:
                tail = tensor.offset + tensor.size
                neuron_size = tail if neuron_size < tail else neuron_size
        return neuron_size

    def __sort_tensor_map(self):
        layer_tensor = OrderedDict()
        weight_tensor = OrderedDict()

        for name, tensor in self.tensor_map.items():
            if tensor.is_weight:
                weight_tensor[name] = tensor
            else:
                layer_tensor[name] = tensor
        layer_list = sorted(layer_tensor.items(), key=lambda obj:obj[1].offset)
        weight_list = sorted(weight_tensor.items(), key=lambda obj:obj[1].offset)

        self.tensor_map.clear()
        for iterm in layer_list:
            self.tensor_map[iterm[0]] = layer_tensor[iterm[0]]
        for iterm in weight_list:
            self.tensor_map[iterm[0]] = weight_tensor[iterm[0]]

    def __parse_normal_val(self, value):
        if value.find(':') != -1:
            v, t = [x.strip("'\" ") for x in value.split(':')]
            if t == 'f32':
                return float(v)
            elif t == 'i32' or t == 'i64':
                return int(v)
            else:
                return v
        elif value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            return value.strip('" ')

    def __parse_array_val(self, value):
        values = value.split(',')
        ret = []
        for v in values:
            ret.append(self.__parse_normal_val(v))
        return ret

    def __parse_attributes(self, attributes):
        stops = [',']
        array = []
        idx = 0
        array.append('')
        for char in attributes:
            if char == stops[-1]:
                del stops[-1]
            if len(stops) == 0:
                array.append('')
                idx += 1
                stops.append(',')
                continue
            if char != ' ':
                array[idx] += char
            if char == '{':
                stops.append('}')
            elif char == '[':
                stops.append(']')
        attrs = {}
        for a in array:
            k, v = a.split('=', 1)
            if v[0] == '{':
                attrs[k] = self.__parse_attributes(v[1:-1])
            elif v[0] == '[':
                attrs[k] = self.__parse_array_val(v[1:-1])
            else:
                attrs[k] = self.__parse_normal_val(v)
        return attrs

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
        _is_weight = (_type == 'load_weight' or _type == 'tl_lg_load_coeff')

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
            for j in range(0, len(ops)):
                if j == i:
                    continue
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
            if not is_cpu_quant(self.ops[i], Op.cpu_ops):
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
            if is_cpu_quant(op, Op.cpu_ops):
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
