#!/usr/bin/env python3
import argparse
import os
import os.path
import csv
import time
import flatbuffers
import hashlib
import struct
import functools
import random

from cvi_toolkit.cvimodel import mlir_parser
from cvi_toolkit.cvimodel import cvi_model as cm

# Version shall be in schema
MAJOR_VER = 1
MIN_VER = 0
SUBMIN_VER = 0

DFT_CHIP = 'cv1835'
DFT_MODEL_FILENAME = 'default.cm'

dtype_map = {
  'fp32': cm.DType.FP32,
  'int32': cm.DType.INT32,
  'uint32': cm.DType.UINT32,
  'bf16': cm.DType.BF16,
  'int16': cm.DType.INT16,
  'uint16': cm.DType.UINT16,
  'int8': cm.DType.INT8,
  'uint8': cm.DType.UINT8
}

dtype_size_map = {
  'fp32': 4,
  'int32': 4,
  'uint32': 4,
  'bf16': 2,
  'int16': 2,
  'uint16': 2,
  'int8': 1,
  'uint8': 1,
}


class Section:
  def __init__(self, builder, name, type, file):
    self.builder = builder
    self.name = name
    self.type = type
    self.file = file

  def build(self, model):
    section_name = self.builder.CreateString(self.name)
    cm.SectionStart(self.builder)
    cm.SectionAddType(self.builder, self.type)
    cm.SectionAddName(self.builder, section_name)
    if self.file:
      with open(self.file, 'rb') as f:
        data = bytearray(f.read())
      cm.SectionAddSize(self.builder, len(data))
      cm.SectionAddOffset(self.builder, model.binary_buf_offset)
      model.binary_buf += data
      model.binary_buf_offset += len(data)
    else:
      cm.SectionAddSize(self.builder, 0)
      cm.SectionAddOffset(self.builder, 0)
    return cm.SectionEnd(self.builder)


class Routine:
  def __init__(self, builder, name, inputs, outputs):
    self.builder = builder
    self.name = name
    self.inputs = inputs
    self.outputs = outputs
    self.section = None

  def _build_inputs_outputs(self):
    input_list = [self.builder.CreateString(x) for x in self.inputs]
    output_list = [self.builder.CreateString(x) for x in self.outputs]
    cm.RoutineStartInTensorsVector(
      self.builder, len(input_list))
    for i in reversed(input_list):
      self.builder.PrependUOffsetTRelative(i)
    tensor_input_list = self.builder.EndVector(len(input_list))

    cm.RoutineStartOutTensorsVector(
      self.builder, len(output_list))
    for o in reversed(output_list):
      self.builder.PrependUOffsetTRelative(o)
    tensor_output_list = self.builder.EndVector(len(output_list))
    return tensor_input_list, tensor_output_list

  def build(self, buider):
    pass


class CpuRoutine(Routine):
  def __init__(self, builder, name, inputs, outputs, func_args):
    Routine.__init__(self, builder, name, inputs, outputs)
    self.func_args = func_args

  def build(self):
    tensor_input_list, tensor_output_list = self._build_inputs_outputs()
    function_args = None
    if self.func_args:
      cm.CpuRoutineStartFunctionArgsVector(self.builder, len(self.func_args))
      for _byte in reversed(self.func_args):
        self.builder.PrependByte(_byte)
      function_args = self.builder.EndVector(len(self.func_args))

    function_name = self.builder.CreateString(self.name)
    cm.CpuRoutineStart(self.builder)
    cm.CpuRoutineAddFunctionSection(self.builder, function_name)
    if function_args:
      cm.CpuRoutineAddFunctionArgs(self.builder, function_args)
    cpu_routine = cm.CpuRoutineEnd(self.builder)

    cm.RoutineStart(self.builder)
    cm.RoutineAddType(self.builder, cm.RoutineType.CPU)
    cm.RoutineAddInTensors(self.builder, tensor_input_list)
    cm.RoutineAddOutTensors(self.builder, tensor_output_list)
    cm.RoutineAddCpuRoutine(self.builder, cpu_routine)
    return cm.RoutineEnd(self.builder)


class TpuRoutine(Routine):
  def __init__(self, builder, name, inputs, outputs, cmdbuf):
    Routine.__init__(self, builder, name, inputs, outputs)
    self.section = Section(builder, name, cm.SectionType.CMDBUF, cmdbuf)

  def build(self):
    tensor_input_list, tensor_output_list = self._build_inputs_outputs()
    section_name = self.builder.CreateString(self.name)
    cm.TpuRoutineStart(self.builder)
    cm.TpuRoutineAddCmdbufSection(self.builder, section_name)
    tpu_routine = cm.TpuRoutineEnd(self.builder)

    cm.RoutineStart(self.builder)
    cm.RoutineAddType(self.builder, cm.RoutineType.TPU)
    cm.RoutineAddInTensors(self.builder, tensor_input_list)
    cm.RoutineAddOutTensors(self.builder, tensor_output_list)
    cm.RoutineAddTpuRoutine(self.builder, tpu_routine)
    return cm.RoutineEnd(self.builder)


class Program:
  def __init__(self, builder, cmdbufs, mlir, verbose):
    self.builder = builder
    self.mlir = mlir
    self.cmdbufs = cmdbufs
    self.sections = []
    self.verbose = verbose

  def __build_dim_vector(self, dims):
    cm.ShapeStartDimVector(self.builder, len(dims))
    for dim in reversed(dims):
      self.builder.PrependInt64(dim)
    shape_dim = self.builder.EndVector(len(dims))
    cm.ShapeStart(self.builder)
    cm.ShapeAddDim(self.builder, shape_dim)
    return cm.ShapeEnd(self.builder)

  def __tensor_id2name(self, ids):
    names = []
    for id in ids:
      names.append(self.mlir.tensor_map[id].name)
    return names

  def __build_neuron_map(self):
    max_neuron_size = 0
    neuron_map = []
    for tensor in self.mlir.tensor_map.values():
      if tensor.is_weight:
        continue
      tensor_name = tensor.name
      tensor_offset = tensor.offset
      dtype = tensor.dtype
      n, c, h, w = tensor.shape
      tensor_dtype = dtype_map[dtype]
      tensor_shape = self.__build_dim_vector((n, c, h, w))
      tensor_stride = self.__build_dim_vector((c * h * w, h * w, w, 1))
      tensor_size = functools.reduce(
        lambda x, y: x * y, (n, c, h, w)) * dtype_size_map[dtype]
      if tensor_offset != -1:
        max_neuron_size = (tensor_offset + tensor_size) if max_neuron_size < (
          tensor_offset + tensor_size) else max_neuron_size
      if self.verbose:
        print(max_neuron_size, tensor_name, tensor_offset,
            tensor_size, dtype, n, c, h, w, dtype_size_map[dtype])

      cm.QuantInfoStart(self.builder)
      cm.QuantInfoAddType(self.builder, 0)
      cm.QuantInfoAddMaxValue(self.builder, 1.0)
      cm.QuantInfoAddMinValue(self.builder, 0.0)
      cm.QuantInfoAddZeroPoint(self.builder, 0)
      cm.QuantInfoAddQscale(self.builder, 1.0)
      tensor_quant = cm.QuantInfoEnd(self.builder)

      tensor_name = self.builder.CreateString(tensor_name)
      cm.TensorStart(self.builder)
      cm.TensorAddTensorId(self.builder, 0)
      cm.TensorAddName(self.builder, tensor_name)
      cm.TensorAddOffset(self.builder, tensor_offset)
      cm.TensorAddDtype(self.builder, tensor_dtype)
      cm.TensorAddShape(self.builder, tensor_shape)
      cm.TensorAddStride(self.builder, tensor_stride)
      cm.TensorAddQuant(self.builder, tensor_quant)
      cm.TensorAddOverwrote(self.builder, tensor.overwrote)
      neuron_map.append(cm.TensorEnd(self.builder))

    cm.ProgramStartTensorMapVector(self.builder, len(neuron_map))
    for tensor in reversed(neuron_map):
      self.builder.PrependUOffsetTRelative(tensor)
    program_neuron_map = self.builder.EndVector(len(neuron_map))
    return program_neuron_map, max_neuron_size

  def __build_inputs_outputs(self):
    input_tensors = []
    for i in self.__tensor_id2name(self.mlir.inputs):
      input_tensors.append(self.builder.CreateString(i))
    cm.ProgramStartInputTensorsVector(
      self.builder, len(input_tensors))
    for itensor in reversed(input_tensors):
      self.builder.PrependUOffsetTRelative(itensor)
    program_input_tensors = self.builder.EndVector(len(input_tensors))

    output_tensors = []
    for o in self.__tensor_id2name(self.mlir.outputs):
      output_tensors.append(self.builder.CreateString(o))
    cm.ProgramStartOutputTensorsVector(
      self.builder, len(output_tensors))
    for otensor in reversed(output_tensors):
      self.builder.PrependUOffsetTRelative(otensor)
    program_output_tensors = self.builder.EndVector(len(output_tensors))
    return program_input_tensors, program_output_tensors

  def __build_routines(self, cmdbufs):
    idx = 0
    routines = []
    for func in self.mlir.functions:
      if self.verbose:
        print(func)
      inputs = self.__tensor_id2name(func.inputs)
      outputs = self.__tensor_id2name(func.outputs)
      if func.cpu_function:
        routine = CpuRoutine(self.builder, func.name, inputs, outputs, func.packed_attr)
      else:
        routine = TpuRoutine(self.builder, func.name, inputs, outputs, cmdbufs[idx])
        self.sections.append(routine.section)
        idx += 1

      routines.append(routine.build())

    cm.ProgramStartRoutinesVector(
      self.builder, len(routines))
    for func in reversed(routines):
      self.builder.PrependUOffsetTRelative(func)
    return self.builder.EndVector(len(routines))

  def build(self):
    program_neuron_map, neuron_size = self.__build_neuron_map()
    program_input_tensors, program_output_tensors = self.__build_inputs_outputs()
    program_routines = self.__build_routines(self.cmdbufs)

    cm.ProgramStart(self.builder)
    cm.ProgramAddBatchNum(self.builder, self.mlir.batch)
    cm.ProgramAddNeuronSize(self.builder, neuron_size)
    cm.ProgramAddInputTensors(self.builder, program_input_tensors)
    cm.ProgramAddOutputTensors(self.builder, program_output_tensors)
    cm.ProgramAddTensorMap(self.builder, program_neuron_map)
    cm.ProgramAddRoutines(self.builder, program_routines)
    return cm.ProgramEnd(self.builder)


class CVIModel:
  def __init__(self, weight, cmdbufs, so_path, so_name, mlir_file, verbose):
    self.mlir = mlir_parser.MlirParser(mlir_file)
    self.builder = flatbuffers.Builder(1024)
    self.program = Program(self.builder, cmdbufs, self.mlir, verbose)
    self.weight = weight
    self.cmdbufs = cmdbufs
    self.so_path = so_path
    self.so_name = so_name
    self.model = bytearray()
    self.binary_buf = bytearray()
    self.binary_buf_offset = 0
    self.verbose = verbose

  def __random_tag(self, length):
    chars = '0123456789abcdef'
    ret = ''
    for i in range(length):
      ret += chars[random.randint(0, len(chars) - 1)]
    return ret

  def __store_model(self, model, binary_buf, output_file):
    model_size = len(model)
    model_tag = 'CviModel'
    payload = model + binary_buf
    m = hashlib.md5()
    m.update(payload)
    md5 = m.digest()
    header = struct.pack('<8sLBB16s2s', model_tag.encode(), len(
      model), MAJOR_VER, MIN_VER, md5, "AA".encode())
    if len(header) != 32:
      raise Exception('header size != 32 bytes')
    if not output_file:
      ts = time.strftime('%Y%m%d%H%M%S', time.localtime())
      output_file = '{}_{}.cm'.format(self.mlir.name, ts, random)
    with open(output_file, 'wb') as f:
      f.write(header)
      f.write(payload)

  def __build_dim_vector(self, dims):
    cm.ShapeStartDimVector(self.builder, len(dims))
    for dim in reversed(dims):
      self.builder.PrependInt64(dim)
    shape_dim = self.builder.EndVector(len(dims))
    cm.ShapeStart(self.builder)
    cm.ShapeAddDim(self.builder, shape_dim)
    return cm.ShapeEnd(self.builder)

  def __build_weight_map(self):
    weight_map = []
    for tensor in self.mlir.tensor_map.values():
      if not tensor.is_weight:
        continue
      name = self.builder.CreateString(tensor.name)
      n,c,h,w = tensor.shape
      tensor_shape = self.__build_dim_vector((n, c, h, w))
      tensor_type = dtype_map[tensor.dtype]
      cm.WeightStart(self.builder)
      cm.WeightAddName(self.builder, name)
      cm.WeightAddOffset(self.builder, tensor.offset)
      cm.WeightAddShape(self.builder, tensor_shape)
      cm.WeightAddType(self.builder, tensor_type)
      cm.WeightAddSize(self.builder, n*c*h*w)
      weight_map.append(cm.WeightEnd(self.builder))
    cm.ModelStartWeightMapVector(self.builder, len(weight_map))
    for w in reversed(weight_map):
      self.builder.PrependUOffsetTRelative(w)
    return self.builder.EndVector(len(weight_map))

  def __build_sections(self):
    sections = []
    weight_section = Section(self.builder, 'weight', cm.SectionType.WEIGHT, self.weight)
    sections.append(weight_section.build(self))

    if self.so_path and self.so_name:
      x86_so = "{}/{}_x86.so".format(self.so_path, self.so_name)
      if os.path.isfile(x86_so):
        print("find custom plugin:{}".format(x86_so))
        x86_so_section = Section(self.builder, "custom", cm.SectionType.FUNC_X86, x86_so)
        sections.append(x86_so_section.build(self))
      arm_so = "{}/{}_arm64.so".format(self.so_path, self.so_name)
      if os.path.isfile(x86_so):
        print("find custom plugin:{}".format(arm_so))
        arm_so_section = Section(self.builder, "custom", cm.SectionType.FUNC_AARCH64, arm_so)
        sections.append(arm_so_section.build(self))

    for section in self.program.sections:
      sections.append(section.build(self))

    cm.ModelStartSectionsVector(self.builder, len(sections))
    for s in reversed(sections):
      self.builder.PrependUOffsetTRelative(s)
    return self.builder.EndVector(len(sections))

  def build(self, output_file):
    now = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
    model_time = self.builder.CreateString(now)
    model_name = self.builder.CreateString(self.mlir.name)

    model_program = self.program.build()
    cm.ModelStartProgramsVector(self.builder, 1)
    self.builder.PrependUOffsetTRelative(model_program)
    model_programs = self.builder.EndVector(1)
    model_weight_map = self.__build_weight_map()
    model_sections = self.__build_sections()

    cm.ModelStart(self.builder)
    model_version = cm.CreateVersion(
      self.builder, MAJOR_VER, MIN_VER, SUBMIN_VER)
    cm.ModelAddVersion(self.builder, model_version)
    cm.ModelAddName(self.builder, model_name)
    cm.ModelAddBuildTime(self.builder, model_time)
    cm.ModelAddWeightMap(self.builder, model_weight_map)
    cm.ModelAddPrograms(self.builder, model_programs)
    cm.ModelAddSections(self.builder, model_sections)
    model = cm.ModelEnd(self.builder)
    self.builder.Finish(model)
    self.__store_model(bytearray(self.builder.Output()),
               self.binary_buf, output_file)


def check_file_existence(x):
  if not os.path.isfile(x):
    raise argparse.ArgumentTypeError('{0} does not exist.'.format(x))
  return x


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This program create CVI model.')
  parser.add_argument('--mlir', required=True, help='mlir model file', type=check_file_existence)
  parser.add_argument('--cmdbuf', required=True, help='combuf files, separated by commas', type=check_file_existence)
  parser.add_argument('--weight', required=True, help='weight file', type=check_file_existence)
  parser.add_argument('--chip', required=False, default=DFT_CHIP, help='Chip, default cv1835')
  parser.add_argument('--output', required=True, help='output cvi.model file')
  parser.add_argument('--plugin_dir', required=False, default=None)
  parser.add_argument('--plugin_name', required=False, default=None)
  parser.add_argument('--verbose', required=False, type=bool, default=False)
  args = parser.parse_args()

  model = CVIModel(args.weight, args.cmdbuf.split(','), args.plugin_dir, args.plugin_name, args.mlir, args.verbose)
  model.build(args.output)
