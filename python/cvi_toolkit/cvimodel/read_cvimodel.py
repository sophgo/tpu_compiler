#!/usr/bin/python3
import argparse
import os
import os.path
import csv
import time
import flatbuffers
import hashlib
import struct
import cvi_model as cm


dtype_map = {
    cm.DType.FP32: 'fp32',
    cm.DType.INT32: 'int32',
    cm.DType.UINT32: 'uint32',
    cm.DType.BF16: 'bf16',
    cm.DType.INT16: 'int16',
    cm.DType.UINT16: 'uint16',
    cm.DType.INT8: 'int8',
    cm.DType.UINT8: 'uint8'
}

section_type_map = {
    cm.SectionType.WEIGHT: 'weight',
    cm.SectionType.CMDBUF: 'cmdbuf',
    cm.SectionType.FUNC_X86: 'func_x86',
    cm.SectionType.FUNC_AARCH64: 'aarch64'
}


def show_model_sections(model, binary_buf):
    print('\nSections:')
    print('{:<6}{:<10}{:<20}{:<12}{:<12}{}'.format(
        'ID', 'TYPE', 'NAME', 'SIZE', 'OFFSET', 'MD5'))
    for i in range(model.SectionsLength()):
        section = model.Sections(i)
        sec_type = section_type_map[section.Type()]
        sec_name = section.Name().decode()
        sec_size = section.Size()
        sec_offset = section.Offset()
        m = hashlib.md5()
        m.update(binary_buf[sec_offset: sec_offset + sec_size])
        md5 = m.hexdigest()
        print('{:<6}{:<10}{:<20}{:<12}{:<12}{}'.format(i,
                                                       sec_type, sec_name,
                                                       sec_size, sec_offset, md5))


def show_programs(model):
    for i in range(model.ProgramsLength()):
        program = model.Programs(i)
        batch_num = program.BatchNum()
        neuron_size = program.NeuronSize()
        print("\nPrograms <{}>".format(i))

        # general info of program
        print("    {:<12}: {}".format('batch_num', batch_num))
        print("    {:<12}: {}".format('neuron_size', neuron_size))
        input_tensors = [program.InputTensors(j).decode()
                         for j in range(program.InputTensorsLength())]
        print("    {:<12}: {}".format('inputs', ','.join(input_tensors)))
        output_tensors = [program.OutputTensors(j).decode()
                          for j in range(program.OutputTensorsLength())]
        print("    {:<12}: {}".format('outputs', ','.join(output_tensors)))
        print("    {:<12}:".format('tensor_map'))
        print("      {:<5}{:<12}{:<6}{:<16}{}".format(
            'ID', 'OFFSET', 'TYPE', 'SHAPE', 'NAME'))
        # tensor map of program
        for j in range(program.TensorMapLength()):
            tensor = program.TensorMap(j)
            shape = [str(tensor.Shape().Dim(k))
                     for k in range(tensor.Shape().DimLength())]
            print("      {:<5}{:<12}{:<6}{:<16}{}".format(
                  j, tensor.Offset(), dtype_map[tensor.Dtype()],
                  ','.join(shape), tensor.Name().decode()))

        # show routines(cpu/tpu) info
        for j in range(program.RoutinesLength()):
            routine = program.Routines(j)
            _str = '  \troutine#{}'.format(j)
            routine_type = 'tpu' if routine.Type() == cm.RoutineType.TPU else 'cpu'
            tensor_inputs = [routine.InTensors(k).decode(
            ) for k in range(routine.InTensorsLength())]
            tensor_outputs = [routine.OutTensors(
                k).decode() for k in range(routine.OutTensorsLength())]
            _str += '\t{}\t{}\t{}'.format(routine_type,
                                          ','.join(tensor_inputs),
                                          ','.join(tensor_outputs))
            if routine_type == 'tpu':
                tpu_routine = routine.TpuRoutine()
                _str += '\tcmdbuf_section:{}'.format(
                    tpu_routine.CmdbufSection().decode())
            else:
                cpu_routine = routine.CpuRoutine()
                _str += '\tfunction_section:{}'.format(
                    cpu_routine.FunctionSection().decode())
            print(_str)


def show_weight_map(model):
    print("\nWeightMap")
    print("    {:<5}{:<12}{:<6}{:<16}{:<16}{}".format(
            'ID', 'OFFSET', 'SIZE', 'TYPE', 'SHAPE', 'NAME'))
    for i in range(model.WeightMapLength()):
        weight = model.WeightMap(i);
        shape = [str(weight.Shape().Dim(k))
                     for k in range(weight.Shape().DimLength())]
        print("    {:<5}{:<12}{:<6}{:<16}{:<16}{}".format(
                  i, weight.Offset(), weight.Size(), dtype_map[weight.Type()],
                  ','.join(shape), weight.Name().decode()))


def extract_programs(model, binary_buf, output_prefix):
    sections = {}
    for i in range(model.SectionsLength()):
        section = model.Sections(i)
        sections[section.Name().decode()] = (
            section.Size(), section.Offset())

    for i in range(model.ProgramsLength()):
        program = model.Programs(i)
        batch_num = program.BatchNum()

        # dump tensor map
        map_file = '{}_program_{}_neuron_map.csv'.format(
            output_prefix, batch_num)
        with open(map_file, 'w') as f:
            for j in range(program.TensorMapLength()):
                tensor = program.TensorMap(j)
                shape = [str(tensor.Shape().Dim(k))
                         for k in range(tensor.Shape().DimLength())]
                f.write('{}, 0x{},{},{}\n'.format(tensor.Name().decode(),
                                                  tensor.Offset, dtype_map[tensor.Dtype()],
                                                  ','.join(shape)))
        print("dump neuron map to {}".format(map_file))

        # dump cmdbuf
        for j in range(program.RoutinesLength()):
            routine = program.Routines(j)
            if routine.Type() != cm.RoutineType.TPU:
                continue
            name = routine.TpuRoutine().CmdbufSection().decode()
            cmdbuf_file = '{}_program_{}_cmdbuf_{}_{}.bin'.format(
                output_prefix, batch_num, j, name)
            with open(cmdbuf_file, 'wb') as f:
                size, offset = sections[name]
                f.write(binary_buf[offset:offset + size])
            print("dump cmdbuf to {}".format(cmdbuf_file))

        # dump weight
        weight_file = '{}_weight.bin'.format(output_prefix)
        with open(weight_file, 'wb') as f:
            size, offset = sections['weight']
            f.write(binary_buf[offset:offset+size])
        print("dump weight to {}".format(weight_file))


def load_model(model_file):
    with open(model_file, 'rb') as f:
        data = f.read()
        header = struct.unpack('<8sLBB16s2s', data[:32])
        tag, body_size, major_v, minor_v, md5, _ = header
        model = cm.Model.GetRootAsModel(data[32: 32 + body_size], 0)
        binary_buf = data[32 + body_size:]

    major = model.Version().Major_()
    minor = model.Version().Minor_()
    sub_minor = model.Version().SubMinor()
    print("Version: {}.{}.{}".format(major, minor, sub_minor))
    print("{} Build at {}".format(
        model.Name().decode(), model.BuildTime().decode()))
    return model, binary_buf


def check_file_existence(x):
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError('{0} does not exist.'.format(x))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program read CVI model.')
    parser.add_argument('--extract', required=False,
                        help='extract cmd, weight, neuron map')
    parser.add_argument('model_file', help='*.cm file',
                        type=check_file_existence)
    args = parser.parse_args()

    model, binary_buf = load_model(args.model_file)
    show_model_sections(model, binary_buf)
    show_programs(model)
    show_weight_map(model)
    if args.extract:
        output_prefix = os.path.splitext(args.model_file)[0]
        extract_programs(model, binary_buf, output_prefix)
