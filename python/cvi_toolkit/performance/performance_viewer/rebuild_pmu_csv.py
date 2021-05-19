#!/usr/bin/python3
import json
import argparse
import datetime
from cvi_toolkit.utils.mlir_parser import MlirParser

def parse_origin_csv(file):
    fields = []
    records = []
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                fields = [x.strip() for x in line.split(',')]
            else:
                records.append([x.strip() for x in line.split(',')])
        return fields, records

def parse_final_mlir(file):
    parser = MlirParser(file)
    op_list = parser.get_all_ops()
    # transfor list to map
    ops = {}
    for op in op_list:
        ops[str(op.loc)] = (op.type, op.shape)
    return ops

def rebuild_pmu_csv(csv, mlir, out_csv):
    fields, records = parse_origin_csv(csv)
    ops = parse_final_mlir(mlir)
    fields.insert(1, 'op_type')
    fields.insert(2, 'op_shape')
    # dump to new csv file
    with open(out_csv, 'w') as f:
        f.write(', '.join(fields))
        f.write('\n')
        for i in range(len(records)):
            loc_ = records[i][0]
            type_, shape_ = ops[loc_]
            shape_ = [str(x) for x in shape_]
            records[i].insert(1, type_)
            records[i].insert(2, 'x'.join(shape_))
            f.write(', '.join(records[i]))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--mlir", required=True, help="final mlir file to codegen")
    parser.add_argument('-c', "--csv", required=True, help="pmu csv file")
    parser.add_argument('-o', "--out_csv", required=True, help="output rebuilt pmu csv file")
    args = parser.parse_args()
    rebuild_pmu_csv(args.csv, args.mlir, args.out_csv)
    print("rebulid pmu csv file to {}".format(args.out_csv))