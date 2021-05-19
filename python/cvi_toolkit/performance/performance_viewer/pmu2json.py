#!/usr/bin/python3
import json
import argparse
import datetime

TPU_FREQ_1835=650

def parse_pmu_line(line):
    pmu_info={}
    x = line.split(",")
    pmu_info['cat'] = "Final MLIR OP Profile"
    pmu_info['pid'] = "0"
    pmu_info['tid'] = "TDMA Op" if (x[0].strip() != "4") else "TIU Op"
    pmu_info['dur'] = float(x[6].strip())*1000/TPU_FREQ_1835
    pmu_info['ts'] = float(x[5].strip())*1000/TPU_FREQ_1835
    pmu_info['ph'] = "X"
    pmu_info['name'] = "OP_" + x[8].strip()

    args = {}
    if x[0].strip() != "4":
        args["OP_Type"] = x[9].strip()
        args["src_address"] = x[10].strip()
        args["dst_address"] = x[11].strip()
        args["trans_fmt"] = x[12].strip()
        args["transpose_md"] = x[13].strip()
        args["cmd_id"] = x[14].strip()
        args["wait_id_tpu"] = x[15].strip()
        args["dst_h_stride"] = x[16].strip()
        args["dst_c_stride_low"] = x[17].strip()
        args["dst_n_stride"] = x[18].strip()
        args["src_h_stride"] = x[19].strip()
        args["src_c_stride_low"] = x[20].strip()
        args["src_n_stride"] = x[21].strip()
        args["dst_c"] = x[22].strip()
        args["dst_h"] = x[25].strip()
        args["dst_w"] = x[24].strip()
        args["src_n"] = x[28].strip()
        args["src_c"] = x[23].strip()
        args["src_h"] = x[27].strip()
        args["src_w"] = x[26].strip()

    pmu_info['args'] = args
    print(pmu_info)
    return pmu_info


class pmu2json:
    def __init__(self, in_file, out_json):
        self.in_file= in_file
        self.out_json = out_json

    def convert(self):
        with open(self.in_file, "r") as f:
            lines = f.readlines()
        pmu_infos=[]
        for line in lines[1:]:
            pmu_info = parse_pmu_line(line)
            pmu_infos.append(pmu_info)
        pmu_json={}
        pmu_json["traceEvents"] = pmu_infos

        with open("out.json", "w+") as outf:
            json.dump(pmu_json, outf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='%(prog)s pmu_dec.csv pmu_dec.json',
        epilog='Takes the pmu description file and produces a pmu json file')

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    p = pmu2json(args.input, args.output)
    p.convert()

    print(args)