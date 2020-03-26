import numpy as np
import onnx
from .model import CVI_Model
from .transform import OnnxConverter
from .build_cvimodel.build_cvimodel import CVIModel as builder
import subprocess
from pathlib import Path


def mlir_traslate(model_file, weight_file, mlirfile):
    subprocess.run(["mlir-translate", "--caffe-to-mlir", model_file,
                    "--caffemodel", weight_file,
                    "-o", mlirfile
                    ])

def mlir_opt(mlirfile, opt_mlirfile, op_info_csv):
    subprocess.run(["mlir-opt",
                    "--assign-layer-id",
                    "--print-tpu-op-info",
                    "--convert-bn-to-scale",
                    "--canonicalize",
                    "--tpu-op-info-filename", op_info_csv,
                    mlirfile,
                    "-o", opt_mlirfile
                    ])

def mlir_import_calibration(mlirfile, cali_mlirfile, threshold_table):
    subprocess.run(["mlir-opt",
                    "--import-calibration-table",
                    "--calibration-table", threshold_table,
                    mlirfile,
                    "-o", cali_mlirfile
                    ])

def mlir_tpu_quant(mlirfile, quant_mlirfile, op_info_csv):
    subprocess.run(["mlir-opt",
                    "--tpu-quant",
                    "--print-tpu-op-info",
                    "--tpu-op-info-filename", op_info_csv,
                    mlirfile,
                    "-o", quant_mlirfile
                    ])

def mlir_lower_opt(mlirfile, opt_mlirfile):
    subprocess.run(["mlir-opt",
                    "--tpu-lower",
                    mlirfile,
                    "-o", opt_mlirfile
                    ])

def mlir_gen_cvimodel(mlirfile, cvi_module):
    cmdbuf_mlir = "cmdbuf_{}".format(mlirfile)
    subprocess.run(["mlir-opt",
                    "--assign-weight-address",
                    "--tpu-weight-address-align=16",
                    "--tpu-weight-map-filename=weight_map.csv",
                    "--tpu-weight-bin-filename=weight.bin",
                    "--assign-neuron-address",
                    "--tpu-neuron-address-align=16",
                    "--tpu-neuron-map-filename=neuron_map.csv",
                    "--assign-layer-id",
                    mlirfile,
                    "-o", cmdbuf_mlir
                    ])
    subprocess.run(["mlir-translate",
                    "--mlir-to-cmdbuf",
                    cmdbuf_mlir,
                    "-o", "cmdbuf.bin"
                    ])
    model_builder = builder("weight.bin", ["cmdbuf.bin"], None, cmdbuf_mlir)
    model_builder.build(cvi_module)


def mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune=False):
    if auto_tune:
        subprocess.run(["cvi_calibration_tool",
                        mlirfile_fp32,
                        dataset,
                        "--output_file", threshold_table,
                        "--auto_tune"
                        ])
    else:
         subprocess.run(["cvi_calibration_tool",
                        mlirfile_fp32,
                        dataset,
                        "--output_file", threshold_table,
                        ])
def run_cvimodel(input_file, cvi_model, output_tensor, all_tensors=False):
     if all_tensors:
        subprocess.run(["cvi_calibration_tool",
                        "--input", input_file,
                        "--model", cvi_model,
                        "--output_file", output_tensor,
                        "--dump-all-tensors"
                        ])
    else:
        subprocess.run(["cvi_calibration_tool",
                        "--input", input_file,
                        "--model", cvi_model,
                        "--output_file", output_tensor,
                        ])
class cvinn(object):
    def __init__(self):
        pass
    def load_model(self, model_type: str, model_file: str,  mlirfile: str, weight_file: str = None):
        if model_type == 'caffe':
            if weight_file == None:
                print("No caffe weight file")
                return -1
            mlirori = "ori_{}".format(mlirfile)
            mlir_traslate(model_file, weight_file, mlirori)
            mlir_opt(mlirori, mlirfile, "{}_op_info.csv".format(model_file.split('.')[0].split('/')[-1]))
            return 0
        elif model_type == 'onnx':
            if not model_file.lower().endswith('.onnx'):
                print("{} is not end with .onnx".format(model_file))
                return -1

            onnx_model = onnx.load(model_file)
            c = OnnxConverter(model_file.split('.')[0].split('/')[-1], onnx_model, mlirfile)
            c.run()
            return 0
        else:
            print("Not support {} type, now support onnx and caffe".format(model_type))
            return -1

    def calibration(self, mlirfile_fp32: str, dataset: str, threshold_table: str, auto_tune=False):
        mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune)
        return 0

    def build_cvimodel(self, mlirfile_fp32: str, cvimodel: str, threshold_table: str, mlirfile_int8: str = None,
                    quant_method: str = "perchannel", cmd_buf: str=None, quant_info=None):
        int8_op_csv = "{}_op_info_int8.csv".format(mlirfile_fp32.split('.')[0].split('/')[-1])
        cali_mlir = "cali_{}".format(mlirfile_fp32)
        mlir_import_calibration(mlirfile_fp32, cali_mlir, threshold_table)

        quant_mlir = "quant_{}".format(mlirfile_fp32)
        mlir_tpu_quant(cali_mlir, quant_mlir, int8_op_csv)

        tg_mlir = "tg_{}".format(mlirfile_fp32)
        mlir_lower_opt(quant_mlir, tg_mlir)
        mlir_gen_cvimodel(tg_mlir, cvimodel)
        return 0

    def tpu_simulation(self, input_file, cvimodel, output_tensor, all_tensors=None):
        run_cvimodel(input_file, cvimodel, output_tensor, all_tensors)
        return 0

    def inference(self, model_type: str, input_data: np.ndarray, model_file=None, weight_file=None, mlirfile=None, all_tensors:str = None):
        net = CVI_Model()
        print(model_type)
        net.load_model(model_type, model_file=model_file, weight_file=weight_file, mlirfile=mlirfile)
        out = net.inference(input_data)
        if all_tensors!=None:
            net.get_all_tensor(all_tensors)
        return out

    def cleanup(self):
        for clean_file in ["*.mlir", "*.bin", "*.csv", "*.cvimodel", "*.npz"]:
            for p in Path(".").glob(clean_file):
                p.unlink()
        return 0

    def dump_quant_info(self):
        return 0
