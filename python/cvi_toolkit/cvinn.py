import numpy as np
import onnx
from .model.ModelFactory import ModelFactory
from .transform import OnnxConverter
from .build_cvimodel import CVIModel as builder
from .calibration.kld_calibrator import KLD_Calibrator_v2
from .calibration.tuner import Tuner_v2
import subprocess
import logging
from pathlib import Path


logger = logging.getLogger(__name__)
std_log_flag = False

if std_log_flag:
    std_output_flag = dict() # empty
else:
    std_output_flag = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}

def checkReturnValue(ret_val: int, func: str):
    if ret_val == 0:
        logger.debug("{} run success".format(func))
    else:
        logger.error("error occured: {}, func: {}".format(ret_val, func))

def mlir_traslate(model_file, weight_file, mlirfile):
    ret = subprocess.run(["mlir-translate", "--caffe-to-mlir", model_file,
                    "--caffemodel", weight_file,
                    "-o", mlirfile
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir_traslate")
    return r_code

def mlir_opt(mlirfile, opt_mlirfile, op_info_csv):
    ret = subprocess.run(["mlir-opt",
                    "--assign-layer-id",
                    "--convert-bn-to-scale",
                    "--canonicalize",
                    "--print-tpu-op-info",
                    "--tpu-op-info-filename", op_info_csv,
                    mlirfile,
                    "-o", opt_mlirfile
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-opt")
    return r_code


def mlir_import_calibration(mlirfile, cali_mlirfile, threshold_table):
    ret = subprocess.run(["mlir-opt",
                    "--import-calibration-table",
                    "--calibration-table", threshold_table,
                    mlirfile,
                    "-o", cali_mlirfile
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-opt, import-calibration-table")
    return r_code

def mlir_tpu_quant(mlirfile, quant_mlirfile, op_info_csv):
    ret = subprocess.run(["mlir-opt",
                    "--tpu-quant",
                    "--print-tpu-op-info",
                    "--tpu-op-info-filename", op_info_csv,
                    mlirfile,
                    "-o", quant_mlirfile
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-opt, mlir_tpu_quant")
    return r_code


def mlir_lower_opt(mlirfile, opt_mlirfile):
    ret = subprocess.run(["mlir-opt",
                    "--tpu-lower",
                    mlirfile,
                    "-o", opt_mlirfile
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-opt, mlir_lower_opt")
    return r_code

def mlir_gen_cvimodel(mlirfile, cvi_module):
    cmdbuf_mlir = "cmdbuf_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
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
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-opt, mlir_to_tg_cmdbuf")
    if r_code != 0:
        return r_code

    ret = subprocess.run(["mlir-translate",
                    "--mlir-to-cmdbuf",
                    cmdbuf_mlir,
                    "-o", "cmdbuf.bin"
                    ], **std_output_flag)
    r_code = ret.returncode
    checkReturnValue(r_code, "mlir-translate, mlir_gen_cmdbuf")
    if r_code != 0:
        return r_code

    model_builder = builder("weight.bin", ["cmdbuf.bin"], None, cmdbuf_mlir, False)
    model_builder.build(cvi_module)
    return 0


def mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune=False):
    if auto_tune:
        subprocess.run(["cvi_calibration_tool",
                        mlirfile_fp32,
                        dataset,
                        "--output_file", threshold_table,
                        "--auto_tune"
                        ], **std_output_flag)
    else:
         subprocess.run(["cvi_calibration_tool",
                        mlirfile_fp32,
                        dataset,
                        "--output_file", threshold_table,
                        ], **std_output_flag)


def run_cvimodel(input_file, cvi_model, output_tensor, all_tensors=True):

    cmd = ["model_runner",
            "--input", input_file,
            "--model", cvi_model,
            "--output", output_tensor,]
    if all_tensors:
        cmd.append("--dump-all-tensors")
    ret = subprocess.run(cmd)
    r_code = ret.returncode
    checkReturnValue(r_code, "model_runner")
    return r_code

class cvinn(object):
    def __init__(self):
        pass
    def convert_model(self, model_type: str, model_file: str,  mlirfile: str, weight_file: str = None, tpu_op_info=None):
        if model_type == 'caffe':
            if weight_file == None:
                print("No caffe weight file")
                return -1
            mlirori = "ori_{}".format(mlirfile)
            mlir_traslate(model_file, weight_file, mlirori)
            if tpu_op_info:
                mlir_opt(mlirori, mlirfile, tpu_op_info)
            else:
                mlir_opt(mlirori, mlirfile, "{}_op_info.csv".format(model_file.split('.')[0].split('/')[-1]))
            return 0
        elif model_type == 'onnx':
            if not model_file.lower().endswith('.onnx'):
                print("{} is not end with .onnx".format(model_file))
                return -1
            mlirori = "ori_{}".format(mlirfile)
            onnx_model = onnx.load(model_file)
            c = OnnxConverter(model_file.split('.')[0].split('/')[-1], onnx_model, mlirori)
            c.run()
            if tpu_op_info:
                mlir_opt(mlirori, mlirfile, tpu_op_info)
            else:
                mlir_opt(mlirori, mlirfile, "{}_op_info.csv".format(model_file.split('.')[0].split('/')[-1]))
            return 0
        else:
            print("Not support {} type, now support onnx and caffe".format(model_type))
            return -1

    def calibration(self, mlirfile_fp32: str, dataset: str, threshold_table: str, pre_func, input_num, histogram_bin_num, auto_tune=False):
        # mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune)
        cvi_model = ModelFactory()
        cvi_model.load_model('mlir', None, mlirfile=mlirfile_fp32)
        calitor = KLD_Calibrator_v2(dataset,pre_func, cvi_model.model, input_num=input_num, histogram_bin_num=histogram_bin_num)
        calitor.do_calibration(threshold_table=threshold_table)
        if auto_tune == True:
            tuner = Tuner_v2(mlirfile_fp32, threshold_table, dataset, preprocess_func=pre_func)
            tuner.run_tune()
        return 0

    def build_cvimodel(self, mlirfile_fp32: str, cvimodel: str, threshold_table: str, mlirfile_int8: str = None,
                    quant_method: str = "perchannel", cmd_buf: str=None, quant_info=None):
        if quant_info:
            int8_op_csv = quant_info
        else:
            int8_op_csv = "{}_op_info_int8.csv".format(mlirfile_fp32.split('.')[0].split('/')[-1])
        cali_mlir = "cali_{}".format(mlirfile_fp32)
        ret = mlir_import_calibration(mlirfile_fp32, cali_mlir, threshold_table)
        if ret != 0:
            logger.error("mlir_import_callibration failed")
            exit(-1)

        if mlirfile_int8:
            quant_mlir = mlirfile_int8
        else:
            quant_mlir = "quant_{}".format(mlirfile_fp32)
        ret = mlir_tpu_quant(cali_mlir, quant_mlir, int8_op_csv)
        if ret != 0:
            logger.error("mlir_tpu_quant failed")
            exit(-1)

        tg_mlir = "tg_{}".format(mlirfile_fp32)
        ret = mlir_lower_opt(quant_mlir, tg_mlir)
        if ret != 0:
            logger.error("mlir_lower_opt failed")

        ret = mlir_gen_cvimodel(tg_mlir, cvimodel)
        if ret != 0:
            logger.error("mlir_gen_cvimodel failed")
            exit(-1)
        return 0

    def tpu_simulation(self, input_file, cvimodel, output_tensor, all_tensors=True):
        ret = run_cvimodel(input_file, cvimodel, output_tensor, all_tensors)
        return ret

    def inference(self, model_type: str, input_npz: str, model_file=None, weight_file=None, mlirfile=None, all_tensors:str = None):
        net = ModelFactory()

        net.load_model(model_type, model_file=model_file, weight_file=weight_file, mlirfile=mlirfile)
        input_data = np.load(input_npz)['input']
        out = net.inference(input_data)

        if all_tensors!=None:
            net.get_all_tensor(input_data, all_tensors)
        return out

    def cleanup(self):
        for clean_file in ["*.mlir", "*.bin", "*.csv", "*.npz"]:
            for p in Path(".").glob(clean_file):
                p.unlink()
        return 0

    def dump_quant_info(self):
        return 0
