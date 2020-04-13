import logging
import subprocess
from ..build_cvimodel import CVIModel as builder

logger = logging.getLogger(__name__)


std_log_flag = False

if std_log_flag:
    std_output_flag = {'capture_output': True}
else:
    std_output_flag = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}

def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        logger.debug("{} run success".format(func))
    else:
        logger.error("error occured: {}, func: {}\nmsg: {}".format(ret.returncode, func, ret))

def mlir_translate(model_file, weight_file, mlirfile):
    ret = subprocess.run(["mlir-translate", "--caffe-to-mlir", model_file,
                    "--caffemodel", weight_file,
                    "-o", mlirfile
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir_translate")
    return ret.returncode

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
    checkReturnValue(ret, "mlir-opt")
    return ret.returncode


def mlir_import_calibration(mlirfile, cali_mlirfile, threshold_table):
    ret = subprocess.run(["mlir-opt",
                    "--import-calibration-table",
                    "--calibration-table", threshold_table,
                    mlirfile,
                    "-o", cali_mlirfile
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, import-calibration-table")
    return ret.returncode

def mlir_tpu_quant(mlirfile, quant_mlirfile, op_info_csv):
    ret = subprocess.run(["mlir-opt",
                    "--tpu-quant",
                    "--print-tpu-op-info",
                    "--tpu-op-info-filename", op_info_csv,
                    mlirfile,
                    "-o", quant_mlirfile
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, mlir_tpu_quant")
    return ret.returncode


def mlir_lower_opt(mlirfile, opt_mlirfile):
    ret = subprocess.run(["mlir-opt",
                    "--tpu-lower",
                    mlirfile,
                    "-o", opt_mlirfile
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, mlir_lower_opt")
    return ret.returncode

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
                    "--convert-cpu-op",
                    mlirfile,
                    "-o", cmdbuf_mlir
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, mlir_to_tg_cmdbuf")
    if ret.returncode != 0:
        return ret.returncode

    ret = subprocess.run(["mlir-translate",
                    "--mlir-to-cmdbuf",
                    cmdbuf_mlir,
                    "-o", "cmdbuf.bin"
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-translate, mlir_gen_cmdbuf")
    if ret.returncode != 0:
        return ret.returncode

    model_builder = builder("weight.bin", ["cmdbuf.bin"], None, None, cmdbuf_mlir, False)
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
    checkReturnValue(ret, "model_runner")
    return ret.returncode