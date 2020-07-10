import logging
import subprocess
from .log_setting import setup_logger

import subprocess
import logging
from pathlib import Path

logger = setup_logger('root')

std_log_flag = logger.level <= logging.DEBUG

if std_log_flag:
    std_output_flag = {'capture_output': True}
else:
    std_output_flag = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.STDOUT}

def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        logger.debug("{} run success".format(func))
    else:
        logger.error("error occured: {}, func: {}\nmsg: {}".format(ret.returncode, func, ret))

def mlir_translate(model_file, weight_file, mlirfile, batch_size=1):

    ret = subprocess.run(["mlir-translate", "--caffe-to-mlir", model_file,
                    "--caffemodel", weight_file,
                    "-o", mlirfile
                    ], **std_output_flag)
    # add multibatch fail
    # "--static-batchsize",batch_size,

    checkReturnValue(ret, "mlir_translate")
    return ret.returncode

def mlir_opt(mlirfile, opt_mlirfile, op_info_csv):
    ret = subprocess.run(["mlir-opt",
                    "--assign-layer-id",
                    "--assign-chip-name",
                    "--chipname=cv183x",
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


def mlir_tpu_quant(mlirfile, quant_mlirfile, op_info_csv, quant_mode="int8"):
    command = [
        "mlir-opt",
        "--tpu-quant",
        "--print-tpu-op-info",
        "--tpu-op-info-filename", op_info_csv,
        mlirfile,
        "-o", quant_mlirfile
    ]
    if quant_mode == "bf16":
        command.insert(2, "--quant-full-bf16")
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "mlir-opt, mlir_tpu_quant")
    return ret.returncode


def mlir_lower_opt(mlirfile, opt_mlirfile):
    lower_mlir = "lw.mlir"
    ret = subprocess.run(["mlir-opt",
                    "--tpu-lower",
                    "--reorder-op",
                    mlirfile,
                    "-o", lower_mlir,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, mlir_lower_opt")
    if ret.returncode != 0:
        return ret.returncode
    opt_mlir = "opt.mlir"
    ret = subprocess.run(["mlir-opt",
                    "--tg-fuse-leakyrelu",
                    "--conv-ic-alignment",
                    lower_mlir,
                    "-o", opt_mlirfile,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, fuse")
    if ret.returncode != 0:
        return ret.returncode
    return 0

def mlir_gen_cvimodel(mlirfile, cvi_module):

    int8_addr = "int8_addr_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--assign-weight-address",
                    "--tpu-weight-address-align=16",
                    "--tpu-weight-map-filename=weight_map.csv",
                    "--tpu-weight-bin-filename=weight.bin",
                    "--assign-neuron-address",
                    "--tpu-neuron-memory-reuse",
                    "--tpu-neuron-address-align=16",
                    "--tpu-neuron-map-filename=neuron_map.csv",
                    mlirfile,
                    "-o", int8_addr
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, int8_addr")
    if ret.returncode != 0:
        return ret.returncode

    int8_addr_func = "int8_addr_func_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--divide-ops-to-func",
                    int8_addr,
                    "-o", int8_addr_func
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, int8_addr_func")
    if ret.returncode != 0:
        return ret.returncode

    ret = subprocess.run(["mlir-translate",
                    "--mlir-to-cvimodel",
                    "--weight-file=weight.bin",
                    int8_addr_func,
                    "-o", cvi_module
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-translate, mlir_gen_cmdbuf")
    if ret.returncode != 0:
        return ret.returncode

    return 0


def mlir_build_cvimodel_no_opt(mlirfile, cvi_model):
    """
        only build cvi model
    """
    addr_mlir = "addr_{}".format(mlirfile)
    command = ["mlir-opt",
               "--assign-weight-address",
               "--tpu-weight-address-align=16",
               "--tpu-weight-map-filename=weight_map.csv",
               "--tpu-weight-bin-filename=weight.bin",
               "--assign-neuron-address",
               "--tpu-neuron-address-align=16",
               "--tpu-neuron-map-filename=neuron_map.csv",
               mlirfile,
               "-o", addr_mlir
               ]
    if std_log_flag:
        logger.debug(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "mlir-opt, int8_addr")
    if ret.returncode != 0:
        return ret.returncode

    fucn_tl_lw = "tl_lw_memopt_func_{}".format(mlirfile)
    # func to tensor
    command = ["mlir-opt",
               "--divide-ops-to-func",
               addr_mlir,
               "-o", fucn_tl_lw
               ]
    if std_log_flag:
        logger.debug(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "mlir-opt, divide-ops-to-func")
    if ret.returncode != 0:
        return ret.returncode

    command = ["mlir-translate",
               "--mlir-to-cvimodel",
               "--weight-file=weight.bin",
               fucn_tl_lw,
               "-o", cvi_model
               ]

    if std_log_flag:
        logger.info(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "mlir-translate, mlir-to-cvimodel")
    if ret.returncode != 0:
        return ret.returncode

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

def gen_bf16_mlir(mlir_src, mlir_target, bf16_layer_table):
    command = ["mlir-opt",
               "--tpu-quant",
               "--quant-int8-mix-bf16-layers-from-file", bf16_layer_table,
               mlir_src,
               "-o", mlir_target
               ]

    if std_log_flag:
        logger.info(command)

    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "mlir-opt, --quant-int8-mix-bf16-layers-from-file")
    if ret.returncode != 0:
        return ret.returncode

def run_cvimodel(input_file, cvi_model, output_tensor, all_tensors=True):

    cmd = ["model_runner",
            "--input", input_file,
            "--model", cvi_model,
            "--batch-num", "1",
            "--output", output_tensor]
    if all_tensors:
        cmd.append("--dump-all-tensors")

    logger.info(cmd)
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "model_runner")
    return ret.returncode
