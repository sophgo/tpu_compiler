import logging
import os
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
    std_output_flag = {'stdout': subprocess.DEVNULL,
                       'stderr': subprocess.STDOUT}


def get_chip_name():
    runchip = os.environ.get('SET_CHIP_NAME', None)
    if not runchip:
        log.warning(
            "no found SET_CHIP_NAME environment value, set 183x as default")
        return "cv183x"
    return runchip


def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        logger.debug("{} run success".format(func))
    else:
        logger.error("[!Error]cmd: {}".format(" ".join(ret.args)))
        logger.error("error occured: {}, func: {}\nmsg: {}".format(
            ret.returncode, func, ret))


def mlir_opt(mlirfile, opt_mlirfile, op_info_csv=None, chip=None):
    if not chip:
        chip = get_chip_name()
    if op_info_csv == None:
        ret = subprocess.run(["tpuc-opt",
                              "--assign-chip-name",
                              "--chipname={}".format(chip),
                              "--convert-bn-to-scale",
                              "--canonicalize",
                              mlirfile,
                              "-o", opt_mlirfile
                              ], **std_output_flag)
    else:
        ret = subprocess.run(["tpuc-opt",
                              "--assign-chip-name",
                              "--chipname={}".format(chip),
                              "--convert-bn-to-scale",
                              "--canonicalize",
                              "--print-tpu-op-info",
                              "--tpu-op-info-filename", op_info_csv,
                              mlirfile,
                              "-o", opt_mlirfile
                              ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt")
    return ret.returncode


def mlir_import_calibration(mlirfile, cali_mlirfile, threshold_table):
    ret = subprocess.run(["tpuc-opt",
                          "--import-calibration-table",
                          "--calibration-table", threshold_table,
                          mlirfile,
                          "-o", cali_mlirfile
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, import-calibration-table")
    return ret.returncode


def mlir_tpu_quant(mlirfile, quant_mlirfile, op_info_csv, quant_mode="int8"):
    command = [
        "tpuc-opt",
        "--tpu-quant",
        "--print-tpu-op-info",
        "--tpu-op-info-filename", op_info_csv,
        mlirfile,
        "-o", quant_mlirfile
    ]
    if quant_mode == "bf16":
        command.insert(2, "--quant-full-bf16")
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, mlir_tpu_quant")
    return ret.returncode

def mlir_int8_quant(mlirfile, int8_mlirfile):
    command = [
        "tpuc-opt",
        "--import-calibration-table",
        "--calibration-table", threshold_table,
        "--tpu-quant",
        mlirfile,
        "-o", int8_mlirfile
    ]
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, mlir_int8_quant")
    return ret.returncode

def mlir_mix_quant(mlirfile, mix_mlirfile,
                   calibration_table, mix_precision_table):
    command = [
        "tpuc-opt",
        "--import-calibration-table",
        "--calibration-table", calibration_table,
        "--assign-chip-name",
        "--tpu-quant",
        "--quant-int8-mix-bf16-layers-from-file", mix_precision_table,
        mlirfile,
        "-o", mix_mlirfile
    ]
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, mix-precision")
    return ret.returncode

def mlir_lower_opt(mlirfile, opt_mlirfile):
    lower_mlir = "lw.mlir"
    ret = subprocess.run(["tpuc-opt",
                          "--tpu-lower",
                          "--reorder-op",
                          mlirfile,
                          "-o", lower_mlir,
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, mlir_lower_opt")
    if ret.returncode != 0:
        return ret.returncode
    opt_mlir = "opt.mlir"
    ret = subprocess.run(["tpuc-opt",
                          "--tg-fuse-leakyrelu",
                          "--conv-ic-alignment",
                          lower_mlir,
                          "-o", opt_mlirfile,
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, fuse")
    if ret.returncode != 0:
        return ret.returncode
    return 0


def mlir_gen_cvimodel(mlirfile, cvi_module):

    int8_addr = "int8_addr_{}".format(mlirfile)
    ret = subprocess.run(["tpuc-opt",
                          "--assign-weight-address",
                          "--tpu-weight-address-align=16",
                          "--tpu-weight-map-filename=weight_map.csv",
                          "--tpu-weight-bin-filename=weight.bin",
                          "--assign-neuron-address",
                          "--tpu-neuron-memory-reuse",
                          "--tpu-neuron-address-align=64",
                          "--tpu-neuron-map-filename=neuron_map.csv",
                          mlirfile,
                          "-o", int8_addr
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, int8_addr")
    if ret.returncode != 0:
        return ret.returncode

    int8_addr_func = "int8_addr_func_{}".format(mlirfile)

    ret = subprocess.run(["tpuc-opt",
                          "--divide-ops-to-func",
                          int8_addr,
                          "-o", int8_addr_func
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, int8_addr_func")
    if ret.returncode != 0:
        return ret.returncode

    ret = subprocess.run(["tpuc-translate",
                          "--mlir-to-cvimodel",
                          "--weight-file=weight.bin",
                          int8_addr_func,
                          "-o", cvi_module
                          ], **std_output_flag)
    checkReturnValue(ret, "tpuc-translate, mlir_gen_cmdbuf")
    if ret.returncode != 0:
        return ret.returncode

    return 0


def mlir_build_cvimodel_no_opt(mlirfile, cvi_model):
    """
        only build cvi model
    """
    addr_mlir = "addr_{}".format(mlirfile)
    command = ["tpuc-opt",
               "--assign-weight-address",
               "--tpu-weight-address-align=16",
               "--tpu-weight-map-filename=weight_map.csv",
               "--tpu-weight-bin-filename=weight.bin",
               "--assign-neuron-address",
               "--tpu-neuron-address-align=64",
               "--tpu-neuron-map-filename=neuron_map.csv",
               mlirfile,
               "-o", addr_mlir
               ]
    if std_log_flag:
        logger.debug(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, int8_addr")
    if ret.returncode != 0:
        return ret.returncode

    fucn_tl_lw = "tl_lw_memopt_func_{}".format(mlirfile)
    # func to tensor
    command = ["tpuc-opt",
               "--divide-ops-to-func",
               addr_mlir,
               "-o", fucn_tl_lw
               ]
    if std_log_flag:
        logger.debug(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, divide-ops-to-func")
    if ret.returncode != 0:
        return ret.returncode

    command = ["tpuc-translate",
               "--mlir-to-cvimodel",
               "--weight-file=weight.bin",
               fucn_tl_lw,
               "-o", cvi_model
               ]

    if std_log_flag:
        logger.info(command)
    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-translate, mlir-to-cvimodel")
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


def gen_bf16_mlir(mlir_src, mlir_target, bf16_layer_table, op_info_csv):
    chip = get_chip_name()
    command = ["tpuc-opt",
               "--assign-chip-name",
               "--chipname={}".format(chip),
               "--tpu-quant",
               "--quant-int8-mix-bf16-layers-from-file", bf16_layer_table,
               "--print-tpu-op-info",
               "--tpu-op-info-filename", op_info_csv,
               mlir_src,
               "-o", mlir_target
               ]

    if std_log_flag:
        logger.info(command)

    ret = subprocess.run(command, **std_output_flag)
    checkReturnValue(ret, "tpuc-opt, --quant-int8-mix-bf16-layers-from-file")
    if ret.returncode != 0:
        return ret.returncode
    return 0


def run_cvimodel(input_file, cvi_model, output_tensor, all_tensors=True,
                 batch_size=1):

    cmd = ["model_runner",
           "--input", input_file,
           "--model", cvi_model,
           "--batch-num", str(batch_size),
           "--output", output_tensor]
    if all_tensors:
        cmd.append("--dump-all-tensors")

    logger.info(cmd)
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "model_runner")
    return ret.returncode
