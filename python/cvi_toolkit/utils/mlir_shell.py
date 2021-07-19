import os
import subprocess
import logging
from pathlib import Path
from .log_setting import setup_logger

logger = setup_logger('root')

std_log_flag = logger.level <= logging.DEBUG
if std_log_flag:
    std_output_flag = {'stdout': subprocess.STDPIPE,
                       'stderr': subprocess.STDOUT}
else:
    std_output_flag = {'stdout': subprocess.DEVNULL,
                       'stderr': subprocess.STDOUT}

def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        logger.debug("{} run success".format(func))
    else:
        logger.error("[!Error]cmd: {}".format(" ".join(ret.args)))
        logger.error("error occured: {}, func: {}\nmsg: {}".format(
            ret.returncode, func, ret))

def mlir_opt(mlirfile, opt_mlirfile, op_info_csv):
    ret = subprocess.run(["tpuc-opt",
                            "--fuse-sigmoid-mul-to-swish",
                            "--convert-bn-to-scale",
                            "--convert-clip-to-relu6",
                            "--canonicalize",
                            "--fuse-relu",
                            "--print-tpu-op-info",
                            "--tpu-op-info-filename", op_info_csv,
                            mlirfile,
                            "-o", opt_mlirfile
                            ], **std_output_flag)
    checkReturnValue(ret, "tpuc-opt")
    return ret.returncode

def mlir_quant(fp32_model, quanted_model, chip_name, op_order_csv,
               all_bf16=False, calib_table=None, mix_table=None):
    cmd = ["tpuc-opt",
           "--assign-chip-name",
           "--chipname", chip_name]
    if all_bf16:
        cmd.extend(["--tpu-quant", "--quant-full-bf16"])
    else:
        assert(calib_table)
        cmd.extend(["--import-calibration-table",
                    "--calibration-table", calib_table,
                    "--tpu-quant"])
        if mix_table and mix_table != '-':
            cmd.extend(['--quant-int8-mix-bf16-layers-from-file', mix_table])

    cmd.extend(["--print-tpu-op-info",
                "--tpu-op-info-filename", op_order_csv,
                fp32_model, "-o", quanted_model])
    logger.debug(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "model_runner")
    return ret.returncode

def mlir_add_preprocess(quanted_mlir, new_mlir, pixel_format, aligned_input=False):
    cmd = ["tpuc-opt",
           "--add-tpu-preprocess",
           "--pixel_format", pixel_format,
           "--canonicalize"]
    if aligned_input:
        cmd.append("--input_aligned=true")
    cmd.extend([quanted_mlir, "-o", new_mlir])
    logger.info(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "mlir_add_preprocess")
    return ret.returncode

def mlir_to_cvimodel(quanted_model, cvimodel,
                     dequant_results_to_fp32=True,
                     expose_bf16_inputs=False,
                     compress_weight=True,
                     append_weight=False):
    cmd = ["mlir_to_cvimodel.sh",
           "-i", quanted_model, "-o", cvimodel,
           "--dequant-results-to-fp32",
           str(dequant_results_to_fp32).lower(),
           "--expose-bf16-inputs",
           str(expose_bf16_inputs).lower(),
           "--compress-weight",
           str(compress_weight).lower(),
           "--append-weight",
           str(append_weight).lower()]
    logger.info(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "mlir_to_cvimodel")
    return ret.returncode

def run_cvimodel(input_file, cvi_model, output_tensor, all_tensors=True):
    cmd = ["model_runner",
           "--input", input_file,
           "--model", cvi_model,
           "--output", output_tensor]
    if all_tensors:
        cmd.append("--dump-all-tensors")

    logger.info(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "model_runner")
    return ret.returncode

def mlir_inference(mlir_model, input_npz, out_npz, all_tensor_npz):
    cmd = [
        "tpuc-interpreter", mlir_model,
        "--tensor-in", input_npz,
        "--dump-all-tensor", all_tensor_npz]
    if out_npz:
        cmd.extend(["--tensor-out", out_npz])
    logger.info(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "tpuc interpreter")
    return ret.returncode

def fp32_blobs_compare(a_npz, b_npz, op_order, tolerance,
                       dequant=False, excepts=None,
                       show_detail=True, mix_precision=False):
    cmd = [
        "cvi_npz_tool.py", "compare", a_npz, b_npz,
        "--op_info", op_order,
        "--tolerance", tolerance]
    if dequant:
        cmd.extend(["--dequant", "--stats_int8_tensor"])
    if excepts:
        cmd.extend(["--except", excepts])
    if mix_precision:
        cmd.extend(['--int8_tensor_close', '0'])
    if show_detail:
        cmd.append('-vv')
    logger.info(" ".join(cmd))
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "compare")
    return ret.returncode
