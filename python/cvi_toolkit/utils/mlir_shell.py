import logging
import subprocess
from ..build_cvimodel import CVIModel as builder
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

def mlir_translate(model_file, weight_file, mlirfile,batch_size=1):

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
    lower_mlir = "lw.mlir"
    ret = subprocess.run(["mlir-opt",
                    "--tpu-lower",
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
                    "-o", opt_mlir,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, fuse")
    if ret.returncode != 0:
        return ret.returncode

     # function argument lower to MemRefType
    memref_mlir = "memref_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--convert-func-to-memref",
                    opt_mlir,
                    "-o", memref_mlir,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, func-to-memref")
    if ret.returncode != 0:
        return ret.returncode

    # op lower to MemRefType
    tg_opt_memref = "tg_opt_memref_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--convert-tg-op-to-memref",
                    memref_mlir,
                    "-o", tg_opt_memref,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, tg_opt_memref")
    if ret.returncode != 0:
        return ret.returncode

    # memory space w/ global memory reuse
    tg_opt_op_memref_addr = "tg_opt_op_memref_addr_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--enable-reuse-global-memory=true",
                    "--assign-neuron-address-memref",
                    "--tpu-neuron-address-align-memref=16",
                    "--tpu-neuron-map-filename-memref=neuron_map_memref_reused.csv",
                    tg_opt_memref,
                    "-o", tg_opt_op_memref_addr,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, tg_opt_op_memref_addr")
    if ret.returncode != 0:
        return ret.returncode

    # tg op back to TensorType
    tg_opt_op_tensor_addr = "tg_opt_op_tensor_addr_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--convert-tg-op-to-tensor",
                    tg_opt_op_memref_addr,
                    "-o", tg_opt_op_tensor_addr,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, tg_opt_op_tensor_addr")
    if ret.returncode != 0:
        return ret.returncode

    # function argument back to TensorType
    # tg_opt_addr = "tg_opt_addr_{}".format(mlirfile)
    ret = subprocess.run(["mlir-opt",
                    "--convert-func-to-tensor",
                    tg_opt_op_tensor_addr,
                    "-o", opt_mlirfile,
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, tg_opt_addr")
    if ret.returncode != 0:
        return ret.returncode

    return ret.returncode

def mlir_gen_cvimodel(mlirfile, cvi_module):
    
    int8_addr = "int8_addr_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--assign-weight-address",
                    "--tpu-weight-address-align=16",
                    "--tpu-weight-map-filename=weight_map.csv",
                    "--tpu-weight-bin-filename=weight.bin",
                    "--assign-neuron-address",
                    "--tpu-neuron-address-align=16",
                    "--tpu-neuron-map-filename=neuron_map.csv",
                    "--convert-cpu-op",
                    mlirfile,
                    "-o", int8_addr
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, int8_addr")
    if ret.returncode != 0:
        return ret.returncode

    deepfusion_tg2tl_la = "deep_fusion_tg2tl_la_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--deep-fusion-tg2tl-la",
                    int8_addr,
                    "-o", deepfusion_tg2tl_la
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, deepfusion_tg2tl_la")
    if ret.returncode != 0:
        return ret.returncode
    
    deep_fusion_tl_la2lw = "deep_fusion_tl_la2lw_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--deep-fusion-tl-la2lw",
                    deepfusion_tg2tl_la,
                    "-o", deep_fusion_tl_la2lw
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, deep_fusion_tl_la2lw")
    if ret.returncode != 0:
        return ret.returncode

    convert_func_to_memref = "convert_func_to_memref_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--convert-func-to-memref",
                    deep_fusion_tl_la2lw,
                    "-o", convert_func_to_memref
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, convert_func_to_memref")
    if ret.returncode != 0:
        return ret.returncode


    convert_tg_op_to_memref = "convert_tg_op_to_memref_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--convert-tg-op-to-memref",
                    convert_func_to_memref,
                    "-o", convert_tg_op_to_memref
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, convert_tg_op_to_memref")
    if ret.returncode != 0:
        return ret.returncode

    enable_reuse_global_memory = "enable_reuse_global_memory_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--enable-reuse-global-memory=true",
                    "--assign-neuron-address-memref",
                    "--tpu-neuron-address-align-memref=16",
                    "--tpu-neuron-map-filename-memref=neuron_map_memopt.csv",
                    convert_tg_op_to_memref,
                    "-o", enable_reuse_global_memory
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, enable_reuse_global_memory")

    if ret.returncode != 0:
        return ret.returncode      

    convert_tg_op_to_tensor = "convert_tg_op_to_tensor_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--convert-tg-op-to-tensor",
                    enable_reuse_global_memory,
                    "-o", convert_tg_op_to_tensor
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, enable_reuse_global_memory")

    if ret.returncode != 0:
        return ret.returncode   

    int8_tl_lw_memopt = "int8_tl_lw_memopt_{}".format(mlirfile)

    ret = subprocess.run(["mlir-opt",
                    "--convert-func-to-tensor",
                    "--convert-cpu-op",
                    convert_tg_op_to_tensor,
                    "-o", int8_tl_lw_memopt
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-opt, int8_tl_lw_memopt")

    if ret.returncode != 0:
        return ret.returncode    


    ret = subprocess.run(["mlir-translate",
                    "--mlir-to-cmdbuf",
                    int8_tl_lw_memopt,
                    "-o", "cmdbuf.bin"
                    ], **std_output_flag)
    checkReturnValue(ret, "mlir-translate, mlir_gen_cmdbuf")
    if ret.returncode != 0:
        return ret.returncode
    
    model_builder = builder("weight.bin", ["cmdbuf.bin"], None, None, int8_tl_lw_memopt, False)
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