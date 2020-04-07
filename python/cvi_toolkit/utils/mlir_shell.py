import logging
import subprocess

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
                    "--print-tpu-op-info",
                    "--convert-bn-to-scale",
                    "--canonicalize",
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