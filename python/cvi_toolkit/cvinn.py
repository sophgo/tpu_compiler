import numpy as np
import onnx
from .model.ModelFactory import ModelFactory
from .transform import OnnxConverter
from .build_cvimodel import CVIModel as builder
from .calibration.kld_calibrator import KLD_Calibrator_v2
from .calibration.tuner import Tuner_v2
from .utils.mlir_shell import checkReturnValue, mlir_translate, mlir_opt, \
                                mlir_import_calibration, mlir_tpu_quant, mlir_lower_opt, mlir_gen_cvimodel, \
                                mlir_calibration, run_cvimodel
from .utils.log_setting import setup_logger

import subprocess
import logging
from pathlib import Path


logger = setup_logger('root')

class cvinn(object):
    def __init__(self):
        pass
    def convert_model(self, model_type: str, model_file: str,  mlirfile: str, weight_file: str = None, tpu_op_info=None,batch_size=1):
        if model_type == 'caffe':
            if weight_file == None:
                print("No caffe weight file")
                return -1
            mlirori = "ori_{}".format(mlirfile)
            ret = mlir_translate(model_file, weight_file, mlirori,batch_size=batch_size)
            if ret != 0:
                logger.error("mlir_translate failed")
                return -1
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

    def calibration(self, mlirfile_fp32: str, dataset: str, threshold_table: str, pre_func, input_num, histogram_bin_num, auto_tune=False,tune_image_num=10):
        # mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune)
        cvi_model = ModelFactory()
        cvi_model.load_model('mlir', None, mlirfile=mlirfile_fp32)
        calitor = KLD_Calibrator_v2(dataset,pre_func, cvi_model.model, input_num=input_num, histogram_bin_num=histogram_bin_num)
        calitor.do_calibration(threshold_table=threshold_table)
        if auto_tune == True:
            tuner = Tuner_v2(mlirfile_fp32, threshold_table, dataset, tune_iteration=tune_image_num,preprocess_func=pre_func)
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
