import numpy as np
import onnx
from .model.ModelFactory import ModelFactory
from .transform import OnnxConverter, TFConverter, CaffeConverter
from .calibration.kld_calibrator import KLD_Calibrator
from .calibration.tuner import AutoTuner
from .utils.mlir_shell import checkReturnValue, mlir_opt, \
                                mlir_import_calibration, mlir_tpu_quant, mlir_lower_opt, mlir_gen_cvimodel, \
    mlir_calibration, run_cvimodel, gen_bf16_mlir, get_chip_name
from .utils.log_setting import setup_logger

import subprocess
import logging
import os
from pathlib import Path


logger = setup_logger('root')
runchip = get_chip_name()

class cvinn(object):
    def __init__(self):
        pass
    def convert_model(self, model_type: str, model_file: str,  mlirfile: str, weight_file: str = None, tpu_op_info=None, batch_size=1, chip=runchip):
        mlirori = "ori_{}".format(mlirfile)
        if model_type == 'caffe':
            if not model_file.lower().endswith('.prototxt'):
                print("{} is not end with .prototxt".format(model_file))
                return -1
            if weight_file == None:
                print("No caffe weight file")
                return -1
            if not weight_file.lower().endswith('.caffemodel'):
                print("{} is not end with .caffemodel".format(weight_file))
                return -1

            c = CaffeConverter(model_file.split(
                '.')[0].split('/')[-1], model_file, weight_file,
                mlirori)
            c.run()
        elif model_type == 'onnx':
            if not model_file.lower().endswith('.onnx'):
                print("{} is not end with .onnx".format(model_file))
                return -1

            c = OnnxConverter(model_file.split(
                '.')[0].split('/')[-1], model_file, mlirori)
            c.run()
        elif model_type == "tensorflow":
            # Savedmodel is directory
            path_to_pb = os.path.join(model_file, "saved_model.pb")
            path_to_pbtxt = os.path.join(model_file, "saved_model.pbtxt")
            print(os.path.exists(path_to_pb))
            if not os.path.exists(path_to_pb) and not os.path.exists(path_to_pbtxt):
                logger.error(
                    "SavedModel file does not exist at: {}/saved_model.pbtxt|saved_model.pb".format(model_file))
                return -1
            c = TFConverter(model_file.split('/')[-1], model_file, mlirori)
            c.run()
        else:
            print("Not support {} type, now support onnx and caffe".format(model_type))
            return -1
        ret = 0
        if tpu_op_info:
            ret = mlir_opt(mlirori, mlirfile, tpu_op_info, chip=chip)
        else:
            ret = mlir_opt(mlirori, mlirfile, "{}_op_info.csv".format(model_file.split('.')[0].split('/')[-1]), chip=chip)
        return ret

    def calibration(self, mlirfile_fp32: str, dataset: str, threshold_table: str, pre_func, input_num, histogram_bin_num, auto_tune=False,tune_image_num=10):
        # mlir_calibration(mlirfile_fp32, dataset, threshold_table, auto_tune)
        cvi_model = ModelFactory()
        cvi_model.load_model('mlir', None, mlirfile=mlirfile_fp32)
        calitor = KLD_Calibrator(image_list=dataset,
            mlir_model=cvi_model.model,
            preprocess_func=pre_func,
            histogram_bin_num=histogram_bin_num)
        thresholds = calitor.do_calibration()
        calitor.dump_threshold_table(threshold_table, thresholds)
        if auto_tune == True:
            tuner = AutoTuner(mlirfile_fp32, threshold_table, dataset, 10,
                              tune_iteration=tune_image_num,preprocess_func=pre_func)
            tuner.run()
        return 0

    def import_cali_table(self, mlirfile_fp32: str, threshold_table: str, mlirfile_cali: str = None):
        ret = mlir_import_calibration(mlirfile_fp32, mlirfile_cali, threshold_table)
        if ret != 0:
            logger.error("mlir_import_callibration failed")
            exit(-1)

    def mlir_quant(self, mlirfile_cali, mlirfile_int8, quant_info=None):
        if quant_info:
            int8_op_csv = quant_info
        else:
            int8_op_csv = "{}_op_info_int8.csv".format(mlirfile_fp32.split('.')[0].split('/')[-1])
        ret = mlir_tpu_quant(mlirfile_cali, mlirfile_int8, int8_op_csv)
        if ret != 0:
            logger.error("mlir_tpu_quant failed")
            exit(-1)

    def build_cvimodel(self, mlirfile_int8: str, cvimodel: str,
                    quant_method: str = "perchannel"):

        tg_mlir = "tg_{}".format(mlirfile_int8)
        ret = mlir_lower_opt(mlirfile_int8, tg_mlir)
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
