import pymlir
from ..utils.mlir_shell import mlir_quant
from ..utils.math_function import cal_sqnr
import cv2
import numpy as np
import sys
import os
import copy
import math
import time
from tqdm import tqdm
from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')

tpu_skip_op = ['tpu.input', 'tpu.quant', 'tpu.layer_norm', 'tpu.softmax'
               'tpu.split', 'tpu.reshape', 'tpu.gru']

class MixQuantModel:
    def __init__(self, fp32_mlir, all_bf16=False,
                 calib_table=None, mix_table=None):
        self.fp32_mlir = fp32_mlir
        self.all_bf16 = all_bf16
        self.calib_table = calib_table
        self.mix_table = mix_table
        self.quanted_mlir_file = '{}.quanted.tune.mlir'.format(fp32_mlir)
        self.model = self._build()

    def _build(self):
        ret = mlir_quant(self.fp32_mlir, self.quanted_mlir_file, "cv183x", "tmp.csv",
                         self.all_bf16, self.calib_table, self.mix_table)
        if ret != 0:
            raise RuntimeError("generate quanted mlir model failed")
        model = pymlir.module()
        model.load(self.quanted_mlir_file)
        return model

    def infer(self, data):
        if type(data) == np.lib.npyio.NpzFile:
            for k, v in data.items():
                self.model.set_tensor(k, v)
            self.model.invoke()
        else:
            self.model.run(data)
        outputs = {}
        for name in self.model.get_output_details():
            outputs[name] = self.model.get_tensor(name)
        return outputs

    def clean(self):
        try:
            mlir_weight_file = self.model.get_weight_file_path()
            del self.model
            os.remove(self.quanted_mlir_file)
            os.remove(mlir_weight_file)
        except:
            pass


class MixPrecSearcher(object):
    def __init__(self, mlir_file, calib_table, image_file_list=[],
                 skip_op=tpu_skip_op, precrocess_func=None, input_num=10):

        if len(image_file_list) == 0:
            raise RuntimeError("Please Set image file list!")

        self.images = image_file_list
        self.fp32_mlir = mlir_file
        self.calib_table = calib_table
        self.skip_ops = skip_op
        self.preprocess_func = precrocess_func
        self.input_data_buffer = dict()
        logger.info("[*] selected images->")
        for i, image in enumerate(self.images):
            logger.info("**** <{}> {}".format(i, image))

    def _dump_all_ops_to_file(self, target_file, ops, excludes=[]):
        with open(target_file, 'w') as f:
            for op in ops:
                if op['name'] not in excludes:
                    f.write(op['name'] + "\n")

    def _is_npz(self, image):
        return True if image.split('.')[-1] == 'npz' else False

    def _fetch_data(self):
        for image in self.images:
            if image not in self.input_data_buffer:
                if self._is_npz(image):
                    x = np.load(image)
                else:
                    x = self.preprocess_func(image)
                self.input_data_buffer[image] = x
            else:
                x = self.input_data_buffer[image]
            yield x

    def _loss(self, preds, gt_preds):
        ret = 0
        cnt = 0
        for op_name in gt_preds:
            a = gt_preds[op_name]
            b = preds[op_name]
            loss = cal_sqnr(a, b)
            if not math.isinf(loss):
                ret += -loss * a.size
                cnt += a.size

        if ret == 0 and cnt == 0:
            return -math.inf
        else:
            return ret / cnt

    def run(self):
        loss_list = list()
        predictions_gt = list()

        # set all layer for bf16
        bf16_model = MixQuantModel(self.fp32_mlir, all_bf16=True)
        for data in self._fetch_data():
            outputs = bf16_model.infer(data)
            predictions_gt.append(outputs)

        self.op_info = bf16_model.model.op_info

        pbar = tqdm(self.op_info)
        for op in pbar:
            pbar.set_description("Processing {}".format(op['name']))
            if op['type'] in self.skip_ops:
                continue

            mix_table = "tmp_mix_table.txt"
            self._dump_all_ops_to_file(mix_table, self.op_info, [op['name']])
            mix_model = MixQuantModel(self.fp32_mlir, False,
                                      self.calib_table, mix_table)
            loss = 0
            for idx, data in enumerate(self._fetch_data()):
                outputs = mix_model.infer(data)
                loss += self._loss(outputs, predictions_gt[idx])
            mix_model.clean()
            loss_list.append((op['name'], loss / len(self.images)))

        bf16_model.clean()
        return sorted(loss_list, key=lambda x: x[1], reverse=True)
