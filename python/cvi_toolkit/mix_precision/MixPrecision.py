import pymlir
from ..utils.mlir_shell import gen_bf16_mlir
import cv2
import numpy as np
import sys, os, copy, shutil, time
from tqdm import tqdm

class MixPrecisior(object):
    def __init__(self, mlir_file, loss_func, data_file=None, skip_op=['tpu.input', 'tpu.quant', 'tpu.cast'],
                precrocess_func=None, input_num=10):

        self.input_num = input_num

        if data_file:
            self.image_txt_list = self.load_data_file(data_file)
            self.image_list = [cv2.imread(i, cv2.IMREAD_COLOR) for i in self.image_txt_list]
        else:
            raise RuntimeError("Please Set input data txt !")

        self.fp32_cali_mlir_file = mlir_file
        self.fp32_cali_model = pymlir.module()
        self.fp32_cali_model.load(mlir_file)
        self.op_layer_list = self.get_op_info_list(self.fp32_cali_model)
        self.op_info = self.fp32_cali_model.op_info
        # clean pybind model
        del self.fp32_cali_model

        self.skip_ops = skip_op

        self.preprocess_func = precrocess_func
        self.loss_func = loss_func

    def load_data_file(self, file_txt):
        image_list = list()
        with open(file_txt, 'r') as f:
            image_list = f.readlines()
            if len(image_list) > self.input_num:
                image_list = image_list[:self.input_num]
        return [img_path.strip() for img_path in image_list]

    def set_preprocess_func(self, precrocess_func):
        self.precrocess_func = precrocess_func

    def get_op_info_list(self, mlir_model):
        return [o['name'] for o in mlir_model.op_info]

    def create_bf16_layer_files(self, bf16_file, layers, exclude_layers = []):
        with open(bf16_file, 'w') as f:
            for bf16_layer in layers:
                if bf16_layer not in exclude_layers:
                    f.write(bf16_layer + "\n")

    def get_layer_name_list(self, exclude_list=[]):
        layer_name_list = []
        for layer in self.op_info:
            if layer['name'] not in exclude_list:
                layer_name_list.append(layer['name'])

        return layer_name_list

    def run(self):
        loss_list = list()
        predictions_gt = list()

        bf16_txt = "bf16_info_file.txt"

        # set all layer for bf16
        self.create_bf16_layer_files(bf16_txt, self.op_layer_list)

        bf16_mlir = "bf16.mlir"
        bf16_quant_tpu_op_info = "bf16_quant_op_info.csv"
        gen_bf16_mlir(self.fp32_cali_mlir_file ,bf16_mlir, bf16_txt, bf16_quant_tpu_op_info)

        # read bf16 mlir
        self.bf16_model = pymlir.module()
        self.bf16_model.load(bf16_mlir)

        for img in self.image_list:
            img = self.preprocess_func(img)
            pred_tensor = self.bf16_model.run(img)
            predictions_gt.append(pred_tensor)
        del self.bf16_model
        print("bf16 inference done")

        pbar = tqdm(self.op_info)
        for layer in pbar:
            if layer['type'] in self.skip_ops:
                continue
            pbar.set_description("Processing {}".format(layer['name']))
            bf16_layer_name_list = self.get_layer_name_list(layer['name'])
            bf16_tmp_txt = "bf16_tmp_file.txt"
            self.create_bf16_layer_files(bf16_tmp_txt, bf16_layer_name_list)

            bf16_tmp_mlir = "bf16_tmp.mlir"
            gen_bf16_mlir(self.fp32_cali_mlir_file ,bf16_tmp_mlir, bf16_tmp_txt, "tmp_quant_op_info.csv")

            self.bf16_model = pymlir.module()
            self.bf16_model.load(bf16_tmp_mlir)

            loss = 0
            for idx, img in enumerate(self.image_list):
                img = self.preprocess_func(img)
                pred_tensor = self.bf16_model.run(img)
                loss += self.loss_func(predictions_gt[idx], pred_tensor)

            del self.bf16_model
            loss_list.append((layer['name'], loss / len(self.image_txt_list)))

        # remove tmp file
        shutil.rmtree(bf16_txt, ignore_errors=True)
        shutil.rmtree(bf16_mlir, ignore_errors=True)
        shutil.rmtree(bf16_tmp_txt, ignore_errors=True)
        shutil.rmtree(bf16_tmp_mlir, ignore_errors=True)

        return sorted(loss_list, key=lambda x: x[1], reverse=True)