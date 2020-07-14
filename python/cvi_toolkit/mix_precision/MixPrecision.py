import pymlir
from ..utils.mlir_shell import gen_bf16_mlir
import cv2
import numpy as np
import sys, os, copy, math, shutil, time
from tqdm import tqdm

class MixPrecisior(object):
    def __init__(self, mlir_file, data_file=None, skip_op=['tpu.input', 'tpu.quant', 'tpu.cast'],
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
        self.op_layer = self.get_op_info()
        self.skip_ops = skip_op

        self.preprocess_func = precrocess_func

    def load_data_file(self, file_txt):
        image_list = list()
        with open(file_txt, 'r') as f:
            image_list = f.readlines()
            if len(image_list) > self.input_num:
                image_list = image_list[:self.input_num]
        return [img_path.strip() for img_path in image_list]

    def set_preprocess_func(self, precrocess_func):
        self.precrocess_func = precrocess_func

    def get_op_info(self):
        op_info = self.fp32_cali_model.op_info
        return [o['name'] for o in op_info]

    def create_bf16_layer_files(self, bf16_file, layers, exclude_layers = []):
        with open(bf16_file, 'w') as f:
            for bf16_layer in layers:
                if bf16_layer not in exclude_layers:
                    f.write(bf16_layer + "\n")

    def cal_sqnr(self, signal_gt, signal_target):
        gt_value = signal_gt.flatten()
        target = signal_target.flatten()
        noise = gt_value - target

        avg_gt = np.sum(gt_value) / gt_value.size
        avg_noise = np.sum(noise) / noise.size

        gt_zero_mean = gt_value - avg_gt
        noise_zero_mean = noise - avg_noise

        var_gt_zero_mean = np.var(gt_zero_mean)
        var_noise_zero_mean = np.var(noise_zero_mean)

        if var_noise_zero_mean == 0.0:
            return 2^31 - 1

        sqnr = 10 * np.log10(var_gt_zero_mean / var_noise_zero_mean)
        return sqnr

    def run(self, output_mlir):
        sqnr_list = list()
        predictions_gt = list()

        bf16_txt = "bf16_info_file.txt"

        # set all layer for bf16
        self.create_bf16_layer_files(bf16_txt, self.op_layer)

        bf16_mlir = "bf16.mlir"
        gen_bf16_mlir(self.fp32_cali_mlir_file ,bf16_mlir, bf16_txt)

        # read bf16 mlir
        self.bf16_model = pymlir.module()
        self.bf16_model.load(bf16_mlir)

        for img in self.image_list:
            img = self.preprocess_func(img)
            res = self.bf16_model.run(img)
            all_tensor = self.bf16_model.get_all_tensor()
            pred_tensor = all_tensor[self.bf16_model.op_info[-1]['name']].flatten()
            predictions_gt.append(pred_tensor)

        print("bf16 inference done")

        pbar = tqdm(self.fp32_cali_model.op_info)
        for layer in pbar:
            if layer['type'] in self.skip_ops:
                continue
            pbar.set_description("Processing {}".format(layer['name']))
            layer_name = layer['name']
            bf16_tmp_txt = "bf16_tmp_file.txt"
            self.create_bf16_layer_files(bf16_tmp_txt, [layer_name])

            bf16_tmp_mlir = "bf16_tmp.mlir"
            gen_bf16_mlir(self.fp32_cali_mlir_file ,bf16_tmp_mlir, bf16_tmp_txt)

            self.bf16_model = pymlir.module()
            self.bf16_model.load(bf16_tmp_mlir)

            sqnr = 0
            for idx, img in enumerate(self.image_list):
                img = self.preprocess_func(img)
                res = self.bf16_model.run(img)
                all_tensor = self.bf16_model.get_all_tensor()
                pred_tensor = all_tensor[self.bf16_model.op_info[-1]['name']].flatten()
                sqnr += self.cal_sqnr(predictions_gt[idx], pred_tensor)

            # print("Layer: {}, SQNR: {}\n\n".format(layer_name, sqnr / len(self.image_txt_list)))

            sqnr_list.append((layer_name, sqnr / len(self.image_txt_list)))

        # remove tmp file
        shutil.rmtree(bf16_txt, ignore_errors=True)
        shutil.rmtree(bf16_mlir, ignore_errors=True)
        shutil.rmtree(bf16_tmp_txt, ignore_errors=True)
        shutil.rmtree(bf16_tmp_mlir, ignore_errors=True)

        return sorted(sqnr_list, key=lambda x: x[1], reverse=True)