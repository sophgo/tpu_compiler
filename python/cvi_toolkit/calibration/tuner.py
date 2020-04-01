#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import numpy as np
import sys, os, copy, math, shutil, time
import pymlir

def parse_threshold_table(threshold_table):
    with open(threshold_table, 'r') as f:
        contents = f.readlines()

    thresholds = {}
    for item in contents:
        item = item.rstrip().split()
        thresholds[item[0]] = float(item[1])

    return thresholds

class Tuner(object):
    def __init__(self, args, preprocess_func):
        self.fp32_model = args.model_file
        self.fp32_module = pymlir.module()
        self.fp32_module.load(args.model_file)

        with open(args.image_list_file,'r') as fp:
            self.all_lines = fp.readlines()

        self.output_path = args.out_path
        self.mlir_opt = os.path.join(args.binary_path, 'mlir-opt')
        self.out_table = os.path.join(self.output_path, "tune_threshold_table")
        self.int8_model = os.path.join(self.output_path, 'tune-int8.mlir')

        self.thresholds = parse_threshold_table(args.input_threshold_table)

        self.enlarge_factor = 1.01
        self.reduce_factor = 0.99

        self.best_threshold = 0
        self.best_diff = 0
        self.limit = int(args.tune_iteration)

        self.preprocess_func = preprocess_func

    def run_tune(self, args):
        time_start = time.time()
        last_best = ''
        for info in self.fp32_module.op_info:
            if info['type'] in ['tpu.input']:
                continue

            op_name = info["name"]
            # Run tuner.py to generate tune models
            best_thres = self.tune_layer(op_name)

            # Move the smallest model
            # shutil.move(os.path.join(self.output_path, op_name.replace('/', '-') + '_thres_' + str(best_thres)), './')
            best_table = os.path.join(self.output_path, op_name.replace('/', '-') + '_thres_' + str(best_thres), "tune_threshold_table")

            # Clear all other models
            # for root, dirs, files in os.walk(self.output_path):
            #     for dir in dirs:
            #         shutil.rmtree(os.path.join(self.output_path, dir))
            #     break
            print('The best tune model: ' + best_table)

            if last_best != '':
                shutil.rmtree(last_best[0:last_best.rfind('/')])

            last_best = str(best_table)
            self.thresholds = parse_threshold_table(last_best)

        time_end = time.time()
        print("Tune time {}".format(time_end - time_start))

    def run_calibration(self, thresholds):
        with open(self.out_table, 'w') as f:
            for item in thresholds:
                f.write(item + ' ' + str(thresholds[item]) + '\n')

        cmd = self.mlir_opt + ' --import-calibration-table --calibration-table ' + self.out_table + \
              ' ' + self.fp32_model + ' -o ' + os.path.join(self.output_path, 'resnet-50-cali.mlir')
        if os.system(cmd) != 0:
            print('Cmd {} execute failed.'.format(cmd))
            exit(-1)

        cmd = self.mlir_opt + ' --tpu-quant \
               ' + os.path.join(self.output_path, 'resnet-50-cali.mlir') + \
              ' -o ' + self.int8_model
        if os.system(cmd) != 0:
            print('Cmd {} execute failed.'.format(cmd))
            exit(-1)

    def tune_layer(self, target_layer):
        ori_diff = sys.float_info.max
        ori_thres = self.thresholds.get(target_layer)
        print('ori_thres={}, ori_diff={}'.format(ori_thres, ori_diff))

        self.best_threshold = ori_thres
        self.best_diff = ori_diff

        self.get_layer_best_threshold(ori_thres, ori_diff, target_layer, self.enlarge_factor)
        self.get_layer_best_threshold(ori_thres, ori_diff, target_layer, self.reduce_factor)

        print('tuning end, layer: {},  best diff with tune: {}/{}, threshold: {}/{}'.format(
            target_layer, self.best_diff, ori_diff, self.best_threshold, ori_thres))

        return self.best_threshold

    def get_layer_best_threshold(self, ori_thres, ori_diff, tune_layer, factor):
        count = 0
        fail_count = 0
        pre_diff = ori_diff

        tune_thresholds = copy.deepcopy(self.thresholds)

        self.net32_inference(tune_layer)

        while fail_count < 3:
            time1 = time.time()
            tune_thres = ori_thres * math.pow(factor, count)
            tune_thresholds[tune_layer] = tune_thres

            print('start tuning: {}, layer: {}, tuning threshold: {}'.format(count + 1, tune_layer, tune_thres))
            self.run_calibration(tune_thresholds)
            diff = self.net8_calculate_diff(tune_layer, tune_thresholds)
            print('end tuning: {}, layer: {}, tuning diff: {}'.format(count + 1, tune_layer, diff))

            if self.best_diff > diff:
                #  Remove previous saved best model/proto
                if os.path.isdir(os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), self.best_threshold))):
                    shutil.rmtree(os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), self.best_threshold)))
                thres_fold = os.path.join(self.output_path, "{}_thres_{}".format(tune_layer.replace('/', '-'), tune_thres))

                try:
                    if not os.path.isdir(thres_fold):
                        os.mkdir(thres_fold)

                    shutil.copy(self.out_table, thres_fold)
                except (OSError, IOError) as e:
                    print(e)

                self.best_diff = diff
                self.best_threshold = tune_thres
                fail_count = 0
            else:
                if pre_diff <= diff:
                    fail_count += 1

            pre_diff = diff
            count += 1
            time2 = time.time()
            print(time2-time1)

    def net32_inference(self, tune_layer):
        self.out32 = {}

        num = 0
        for line in self.all_lines:
            x = self.preprocess_func(line)
            x = np.expand_dims(x, axis=0)
            _ = self.fp32_module.run(x)
            data = self.fp32_module.get_tensor(tune_layer)
            self.out32[num] = data

            num += 1
            if num >= self.limit:
                break

    def net8_calculate_diff(self, tune_layer, tune_thresholds):
        int8_module = pymlir.module()
        int8_module.load(self.int8_model)

        num = 0
        layer_dist = 0
        for line in self.all_lines:
            x = self.preprocess_func(line)
            x = np.expand_dims(x, axis=0)
            _ = int8_module.run(x)
            out = int8_module.get_tensor(tune_layer)

            out = out * tune_thresholds[tune_layer] / 128.0
            layer_dist += np.linalg.norm(self.out32[num] - out)

            num += 1
            if num >= self.limit:
                break

        os.remove(int8_module.get_weight_file_path())

        return layer_dist / self.limit
