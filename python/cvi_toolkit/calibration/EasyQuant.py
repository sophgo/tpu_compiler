import pymlir
import numpy as np


class EasyQuant(object):
    def __init__(self, mlir_file):
        self.model = pymlir.module()
        self.model.load(mlir_file)

    def findWeightMaxPerChannel(self):
        weightData = self.model.getWeightData()
        weightPerChannel = dict()
        for name, wd in weightData.items():
            wd_shape = wd.shape
            wd = np.absolute(wd)
            wd = np.reshape(wd, (wd_shape[0], -1))
            per_channel_max = np.amax(wd, axis=1)
            weightPerChannel[name] = per_channel_max
        return weightPerChannel

    # def findWeightMaxPerLayer(self):

    def updateCalibrationTable(self, calibration_table_file: str):
        wegihtDataPerChannel = self.findWeightMaxPerChannel()
        with open(calibration_table_file, 'a') as writer:
            for name, data in wegihtDataPerChannel.items():
                data_list_str = ""
                for i in data:
                    data_list_str += "{:.5f} ".format(i)
                threshold_info = "weight {} {}\n".format(name, data_list_str)
                writer.write(threshold_info)
