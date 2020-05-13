from jinja2 import Template
import os

yml_format = """
#data preprocess paremeter please set to null if no need.
output_file: {{ NET }}.cvimodel

data_preprocess:
    image_resize_dim: {{ IMAGE_RESIZE_DIMS }}
    net_input_dims: {{ NET_INPUT_DIMS }}
    # for mean , you should only set one of "image_mean" "channel_mean" and "mean_file"
    #image mean value
    #image_mean: 20.5
    #perchanel mean value
    channel_mean: {{ MEAN }}
    #mean file
    #mean_file: /home/hongjun/models/data/ilsvrc_2012_mean.npy
    std: {{ STD }}

    #Multiply raw input by this scale before preprocession
    raw_scale: {{ RAW_SCALE }}
    #Multiply input features by this scale to finish preprocession
    input_scale: {{ INPUT_SCALE }}
    #RGB order: rgb or bgr , default is bgr the same as opencv
    RGB_order: {{ MODEL_CHANNEL_ORDER }}
    data_format: {{ DATA_FORMAT }}
    #not implement yet
    Standardization: null
    #letterbox resize, not implement yet
    LetterBox: False
    #npz input directly , do nothing for preprocessing, just use npz input
    npy_input:

    input_file: {{ REGRESSION_PATH }}/data/cat.jpg
    output_npz:  {{ NET }}_in_fp32.npz

#step 1: model convert
Convert_model:
    #framework type: caffe or onnx
    framework_type:  {{ MODEL_TYPE }}
    #for onnx model, please set model file only
    weight_file: {{ MODEL_DAT }}
    model_file: {{ MODEL_DEF }}

#step2: do calibration
Calibration:
    # You can import calibration table directly or do calibration.
    calibraion_table: {{ CALI_TABLE }}

    # Do caliration parameters, if you import calibration table below parameters will be ignored.
    Dataset: /tmp/input.txt
    #How many images you want to use to do calibration, if more the num if Dataset, Dataset num will be used
    image_num: 100
    histogram_bin_num: 2048

#step3: do quantization
Quantization:
    per_channel: true
    per_tensor: true
    asymmetric: false
    symmetric: true

#step4: do accuracy test
Accuracy_test:
    # Enable  perlayer tensor fp32 similarity check to make sure convert IR result is correct.
    FP32_Accuracy_test: True
    Tolerance_FP32:
        - 0.98
        - 0.98
        - 0.98
    # Enable perlayer tensor int8 similarity check to make sure quantization result is correct.
    INT8_Accuracy_test: True
    Tolerance_INT8:
        - 0
        - 0
        - 0
    {% if EXCEPTS != "-" %}
    excepts: {{ EXCEPTS }}
    {% endif %}

#step5: do tpu offline simulation to check accuracy.

# Do cvimodel TPU simulation or not.
# This will do accuracy test by default by comparing with INT8 result after quantization.
# Return fail if not get exactly same result.
Simulation: true

#other parameter

#Do compression or not when do data store and load.
vlc_compress(VLC): true


"""


tm = Template(yml_format)

print(os.environ['NET'])
print(os.environ['MODEL_DEF'])

msg = tm.render(os.environ)
print(msg)

with open("{}.yml".format(os.environ['NET']), "w") as f:
    f.write(msg)
