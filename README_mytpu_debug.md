# mlir-tpu

## dump all tensor

```
$ ./bin/mlir-tpu-interpreter resnet-50-quant-int8-addr2.mlir \
    --tensor-in test_cat_in_fp32.bin \
    --tensor-out out-quant-int8.bin \
    --dump-all-tensor=tensor_all_quant-int8.npz

$ python ./npz_list.py tensor_all_quant-int8.npz
```

## get quantized input data

dump, save, and show
```
$ python ./npz_dump.py tensor_all_quant-int8.npz data_quant

$ python ./npz_to_bin.py tensor_all_quant-int8.npz data_quant
$ mv data_quant.bin test_cat_in_quant-int8.bin
$ python ./bin_fp32_to_int8.py test_cat_in_quant-int8.bin test_cat_in_int8.bin 1 3 224 224

$ python ./bin_dump.py test_cat_in_int8.bin int8 1 3 224 224
```
Note `test_cat_in_quant-int8.bin` is still saved in float32

## run test_bmnet

```
$ ./test/test_bmnet \
    test_cat_in_int8.bin \
    ~/work/llvm-project/build/ResNet-50-model.bin \
    ~/work/llvm-project/build/cmdbuf.bin \
    out_new.bin \
    1000 150528 25542640 1
```

With fill neuron dump
```
$ ./test/test_bmnet \
    test_cat_in_int8.bin \
    ~/work/llvm-project/build/ResNet-50-model.bin \
    ~/work/llvm-project/build/cmdbuf.bin \
    out_new.bin \
    25542640 0 25542640 1
```

```
# extract input
$ python ./bin_extract.py out_new.bin out_data_quant.bin int8 0 150528
$ python ./bin_dump.py out_data_quant.bin int8 1 3 224 224
# ref
$ python ./npz_dump.py tensor_all_quant-int8.npz data_quant


# extract conv1 (0x01797ff0 = 24739824)
$ python ./bin_extract.py out_new.bin out_conv1.bin int8 24739824 802816
$ python ./bin_dump.py out_conv1.bin int8 1 64 112 112
# ref
$ python ./npz_dump.py tensor_all_quant-int8.npz scale_conv1


# extract conv1_relu (0x016d3ff0 = 23937008)
$ python ./bin_extract.py out_new.bin out_conv1_relu.bin int8 23937008 802816
$ python ./bin_dump.py out_conv1_relu.bin int8 1 64 112 112
# ref
$ python ./npz_dump.py tensor_all_quant-int8.npz conv1_relu


# extract pool1 (0x016a2ff0 = 23736304)
$ python ./bin_extract.py out_new.bin out_pool1.bin int8 23736304 200704
$ python ./bin_dump.py out_pool1.bin int8 1 64 56 56
# ref
$ python ./npz_dump.py tensor_all_quant-int8.npz pool1

```

```
# weight, conv1 filter
$ python ./bin_extract.py out_new.bin out_conv1_0.bin 0x01856f10 9408
$ python ./bin_extract.py out_new.bin out_conv1_1.bin 0x01856ed0 64
$ python ./bin_extract.py out_new.bin out_conv1_2.bin 0x01856ec0 1
```

## get the neuron map

```
[data_quant                          ][  150528] : [ 0x00000000 --> 0x00024c00 ]
[fc1000                              ][    1008] : [ 0x00024c00 --> 0x00024ff0 ]
[pool5                               ][    2048] : [ 0x00024ff0 --> 0x000257f0 ]
[res5c_relu                          ][  100352] : [ 0x000257f0 --> 0x0003dff0 ]
[res5c                               ][  100352] : [ 0x0003dff0 --> 0x000567f0 ]
[scale5c_branch2c                    ][  100352] : [ 0x000567f0 --> 0x0006eff0 ]
[res5c_branch2b_relu                 ][   25088] : [ 0x0006eff0 --> 0x000751f0 ]
[scale5c_branch2b                    ][   25088] : [ 0x000751f0 --> 0x0007b3f0 ]
[res5c_branch2a_relu                 ][   25088] : [ 0x0007b3f0 --> 0x000815f0 ]
[scale5c_branch2a                    ][   25088] : [ 0x000815f0 --> 0x000877f0 ]
[res5b_relu                          ][  100352] : [ 0x000877f0 --> 0x0009fff0 ]
[res5b                               ][  100352] : [ 0x0009fff0 --> 0x000b87f0 ]
[scale5b_branch2c                    ][  100352] : [ 0x000b87f0 --> 0x000d0ff0 ]
[res5b_branch2b_relu                 ][   25088] : [ 0x000d0ff0 --> 0x000d71f0 ]
[scale5b_branch2b                    ][   25088] : [ 0x000d71f0 --> 0x000dd3f0 ]
[res5b_branch2a_relu                 ][   25088] : [ 0x000dd3f0 --> 0x000e35f0 ]
[scale5b_branch2a                    ][   25088] : [ 0x000e35f0 --> 0x000e97f0 ]
[res5a_relu                          ][  100352] : [ 0x000e97f0 --> 0x00101ff0 ]
[res5a                               ][  100352] : [ 0x00101ff0 --> 0x0011a7f0 ]
[scale5a_branch2c                    ][  100352] : [ 0x0011a7f0 --> 0x00132ff0 ]
[res5a_branch2b_relu                 ][   25088] : [ 0x00132ff0 --> 0x001391f0 ]
[scale5a_branch2b                    ][   25088] : [ 0x001391f0 --> 0x0013f3f0 ]
[res5a_branch2a_relu                 ][   25088] : [ 0x0013f3f0 --> 0x001455f0 ]
[scale5a_branch2a                    ][   25088] : [ 0x001455f0 --> 0x0014b7f0 ]
[scale5a_branch1                     ][  100352] : [ 0x0014b7f0 --> 0x00163ff0 ]
[res4f_relu                          ][  200704] : [ 0x00163ff0 --> 0x00194ff0 ]
[res4f                               ][  200704] : [ 0x00194ff0 --> 0x001c5ff0 ]
[scale4f_branch2c                    ][  200704] : [ 0x001c5ff0 --> 0x001f6ff0 ]
[res4f_branch2b_relu                 ][   50176] : [ 0x001f6ff0 --> 0x002033f0 ]
[scale4f_branch2b                    ][   50176] : [ 0x002033f0 --> 0x0020f7f0 ]
[res4f_branch2a_relu                 ][   50176] : [ 0x0020f7f0 --> 0x0021bbf0 ]
[scale4f_branch2a                    ][   50176] : [ 0x0021bbf0 --> 0x00227ff0 ]
[res4e_relu                          ][  200704] : [ 0x00227ff0 --> 0x00258ff0 ]
[res4e                               ][  200704] : [ 0x00258ff0 --> 0x00289ff0 ]
[scale4e_branch2c                    ][  200704] : [ 0x00289ff0 --> 0x002baff0 ]
[res4e_branch2b_relu                 ][   50176] : [ 0x002baff0 --> 0x002c73f0 ]
[scale4e_branch2b                    ][   50176] : [ 0x002c73f0 --> 0x002d37f0 ]
[res4e_branch2a_relu                 ][   50176] : [ 0x002d37f0 --> 0x002dfbf0 ]
[scale4e_branch2a                    ][   50176] : [ 0x002dfbf0 --> 0x002ebff0 ]
[res4d_relu                          ][  200704] : [ 0x002ebff0 --> 0x0031cff0 ]
[res4d                               ][  200704] : [ 0x0031cff0 --> 0x0034dff0 ]
[scale4d_branch2c                    ][  200704] : [ 0x0034dff0 --> 0x0037eff0 ]
[res4d_branch2b_relu                 ][   50176] : [ 0x0037eff0 --> 0x0038b3f0 ]
[scale4d_branch2b                    ][   50176] : [ 0x0038b3f0 --> 0x003977f0 ]
[res4d_branch2a_relu                 ][   50176] : [ 0x003977f0 --> 0x003a3bf0 ]
[scale4d_branch2a                    ][   50176] : [ 0x003a3bf0 --> 0x003afff0 ]
[res4c_relu                          ][  200704] : [ 0x003afff0 --> 0x003e0ff0 ]
[res4c                               ][  200704] : [ 0x003e0ff0 --> 0x00411ff0 ]
[scale4c_branch2c                    ][  200704] : [ 0x00411ff0 --> 0x00442ff0 ]
[res4c_branch2b_relu                 ][   50176] : [ 0x00442ff0 --> 0x0044f3f0 ]
[scale4c_branch2b                    ][   50176] : [ 0x0044f3f0 --> 0x0045b7f0 ]
[res4c_branch2a_relu                 ][   50176] : [ 0x0045b7f0 --> 0x00467bf0 ]
[scale4c_branch2a                    ][   50176] : [ 0x00467bf0 --> 0x00473ff0 ]
[res4b_relu                          ][  200704] : [ 0x00473ff0 --> 0x004a4ff0 ]
[res4b                               ][  200704] : [ 0x004a4ff0 --> 0x004d5ff0 ]
[scale4b_branch2c                    ][  200704] : [ 0x004d5ff0 --> 0x00506ff0 ]
[res4b_branch2b_relu                 ][   50176] : [ 0x00506ff0 --> 0x005133f0 ]
[scale4b_branch2b                    ][   50176] : [ 0x005133f0 --> 0x0051f7f0 ]
[res4b_branch2a_relu                 ][   50176] : [ 0x0051f7f0 --> 0x0052bbf0 ]
[scale4b_branch2a                    ][   50176] : [ 0x0052bbf0 --> 0x00537ff0 ]
[res4a_relu                          ][  200704] : [ 0x00537ff0 --> 0x00568ff0 ]
[res4a                               ][  200704] : [ 0x00568ff0 --> 0x00599ff0 ]
[scale4a_branch2c                    ][  200704] : [ 0x00599ff0 --> 0x005caff0 ]
[res4a_branch2b_relu                 ][   50176] : [ 0x005caff0 --> 0x005d73f0 ]
[scale4a_branch2b                    ][   50176] : [ 0x005d73f0 --> 0x005e37f0 ]
[res4a_branch2a_relu                 ][   50176] : [ 0x005e37f0 --> 0x005efbf0 ]
[scale4a_branch2a                    ][   50176] : [ 0x005efbf0 --> 0x005fbff0 ]
[scale4a_branch1                     ][  200704] : [ 0x005fbff0 --> 0x0062cff0 ]
[res3d_relu                          ][  401408] : [ 0x0062cff0 --> 0x0068eff0 ]
[res3d                               ][  401408] : [ 0x0068eff0 --> 0x006f0ff0 ]
[scale3d_branch2c                    ][  401408] : [ 0x006f0ff0 --> 0x00752ff0 ]
[res3d_branch2b_relu                 ][  100352] : [ 0x00752ff0 --> 0x0076b7f0 ]
[scale3d_branch2b                    ][  100352] : [ 0x0076b7f0 --> 0x00783ff0 ]
[res3d_branch2a_relu                 ][  100352] : [ 0x00783ff0 --> 0x0079c7f0 ]
[scale3d_branch2a                    ][  100352] : [ 0x0079c7f0 --> 0x007b4ff0 ]
[res3c_relu                          ][  401408] : [ 0x007b4ff0 --> 0x00816ff0 ]
[res3c                               ][  401408] : [ 0x00816ff0 --> 0x00878ff0 ]
[scale3c_branch2c                    ][  401408] : [ 0x00878ff0 --> 0x008daff0 ]
[res3c_branch2b_relu                 ][  100352] : [ 0x008daff0 --> 0x008f37f0 ]
[scale3c_branch2b                    ][  100352] : [ 0x008f37f0 --> 0x0090bff0 ]
[res3c_branch2a_relu                 ][  100352] : [ 0x0090bff0 --> 0x009247f0 ]
[scale3c_branch2a                    ][  100352] : [ 0x009247f0 --> 0x0093cff0 ]
[res3b_relu                          ][  401408] : [ 0x0093cff0 --> 0x0099eff0 ]
[res3b                               ][  401408] : [ 0x0099eff0 --> 0x00a00ff0 ]
[scale3b_branch2c                    ][  401408] : [ 0x00a00ff0 --> 0x00a62ff0 ]
[res3b_branch2b_relu                 ][  100352] : [ 0x00a62ff0 --> 0x00a7b7f0 ]
[scale3b_branch2b                    ][  100352] : [ 0x00a7b7f0 --> 0x00a93ff0 ]
[res3b_branch2a_relu                 ][  100352] : [ 0x00a93ff0 --> 0x00aac7f0 ]
[scale3b_branch2a                    ][  100352] : [ 0x00aac7f0 --> 0x00ac4ff0 ]
[res3a_relu                          ][  401408] : [ 0x00ac4ff0 --> 0x00b26ff0 ]
[res3a                               ][  401408] : [ 0x00b26ff0 --> 0x00b88ff0 ]
[scale3a_branch2c                    ][  401408] : [ 0x00b88ff0 --> 0x00beaff0 ]
[res3a_branch2b_relu                 ][  100352] : [ 0x00beaff0 --> 0x00c037f0 ]
[scale3a_branch2b                    ][  100352] : [ 0x00c037f0 --> 0x00c1bff0 ]
[res3a_branch2a_relu                 ][  100352] : [ 0x00c1bff0 --> 0x00c347f0 ]
[scale3a_branch2a                    ][  100352] : [ 0x00c347f0 --> 0x00c4cff0 ]
[scale3a_branch1                     ][  401408] : [ 0x00c4cff0 --> 0x00caeff0 ]
[res2c_relu                          ][  802816] : [ 0x00caeff0 --> 0x00d72ff0 ]
[res2c                               ][  802816] : [ 0x00d72ff0 --> 0x00e36ff0 ]
[scale2c_branch2c                    ][  802816] : [ 0x00e36ff0 --> 0x00efaff0 ]
[res2c_branch2b_relu                 ][  200704] : [ 0x00efaff0 --> 0x00f2bff0 ]
[scale2c_branch2b                    ][  200704] : [ 0x00f2bff0 --> 0x00f5cff0 ]
[res2c_branch2a_relu                 ][  200704] : [ 0x00f5cff0 --> 0x00f8dff0 ]
[scale2c_branch2a                    ][  200704] : [ 0x00f8dff0 --> 0x00fbeff0 ]
[res2b_relu                          ][  802816] : [ 0x00fbeff0 --> 0x01082ff0 ]
[res2b                               ][  802816] : [ 0x01082ff0 --> 0x01146ff0 ]
[scale2b_branch2c                    ][  802816] : [ 0x01146ff0 --> 0x0120aff0 ]
[res2b_branch2b_relu                 ][  200704] : [ 0x0120aff0 --> 0x0123bff0 ]
[scale2b_branch2b                    ][  200704] : [ 0x0123bff0 --> 0x0126cff0 ]
[res2b_branch2a_relu                 ][  200704] : [ 0x0126cff0 --> 0x0129dff0 ]
[scale2b_branch2a                    ][  200704] : [ 0x0129dff0 --> 0x012ceff0 ]
[res2a_relu                          ][  802816] : [ 0x012ceff0 --> 0x01392ff0 ]
[res2a                               ][  802816] : [ 0x01392ff0 --> 0x01456ff0 ]
[scale2a_branch2c                    ][  802816] : [ 0x01456ff0 --> 0x0151aff0 ]
[res2a_branch2b_relu                 ][  200704] : [ 0x0151aff0 --> 0x0154bff0 ]
[scale2a_branch2b                    ][  200704] : [ 0x0154bff0 --> 0x0157cff0 ]
[res2a_branch2a_relu                 ][  200704] : [ 0x0157cff0 --> 0x015adff0 ]
[scale2a_branch2a                    ][  200704] : [ 0x015adff0 --> 0x015deff0 ]
[scale2a_branch1                     ][  802816] : [ 0x015deff0 --> 0x016a2ff0 ]
[pool1                               ][  200704] : [ 0x016a2ff0 --> 0x016d3ff0 ]
[conv1_relu                          ][  802816] : [ 0x016d3ff0 --> 0x01797ff0 ]
[scale_conv1                         ][  802816] : [ 0x01797ff0 --> 0x0185bff0 ]
```
