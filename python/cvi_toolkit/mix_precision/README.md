# SQNR search for practical Mix Precision under int8/bf16

We base bf16 then turn off one layer to find highest sqnr 

# find proper bf16 layers
```sh
# your network is mobilenetv3_pytorch
export NET=mobilenetv3_pytorch

cd ${MLIR_SRC_PATH}/${NET}

# find bf16 layers
# you should prepare ${NET}_quant_int8_multiplier.mlir/${NET}_op_info.csv, mayber do \${NET}_regression_1_fp32 and \${NET}_regression_2_int8 at first
../python/cvi_toolkit/mix_precision/find_mix_precision.py --all_layers_name_csv_file

${NET}_op_info.csv --layers_column_name input --net_name ${NET} --gen_cmd_script ../python/cvi_toolkit/mix_precision/gen_mix_precision.sh --model ${NET}_quant_int8_multiplier.mlir
```

# dependency

you should prepare *ilsvrc12_256* and *imagenet_synset_to_human_label_map.txt* from nas path /ai/dataset_zoo/imagenet/ILSVRC2012_256/{ilsvrc12_256.zip,caffe_imagenet_synset_to_human_label_map.txt}
