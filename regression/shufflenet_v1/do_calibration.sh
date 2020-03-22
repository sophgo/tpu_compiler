set -e

CALI_PATH=$REGRESSION_PATH/shufflenet_v1

#input txt
#python3 gen_data_list.py ~/data/dataset/imagenet/img_val_extracted/val 5000 input.txt

python $MLIR_SRC_PATH/python/calibration/run_calibration.py \
    $CALI_PATH/shufflenet/shufflenet_opt.mlir \
    $REGRESSION_PATH/mobilenet_v2/data/cali_list_imagenet_1000.txt \
    --output_file=$CALI_PATH/data/shufflenet_threshold_table \
    --net_input_dims 224,224 \
    --mean 103.94,116.78,123.68 \
    --input_scale 0.017 \
    --raw_scale 255.0 \
    --input_num=1000
