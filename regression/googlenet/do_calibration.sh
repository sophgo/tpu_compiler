set -e

CALI_PATH=$REGRESSION_PATH/googlenet

#input txt
#python3 gen_data_list.py ~/data/dataset/imagenet/img_val_extracted/val 5000 input.txt

python $MLIR_SRC_PATH/python/cvi_toolkit/calibration/run_calibration.py \
    $CALI_PATH/googlenet/googlenet_opt.mlir \
    $REGRESSION_PATH/mobilenet_v2/data/cali_list_imagenet_1000.txt \
    --output_file=$CALI_PATH/data/googlenet_calibration_table \
    --net_input_dims 224,224 \
    --mean 104,117,123 \
    --input_num=1000
