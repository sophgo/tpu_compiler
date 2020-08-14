#!/bin/bash
set -e

usage()
{
   echo ""
   echo "Usage: convert_model.sh [-i model_file] [-d model_data] [-t model_type]"
   echo "                        [-b batch_size] [-q quant_cali_table] [-o output_cvimodel]"
   echo "                        [-l do_layer_group] [-p do_fused_preprocess]"
   echo "                        [-r raw_scale] [-m mean] [-s std] [-a input_scale] [-w channel_order]"
   echo "                        [-z image_dims] [-c do_crop] [-f crop_offset]"
   echo -e "\t-i Model file (prototxt or onnx or tf)"
   echo -e "\t-d Model data file (caffe only, .caffemodel file)"
   echo -e "\t-t Model type (caffe|onnx|tf)"
   echo -e "\t-b Batch size                                  [default: 1]"
   echo -e "\t-q Quant calibration table file"
   echo -e "\t-o Output cvimodel file"
   echo -e "\t-l Do layergroup optimization                  [default: 1]"
   echo -e "\t-p Do fused preprocess                         [default: 0]"
   echo -e "\t-r Fused preprocess raw scale                  [default: 255.0]"
   echo -e "\t-m Fused preprocess mean                       [default: 0.0,0.0,0.0]"
   echo -e "\t-s Fused preprocess std                        [default: 1.0,1.0,1.0]"
   echo -e "\t-a Fused preprocess input scale                [default: 1.0,1.0,1.0]"
   echo -e "\t-w Fused preprocess channel order (rgb|bgr)    [default: bgr]"
   echo -e "\t-c Fused preprocess do crop                    [default: 0] "
   echo -e "\t-z Fused preprocess crop image size            [default: [none], use net dim]"
   echo -e "\t-f Fused preprocess crop offset                [default: [none], center]"
   echo -e "\t-h help"
   exit 1
}

bs="1"
do_layergroup="1"
do_fused_preprocess="0"
do_crop="0"

while getopts "i:d:t:b:q:o:l:pr:m:s:a:w:cz:f:h" opt
do
  case "$opt" in
    i ) model_def="$OPTARG" ;;
    d ) model_data="$OPTARG" ;;
    t ) model_type="$OPTARG" ;;
    b ) bs="$OPTARG" ;;
    q ) cali_table="$OPTARG" ;;
    o ) output="$OPTARG" ;;
    l ) do_layergroup="$OPTARG" ;;
    p ) do_fused_preprocess="1" ;;
    r ) raw_scale="$OPTARG" ;;
    m ) mean="$OPTARG" ;;
    s ) std="$OPTARG" ;;
    a ) input_scale="$OPTARG" ;;
    w ) channel_order="$OPTARG" ;;
    c ) do_crop="1" ;;
    z ) image_size="$OPTARG" ;;
    f ) crop_offset="$OPTARG" ;;
    h ) usage ;;
  esac
done
