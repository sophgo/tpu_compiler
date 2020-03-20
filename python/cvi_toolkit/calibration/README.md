# calibration tool

## How to run calibration


```
python3 run_calibration.py resnet50 ./resnet-50.mlir ./input.txt
python3 run_calibration.py yolo_v3 ./yolo_v3_416_opt.mlir ./input_coco_100.txt --input_num=100
```

run_calibration will generate the threshold table. The input.txt contains the calibration data path line by line.


### How to run auto tune

```
python3 run_tune.py ./resnet-50-opt.mlir ./resnet50_threshold_table ./input.txt ../build/bin --out_path ./result_tune --tune_iteration 1
```
