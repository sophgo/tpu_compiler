
# Tasks

## 1. Upload to code server, and setup basic CI

## 2. Resnet-50 Per-Channel with Multiplier pass PXP

## 3. Resnet-50 with compression support

## 4. Optimize Resnet-50 performance

## 5. Write BF16 backend, and pass Resnet-50

## -------------- Tapeout (30/11)   ------------------

## * more network: Mobilenet-V2, SSD, YOLOv3

## * Calibration using pymlir

## * Accuracy tuning & regression: Mobilenet-V2, YOLOv3, u-net

## * Handle meta info, build bmodel

## * CPU ops integration

## -------------- Chip Back (31/01) ----------------

## * Add basic ONNX frontend

## * add basic TFLite frontend

## * Code cleanup for frontend open source

## -------------- V0.1 Release ---------------

# Backlog

## * update llvm-project and mlir to the latest

## * Use DenseElementAttr and elideElementsAttrIfLarger to hold weight

(but BF16 is not supported well)

## * change value_map_t to support different data type

## * Pass Manager to hold the whole sequance

## * Multiple inputs/outputs support (npz file)

## * Python Wrapper enhancement (handle shapes, multiple input/outputs)

## * Refactor: Remove Caffe linking (for RTTI & Exception)

## * Refactor: CNPY (to support tensor other than "float32")

## * Refactor: all tensor strongly comply with declared type

(right now all "float32" everywhere)

## * Refactor: calculate all rshift/multiplier in Quantization process

(right now, some are calucated in interpreter or translater)

## * Refactor: Share Weight buffer with TensorFile, to save memory footprint

## * Frontend: MobileNet & YOLO accruracy test and tuning

## * Refactor: more explicit weight transpose (if needed) before generate weight.bin

## * pool ceil mode

## * relu6, update in threshold_y of conv

## * prelu/leaky relu accuracy testing (convert to relu)

## * Refactor backend interface

## * Refactor interpreter/translate code, reuse Op parameters parser
