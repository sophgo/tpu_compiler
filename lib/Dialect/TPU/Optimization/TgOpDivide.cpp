//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace mlir;
#define NPU_NUM (MInfo::lane_num)
#define LOCAL_MEM_SIZE (MInfo::lmem_per_lane)

namespace {
class TgOpDividePass : public mlir::PassWrapper<TgOpDividePass, FunctionPass> {
public:
  typedef struct {
    Operation *op;
    std::set<Operation *> op_set;
    int64_t max_size;
  } main_op_t;

  typedef struct {
    int out_h_start;
    int out_h;
    int in_h_start;
    int in_h;
    int num_backward;
    int num_forward;
    Operation *op; // op after slice;
  } h_slice_t;

  typedef struct {
    std::vector<h_slice_t> slice;
    int num_uses;
    int num_input;
    std::vector<int64_t> shape; // output shape
  } op_info_t;

public:
  explicit TgOpDividePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  static bool support(Operation *op) {
    if (isa<tpu::TG_INT8_Conv2DOp>(op) ||
        isa<tpu::TG_INT8_EltwiseAddOp>(op) ||
        isa<tpu::TG_INT8_PoolMax2DOp>(op)) {
      return true;
    }
    return false;
  }

  static bool update_box(std::set<Operation *> &op_box, main_op_t &main_op) {
    if (op_box.size() == 1) {
      return true;
    }
    if (op_box.empty()) {
      return false;
    }
    Operation *update_op = *op_box.begin();
    int min_layer_id = getOpLayerId(update_op);
    for (auto op : op_box) {
      auto id = getOpLayerId(op);
      if (id < min_layer_id) {
        min_layer_id = id;
        update_op = op;
      }
    }
    op_box.erase(update_op);
    for (auto &use : update_op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      if (!support(sub_op)) {
        return false;
      }
      op_box.insert(sub_op);
      main_op.op_set.insert(sub_op);
      auto size = getTensorSize(sub_op->getResult(0));
      if (main_op.max_size < size) {
        main_op.max_size = size;
      }
    }
    return update_box(op_box, main_op);
  }

  static inline std::string getOpName(Operation *op) {
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
    return tpuOp.getOpName().str();
  }

  inline bool start_slice() { return slice_idx == 0; }
  inline bool end_slice() { return slice_idx == num_slice - 1; }

  static Operation *getNextMainOp(Operation *op, main_op_t &main_op) {
    std::set<Operation *> op_box;
    for (auto &use : op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      if (!support(sub_op)) {
        return nullptr;
      }
      op_box.insert(sub_op);
      main_op.op_set.insert(sub_op);
      auto size = getTensorSize(sub_op->getResult(0));
      if (main_op.max_size < size) {
        main_op.max_size = size;
      }
    }
    bool ret = update_box(op_box, main_op);
    if (ret == false) {
      return nullptr;
    }
    return *op_box.begin();
  }

  bool init_main_op(FuncOp &fn) {
    Operation *in_op = nullptr;
    bool multi_input = false;
    fn.walk([&](tpu::InputOp inputOp) {
      if (in_op == nullptr) {
        in_op = inputOp.getOperation();
      } else if (!multi_input) {
        // not support multi inpult
        multi_input = true;
      }
    });
    if (multi_input) {
      return false;
    }
    Operation *next_op = in_op;
    do {
      next_op = getNextOp(next_op);
    } while (next_op != nullptr && false == support(next_op));
    if (next_op == nullptr) {
      return false;
    }

    auto size = getTensorSize(next_op->getResult(0));
    main_op_t info = {
        .op = next_op,
        .op_set = {next_op},
        .max_size = size,
    };
    main_ops.clear();
    max_size = 0;
    do {
      if (max_size < info.max_size) {
        max_size = info.max_size;
      }
      main_ops.push_back(info);
      info.op = nullptr;
      info.max_size = 0;
      info.op_set.clear();
      next_op = getNextMainOp(next_op, info);
      info.op = next_op;
    } while (next_op != nullptr);
    // make sure max_size is really too large
    if (max_size <= (int)(NPU_NUM * LOCAL_MEM_SIZE)) {
      return false;
    }
    if (main_ops.size() < 2) {
      return false;
    }
    start_idx = 0;
    end_idx = main_ops.size() - 1;

    init_last_size(fn);
    if (false == update_op_set()) {
      return false;
    }
    for (auto op : op_set) {
      op_info_t op_info;
      op_info.num_input = 0;
      op_info.num_uses = 0;
      op_info.slice.clear();
      op_info.shape = getTensorShape(op->getResult(0));
      if (op_info.shape.size() < 3) {
        return false;
      }
      for (auto &use : op->getResult(0).getUses()) {
        if (op_set.find(use.getOwner()) != op_set.end()) {
          op_info.num_uses++;
        }
      }
      for (auto input : op->getOperands()) {
        if (op_set.find(input.getDefiningOp()) != op_set.end()) {
          op_info.num_input++;
        }
      }
      op_h_map[op] = op_info;
    }
    return true;
  }

  bool update_op_set() {
    auto start_size = main_ops[start_idx].max_size;
    if (start_size <= last_size) {
      while (start_idx < end_idx &&
             main_ops[start_idx + 1].max_size <= last_size) {
        start_idx++;
      }
    }
    auto end_size = main_ops[end_idx].max_size;
    if (end_size <= last_size) {
      while (start_idx < end_idx &&
             main_ops[end_idx - 1].max_size <= last_size) {
        end_idx--;
      }
    }
    if (start_idx >= end_idx) {
      return false;
    }
    op_set.clear();
    for (int i = start_idx; i <= end_idx; i++) {
      auto ops = main_ops[i].op_set;
      op_set.insert(ops.begin(), ops.end());
    }
    return true;
  }

  void init_last_size(FuncOp &fn) {
    op_set.clear();
    for (auto &info : main_ops) {
      op_set.insert(info.op_set.begin(), info.op_set.end());
    }
    last_size = getTensorSize(main_ops[end_idx].op->getResult(0));
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect()->getNamespace() != "tpu" ||
          isa<tpu::LoadWeightOp>(op) || isa<tpu::WeightFileOp>(op) ||
          isa<tpu::NoneOp>(op) || isa<tpu::InputOp>(op) || isa<ReturnOp>(op) ||
          isa<FuncOp>(op)) {
      } else if (op_set.find(op) != op_set.end()) {
      } else {
        auto size = getTensorSize(op->getResult(0));
        if (last_size < size) {
          last_size = size;
        }
      }
    });
  }

  bool backward(Operation *op, int h_start, int h) {
    h_slice_t slice = {.out_h_start = h_start,
                       .out_h = h,
                       .in_h_start = h_start,
                       .in_h = h,
                       .num_backward = 1,
                       .num_forward = 0,
                       .op = nullptr};
    if (op_set.find(op) == op_set.end()) {
      // input
      return true;
    }
    auto &op_info = op_h_map[op];
    if (op_info.shape[2] < h * 2) {
      return false;
    }
    bool exist = (op_info.slice.size() > slice_idx);
    if (!exist) {
      op_info.slice.push_back(slice);
    }
    auto &s = op_info.slice[slice_idx];
    if (exist) {
      if (s.out_h < h) {
        s.out_h = h;
      }
      if (s.out_h_start > h_start) {
        s.out_h_start = h_start;
      }
      s.num_backward++;
    }
    // check all sub ops has backward, then do backward
    if (s.num_backward < op_info.num_uses) {
      return true;
    }

    // do backward
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_EltwiseAddOp>(op)) {
      auto do_early_stride = cast_op.do_early_stride();
      auto h_stride = cast_op.early_stride_h();
      if (do_early_stride) {
        s.in_h_start = s.out_h_start * h_stride;
        s.in_h = s.out_h * h_stride;
      } else {
        s.in_h_start = s.out_h_start;
        s.in_h = s.out_h;
      }
      for (auto input : cast_op.inputs()) {
        if (false == backward(input.getDefiningOp(), s.in_h_start, s.in_h)) {
          return false;
        }
      }
      return true;
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_Conv2DOp>(op)) {
      bool is_dw, with_bias, do_relu;
      int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w;
      int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;
      parseConvParam(cast_op.param(), false, cast_op.input(), cast_op.output(),
                     n, ic, ih, iw, oc, oh, ow, g, kh, kw,
                     ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                     with_bias, do_relu, pad_value);

      if (dh > 1) {
        kh = dh * (kh - 1) + 1;
      }
      s.in_h_start = (start_slice() ? 0 : s.out_h_start * sh - pt);
      s.in_h_start = std::max(s.in_h_start, 0);
      if (start_slice()) {
        s.in_h = (s.out_h - 1) * sh + kh - pt;
      } else if (end_slice()) {
        s.in_h = ih - s.in_h_start;
      } else {
        s.in_h = (s.out_h - 1) * sh + kh;
      }
      s.in_h = std::min(s.in_h, ih);
      return backward(cast_op.input().getDefiningOp(), s.in_h_start, s.in_h);
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_PoolMax2DOp>(op)) {
      int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
      bool is_global, do_relu, count_include_pad;
      parsePoolParam(cast_op.param(), cast_op.input(), cast_op.output(), n, c,
                     ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                     is_global, do_relu, count_include_pad);
      s.in_h_start = (start_slice() ? 0 : s.out_h_start * sh - pt);
      s.in_h_start = std::max(s.in_h_start, 0);
      if (start_slice()) {
        s.in_h = (s.out_h - 1) * sh + kh - pt;
      } else if (end_slice()) {
        s.in_h = ih - s.in_h_start;
      } else {
        s.in_h = (s.out_h - 1) * sh + kh;
      }
      s.in_h = std::min(s.in_h, ih);
      return backward(cast_op.input().getDefiningOp(), s.in_h_start, s.in_h);
    }
    return false;
  }

  bool do_backward() {
    auto last_op = main_ops[end_idx].op;
    auto shape = getTensorShape(last_op->getResult(0));
    int last_h = shape[2];
    num_slice = (max_size + last_size - 1) / last_size;
    num_slice = std::min(32u, std::min(num_slice, (uint32_t)(last_h / 3)));
    if (num_slice < 2) {
      return false;
    }
    int h_step = (last_h + num_slice - 1) / num_slice;
    slice_idx = 0;
    for (int h_pos = 0; h_pos < last_h; h_pos += h_step, ++slice_idx) {
      auto h = std::min(h_step, last_h - h_pos);
      if (false == backward(last_op, h_pos, h)) {
        return false;
      }
    }
    for (auto &op : op_set) {
      if (op_h_map[op].slice.size() != num_slice) {
        // make sure all ops backward
        return false;
      }
    }
    return true;
  }

  bool do_slice(FuncOp &fn) {
    while (do_backward() == false) {
      auto size = main_ops[end_idx].max_size;
      end_idx--;
      if (last_size < size) {
        last_size = size;
        if (last_size * 2 > max_size) {
          return false;
        }
        if (false == update_op_set()) {
          return false;
        }
      }
      if (start_idx >= end_idx) {
        return false;
      }
      for (auto &op_info : op_h_map) {
        op_info.second.slice.clear();
      }
    }
    return true;
  }

  template <typename T>
  void copy_tensor_inner(StringRef from_name, StringRef to_name,
                         TensorType type, TensorFile *wTF) {
    auto tensor = wTF->readTensor<T>(from_name, type);
    wTF->addTensor<T>(to_name, tensor->data(), type);
    if (end_slice()) {
      wTF->deleteTensor<T>(from_name);
    }
  }

  void copy_tensor(StringRef from_name, StringRef to_name, TensorType type,
                   TensorFile *wTF) {
    auto bitwidth = type.getElementType().getIntOrFloatBitWidth();
    switch (bitwidth) {
    case 8:
      copy_tensor_inner<uint8_t>(from_name, to_name, type, wTF);
      break;
    case 16:
      copy_tensor_inner<uint16_t>(from_name, to_name, type, wTF);
      break;
    case 32:
      copy_tensor_inner<uint32_t>(from_name, to_name, type, wTF);
      break;
    default:
      llvm_unreachable("unknow bitwidth");
    }
  }

  Value copy_weight(OpBuilder &builder, Operation *op, Value weight) {
    auto wTF = getWeightTensorFile(op);
    auto wFV = getWeightFileValue(op);
    auto weight_op = dyn_cast<tpu::LoadWeightOp>(weight.getDefiningOp());
    if (!weight_op) {
      return weight;
    }
    auto name = weight_op.name().str() + "_tod_" + std::to_string(slice_idx);
    auto type = weight_op.getResult().getType().template cast<TensorType>();
    copy_tensor(weight_op.name(), name, type, wTF);
    if (end_slice()) {
      weight_set.insert(weight.getDefiningOp());
    }
    std::vector<NamedAttribute> attrs;
    for (auto &pair : weight_op->getAttrs()) {
      if (pair.first == "name") {
        attrs.push_back(
            builder.getNamedAttr("name", builder.getStringAttr(name)));
      } else {
        attrs.push_back(builder.getNamedAttr(pair.first.c_str(), pair.second));
      }
    }
    return builder.create<tpu::LoadWeightOp>(op->getLoc(), type,
                                             ArrayRef<Value>{wFV},
                                             ArrayRef<NamedAttribute>{attrs});
  }

  Value adjust_input(OpBuilder &builder, Operation * op, Value input, h_slice_t &s) {
    auto input_op = input.getDefiningOp();
    auto shape = getTensorShape(input_op->getResult(0));
    if (op_set.find(input_op) == op_set.end() || op == main_ops[start_idx].op) {
      if (shape[2] == s.in_h) {
        return input;
      }
      return create_crop_op(builder, input_op, s.in_h_start, s.in_h);
    }
    auto &s_in = op_h_map[input_op].slice[slice_idx];
    if (s.in_h == s_in.out_h && s.in_h_start == s_in.out_h_start) {
      return s_in.op->getResult(0);
    }
    return create_crop_op(builder, s_in.op, s.in_h_start - s_in.out_h_start,
                          s.in_h);
  }

  void forward(OpBuilder &builder, Operation *op) {
    if (op_set.find(op) == op_set.end()) {
      return;
    }
    auto &op_info = op_h_map[op];
    auto &s = op_info.slice[slice_idx];
    s.num_forward++;
    if (s.num_forward < op_info.num_input && op != main_ops[start_idx].op) {
      return;
    }

    auto origin_shape = getTensorShape(op->getResult(0));
    auto output_shape = origin_shape;
    output_shape[2] = s.out_h;
    auto type = RankedTensorType::get(
        output_shape,
        op->getResult(0).getType().cast<TensorType>().getElementType());
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_EltwiseAddOp>(op)) {
      for (auto input : cast_op.inputs()) {
        auto in = adjust_input(builder, op, input, s);
        operands.push_back(in);
      }
      std::string name =
          cast_op.name().str() + "_tod_" + std::to_string(slice_idx);
      for (auto &pair : op->getAttrs()) {
        if (pair.first == "name") {
          attrs.push_back(
              builder.getNamedAttr("name", builder.getStringAttr(name)));
        } else {
          attrs.push_back(
              builder.getNamedAttr(pair.first.c_str(), pair.second));
        }
      }
      auto newOp = builder.create<tpu::TG_INT8_EltwiseAddOp>(
          op->getLoc(), type, ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      s.op = newOp.getOperation();
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::TG_INT8_Conv2DOp>(op)) {
      auto in = adjust_input(builder, op, cast_op.input(), s);
      operands.push_back(in);
      auto filter = copy_weight(builder, op, cast_op.filter());
      auto pc_info = copy_weight(builder, op, cast_op.pc_info());
      operands.push_back(filter);
      operands.push_back(pc_info);
      std::string name =
          cast_op.name().str() + "_tod_" + std::to_string(slice_idx);

      auto p = cast_op.paramAttr();
      int pad_t = p.padding_t().getInt();
      if (start_slice() == false && origin_shape[2] != s.out_h) {
        pad_t = 0;
      }
      int pad_b = p.padding_b().getInt();
      if (end_slice() == false && origin_shape[2] != s.out_h) {
        pad_b = 0;
      }

      for (auto &pair : op->getAttrs()) {
        if (pair.first == "name") {
          attrs.push_back(
              builder.getNamedAttr("name", builder.getStringAttr(name)));
        } else if (pair.first == "param") {
          attrs.push_back(builder.getNamedAttr(
              "param",
              tpu::ConvParam::get(p.kernel_h(), p.kernel_w(),
                  p.stride_h(), p.stride_w(), p.padding(), p.dilation_h(),
                  p.dilation_w(), builder.getI32IntegerAttr(pad_t),
                  builder.getI32IntegerAttr(pad_b), p.padding_l(),
                  p.padding_r(), p.group(), p.is_dw(), p.with_bias(),
                  p.do_relu(), p.ins(), p.pad_value(), &getContext())));
        } else {
          attrs.push_back(
              builder.getNamedAttr(pair.first.c_str(), pair.second));
        }
      }
      auto newOp = builder.create<tpu::TG_INT8_Conv2DOp>(
          op->getLoc(), type, ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      s.op = newOp.getOperation();
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::TG_INT8_PoolMax2DOp>(op)) {
      auto in = adjust_input(builder, op, cast_op.input(), s);
      operands.push_back(in);
      std::string name =
          cast_op.name().str() + "_tod_" + std::to_string(slice_idx);
      auto p = cast_op.paramAttr();
      int pad_t = p.padding_t().getInt();
      if (start_slice() == false && origin_shape[2] != s.out_h) {
        pad_t = 0;
      }
      int pad_b = p.padding_b().getInt();
      if (end_slice() == false && origin_shape[2] != s.out_h) {
        pad_b = 0;
      }
      for (auto &pair : op->getAttrs()) {
        if (pair.first == "name") {
          attrs.push_back(
              builder.getNamedAttr("name", builder.getStringAttr(name)));
        } else if (pair.first == "param") {
          attrs.push_back(builder.getNamedAttr(
              "param",
              tpu::PoolParam::get(
                  p.kernel_h(), p.kernel_w(), builder.getI32IntegerAttr(pad_t),
                  builder.getI32IntegerAttr(pad_b), p.padding_l(),
                  p.padding_r(), p.pad_value(), p.stride_h(), p.stride_w(),
                  p.do_relu(), p.count_include_pad(), &getContext())));
        } else {
          attrs.push_back(
              builder.getNamedAttr(pair.first.c_str(), pair.second));
        }
      }
      auto newOp = builder.create<tpu::TG_INT8_PoolMax2DOp>(
          op->getLoc(), type, ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      s.op = newOp.getOperation();
    }
    for (auto &use : op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      forward(builder, sub_op);
    }
    if (end_slice()) {
      op_to_erase.push_back(op);
    }
  }

  Value create_crop_op(OpBuilder &builder, Operation *op, int h_start,
                       int h_slice) {
    auto shape = getTensorShape(op->getResult(0));
    std::vector<int> crop_shape(shape.begin(), shape.end());
    crop_shape[2] = h_slice;
    std::vector<int> offset(shape.size(), 0);
    offset[2] = h_start;
    std::string name = getOpName(op) + "_tod_crop_" + std::to_string(slice_idx);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
    attrs.push_back(builder.getNamedAttr("crop_offset", builder.getI32ArrayAttr(offset)));
    std::vector<Value> operands;
    operands.push_back(op->getResult(0));
    shape[2] = h_slice;
    auto type = RankedTensorType::get(
        shape, op->getResult(0).getType().cast<TensorType>().getElementType());
    return builder.create<tpu::TG_INT8_CropOp>(op->getLoc(), type,
                                               ArrayRef<Value>{operands},
                                               ArrayRef<NamedAttribute>{attrs});
  }

  void concat_all(FuncOp &fn, OpBuilder &builder) {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto last_op = main_ops[end_idx].op;
    auto &slice = op_h_map[last_op].slice;
    for (auto &s : slice) {
      operands.push_back(s.op->getResult(0));
    }
    std::string name = getOpName(last_op);
    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
    attrs.push_back(builder.getNamedAttr("axis", builder.getI32IntegerAttr(2)));
    auto new_op = builder.create<tpu::TG_INT8_ConcatOp>(
        last_op->getLoc(), last_op->getResult(0).getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    last_op->replaceAllUsesWith(new_op.getOperation());
    op_to_erase.insert(op_to_erase.end(), weight_set.begin(), weight_set.end());
    for (auto &op : op_to_erase) {
      op->erase();
    }
  }

  void do_process(FuncOp &fn, OpBuilder &builder) {
    llvm::errs() << "============ tg op divide ===========================\n";
    for (int i = start_idx; i <= end_idx; i++) {
      auto &info = main_ops[i];
      llvm::errs() << "op:" << getOpName(info.op) << ", size: " << info.max_size
                   << "\n";
    }
    llvm::errs() << "max_size: " << max_size << ", last_size: " << last_size
                 << "\ndivide to [" << num_slice << "] pieces\n";
    builder.setInsertionPointAfter(main_ops[end_idx].op);
    for (slice_idx = 0; slice_idx < num_slice; slice_idx++) {
      forward(builder, main_ops[start_idx].op);
    }
    concat_all(fn, builder);
  }

  void runOnFunction() override {
    auto fn = getFunction();
    MInfo::getChipInfo(fn);
    auto *context = &getContext();
    auto builder = OpBuilder(context);
    if (init_main_op(fn) == false) {
      llvm::errs() << "tg-op-divide op set failed\n";
      return;
    }
    if (do_slice(fn) == false) {
      llvm::errs() << "tg-op-divide slice failed\n";
      return;
    }
    do_process(fn, builder);
  }

private:
  uint32_t num_slice;
  uint32_t slice_idx;
  std::vector<main_op_t> main_ops;
  std::map<Operation *, op_info_t> op_h_map;
  std::set<Operation *> op_set;
  std::set<Operation *> weight_set;
  std::vector<Operation *> op_to_erase;
  int start_idx;
  int end_idx;
  int64_t max_size;
  int64_t last_size;
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createTgOpDividePass() {
  return std::make_unique<TgOpDividePass>();
}
