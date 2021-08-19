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
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace mlir;
#define EU_NUM (MInfo::eu_num)
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
    Operation *op; // op after slice;

  } h_slice_t;

  typedef struct {
    std::vector<h_slice_t> slice;
    int num_uses;
  } op_info_t;

public:
  explicit TgOpDividePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  static bool support(Operation *op) {
    if (isa<tpu::TG_INT8_PC_Conv2DOp>(op) ||
        isa<tpu::TG_INT8_EltwiseAddOp>(op)) {
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

  bool init_op_set(FuncOp &fn) {
    Operation *in_op = nullptr;
    bool multi_input = false;
    op_data.clear();
    op_set.clear();
    max_size = 0;
    fn.walk([&](tpu::InputOp inputOp) {
      if (in_op == nullptr) {
        in_op = inputOp.getOperation();
      } else {
        multi_input = true;
      }
    });
    if (multi_input) {
      return false;
    }
    auto next_op = getNextOp(in_op);
    while (next_op != nullptr) {
      if (support(next_op)) {
        break;
      }
      next_op = getNextOp(next_op);
    }
    if (next_op == nullptr) {
      return false;
    }
    start_op = next_op;
    auto size = getTensorSize(next_op->getResult(0));
    main_op_t info = {
        .op = next_op,
        .op_set = {next_op},
        .max_size = size,
    };
    do {
      info.op = next_op;
      if (max_size < info.max_size) {
        max_size = info.max_size;
      }
      op_data.push_back(info);
      info.max_size = 0;
      info.op_set.clear();
      info.op = nullptr;
      next_op = getNextMainOp(next_op, info);
    } while (next_op != nullptr);
    last_op = op_data.back().op;
    for (auto &use : last_op->getResult(0).getUses()) {
      last_uses.push_back(use.getOwner());
    }

    // make sure max_size is really too large
    if (max_size <= (int)(NPU_NUM * LOCAL_MEM_SIZE)) {
      return false;
    }
    if (op_data.size() <= 1) {
      return false;
    }
    for (auto &info : op_data) {
      op_set.insert(info.op_set.begin(), info.op_set.end());
    }
    for (auto op : op_set) {
      op_info_t op_info;
      auto iter = op->getResult(0).use_begin();
      auto end = op->getResult(0).use_end();
      for (op_info.num_uses = 0; iter != end; iter++, op_info.num_uses++)
        ;
      op_h_map[op] = op_info;
    }
    return true;
  }

  bool decide_piece(FuncOp fn) {
    auto last_size = getTensorSize(last_op->getResult(0));
    auto last_shape = getTensorShape(last_op->getResult(0));
    uint32_t last_h = last_shape[2];
    num_piece = 1;
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
    num_piece = (max_size + last_size - 1) / last_size;
    num_piece = std::min(32u, std::min(num_piece, last_h));
    if (num_piece < 2) {
      return false;
    }
    return true;
  }

  bool backward(Operation *op, int h_start, int h) {
    h_slice_t slice = {.out_h_start = h_start,
                       .out_h = h,
                       .in_h_start = h_start,
                       .in_h = h,
                       .num_backward = 1,
                       .op = nullptr};
    if (op_set.find(op) == op_set.end()) {
      if (getNextOp(op) == start_op) {
        first_slice.push_back(slice);
        return true;
      }
      return false;
    }
    auto &op_info = op_h_map[op];
    if (op_info.slice.size() <= current_idx) {
      op_info.slice.push_back(slice);
    } else {
      auto &index = op_info.slice[current_idx];
      if (index.out_h < h) {
        index.out_h = h;
      }
      if (index.out_h_start > h_start) {
        index.out_h_start = h_start;
      }
      index.num_backward++;
    }
    auto &index = op_info.slice[current_idx];
    // check all sub ops has backward, then do backward
    if (op != last_op && index.num_backward < op_info.num_uses) {
      return true;
    }

    // do backward
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_EltwiseAddOp>(op)) {
      auto do_early_stride = cast_op.do_early_stride();
      auto h_stride = cast_op.early_stride_h();
      if (do_early_stride) {
        index.in_h_start = index.out_h_start * h_stride;
        index.in_h = index.out_h * h_stride;
      } else {
        index.in_h_start = index.out_h_start;
        index.in_h = index.out_h;
      }
      for (auto input : cast_op.inputs()) {
        if (false ==
            backward(input.getDefiningOp(), index.in_h_start, index.in_h)) {
          return false;
        }
      }
      return true;
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_PC_Conv2DOp>(op)) {
      bool is_dw, with_bias, do_relu;
      int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w;
      int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;
      parseConvParam(cast_op.param(), false, cast_op.input(), cast_op.output(),
                     cast_op.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw,
                     ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                     with_bias, do_relu, pad_value);

      if (dh > 1) {
        kh = dh * (kh - 1) + 1;
      }
      index.in_h_start = (current_idx == 0 ? 0 : index.out_h_start * sh - pt);
      if (current_idx == 0) {
        index.in_h = (index.out_h - 1) * sh + kh - pt;
      } else if (current_idx == num_piece - 1) {
        index.in_h = ih - index.in_h_start;
      } else {
        index.in_h = (index.out_h - 1) * sh + kh;
      }
      return backward(cast_op.input().getDefiningOp(), index.in_h_start,
                      index.in_h);
    }
    return false;
  }

  bool do_slice() {
    auto shape = getTensorShape(last_op->getResult(0));
    if (shape.size() < 3) {
      return false;
    }
    int last_h = shape[2];
    int h_step = (last_h + num_piece - 1) / num_piece;
    current_idx = 0;
    for (int h_pos = 0; h_pos < last_h; h_pos += h_step, ++current_idx) {
      auto h = std::min(h_step, last_h - h_pos);
      if (false == backward(last_op, h_pos, h)) {
        return false;
      }
    }
    if (first_slice.size() != num_piece) {
      return false;
    }
    for (auto &pair : op_h_map) {
      if (pair.second.slice.size() != num_piece) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  void copy_tensor_inner(StringRef from_name, StringRef to_name,
                         TensorType type, TensorFile *wTF) {
    auto tensor = wTF->readTensor<T>(from_name, type);
    wTF->addTensor<T>(to_name, tensor->data(), type);
    if (current_idx == num_piece - 1) {
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
    auto name = weight_op.name().str() + "_tod_" + std::to_string(current_idx);
    auto type = weight_op.getResult().getType().template cast<TensorType>();
    copy_tensor(weight_op.name(), name, type, wTF);
    if (current_idx == num_piece - 1) {
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

  void forward(OpBuilder &builder, Operation *op) {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    Operation *input_op;
    if (op == start_op) {
      input_op = first_slice[current_idx].op;
    } else {
      auto tmp = op->getOperand(0).getDefiningOp();
      input_op = op_h_map[tmp].slice[current_idx].op;
    }
    auto &current = op_h_map[op].slice[current_idx];
    auto input_shape = getTensorShape(op->getResult(0));
    auto output_shape = input_shape;
    output_shape[2] = current.out_h;
    auto type = RankedTensorType::get(
        output_shape,
        op->getResult(0).getType().cast<TensorType>().getElementType());
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::TG_INT8_EltwiseAddOp>(op)) {
      // make sure all input ops are ready
      for (auto input : cast_op.inputs()) {
        auto &slice = op_h_map[input.getDefiningOp()].slice[current_idx];
        if (slice.op == nullptr) {
          return;
        }
      }
      for (auto input : cast_op.inputs()) {
        auto &slice = op_h_map[input.getDefiningOp()].slice[current_idx];
        if (current.in_h == slice.out_h &&
            current.in_h_start == slice.out_h_start) {
          operands.push_back(slice.op->getResult(0));
        } else {
          auto crop_op = create_crop_op(builder, slice.op,
                                        current.in_h_start - slice.out_h_start,
                                        current.in_h);
          operands.push_back(crop_op);
        }
      }
      std::string name =
          cast_op.name().str() + "_tod_" + std::to_string(current_idx);
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
      current.op = newOp.getOperation();
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::TG_INT8_PC_Conv2DOp>(op)) {
      auto filter = copy_weight(builder, op, cast_op.filter());
      auto pc_info = copy_weight(builder, op, cast_op.pc_info());
      operands.push_back(input_op->getResult(0));
      operands.push_back(filter);
      operands.push_back(pc_info);
      std::string name =
          cast_op.name().str() + "_tod_" + std::to_string(current_idx);

      auto p = cast_op.paramAttr();
      int pad_t = p.padding_t().getInt();
      if (current_idx != 0) {
        pad_t = 0;
      }
      int pad_b = p.padding_b().getInt();
      if (current_idx != num_piece - 1) {
        pad_b = 0;
      }

      for (auto &pair : op->getAttrs()) {
        if (pair.first == "name") {
          attrs.push_back(
              builder.getNamedAttr("name", builder.getStringAttr(name)));
        } else if (pair.first == "param") {
          attrs.push_back(builder.getNamedAttr(
              "param",
              tpu::ConvParam::get(
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
      auto newOp = builder.create<tpu::TG_INT8_PC_Conv2DOp>(
          op->getLoc(), type, ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});
      current.op = newOp.getOperation();
    }
    if (op == last_op) {
      return;
    }
    for (auto &use : op->getResult(0).getUses()) {
      auto sub_op = use.getOwner();
      forward(builder, sub_op);
    }
  }

  Value create_crop_op(OpBuilder &builder, Operation *op, int h_start,
                       int h_slice) {
    auto shape = getTensorShape(op->getResult(0));
    std::vector<int> crop_shape(shape.begin(), shape.end());
    crop_shape[2] = h_slice;
    std::vector<int> offset(shape.size(), 0);
    offset[2] = h_start;
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
    std::string name =
        tpuOp.getOpName().str() + "_tod_" + std::to_string(current_idx);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
    attrs.push_back(builder.getNamedAttr("crop_shape",
                                         builder.getI32ArrayAttr(crop_shape)));
    attrs.push_back(
        builder.getNamedAttr("crop_offset", builder.getI32ArrayAttr(offset)));
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
    auto &slice = op_h_map[last_op].slice;
    for (auto &s : slice) {
      operands.push_back(s.op->getResult(0));
    }
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(last_op);
    std::string name = tpuOp.getOpName().str();
    attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
    attrs.push_back(builder.getNamedAttr("axis", builder.getI32IntegerAttr(2)));
    auto new_op = builder.create<tpu::TG_INT8_ConcatOp>(
        last_op->getLoc(), last_op->getResult(0).getType(),
        ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    for (auto &use : last_uses) {
      for (uint32_t i = 0; i < use->getNumOperands(); i++) {
        if (use->getOperand(i).getDefiningOp() == last_op) {
          use->setOperand(i, new_op);
        }
      }
    }
#if 0
    // needn't erase, or will corrupt
    for (auto &op : op_set) {
      op->erase();
    }
    for (auto &op : weight_set) {
      op->erase();
    }
#endif
  }

  void do_process(FuncOp &fn, OpBuilder &builder) {
    llvm::errs() << "============ tg op divide ===========================\n";
    for (auto &info : op_data) {
      auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(info.op);
      llvm::errs() << "op:" << tpuOp.getOpName() << ", size: " << info.max_size
                   << "\n";
    }
    llvm::errs() << "divide to [" << num_piece << "] pieces\n";
    auto input_op = start_op->getOperand(0);
    builder.setInsertionPointAfter(last_op);
    for (uint32_t i = 0; i < num_piece; i++) {
      current_idx = i;
      auto crop_op =
          create_crop_op(builder, input_op.getDefiningOp(),
                         first_slice[i].out_h_start, first_slice[i].out_h);
      first_slice[i].op = crop_op.getDefiningOp();
      forward(builder, start_op);
    }
    concat_all(fn, builder);
  }

  void runOnFunction() override {
    auto fn = getFunction();
    MInfo::getChipInfo(fn);
    auto *context = &getContext();
    auto builder = OpBuilder(context);
    if (init_op_set(fn) == false) {
      llvm::errs() << "tg-op-divide op set failed\n";
      return;
    }
    if (decide_piece(fn) == false) {
      llvm::errs() << "tg-op-divide piece failed\n";
      return;
    }
    if (do_slice() == false) {
      llvm::errs() << "tg-op-divide slice failed\n";
      return;
    }
    do_process(fn, builder);
  }

private:
  uint32_t num_piece;
  uint32_t current_idx;
  std::vector<main_op_t> op_data;
  std::map<Operation *, op_info_t> op_h_map;
  std::set<Operation *> op_set;
  std::set<Operation *> weight_set;
  std::vector<Operation *> last_uses;
  Operation *start_op;
  Operation *last_op;
  std::vector<h_slice_t> first_slice;
  int64_t max_size;
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createTgOpDividePass() {
  return std::make_unique<TgOpDividePass>();
}
