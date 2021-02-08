/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_IMLAYER_H
#define GROUPOPS_IMLAYER_H

#include <fstream>
#include <string>
#include <iostream>
#include <set>
#include <algorithm>
#include "mlir/IR/Builders.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Tensor.hpp"

namespace mlir {

typedef enum {
  IR_ABS = 0,
  IR_CONVOLUTION,
  IR_DECONVOLUTION,
  IR_POOLING,
  IR_Scale,
  IR_LRN,
  IR_INNERPRODUCT,
  IR_RELU,
  IR_CONCAT,
  IR_QUANT,
  IR_BATCHNORM,
  IR_SCALE,
  IR_MAC,
  IR_ELTWISE,
  IR_PRELU,
  IR_LEAKY_RELU,
  IR_ACTIVATION,
  IR_UPSAMPLE,
  IR_SHUFFLECHANNEL,
  IR_SLICE,
  IR_ARITHMETIC,
  IR_PAD,
  IR_CROP,
  IR_JOIN,
  IR_MULTIINPUT,
  IR_CAST,
  IR_ZERO_MASK,
  IR_MATMUL,
  IR_OTHER,
} IR_TYPE;

class ImLayer {
 public:
  ImLayer(IR_TYPE type, Operation *op, bool fusible = false);
  virtual ~ImLayer();

  IR_TYPE type() const { return type_; }

  const std::string &name() const { return name_; }

  int layer_id() const { return layer_id_; }

  Operation *op() const { return op_; }

  void set_id(int id) { id_ = id; }
  int id() const { return id_; }

  void set_type(IR_TYPE type) { type_ = type; }
  const std::string &type_name() const { return type_name_; }

  void add_in_tensor(int n, int c, int h, int w, int unit_size, std::string& storage, const std::string &name,
                     tensor_type_t type);

  void add_in_tensor(ShapedType* shape, const std::string &name, tensor_type_t type);

  void add_in_tensor(Value op, tensor_type_t type);

  void add_out_tensor(Value op, tensor_type_t type, std::string storage = "INT8");

  void add_imm_tensor(const std::shared_ptr<Tensor> associcate, int count, const std::string &name);
  // // clear temp_data if has.
  virtual void clear_temp_data() {}

  static std::shared_ptr<ImLayer> create(Operation *op);
  static std::string getStorage(Value v);
  static void register_it(std::shared_ptr<ImLayer> &layer);
  static void unregister_all();
  static std::vector<std::shared_ptr<ImLayer>> layers;


  bool is_inplace_layer;
  bool do_relu;
  std::vector<std::shared_ptr<Tensor>> in_tensors;
  std::vector<std::shared_ptr<Tensor>> out_tensors;
  std::vector<std::shared_ptr<Tensor>> imm_tensors;
  bool is_tg_layer;
  bool fusible;  // if could fuse to other IRs.

 protected:
  int id_;
  IR_TYPE type_;
  std::string type_name_;
  std::string name_;
  Operation *op_;
  int layer_id_; // keep in mlir SSA
};

class ImConv : public ImLayer {
 public:
  explicit ImConv(Operation *op);
};

class ImPooling : public ImLayer {
 public:
  explicit ImPooling(Operation *op);
};

class ImInnerproduct : public ImLayer {
 public:
  explicit ImInnerproduct(Operation *op);
};

class ImEltwise : public ImLayer {
 public:
  explicit ImEltwise(Operation *op);
};

class ImBatchnorm : public ImLayer {
 public:
  explicit ImBatchnorm(Operation *op);
};

class ImScale : public ImLayer {
 public:
  explicit ImScale(Operation *op);
};

class ImMac : public ImLayer {
 public:
  explicit ImMac(Operation *op);
};

class ImActivation : public ImLayer {
 public:
  explicit ImActivation(Operation *op);
};

class ImAbs : public ImLayer {
 public:
  explicit ImAbs(Operation *op);
};



class ImLrn : public ImLayer {
 public:
  explicit ImLrn(Operation *op);
};

class ImConcat : public ImLayer {
 public:
  explicit ImConcat(Operation *op);
  std::set<int> ignored_bottoms;

  void clear_temp_data() override { ignored_bottoms.clear(); }
};

class ImUpsample : public ImLayer {
 public:
  explicit ImUpsample(Operation *op);
};

class ImPRelu : public ImLayer {
 public:
  explicit ImPRelu(Operation *op);
};

class ImLeakyRelu : public ImLayer {
 public:
  explicit ImLeakyRelu(Operation *op);
};

class ImDeconv : public ImLayer {
 public:
  explicit ImDeconv(Operation *op);
};

class ImShuffleChannel : public ImLayer {
 public:
  explicit ImShuffleChannel(Operation *op);
};

class ImSlice : public ImLayer {
 public:
  explicit ImSlice(Operation *op);
};

class ImArithmetic : public ImLayer {
 public:
  explicit ImArithmetic(Operation *op);
};

class ImQuantization : public ImLayer {
 public:
  explicit ImQuantization(Operation *op);
};

class ImPad : public ImLayer {
 public:
  explicit ImPad(Operation *op);
};

class ImCrop : public ImLayer {
 public:
  explicit ImCrop(Operation *op);
};

class ImRelu : public ImLayer {
 public:
  explicit ImRelu(Operation *op);
};

class ImCast : public ImLayer {
 public:
  explicit ImCast(Operation *op);
};

class ImCommon : public ImLayer {
 public:
  ImCommon(Operation *op, bool inplace_compute, IR_TYPE type = IR_OTHER);
};

class ImQuant : public ImLayer {
 public:
  explicit ImQuant(Operation *op);
};

class ImZeroMask : public ImLayer {
 public:
  explicit ImZeroMask(Operation *op);
};

class ImMatMul : public ImLayer {
  public:
   explicit ImMatMul(Operation *op);
};

}
#endif
