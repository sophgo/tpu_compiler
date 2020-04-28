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
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Tensor.hpp"

namespace mlir {

using namespace std;

typedef enum {
  IR_CONVOLUTION = 0,
  IR_DECONVOLUTION,
  IR_POOLING,
  IR_LRN,
  IR_INNERPRODUCT,
  IR_RELU,
  IR_CONCAT,
  IR_BATCHNORM,
  IR_SCALE,
  IR_MAC,
  IR_ELTWISE,
  IR_PRELU,
  IR_ACTIVATION,
  IR_UPSAMPLE,
  IR_SHUFFLECHANNEL,
  IR_SLICE,
  IR_ARITHMETIC,
  IR_QUANTIZATION,
  IR_JOIN,
  IR_MULTIINPUT,
  IR_OTHER,
} IR_TYPE;

class ImLayer {
 public:
  ImLayer(IR_TYPE type, Operation *op, bool fusible = false);
  virtual ~ImLayer();

  IR_TYPE type() const { return type_; }

  const string &name() const { return name_; }

  Operation *op() const { return op_; }

  void set_id(int id) { id_ = id; }
  int id() const { return id_; }

  void set_type(IR_TYPE type) { type_ = type; }
  const string &type_name() const { return type_name_; }

  void add_in_tensor(int n, int c, int h, int w, int unit_size, string& storage, const string &name,
                     tensor_type_t type, gaddr_t gaddr = 0xFFFFFFFF);

  void add_in_tensor(ShapedType* shape, const string &name, tensor_type_t type,
                     gaddr_t gaddr = 0xFFFFFFFF);

  void add_in_tensor(Value * op, tensor_type_t type, gaddr_t gddr = 0xFFFFFFFF);

  void add_out_tensor(Value * op, tensor_type_t type, gaddr_t gaddr = 0xFFFFFFFF);

  void add_imm_tensor(const shared_ptr<Tensor> associcate, int count, const string &name);
  // // clear temp_data if has.
  virtual void clear_temp_data() {}

  static shared_ptr<ImLayer> create(Operation *op);
  static void register_it(shared_ptr<ImLayer> &layer);
  static void unregister_all();
  static vector<shared_ptr<ImLayer>> layers;


  bool is_inplace_layer;
  bool do_relu;
  vector<shared_ptr<Tensor>> in_tensors;
  vector<shared_ptr<Tensor>> out_tensors;
  vector<shared_ptr<Tensor>> imm_tensors;
  bool is_tg_layer;
  bool fusible;  // if could fuse to other IRs.

 protected:
  int id_;
  IR_TYPE type_;
  string type_name_;
  string name_;
  Operation *op_;
};

class ImConv : public ImLayer {
 public:
  explicit ImConv(Operation *op);
  bool conv1x1_to_fc;
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

class ImLrn : public ImLayer {
 public:
  explicit ImLrn(Operation *op);
};

class ImConcat : public ImLayer {
 public:
  explicit ImConcat(Operation *op);
  set<int> ignored_bottoms;

  void clear_temp_data() override { ignored_bottoms.clear(); }
};

class ImUpsample : public ImLayer {
 public:
  explicit ImUpsample(Operation *op);
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

class ImCommon : public ImLayer {
 public:
  ImCommon(Operation *op, bool inplace_compute, IR_TYPE type = IR_OTHER);
};

}
#endif
