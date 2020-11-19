#ifndef GROUPOPS_NETGRAPH_H
#define GROUPOPS_NETGRAPH_H

#include <fstream>
#include <iostream>
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

#include "utils.hpp"
#include "ImLayer.hpp"

namespace mlir {

class NetGraph {
 public:
  explicit NetGraph(FuncOp* fn){};
  ~NetGraph();

  void parse_graph(FuncOp * fn);
  const ImLayer* get_layer_by_op(Operation * op);
//   // get tensor type
  tensor_type_t get_tensor_type(int tensor_id);
  int getImLayerSize();
  bool layer_inplace_compute(int layer_id);

//   // get layer parameter
  const ImLayer* get_layer_by_id(int layer_id);
  const std::vector<int>& get_in_tensors_of_layer(int layer_id);
  const std::vector<int>& get_out_tensors_of_layer(int layer_id);

  int get_tensor_nums(int tensor_id);
  int get_tensor_channels(int tensor_id);
  int get_tensor_height(int tensor_id);
  int get_tensor_width(int tensor_id);
  int get_tensor_unit_size(int tensor_id);
  void get_tensor_dim(int tensor_id, int* tensor_dim);
  gaddr_t get_tensor_tsm_mem(int tensor_id);
  int get_tensor_local_offset(int tensor_id);

//   // get tensor parameter
  Tensor* get_tensor_by_id(int id);
  void set_tensor_local_offest(int tensor_id, int local_mem_offset);
  void set_tensor_tsm_offest(int tensor_id, gaddr_t gaddr);

  void set_tensor_num_height_slice(int tensor_id, int n_idx, int n_slice, int h_idx, int h_slice,
                                   bool h_slice_skip_first, bool h_slice_skip_last);
  void set_tensor_height_slice_max(int tensor_id, int h_slice_max);

  int get_tensor_from_layer(int tensor_id);
  const std::vector<int>& get_tensor_to_layer(int tensor_id);

  bool is_concat_special_case(int layer_id, int tid, int cluster_size = 0);

  bool is_concat_optimized_case(int layer_id, int tid, int cluster_size = 0);

//  private:
  std::map<int, std::pair<std::vector<int>, std::vector<int> > > layer_id_to_inout_tensor;
  std::map<int, std::vector<int> > tensor_to_layer_id;
  std::map<int, int> tensor_from_layer_id;
  std::vector<int> dummy;
};

}

#endif