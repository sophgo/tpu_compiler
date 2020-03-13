#ifndef GROUPOPS_GRAPH_H
#define GROUPOPS_GRAPH_H

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


namespace mlir {
class Node;

class Edge;

class LayerOp {
public:
  LayerOp(Operation * op){
    op_ = op;
  }

  int64_t top_size();
  int64_t bottom_size();
  uint64_t id();
  // get top tensor name
  llvm::StringRef top(int idx);
  llvm::StringRef bottom(int idx);
  llvm::StringRef name();


private:
  Operation * op_;
};


class NetParameter {
public:
  NetParameter() {};
private:

};

class Graph {
 public:
  Graph();
  Graph(Graph *);
  Graph(FuncOp *op);
  virtual ~Graph();

  void build_start();
  void build_end();
  void input_from_function(FuncOp *op);
  Node *add_node(Operation *op);
  // void build_output_nodes(std::vector<Node *>& outputs);
  // Node *traverse_start();
  // Node *traverse_next();
  // void traverse_end();
  // void traverse_quit();
  // // suspend traversing current branch,
  // // switch to another one.
  // bool traverse_suspend();
  // void remove_node(Node *node);
  // void insert_node(const LayerOp &op);
  // void insert_before_node(const LayerOp &op);
  // void delete_node(Node *node);
  // void set_name(std::string name) { name_ = name; }
  const std::string& name() { return name_; }
  // void fixup_caffe_net_crop(NetParameter &param, NetParameter *param_fixed);
  // void output_to_net(NetParameter *out_net);
  // void input_from_net(NetParameter *in_net);
  // void dump(const std::string& tag);
  // std::map<std::string, std::shared_ptr<Edge>>& edge_list() { return edge_list_; }
private:

  Node *root_node_ = nullptr;
  std::string name_;
  //
  // build graph and traverse are exclusive states
  // can not do at the same time
  //
  bool is_doing_build_ = false;
  // bool is_doing_traverse_ = false;
  // Node *node_cur_ = nullptr;
  // std::vector<Node *> split_list_;
  // std::vector<Node *> suspend_list_;
  std::vector<Node *> nodes_;
  std::map<std::string, std::shared_ptr<Edge>> edge_list_;

  // bool is_split_node(Node *node);
  // bool is_merge_node(Node *node);
  // bool is_child_of(Node *parent, Node *child);
};

class Edge {
public:
  Edge(const std::string& name)
    : used(nullptr), defined(nullptr), name_(name) {}
  std::string& name() { return name_; }
  Node* used;
  Node* defined;
  //gaddr_t addr;
private:
  std::string name_;
  //TensorShape shape_;
};

class Node {
 public:
  Node(Graph* graph) : graph_(graph) {}
  Node(Graph* graph, LayerOp* op)
     : graph_(graph),op_(op) {}
  // virtual ~Node() {}
  // int prev_size() { return op_.bottom_size(); }
  // int next_size() { return op_.top_size(); }
  // Node *prev(int index) {
  //   auto name = op_.bottom(index);
  //   auto edge = graph_->edge_list()[name];
  //   CHECK(edge->defined != nullptr) << "undefined edge:" << name;
  //   return edge->defined;
  // }

  // Node *next(int index) {
  //   auto name = op_.top(index);
  //   auto edge = graph_->edge_list()[name];
  //   CHECK(edge->used != nullptr) << "unused edge: " << name;
  //   return edge->used;
  // }

  // void set_prev(Node *node, const std::string& name) {
  //   auto edge = graph_->edge_list()[name];
  //   edge->defined = node;
  //   edge->used = this;
  // }

  // void set_next(Node *node, const std::string& name) {
  //   auto edge = graph_->edge_list()[name];
  //   edge->defined = this;
  //   edge->used = node;
  // }

  // // if a node has more than one edges and
  // // not all of his edges are used by other nodes.
  // bool is_dangling_node() {
  //   int dangling = 0;
  //   for (auto& top : op_.top()) {
  //     auto edge = graph_->edge_list()[top];
  //     if (edge->used == nullptr)
  //       dangling ++;
  //   }
  //   if (dangling)
  //     return false;
  //   return true;
  // }

  // bool is_top_used(int top_index) {
  //   auto edge = graph_->edge_list()[op_.top(top_index)];
  //   if (edge->used == nullptr) {
  //     return false;
  //   }
  //   return true;
  // }

  LayerOp *op() { return op_; }
  // std::string op_name() { return op_.name(); }
  // std::string op_type() { return op_.type(); }
  // // for graph traverse
  // int cur_index() { return cur_index_; }
  // void inc_cur_index() { cur_index_++; }
  // void reset_cur_index() { cur_index_ = 0; }

 private:
  Graph* graph_;
  LayerOp *op_;
  // for graph traverse
  int cur_index_ = 0;
};


}

#endif