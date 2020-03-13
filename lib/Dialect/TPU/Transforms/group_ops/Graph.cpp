#include "Graph.hpp"

namespace mlir {

// LayerOp

int64_t LayerOp::top_size() {
  return op_->getNumResults();
}

int64_t LayerOp::bottom_size() {
  return op_->getNumOperands();
}

llvm::StringRef LayerOp::top(int idx) {
  auto op_top = op_->getResult(idx)->getDefiningOp();
  if (op_top) {
    auto name = mlir::getOpName(op_top);
    return name;
  }
  else
    return llvm::StringRef();
}

llvm::StringRef LayerOp::bottom(int idx) {
  auto op_bottom = op_->getOperand(idx)->getDefiningOp();
  if (op_bottom) {
    auto name = mlir::getOpName(op_bottom);
    return name;
  }
  else
    return llvm::StringRef();
}

llvm::StringRef LayerOp::name() {
  auto op_name = mlir::getOpName(op_);
  return op_name;
}

uint64_t LayerOp::id() {
  uint64_t layer_id = mlir::getOpLayerId(op_);
  return layer_id;
}

// Graph
Graph::Graph() {}
Graph::~Graph() {}

// construct a graph from the mlir module
Graph::Graph(FuncOp *fn) { input_from_function (fn); }

void Graph::build_start() {
  assert(is_doing_build_ == false);
  root_node_ = nullptr;
  is_doing_build_ = true;
}

void Graph::build_end() {
  assert(is_doing_build_ == true);
  is_doing_build_ = false;
}

void Graph::input_from_function(FuncOp *fn) {
  build_start();
  fn->walk([&](Operation * op) {
    // skip func
    if (isa<tpu::LoadWeightOp>(op) || isa<tpu::LoadFileOp>(op) ||
        isa<tpu::NoneOp>(op)) {;}
    else
      add_node(op);
  });
  build_end();
  //name_ = in_net->name();
}

Node *Graph::add_node(Operation *p) {
  LayerOp * op = new LayerOp(p);
  assert(is_doing_build_ ==  true);

  Node *new_node = new Node(this, op);

  if (root_node_ == nullptr) {
    root_node_ = new_node;
  }

  for (int i = 0; i < op->top_size(); i++) {
    auto name = op->top(i);
    llvm::errs() << "top name :" << name << "\n";
    assert(edge_list_.find(name) == edge_list_.end());

    auto edge = std::make_shared<Edge>(name);
    edge->defined = new_node;
    edge_list_[name] = edge;
  }

  for (int i = 0; i < op->bottom_size(); i++) {
    auto name = op->bottom(i);
    llvm::errs() << "bottom name " << name << "\n";
    if (!name.empty()) {
      assert(edge_list_[name]->used == nullptr);
      edge_list_[name]->used = new_node;
    }
  }

  // // If the added node does not have a id, give one.
  // assert(new_node->op()->has_id());
      // << "Node " << new_node->op()->name() << " does not contain an id.";
  for (size_t i = 0; i < nodes_.size(); i++) {
    assert(nodes_[i]->op()->id() != new_node->op()->id());
        // << "Oops. Duplicated id " << new_node->op()->id() << " for node " << nodes_[i]->op()->name()
        // << " and " << new_node->op()->name();
  }

  nodes_.push_back(new_node);
  return new_node;
}

}
