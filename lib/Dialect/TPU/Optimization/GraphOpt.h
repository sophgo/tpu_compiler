#pragma once

#include <memory>
#include <vector>
#include <list>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

struct Vertex {
  int id = -1;
  Operation *op = nullptr;
  std::vector<Operation *> depend_ops;
  std::vector<int> parents;
  std::vector<int> children;
  size_t size = 0;
  int visit_count = 0;
  bool visited = false;
  bool marked = false;
  bool fusible = false;
  bool fused = false;
};

class Graph {
public:
  virtual ~Graph();
  virtual int add_vertex(Operation *op);
  virtual void print();
  virtual int rearrang(std::list<std::shared_ptr<Vertex> > &vertex_list);
  virtual int rebuild_op(const std::list<std::shared_ptr<Vertex>> &vertex_list);

  size_t get_op_size(Operation *op);
  bool check_fusible(Operation *op);

public:
  std::vector<std::vector<std::shared_ptr<Vertex> > > edges;
  std::map<Operation *, std::shared_ptr<Vertex>> vertics;
  std::list<std::shared_ptr<Vertex>> head;
  std::list<std::shared_ptr<Vertex>> tail;
  uint32_t vertex_num = 0;
};

class GraphFuseContinousOp : public Graph {
public:
  int rearrang(std::list<std::shared_ptr<Vertex> > &vertex_list) override;
};

class GraphFuseContinousOp2 : public Graph {
public:
  int rearrang(std::list<std::shared_ptr<Vertex> > &vertex_list) override;
};

class GraphFuseOp : public Graph {
public:
  int rearrang(std::list<std::shared_ptr<Vertex> > &vertex_list) override;
};

struct GraphFactory {
  static std::unique_ptr<Graph> getInstance(int type) {
    switch (type) {
    case 1:
      return std::unique_ptr<Graph>(new Graph());
    case 2:
      return std::unique_ptr<Graph>(dynamic_cast<Graph *>(new GraphFuseContinousOp()));
    case 3:
      return std::unique_ptr<Graph>(dynamic_cast<Graph *>(new GraphFuseContinousOp2()));
    case 4:
      return std::unique_ptr<Graph>(dynamic_cast<Graph *>(new GraphFuseOp()));
    default:
      llvm::errs() << "unsupport graph opt type!\n";
      return std::unique_ptr<Graph>(nullptr);
    }
  }
};

} //namespace mlir