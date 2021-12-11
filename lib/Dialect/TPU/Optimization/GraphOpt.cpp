#include "GraphOpt.h"
#define DEBUG_TYPE "graph_opt"

namespace mlir {

static llvm::cl::OptionCategory clOptionsCategory("Graph Optimizer Option");
static llvm::cl::opt<int> clGraphOptType(
    "graph-opt-method",
    llvm::cl::desc("choose graph optimizer method. range[1-4]"),
    llvm::cl::init(2),
    llvm::cl::cat(clOptionsCategory));

Graph::~Graph() {}

int Graph::add_vertex(Operation *op) {
  if (!op) {
    return 0;
  }
  if (vertics.end() != vertics.find(op)) {
    assert(0);
    return -1;
  }
  if (isa<tpu::LoadWeightOp>(op)) {
    return 0;
  }

  std::shared_ptr<Vertex> vertex = std::make_shared<Vertex>();
  vertex->op = op;
  if (isa<tpu::NoneOp>(op) || isa<tpu::WeightFileOp>(op)) {
    head.emplace_back(vertex);
    vertics[op] = vertex;
    return 0;
  }
  if (isa<FuncOp>(op)) {
    tail.emplace_back(vertex);
    vertics[op] = vertex;
    return 0;
  }

  vertex->id = vertex_num++;
  for (auto operand : op->getOperands()) {
    auto opdOp = operand.getDefiningOp();
    if (!opdOp || isa<tpu::NoneOp>(opdOp) || isa<tpu::WeightFileOp>(opdOp) ||
        isa<FuncOp>(opdOp)) {
      continue;
    } else if (isa<tpu::LoadWeightOp>(opdOp)) {
      vertex->depend_ops.emplace_back(opdOp);
    } else {
      auto iter = vertics.find(opdOp);
      if (vertics.end() == iter) {
        assert(0 && "error add vertex!");
        return -1;
      }
      vertex->parents.emplace_back(iter->second->id);
      iter->second->children.emplace_back(vertex->id);
    }
  }
  for (auto id : vertex->parents) {
    edges[id].emplace_back(vertex);
  }
  if (!isa<ReturnOp>(op)) {
    vertex->size = get_op_size(op);
  }
  vertics[op] = vertex;
  edges.emplace_back();
  edges.back().emplace_back(vertex);
  assert(vertex_num == edges.size());
  return 0;
}

void Graph::print() {
  llvm::errs() << "==========PRINT graph===============\n";
  int idx = 0;
  for (auto &v : head) {
    llvm::errs() << "H :" << idx++ << "\t" << v->op->getName().getStringRef()
                 << "\n";
  }
  for (uint32_t i = 0; i < edges.size(); ++i) {
    llvm::errs() << "V " << i << "\t:";
    for (auto &v : edges[i]) {
      llvm::errs() << v->id << ",";
    }
    llvm::errs() << "\t" << (*edges[i].begin())->op->getName().getStringRef()
                 << ":" << getOpName((*edges[i].begin())->op) << "\n";
  }
  idx = 0;
  for (auto &v : tail) {
    llvm::errs() << "T :" << idx++ << "\t" << v->op->getName().getStringRef()
                 << "\n";
  }
  llvm::errs() << "==========PRINT graph end===============\n";
}

size_t Graph::get_op_size(Operation *op) {
  size_t dsize = 1;
  auto type = op->getResult(0).getType().template cast<TensorType>();
  std::vector<int64_t> shape = type.getShape();
  auto count = std::accumulate(std::begin(shape), std::end(shape), 1,
                               std::multiplies<>());
  auto elementType = type.getElementType();
  if (elementType.isF32()) {
    dsize = sizeof(float);
  } else if (elementType.isInteger(8)) {
    dsize = sizeof(int8_t);
  } else if (elementType.isInteger(16)) {
    dsize = sizeof(int16_t);
  } else if (elementType.isInteger(32)) {
    dsize = sizeof(int32_t);
  } else if (elementType.isBF16()) {
    dsize = sizeof(uint16_t);
  } else {
    llvm_unreachable("unsupported data type");
  }
  return count * dsize;
}

bool Graph::check_fusible(Operation *op) {
  if (isa<tpu::TG_INT8_AbsOp>(op) || isa<tpu::TG_BF16_AbsOp>(op) ||
      isa<tpu::TG_INT8_Conv2DOp>(op) || isa<tpu::TG_BF16_Conv2DOp>(op) ||
      isa<tpu::TG_INT8_DeConv2DOp>(op) || isa<tpu::TG_BF16_DeConv2DOp>(op) ||
      isa<tpu::TG_INT8_EltwiseAddOp>(op) ||
      isa<tpu::TG_INT8_EltwiseMulOp>(op) ||
      isa<tpu::TG_BF16_EltwiseAddOp>(op) ||
      isa<tpu::TG_BF16_EltwiseMulOp>(op) ||
      isa<tpu::TG_INT8_FullyConnectedOp>(op) ||
      isa<tpu::TG_BF16_FullyConnectedOp>(op) ||
      isa<tpu::TG_INT8_MatMulOp>(op) || isa<tpu::TG_BF16_MatMulOp>(op) ||
      isa<tpu::ReshapeOp>(op) || isa<tpu::TG_INT8_PoolAvg2DOp>(op) ||
      isa<tpu::TG_INT8_PoolMax2DOp>(op) || isa<tpu::TG_BF16_PoolAvg2DOp>(op) ||
      isa<tpu::TG_BF16_PoolMax2DOp>(op) || isa<tpu::TG_INT8_ConcatOp>(op) ||
      isa<tpu::TG_BF16_ConcatOp>(op) || isa<tpu::TG_INT8_LutOp>(op) ||
      isa<tpu::TG_BF16_LutOp>(op) || isa<tpu::TG_INT8_PReluOp>(op) ||
      isa<tpu::TG_BF16_PReluOp>(op) || isa<tpu::TG_INT8_ShuffleChannelOp>(op) ||
      isa<tpu::TG_BF16_ShuffleChannelOp>(op) || isa<tpu::TG_INT8_SwapChannelOp>(op) ||
      isa<tpu::TG_BF16_SwapChannelOp>(op) || isa<tpu::TG_INT8_LrnOp>(op) ||
      isa<tpu::TG_BF16_LrnOp>(op) || isa<tpu::TG_INT8_ScaleOp>(op) ||
      isa<tpu::TG_BF16_ScaleOp>(op) || isa<tpu::TG_INT8_ScaleLutOp>(op) ||
      isa<tpu::TG_INT8_MulConstOp>(op) || isa<tpu::TG_BF16_MulConstOp>(op) ||
      isa<tpu::TG_INT8_UpsampleOp>(op) || isa<tpu::TG_BF16_UpsampleOp>(op) ||
      isa<tpu::TG_INT8_LeakyReluOp>(op) || isa<tpu::TG_BF16_LeakyReluOp>(op) ||
      isa<tpu::TG_INT8_PadOp>(op) || isa<tpu::TG_BF16_PadOp>(op) ||
      isa<tpu::TG_INT8_CropOp>(op) || isa<tpu::TG_BF16_CropOp>(op) ||
      isa<tpu::TG_INT8_ReluOp>(op) || isa<tpu::TG_BF16_ReluOp>(op) ||
      isa<tpu::GenericCpuOp>(op) || isa<tpu::TG_QuantOp>(op) ||
      isa<tpu::InputOp>(op)) {
    return true;
  }
  return false;
}

int Graph::rebuild_op(const std::list<std::shared_ptr<Vertex>> &vertex_list) {
  if (vertex_list.empty()) {
    return 0;
  }
  auto block = (*vertex_list.begin())->op->getBlock();
  for (auto iter = vertex_list.rbegin(); iter != vertex_list.rend(); ++iter) {
    auto &vertex = *iter;
    vertex->op->moveBefore(&(*block->begin()));
    for (auto depend_iter = vertex->depend_ops.rbegin();
         depend_iter != vertex->depend_ops.rend(); ++depend_iter) {
      (*depend_iter)->moveBefore(&(*block->begin()));
    }
  }
  return 0;
}

int Graph::rearrang(std::list<std::shared_ptr<Vertex>> &vertex_list) {
  std::list<int> queue;
  vertex_list.clear();

  if (edges.empty()) {
    return 0;
  }

  queue.emplace_front(0);
  while (!queue.empty()) {
    int cur_id = queue.front();
    queue.pop_front();

    auto &vertex = edges[cur_id][0];

    // already visited
    if (vertex->visited) {
      continue;
    }

    // visit parent vertex first
    if (vertex->visit_count < static_cast<int>(vertex->parents.size())) {
      for (auto id : vertex->parents) {
        auto &parents_vertex = edges[id][0];
        if (!parents_vertex->marked) {
          queue.emplace_back(parents_vertex->id);
          parents_vertex->marked = true;
        }
      }
      queue.emplace_back(cur_id);
    } else {
      // visit this vertex
      vertex->visited = true;
      for (uint32_t i = 1; i < edges[cur_id].size(); ++i) {
        auto &children_vertex = edges[cur_id][i];
        children_vertex->visit_count += 1;
        if (!children_vertex->marked) {
          queue.emplace_back(children_vertex->id);
          children_vertex->marked = true;
        }
      }
      vertex_list.emplace_back(vertex);
    }
  }

  for (auto &head_v : head) {
    vertex_list.emplace_front(head_v);
  }
  // for (auto &tail_v : tail) {
  //  vertex_list.emplace_back(tail_v);
  //}
  return 0;
}

int GraphFuseContinousOp::rearrang(std::list<std::shared_ptr<Vertex>> &vertex_list) {
  std::list<int> queue;
  vertex_list.clear();

  if (edges.empty()) {
    return 0;
  }

  // set fusible
  for (uint32_t i = 1; i < edges.size(); ++i) {
    edges[i][0]->fusible = check_fusible(edges[i][0]->op);
  }

  std::list<int> fused_list;
  for (uint32_t i = 1; i < edges.size(); ++i) {
    if (!edges[i][0]->fusible || edges[i][0]->fused) {
      continue;
    }
    fused_list.emplace_back(i);

    // find fusible parent vertex
    if (edges[i][0]->parents.size() == 1) {
      int cur_id = edges[i][0]->parents[0];
      while (1 == edges[cur_id][0]->children.size() &&
             edges[cur_id][0]->fusible && 0 != cur_id) {
        fused_list.emplace_front(cur_id);
        if (edges[cur_id][0]->parents.size() == 1) {
          cur_id = edges[cur_id][0]->parents[0];
        } else {
          break;
        }
      }
    }

    // find fusible children vertex
    if (1 == edges[i][0]->children.size()) {
      int cur_id = edges[i][1]->id;
      while (1 == edges[cur_id][0]->parents.size() &&
             edges[cur_id][0]->fusible) {
        fused_list.emplace_back(cur_id);
        if (1 == edges[cur_id][0]->children.size()) {
          cur_id = edges[cur_id][1]->id;
        } else {
          break;
        }
      }
    }

    if (fused_list.size() > 1) {
      int target_id = fused_list.back();
      auto &target_vertex = edges[target_id][0];
      target_vertex->fused = true;
      std::list<Operation *> depend_ops;

      auto iter = fused_list.rbegin();
      ++iter;
      for (; iter != fused_list.rend(); ++iter) {
        int cur_id = *iter;
        auto &cur_vertex = edges[cur_id][0];
        cur_vertex->fused = true;
        depend_ops.emplace_front(cur_vertex->op);
        for (auto &op : cur_vertex->depend_ops) {
          depend_ops.emplace_front(op);
        }
      }
      target_vertex->depend_ops.insert(target_vertex->depend_ops.begin(),
                                       depend_ops.begin(), depend_ops.end());

      int head_id = fused_list.front();
      auto &head_vertex = edges[head_id][0];
      for (auto &parent : head_vertex->parents) {
        for (auto &child : edges[parent][0]->children) {
          if (child == head_id) {
            child = target_id;
            break;
          }
        }
      }

      target_vertex->parents = head_vertex->parents;
    }
    fused_list.clear();
  }

  queue.emplace_front(0);
  while (!queue.empty()) {
    int cur_id = queue.front();
    queue.pop_front();

    auto &vertex = edges[cur_id][0];

    // already visited
    if (vertex->visited) {
      continue;
    }

    // visit parent vertex first
    if (vertex->visit_count < static_cast<int>(vertex->parents.size())) {
      for (auto id : vertex->parents) {
        auto &parents_vertex = edges[id][0];
        if (parents_vertex->marked) {
          continue;
        }
        queue.emplace_back(parents_vertex->id);
        parents_vertex->marked = true;
      }
      queue.emplace_back(cur_id);
    } else {
      // visit this vertex
      vertex->visited = true;
      for (auto child_id : vertex->children) {
        auto &children_vertex = edges[child_id][0];
        children_vertex->visit_count += 1;
        if (children_vertex->marked) {
          continue;
        }
        if (vertex->fusible && children_vertex->fusible) {
          queue.emplace_front(child_id);
        } else {
          queue.emplace_back(child_id);
        }
        children_vertex->marked = true;
      }
      vertex_list.emplace_back(vertex);
    }
  }

  for (auto &head_v : head) {
    vertex_list.emplace_front(head_v);
  }
  return 0;
}

int GraphFuseContinousOp2::rearrang(std::list<std::shared_ptr<Vertex>> &vertex_list) {
  std::list<int> queue;
  vertex_list.clear();

  if (edges.empty()) {
    return 0;
  }

  // set fusible
  for (uint32_t i = 1; i < edges.size(); ++i) {
    edges[i][0]->fusible = check_fusible(edges[i][0]->op);
  }

  std::list<int> fused_list;
  for (uint32_t i = 1; i < edges.size(); ++i) {
    if (!edges[i][0]->fusible || edges[i][0]->fused) {
      continue;
    }
    fused_list.emplace_back(i);

    // find fusible parent vertex
    if (edges[i][0]->parents.size() == 1) {
      int cur_id = edges[i][0]->parents[0];
      while (1 == edges[cur_id][0]->children.size() &&
             edges[cur_id][0]->fusible && 0 != cur_id) {
        fused_list.emplace_front(cur_id);
        if (edges[cur_id][0]->parents.size() == 1) {
          cur_id = edges[cur_id][0]->parents[0];
        } else {
          break;
        }
      }
    }

    // find fusible children vertex
    if (1 == edges[i][0]->children.size()) {
      int cur_id = edges[i][1]->id;
      while (1 == edges[cur_id][0]->parents.size() &&
             edges[cur_id][0]->fusible) {
        fused_list.emplace_back(cur_id);
        if (1 == edges[cur_id][0]->children.size()) {
          cur_id = edges[cur_id][1]->id;
        } else {
          break;
        }
      }
    }

    if (fused_list.size() > 1) {
      int target_id = fused_list.back();
      auto &target_vertex = edges[target_id][0];
      target_vertex->fused = true;
      std::list<Operation *> depend_ops;

      auto iter = fused_list.rbegin();
      ++iter;
      for (; iter != fused_list.rend(); ++iter) {
        int cur_id = *iter;
        auto &cur_vertex = edges[cur_id][0];
        cur_vertex->fused = true;
        depend_ops.emplace_front(cur_vertex->op);
        for (auto &op : cur_vertex->depend_ops) {
          depend_ops.emplace_front(op);
        }
      }
      target_vertex->depend_ops.insert(target_vertex->depend_ops.begin(),
                                       depend_ops.begin(), depend_ops.end());

      int head_id = fused_list.front();
      auto &head_vertex = edges[head_id][0];
      for (auto &parent : head_vertex->parents) {
        for (auto &child : edges[parent][0]->children) {
          if (child == head_id) {
            child = target_id;
            break;
          }
        }
      }

      target_vertex->parents = head_vertex->parents;
    }
    fused_list.clear();
  }

  queue.emplace_front(0);
  while (!queue.empty()) {
    int cur_id = queue.front();
    queue.pop_front();

    auto &vertex = edges[cur_id][0];

    // already visited
    if (vertex->visited) {
      continue;
    }

    // visit parent vertex first
    if (vertex->visit_count < static_cast<int>(vertex->parents.size())) {
      for (auto id : vertex->parents) {
        auto &parents_vertex = edges[id][0];
        if (parents_vertex->marked) {
          continue;
        }
        queue.emplace_back(parents_vertex->id);
        parents_vertex->marked = true;
      }
      queue.emplace_back(cur_id);
    } else {
      // visit this vertex
      vertex->visited = true;
      for (auto child_id : vertex->children) {
        auto &children_vertex = edges[child_id][0];
        children_vertex->visit_count += 1;
        if (children_vertex->marked) {
          continue;
        }
        queue.emplace_back(child_id);
        children_vertex->marked = true;
      }
      vertex_list.emplace_back(vertex);
    }
  }

  for (auto &head_v : head) {
    vertex_list.emplace_front(head_v);
  }
  return 0;
}

int GraphFuseOp::rearrang(std::list<std::shared_ptr<Vertex>> &vertex_list) {
  std::list<int> queue;
  vertex_list.clear();

  if (edges.empty()) {
    return 0;
  }

  // set fusible
  for (uint32_t i = 1; i < edges.size(); ++i) {
    edges[i][0]->fusible = check_fusible(edges[i][0]->op);
  }

  queue.emplace_front(0);
  while (!queue.empty()) {
    int cur_id = queue.front();
    queue.pop_front();

    auto &vertex = edges[cur_id][0];

    // already visited
    if (vertex->visited) {
      continue;
    }

    // visit parent vertex first
    if (vertex->visit_count < static_cast<int>(vertex->parents.size())) {
      for (auto id : vertex->parents) {
        auto &parents_vertex = edges[id][0];
        if (parents_vertex->marked) {
          continue;
        }
        queue.emplace_back(parents_vertex->id);
        parents_vertex->marked = true;
      }
      queue.emplace_back(cur_id);
    } else {
      // visit this vertex
      vertex->visited = true;
      for (auto child_id : vertex->children) {
        auto &children_vertex = edges[child_id][0];
        children_vertex->visit_count += 1;
        if (children_vertex->marked) {
          continue;
        }
        if (children_vertex->size < vertex->size ||
            (vertex->fusible && children_vertex->fusible)) {
          queue.emplace_front(child_id);
        } else {
          queue.emplace_back(child_id);
        }
        children_vertex->marked = true;
      }
      vertex_list.emplace_back(vertex);
    }
  }

  for (auto &head_v : head) {
    vertex_list.emplace_front(head_v);
  }
  return 0;
}

class GraphOptPass : public mlir::PassWrapper<GraphOptPass, FunctionPass> {
public:
  explicit GraphOptPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    std::unique_ptr<Graph> graph = GraphFactory::getInstance(clGraphOptType);
    if (!graph) {
      return;
    }

    auto fn = getFunction();
    fn->walk([&](Operation *op) {
      if (0 != graph->add_vertex(op)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    LLVM_DEBUG(graph->print());

    std::list<std::shared_ptr<Vertex> > vertex_list;
    graph->rearrang(vertex_list);
    LLVM_DEBUG(print_vertex(vertex_list););
    graph->rebuild_op(vertex_list);
  }

  void print_vertex(std::list<std::shared_ptr<Vertex>> &vertex_list) {
    for (auto &vertex : vertex_list) {
      llvm::errs() << vertex->id << "->";
    }
    llvm::errs() << "\n";
  }

private:
  llvm::raw_ostream &os;
};

std::unique_ptr<mlir::Pass> createGraphOptPass() {
  return std::make_unique<GraphOptPass>();
}

} // namespace mlir