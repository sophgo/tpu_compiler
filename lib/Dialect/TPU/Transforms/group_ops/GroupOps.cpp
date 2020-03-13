

#include "utils.hpp"
#include "NetGraph.hpp"
#include "GroupOptimizer.hpp"
namespace mlir {


class GroupOpsPass : public FunctionPass<GroupOpsPass> {
public:
  explicit GroupOpsPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    process_fn(&fn, context);
  }

  void process_fn(FuncOp *fn, MLIRContext* context);
private:
  llvm::raw_ostream &os;

};

void GroupOpsPass :: process_fn(FuncOp *fn, MLIRContext * context) {
  NetGraph * net_graph = new NetGraph(fn);
  net_graph->parse_graph(fn);

  auto optimizer = new GroupOptimizer(net_graph);
  optimizer->optimize();

  optimizer->build_fn(fn, context, optimizer);
}

std::unique_ptr<OpPassBase<FuncOp>> createGroupOpsPass() {
    return std::make_unique<GroupOpsPass>();
}

static PassRegistration<GroupOpsPass>
    pass("group-ops",
         "Group ops together to speedup");
}