

#include "utils.hpp"
#include "NetGraph.hpp"
#include "GroupOptimizer.hpp"

#define DEBUG_TYPE "group_ops"

namespace mlir {

class GroupOpsPass : public FunctionPass<GroupOpsPass> {
public:
  explicit GroupOpsPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    std::string getRunChipType;
    MInfo Machineinfo;
    get_cvichip_name(getRunChipType);
    Machineinfo.getChipInfo(getRunChipType.c_str());
    assert(MInfo::version && "refer to set-chip");
    auto fn = getFunction();
    auto *context = &getContext();
    pre_process(&fn, context);
    process_fn(&fn, context);
  }

  void process_fn(FuncOp* fn, MLIRContext * context);
  void pre_process(FuncOp* fn, MLIRContext * context);
private:
  llvm::raw_ostream &os;

};

void GroupOpsPass::process_fn(FuncOp *fn, MLIRContext * context) {
  NetGraph * net_graph = new NetGraph(fn);
  net_graph->parse_graph(fn);

  auto optimizer = new GroupOptimizer(net_graph, fn, context);
  optimizer->optimize();

  optimizer->build_fn(context);
}

template <typename OpTy>
struct fuseLeakyReluPattern : public RewritePattern {
  fuseLeakyReluPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite
    (Operation *op, PatternRewriter &rewriter) const override {
    auto conv_op = cast<OpTy>(op);
    assert(conv_op);

    if (conv_op.fused_leaky())
      return matchFailure();
    // if fuse with next inst
    if (conv_op.fuse_next()) {
      Operation * relu_op = getNextOp(op);
      auto relu = dyn_cast<tpu::TG_INT8_LeakyReluOp>(relu_op);
      conv_op.setAttr("negative_slope", relu.negative_slopeAttr());
      if (relu.rshift_pos().hasValue())
        conv_op.setAttr("rshift_pos", relu.rshift_posAttr());
      if (relu.m_i8_pos().hasValue())
        conv_op.setAttr("m_i8_pos", relu.m_i8_posAttr());
      if (relu.rshift_neg().hasValue())
        conv_op.setAttr("rshift_neg", relu.rshift_negAttr());
      if (relu.m_i8_neg().hasValue())
        conv_op.setAttr("m_i8_neg", relu.m_i8_negAttr());
      conv_op.setAttr("fused_leaky",rewriter.getBoolAttr(true));
      conv_op.setAttr("name", relu.nameAttr());

      // delete leaky relu
      relu_op->replaceAllUsesWith(conv_op);
    }
    return matchSuccess();
  }
};

void GroupOpsPass::pre_process(FuncOp* fn, MLIRContext *context){
  OwningRewritePatternList patterns;
  patterns.clear();
  patterns.insert<
    fuseLeakyReluPattern<tpu::TG_INT8_PC_Conv2DOp>
  >(context);
  applyPatternsGreedily(*fn, patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createGroupOpsPass() {
    return std::make_unique<GroupOpsPass>();
}

static PassRegistration<GroupOpsPass>
    pass("group-ops",
         "Group ops together to speedup");
}