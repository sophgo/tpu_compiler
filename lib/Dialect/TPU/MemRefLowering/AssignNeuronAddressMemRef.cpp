#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct AssignAllocPattern : public RewritePattern {
  AssignAllocPattern(MLIRContext *context, uint64_t *pos,
                     llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(AllocOp::getOperationName(), 1, context), pos_(pos),
        map_os_(map_os), alignment_(alignment) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto resultType = op->getResult(0)->getType();
    auto memRefType = resultType.dyn_cast<MemRefType>();
    assert(memRefType && "Expect result is MemRefType");
    if (!memRefType)
      return matchFailure();

    enum {ActivationMemorySpace = 2};

    auto memorySpace = memRefType.getMemorySpace();
    if (memorySpace != ActivationMemorySpace)
      return matchFailure();

    auto elementType = memRefType.getElementType();
    assert(elementType.isIntOrFloat() && "Expect result is int or float");
    if (!elementType.isIntOrFloat())
      return matchFailure();

    std::vector<int64_t> shape = memRefType.getShape();
    uint32_t dataTypeSize = elementType.getIntOrFloatBitWidth() / 8;
    uint32_t allocatedSize = dataTypeSize;
    for (unsigned i = 0; i < shape.size(); ++i) {
      allocatedSize *= shape[i];
    }

    std::string dtype;
    if (elementType.dyn_cast<IntegerType>())
      dtype = "int";
    else if (elementType.isBF16())
      dtype = "bf";
    else
      dtype = "f";

    // pad to alignment
    if (allocatedSize % alignment_) {
      allocatedSize = allocatedSize + alignment_ - (allocatedSize % alignment_);
    }

    llvm::dbgs() << "op " << op->getName() << "\n";

    // Update neuron address of allocated tpu op
    for (auto &user : op->getResult(0)->getUses()) {
      Operation *user_op = user.getOwner();
      auto lastOperand = user_op->getOperand(user_op->getNumOperands() - 1);

      llvm::dbgs() << "  user_op " << user_op->getName() << "\n";

      if (user_op->getName().getDialect().str() == "tpu" &&
          lastOperand == op->getResult(0)) {

        // FIXME: skip tg_memref_quant, fc_reshape
        if (user_op->getName().getStringRef().str() == "tpu.tg_memref_quant" ||
            user_op->getName().getStringRef().str() == "tpu.tg_memref_reshape") {
          allocatedSize = 0;
        }

        auto curPos = *pos_;
        auto newPos = curPos + allocatedSize;

        // expand to dims=4
        while (shape.size() < 4)
          shape.insert(shape.begin(), 1);
        map_os_ << getOpName(user_op) << ","
                << llvm::format_hex(curPos, 10) << ","
                << dtype << elementType.getIntOrFloatBitWidth() << ","
                << shape[0] << "," << shape[1] << ","
                << shape[2] << "," << shape[3] << "\n";

        llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                      getOpName(user_op).str().c_str(), allocatedSize)
                    << llvm::format_hex(curPos, 10) << " --> "
                    << llvm::format_hex(newPos, 10) << " ]\n";
        setOpAddress(user_op, curPos);
        *pos_ = newPos;

        return matchFailure();
      }
    }

    return matchFailure();
  }

  uint64_t *pos_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;

};

static llvm::cl::opt<size_t> clNeuronAlignmentMemRef(
    "tpu-neuron-address-align-memref",
    llvm::cl::desc("Specify the alignment for neuron"),
    llvm::cl::init(16));

static llvm::cl::opt<std::string> clNeuronMapFilenameMemRef(
    "tpu-neuron-map-filename-memref",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

class AssignNeuronAddressMemRefPass :
    public FunctionPass<AssignNeuronAddressMemRefPass> {
public:
  explicit AssignNeuronAddressMemRefPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // create a map file
    std::unique_ptr<llvm::ToolOutputFile> neuronMapFile = nullptr;
    if (clNeuronMapFilenameMemRef != "-") {
      std::string errorMessage;
      neuronMapFile = openOutputFile(clNeuronMapFilenameMemRef, &errorMessage);
      if (!neuronMapFile) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
    }

    uint64_t pos = 0;
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    ConversionTarget target(getContext());

    patterns.insert<AssignAllocPattern
        >(context, &pos, neuronMapFile->os(), clNeuronAlignmentMemRef);
    if (failed(applyPartialConversion(fn, target, patterns)))
      return signalPassFailure();

    if (neuronMapFile) {
      neuronMapFile->keep();
    }
  }

private:
  llvm::raw_ostream &os;
};

} // anonymous space

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createAssignNeuronAddressMemRefPass() {
  return std::make_unique<AssignNeuronAddressMemRefPass>();
}

static PassRegistration<AssignNeuronAddressMemRefPass>
    pass("assign-neuron-address-memref", "Assign address to each neuron");
