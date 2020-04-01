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
static llvm::cl::opt<size_t> clNeuronAlignmentMemRef(
    "tpu-neuron-address-align-memref",
    llvm::cl::desc("Specify the alignment for neuron"),
    llvm::cl::init(16));

static llvm::cl::opt<std::string> clNeuronMapFilenameMemRef(
    "tpu-neuron-map-filename-memref",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> clNeuronMapEnableRecycleMemRef(
    "enable-tpu-neuron-map-recyle-memref",
    llvm::cl::desc("Enable neuron recycle"),
    llvm::cl::init(false));

struct AssignNeuronAddressMemRefPass :
    public FunctionPass<AssignNeuronAddressMemRefPass> {
  void runOnFunction() override;
  void handleAllocOp(Operation *opInst);
  void handleDeallocOp(Operation *opInst);
  void handleQuantOp(Operation *opInst);
  void handleSliceOp(Operation *opInst);
  Operation *findTpuOpFromAllocOp(Operation *op);
  Operation *findTpuOpFromDeallocOp(Operation *op);

  struct NeuronInfo {
    llvm::StringRef name;
    int layerId;
    uint64_t offset;
    uint64_t size;
  };

  // FIXME: defined in dialect
  static constexpr int ActivationMemorySpace = 2;

  bool isMemoryAliasedOp(Operation *op);
  bool isMemoryAliasedOpHandled(Operation *op);
  bool isReuseRecycledNeuron(Operation *op, uint64_t allocatedSize,
                             uint64_t &offset);
  void dumpNeuronInfo(std::vector<NeuronInfo> &neuronLlist);
  bool findNeuronByLayerId(std::vector<NeuronInfo> &neuronList,
                           int layerId);
  bool findNeuronByName(std::vector<NeuronInfo> &neuronList,
                        llvm::StringRef name);
  void sortRecycledNeuronBySize(void);
  llvm::raw_ostream &os() { return *map_os_; }

  bool recycledEnabled;
  uint64_t pos_;
  llvm::raw_ostream *map_os_;
  size_t alignment_;
  std::vector<NeuronInfo>usedList;
  std::vector<NeuronInfo>recycledList;
};

} // anonymous space

Operation *AssignNeuronAddressMemRefPass::findTpuOpFromAllocOp(Operation *op) {
  llvm::dbgs() << "  findTpuOpFromAllocOp op " << op->getName() << "\n";

  // AllocOp has only one result.
  Operation *firstUseOp = nullptr;
  for (auto &user : op->getResult(0)->getUses()) {
    Operation *userOp = user.getOwner();
    auto lastOperand = userOp->getOperand(userOp->getNumOperands() - 1);

    llvm::dbgs() << "    userOp " << userOp->getName() << "\n";

    // Find corresponding tpu op.
    if (userOp->getName().getDialect().str() == "tpu" &&
        op->getNumResults() && lastOperand == op->getResult(0)) {
      if (firstUseOp) {
        if (userOp->isBeforeInBlock(firstUseOp)) {
          firstUseOp = userOp;
        }
      } else {
        firstUseOp = userOp;
      }
    }
  }

  if (firstUseOp)
    llvm::dbgs() << "    firstUseOp " << firstUseOp->getName() << "\n";
  else
    llvm::dbgs() << "    no firstUseOp\n";

  return firstUseOp;
}

Operation *
AssignNeuronAddressMemRefPass::findTpuOpFromDeallocOp(Operation *op) {
  // DeallocOp has only on operand.
  Operation *firstUseOp = nullptr;
  for (auto &user : op->getOperand(0)->getUses()) {
    Operation *userOp = user.getOwner();
    auto lastOperand = userOp->getOperand(userOp->getNumOperands() - 1);

    // Find coresponding tpu op.
    if (userOp->getName().getDialect().str() == "tpu" &&
        lastOperand == op->getOperand(0)) {
      if (firstUseOp) {
        if (userOp->isBeforeInBlock(firstUseOp)) {
          firstUseOp = userOp;
        }
      } else {
        firstUseOp = userOp;
      }
    }
  }
  return firstUseOp;
}

bool AssignNeuronAddressMemRefPass::isMemoryAliasedOp(Operation *op) {
  // FIXME: skip fc_reshape
  // Can replace AllocOp with ViewOp ?
  if (dyn_cast<tpu::TG_MemRef_ReshapeOp>(op)) {
    // No gaddr, skip it
    return true;
  } else if (dyn_cast<tpu::TG_INT8_SliceOp>(op) ||
             dyn_cast<tpu::TG_BF16_SliceOp>(op)) {
    auto resultType = op->getResult(0)->getType();
    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    auto batch = tensorType.getShape()[0];

    // Avoid copy when batch = 1
    if (batch == 1) {
      return true;
    }
  }

  return false;
}

bool AssignNeuronAddressMemRefPass::isMemoryAliasedOpHandled(Operation *op) {
   llvm::dbgs() << "isMemoryAliasedOpHandled op " << op->getName() << "\n";

  // FIXME: skip fc_reshape
  // Can replace AllocOp with ViewOp ?
  // if (op->getName().getStringRef().str() == "tpu.tg_memref_reshape") {
  if (dyn_cast<tpu::TG_MemRef_ReshapeOp>(op)) {
    // No gaddr, skip it
    return true;
  } else if (dyn_cast<tpu::TG_MemRef_INT8_SliceOp>(op) ||
             dyn_cast<tpu::TG_MemRef_BF16_SliceOp>(op)) {
    auto resultType = op->getOperand(op->getNumOperands()-1)->getType();
    auto memRefType = resultType.dyn_cast<MemRefType>();
    std::vector<int64_t> shape = memRefType.getShape();
    auto batch = shape[0];

    auto elementType = memRefType.getElementType();
    uint64_t dataTypeSize = elementType.getIntOrFloatBitWidth() / 8;
    uint64_t allocatedSize = dataTypeSize;
    for (unsigned i = 0; i < shape.size(); ++i)
      allocatedSize *= shape[i];

    std::string dtype;
    if (elementType.dyn_cast<IntegerType>())
      dtype = "int";
    else if (elementType.isBF16())
      dtype = "bf";
    else
      dtype = "f";

    uint64_t baseGAddr = pos_;

    // Reuse memory when batch = 1
    if (batch == 1) {
      auto operandOp = op->getOperand(0)->getDefiningOp();
      if (dyn_cast<AllocOp>(operandOp)) {
        // Previous op is lowed op
        auto prevOp = findTpuOpFromAllocOp(op->getOperand(0)->getDefiningOp());
        assert(prevOp && "Not tpu op belong to allocOp");
        baseGAddr = getOpAddress(prevOp);
      } else if (dyn_cast<tpu::TG_TensorToMemRefOp>(operandOp)) {
        // Previous op is not lowed op
        auto prevOp = operandOp->getOperand(0)->getDefiningOp();
        baseGAddr = getOpAddress(prevOp);
      } else {
        assert("Unexpected previous Op of SliceOp");
      }
    }

    uint64_t axis = 0;
    uint64_t offset = 0;
    if (auto tpuOp = dyn_cast<tpu::TG_MemRef_INT8_SliceOp>(op)) {
      axis = tpuOp.axis().getLimitedValue();
      offset = tpuOp.offset().getLimitedValue();
    } else if (auto tpuOp = dyn_cast<tpu::TG_MemRef_BF16_SliceOp>(op)) {
      axis = tpuOp.axis().getLimitedValue();
      offset = tpuOp.offset().getLimitedValue();
    }

    offset *= dataTypeSize;
    for (uint64_t i = axis + 1; i < shape.size(); ++i) {
      offset *= shape[i];
    }

    uint64_t curPos = baseGAddr + offset;

    // pad to alignment
    allocatedSize = llvm::alignTo(allocatedSize, alignment_);
    auto tpuOpIf = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
    os() << tpuOpIf.getOpName().str().c_str() << ","
         << llvm::format_hex(curPos, 10) << ","
            << dtype << elementType.getIntOrFloatBitWidth() << ","
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                  tpuOpIf.getOpName().str().c_str(), allocatedSize)
                << llvm::format_hex(curPos, 10) << " --> "
                << llvm::format_hex(curPos+allocatedSize, 10) << " ]\n";

    setOpAddress(op, curPos);
    return true;
  }

  return false;
}

bool AssignNeuronAddressMemRefPass::isReuseRecycledNeuron(
    Operation *op, uint64_t allocatedSize, uint64_t &offset) {
  if (!recycledEnabled)
    return false;

  if (!recycledList.size())
    return false;

  // Only sort when needed
  sortRecycledNeuronBySize();

  for (auto it = recycledList.begin(); it != recycledList.end(); ++it) {
    if (it->size >= allocatedSize) {
      offset = it->offset;
      recycledList.erase(it);
      return true;
    }
  }

  return false;
}

void AssignNeuronAddressMemRefPass::dumpNeuronInfo(
    std::vector<NeuronInfo> &neuronLlist) {
  for (auto &it : neuronLlist) {
    llvm::dbgs() << "    name " << it.name
                 << ", layerId " << it.layerId
                 << ", offset " << llvm::format_hex(it.offset, 10)
                 << ", size " << it.size
                 << "\n";
  }
}

void AssignNeuronAddressMemRefPass::sortRecycledNeuronBySize(void) {
  if (recycledList.size() < 2)
    return;

  llvm::array_pod_sort(recycledList.begin(), recycledList.end(),
                       [](const auto *lhs, const auto *rhs) {
                         return llvm::array_pod_sort_comparator<uint64_t>(
                             &lhs->size, &rhs->size);
                      });
}

bool AssignNeuronAddressMemRefPass::findNeuronByLayerId(
    std::vector<NeuronInfo> &neuronList, int layerId) {
  for (auto &it : neuronList) {
    if (it.layerId == layerId)
      return true;
  }

  return false;
}

bool AssignNeuronAddressMemRefPass::findNeuronByName(
    std::vector<NeuronInfo> &neuronList, llvm::StringRef name) {
  for (auto &it : neuronList) {
    if (!name.compare(it.name)) {
      return true;
    }
  }

  return false;
}

void AssignNeuronAddressMemRefPass::handleAllocOp(Operation *opInst) {
  auto resultType = opInst->getResult(0)->getType();
  auto memRefType = resultType.dyn_cast<MemRefType>();

  // Check activation memory space
  auto memorySpace = memRefType.getMemorySpace();
  if (memorySpace != ActivationMemorySpace)
    return;

  // Should be integer(int8) or float(float, bf16)
  auto elementType = memRefType.getElementType();
  assert(elementType.isIntOrFloat() && "Expect result is int or float");
  if (!elementType.isIntOrFloat())
    return;

  std::vector<int64_t> shape = memRefType.getShape();
  uint64_t dataTypeSize = elementType.getIntOrFloatBitWidth() / 8;
  uint64_t allocatedSize = dataTypeSize;
  for (unsigned i = 0; i < shape.size(); ++i)
    allocatedSize *= shape[i];

  std::string dtype;
  if (elementType.dyn_cast<IntegerType>())
    dtype = "int";
  else if (elementType.isBF16())
    dtype = "bf";
  else
    dtype = "f";

  // pad to alignment
  allocatedSize = llvm::alignTo(allocatedSize, alignment_);

  auto tpuOp = findTpuOpFromAllocOp(opInst);
  if (!tpuOp)
    return;

  if (isMemoryAliasedOpHandled(tpuOp))
    return;

  uint64_t curPos = pos_;
  uint64_t newPos = curPos + allocatedSize;

  if (isReuseRecycledNeuron(tpuOp, allocatedSize, curPos)) {
    newPos = pos_;
  }

  auto tpuOpIf = llvm::dyn_cast<tpu::TpuOpCommonInterface>(tpuOp);
  if (!tpuOpIf) {
    llvm::errs() << "Error ! tpuOp " << tpuOp->getName()
                 << " does not have common interface\n";
  }
  assert(tpuOpIf && "Expect tpu op has common interface");

  // expand to dims=4
  while (shape.size() < 4)
    shape.insert(shape.begin(), 1);
  os() << tpuOpIf.getOpName().str().c_str() << ","
       << llvm::format_hex(curPos, 10) << ","
          << dtype << elementType.getIntOrFloatBitWidth() << ","
          << shape[0] << "," << shape[1] << ","
          << shape[2] << "," << shape[3] << "\n";

  llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                tpuOpIf.getOpName().str().c_str(), allocatedSize)
              << llvm::format_hex(curPos, 10) << " --> "
              << llvm::format_hex(curPos+allocatedSize, 10) << " ]\n";
  setOpAddress(tpuOpIf, curPos);
  pos_ = newPos;

  NeuronInfo neuronInfo;
  neuronInfo.name = tpuOpIf.getOpName();
  neuronInfo.layerId = tpuOpIf.getOpLayerId();
  neuronInfo.offset = curPos;
  neuronInfo.size = allocatedSize;
  usedList.push_back(neuronInfo);

}

void AssignNeuronAddressMemRefPass::handleDeallocOp(Operation *opInst) {
  auto resultType = opInst->getOperand(0)->getType();
  auto memRefType = resultType.dyn_cast<MemRefType>();

  // Check activation memory space
  auto memorySpace = memRefType.getMemorySpace();
  if (memorySpace != ActivationMemorySpace)
    return;

  auto tpuOp = findTpuOpFromDeallocOp(opInst);
  if (!tpuOp)
    return;

  auto tpuOpIf = llvm::dyn_cast<tpu::TpuOpCommonInterface>(tpuOp);
  if (!tpuOpIf)
    llvm::errs() << "tpuOp " << tpuOp->getName() << " does not have common interface\n";
  assert(tpuOpIf && "Expect tpu op has common interface");

  auto layerId = tpuOpIf.getOpLayerId();
  auto opName = tpuOpIf.getOpName();

  for (auto it = usedList.begin(); it != usedList.end(); ++it) {
    // if (it->layerId == layerId) {
    if (!opName.compare(it->name)) {

      // auto isLayerIdExisted = findNeuronByLayerId(recycledList, layerId);
      auto isLayerIdExisted = findNeuronByName(recycledList, opName);
      if (isLayerIdExisted) {
        llvm::errs() << "Error ! name " << opName << "already exists.\n"
                     << "dump error recycledList before insersion\n";
        dumpNeuronInfo(recycledList);
        assert(0);
      }

      if (!isMemoryAliasedOp(tpuOp))
        recycledList.push_back(*it);

      usedList.erase(it);

      break;
    }
  }
}

void AssignNeuronAddressMemRefPass::handleQuantOp(Operation *opInst) {
  // Reserved space for both input/quantized input.
  // There is no explicit place to assign input.
  auto inputType = opInst->getOperand(0)->getType().dyn_cast<RankedTensorType>();
  assert(inputType && "Expect input is ranked tensor type.");
  if (!inputType)
    return;

  // Should be integer(int8) or float(float, bf16)
  auto elementType = inputType.getElementType();
  assert(elementType.isIntOrFloat() && "Expect input is int or float");
  if (!elementType.isIntOrFloat())
    return;

  std::vector<int64_t> shape = inputType.getShape();
  uint64_t dataTypeSize = elementType.getIntOrFloatBitWidth() / 8;
  uint64_t allocatedSize = dataTypeSize;
  for (unsigned i = 0; i < shape.size(); ++i)
    allocatedSize *= shape[i];

  std::string dtype;
  if (elementType.dyn_cast<IntegerType>())
    dtype = "int";
  else if (elementType.isBF16())
    dtype = "bf";
  else
    dtype = "f";

  // pad to alignment
  allocatedSize = llvm::alignTo(allocatedSize, alignment_);

  uint64_t curPos = pos_;
  uint64_t newPos = curPos + allocatedSize;

  auto tpuOpIf = llvm::dyn_cast<tpu::TpuOpCommonInterface>(opInst);
  if (!tpuOpIf) {
    llvm::errs() << "Error ! tpuOp " << opInst->getName()
                 << " does not have common interface\n";
  }
  assert(tpuOpIf && "Expect tpu op has common interface");

  llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                tpuOpIf.getOpName().str().c_str(), allocatedSize)
              << llvm::format_hex(curPos, 10) << " --> "
              << llvm::format_hex(newPos, 10) << " ]\n";
  setOpAddress(tpuOpIf, curPos);
  pos_ = newPos;
}

void AssignNeuronAddressMemRefPass::runOnFunction() {
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

  recycledEnabled = clNeuronMapEnableRecycleMemRef;
  pos_ = 0;
  alignment_ = clNeuronAlignmentMemRef;
  map_os_ = &neuronMapFile->os();

  getFunction().walk([&](Operation *opInst) {
    if (dyn_cast<AllocOp>(opInst)) {
      handleAllocOp(opInst);
    } else if (dyn_cast<DeallocOp>(opInst)) {
      handleDeallocOp(opInst);
    } else if (dyn_cast<tpu::QuantOp>(opInst)) {
      handleQuantOp(opInst);
    }
  });

  if (neuronMapFile) {
    neuronMapFile->keep();
  }

  llvm::dbgs() << "total neuron size " << pos_ << "\n";
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createAssignNeuronAddressMemRefPass() {
  return std::make_unique<AssignNeuronAddressMemRefPass>();
}

static PassRegistration<AssignNeuronAddressMemRefPass>
    pass("assign-neuron-address-memref", "Assign address to each neuron");
