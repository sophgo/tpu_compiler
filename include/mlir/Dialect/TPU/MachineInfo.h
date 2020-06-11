#ifndef _TP_MACHINE_INFO_H_
#define _TP_MACHINE_INFO_H_

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir{

class MInfo : public FunctionPass<MInfo> {
public:
  static uint32_t version;
  static uint32_t lane_num;
  static uint32_t eu_num;
  static uint64_t lmem_per_lane;

  explicit MInfo() {};
  void runOnFunction();
  void getChipInfo(const char* name);

  static uint64_t getSizePerLane(int n, int c, int h, int w, bool eu_align) {
    assert(version && "refer to chip-type");

    uint64_t channelPerLane = llvm::alignTo(c, lane_num) / lane_num;
    uint64_t bytesPerChannel = h * w;
    if (eu_align) {
      bytesPerChannel = llvm::alignTo(bytesPerChannel, eu_num);
    }
    // total number align to eu_num is mandatory
    return llvm::alignTo(n * channelPerLane * bytesPerChannel, eu_num);
  }
};

}

#define get_cvichip_name(chipname) \
do {           \
  getFunction().walk([&](Operation *op) { \
    if (op->getName().getDialect().str() != "tpu" \
        || isa<tpu::WeightFileOp>(op) \
        || isa<tpu::LoadWeightOp>(op) \
        || isa<tpu::NoneOp>(op)) { \
      /* no need to assign*/ \
    } else { \
        std::string clRunChipType = getChipName(op); \
      if(!clRunChipType.empty() && (clRunChipType.compare("CPU") != 0) && (clRunChipType.compare("NONE") != 0)) { \
        chipname = clRunChipType; \
      } \
    } \
  }); \
}while(0)

#endif /* _TP_MACHINE_INFO_H_ */

