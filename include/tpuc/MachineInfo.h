#ifndef _TP_MACHINE_INFO_H_
#define _TP_MACHINE_INFO_H_

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MathExtras.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
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

class MInfo {
public:
  static void getChipInfo(FuncOp op);
  static void getChipInfo(std::string chipName);

  static uint32_t version;
  static uint32_t lane_num;
  static uint32_t eu_num;
  static uint64_t lmem_per_lane;
  static uint32_t lmem_bank_num;

  static int MAX_TIU_BATCH;
  static int MAX_TIU_CHANNEL;
  static int MAX_TIU_HEIGHT;
  static int MAX_TIU_WIDTH;

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

#endif /* _TP_MACHINE_INFO_H_ */

