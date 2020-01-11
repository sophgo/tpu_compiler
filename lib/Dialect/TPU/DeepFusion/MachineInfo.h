#ifndef _TP_MACHINE_INFO_H_
#define _TP_MACHINE_INFO_H_

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MathExtras.h"

// TODO: move to backend
class MInfo {
public:
  static const int lane_num = 32;
  static const int eu_num = 16;
  static const uint64_t lmem_per_lane = 32 * 1024;

  static uint64_t getSizePerLane(int n, int c, int h, int w, bool eu_align) {
    uint64_t channelPerLane = llvm::alignTo(c, lane_num) / lane_num;
    uint64_t bytesPerChannel = h * w;
    if (eu_align) {
      bytesPerChannel = llvm::alignTo(bytesPerChannel, eu_num);
    }
    // total number align to eu_num is mandatory
    return llvm::alignTo(n * channelPerLane * bytesPerChannel, eu_num);
  }
};

#endif /* _TP_MACHINE_INFO_H_ */
