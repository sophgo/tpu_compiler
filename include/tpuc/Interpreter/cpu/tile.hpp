#ifndef INTERPRETER_CPU_TILE_H
#define INTERPRETER_CPU_TILE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class TileOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUTileOp";

  TileOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  int32_t axis;
  int32_t tiles;

};
} // namespace mlir
#endif