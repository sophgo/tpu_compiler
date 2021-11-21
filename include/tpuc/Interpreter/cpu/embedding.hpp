#ifndef INTERPRETER_CPU_EMBEDDING_H
#define INTERPRETER_CPU_EMBEDDING_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void embedding(int *index, float *table, float *output,
               std::vector<int64_t> &indexShape,
               std::vector<int64_t> &tableShape,
               std::vector<int64_t> &topShape);

class EmbeddingOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEmbeddingOp";

  EmbeddingOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData table_data;
  SyncedData scale_data;
  SyncedData zeropoint_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape table_shape;
  SyncedDataShape output_shape;
  bool mix_bf16;
};
} // namespace mlir
#endif
