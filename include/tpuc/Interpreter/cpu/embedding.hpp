#ifndef INTERPRETER_CPU_EMBEDDING_H
#define INTERPRETER_CPU_EMBEDDING_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void embedding(int *index,  float *table, float *output,
               std::vector<int64_t> &indexShape, std::vector<int64_t> &tableShape,
               std::vector<int64_t> &topShape);

class EmbeddingOpKernel : public CPUOpKernel<EmbeddingOpKernel> {
public:
  static constexpr const char *OpName = "CPUEmbeddingOp";

  EmbeddingOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData table_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape table_shape;
  SyncedDataShape output_shape;

};
} // namespace mlir
#endif
