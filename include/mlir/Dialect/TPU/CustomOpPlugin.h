#ifndef CVI_CUSTOM_OP_PLUGIN_H_
#define CVI_CUSTOM_OP_PLUGIN_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include <map>
#include "llvm/Support/DynamicLibrary.h"
#include "mlir/Dialect/TPU/CustomOpParam.h"
#include "mlir/Dialect/TPU/CustomOp.h"

namespace cvi {

using CustomOpCreateFn = CustomOp *(*) (OpParam &);

class CustomOpPlugin {
public:
  void int8Interpret(const char *opName, OpParam &param,
                     std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                     std::vector<std::vector<int64_t>> &operand_shapes,
                     std::shared_ptr<std::vector<float>> &result_tensor,
                     std::vector<int64_t> &result_shape);

  void fp32Interpret(const char *opName, OpParam &param,
                     std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                     std::vector<std::vector<int64_t>> &operand_shapes,
                     std::shared_ptr<std::vector<float>> &result_tensor,
                     std::vector<int64_t> &result_shape);

  void bf16Interpret(const char *opName, OpParam &param,
                     std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                     std::vector<std::vector<int64_t>> &operand_shapes,
                     std::shared_ptr<std::vector<float>> &result_tensor,
                     std::vector<int64_t> &result_shape);

  void int8Quant(const char *opName, OpParam &param, OpParam *quant,
                 float prev_threshold);

  void bf16Quant(const char *opName, OpParam &param, OpParam *quant,
                 float prev_threshold);

  void int8CodeGen(const char *opName, OpParam &param, void *ctx,
               std::vector<std::vector<int64_t>> &operand_shapes,
               std::vector<uint64_t> &operand_gaddrs, std::vector<int64_t> &result_shape,
               uint64_t result_gaddr, int layer_id);

  void bf16CodeGen(const char *opName, OpParam &param, void *ctx,
               std::vector<std::vector<int64_t>> &operand_shapes,
               std::vector<uint64_t> &operand_gaddrs, std::vector<int64_t> &result_shape,
               uint64_t result_gaddr, int layer_id);

  static CustomOpPlugin *load(std::string pluginFile = "");

private:
  CustomOpPlugin(std::string pluginFile);
  CustomOp *loadCustomOp(std::string opName, OpParam &param);

  llvm::sys::DynamicLibrary library;
  std::map<std::string, CustomOpCreateFn> customOps;
  static CustomOpPlugin *plugin;
};

} // namespace cvi
#endif