#ifndef CVI_CUSTOM_OP_PLUGIN_H_
#define CVI_CUSTOM_OP_PLUGIN_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include <map>
#include "llvm/Support/DynamicLibrary.h"
#include "tpuc/CustomOpParam.h"
#include "tpuc/CustomOp.h"

namespace cvi {

using CustomOpCreateFn = CustomOp *(*) (OpParam &);

class CustomOpPlugin {
public:

  void fp32Interpret(const char *opName, OpParam &param,
                     std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                     std::vector<std::vector<int64_t>> &operand_shapes,
                     std::shared_ptr<std::vector<float>> &result_tensor,
                     std::vector<int64_t> &result_shape);

  void bf16Quant(const char *opName, OpParam &param, OpParam *quant);

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