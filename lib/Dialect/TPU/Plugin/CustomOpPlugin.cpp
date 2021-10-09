#include "llvm/Support/CommandLine.h"
#include "tpuc/CustomOpPlugin.h"

namespace cvi {

static llvm::cl::opt<std::string>
    clCustomOpPlugin("custom-op-plugin",
        llvm::cl::desc("Specify the library to custom op"));

CustomOpPlugin::CustomOpPlugin(std::string path) {
  std::string ErrorMsg;
  library = llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &ErrorMsg);
  if (!library.isValid()) {
    ErrorMsg = "Failed to load customer interpret plugin from path:" + path +
               ", reason:" + ErrorMsg + "\n";
    llvm_unreachable(ErrorMsg.c_str());
  }
  llvm::errs() << "load custom op plugin from " << path << "\n";
}

CustomOp* CustomOpPlugin::loadCustomOp(std::string opName, OpParam &param) {
  auto it = customOps.find(opName);
  if (it == customOps.end()) {
    auto funcName = "CustomOp" + opName + "Create";
    auto func = reinterpret_cast<CustomOpCreateFn>(
        library.getAddressOfSymbol(funcName.c_str()));
    if (!func) {
      llvm_unreachable(("failed to find " + funcName).c_str());
    }
    customOps[opName] = func;
  }
  return customOps[opName](param);
}

void CustomOpPlugin::fp32Interpret(
    const char *opName, OpParam &param,
    std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
    std::vector<std::vector<int64_t>> &operand_shapes,
    std::shared_ptr<std::vector<float>> &result_tensor,
    std::vector<int64_t> &result_shape) {
  auto op = loadCustomOp(opName, param);
  assert(op);
  op->interpretFp32(operand_tensors, operand_shapes, result_tensor, result_shape);
  delete op;
}

void CustomOpPlugin::bf16Quant(const char *opName, OpParam &param, OpParam *quant) {
  auto op = loadCustomOp(opName, param);
  assert(op);
  op->quantizeBf16();
  delete op;
}

void CustomOpPlugin::bf16CodeGen(const char *opName, OpParam &param, void *ctx,
                                 std::vector<std::vector<int64_t>> &operand_shapes,
                                 std::vector<uint64_t> &operand_gaddrs,
                                 std::vector<int64_t> &result_shape,
                                 uint64_t result_gaddr, int layer_id) {
  auto op = loadCustomOp(opName, param);
  assert(op);
  op->codeGenBf16(ctx, operand_shapes, operand_gaddrs, result_shape,
                 result_gaddr, layer_id);
}

CustomOpPlugin* CustomOpPlugin::plugin = nullptr;

CustomOpPlugin *CustomOpPlugin::load(std::string pluginFile) {
  if (plugin)
    return plugin;
  if (clCustomOpPlugin.empty() && pluginFile.empty()) {
    llvm::errs() << "need to add --custom-op-plugin to load custom op plugin\n";
    return nullptr;
  }
  plugin = new CustomOpPlugin(clCustomOpPlugin.empty() ? pluginFile : clCustomOpPlugin);
  return plugin;
}


} // namespace cvi