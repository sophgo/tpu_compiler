#include "tpuc/Interpreter/cpukernel.h"


namespace mlir {

CPUOpKernel::CPUOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping, bool hasOpds) {
  auto type = op.getResult(0).getType().cast<TensorType>();
  this->shape = type.getShape();
  this->name = getOpName(&op).str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (hasOpds) {
    assignOperandTensors(op, valueMapping, weightMapping);
  }
  assignResultTensor(op, valueMapping);
  signature = generateSignature(op);
  this->op = op.getResult(0).getDefiningOp();
}

void CPUOpKernel::dump() {
  std::string shape_str;
  if (this->shape.size() == 0) {
    llvm_unreachable("No shape");
  }
  for (auto &i : this->shape) {
    shape_str = shape_str + std::to_string(i) + " ";
  }
  llvm::outs() << this->op_type << "\n";
  llvm::outs() << "\tName: " << this->name << "\n";
  llvm::outs() << "\tShape: " << shape_str << "\n";
  llvm::outs() << "\tDataType: " << this->get_data_type() << "\n";
}

std::string CPUOpKernel::generateSignature(Operation &op) {
  std::string signature;
  std::string s;
  llvm::raw_string_ostream os(s);
  op.print(os);
  auto str = os.str();
  for (int i = 0; i < (int)str.size(); i++) {
    if (str[i] == ')') {
      signature = str.substr(i + 1);
      break;
    }
  }
  return signature;
}

void CPUOpKernel::assignOperandTensors(Operation &op,
                                       const value_map_t &valueMapping,
                                       const weight_map_t &weightMapping) {
  for (auto opd : op.getOperands()) {
    if (isTensorNone(opd)) {
      opdTensors.push_back(nullptr);
      continue;
    } else if (isa<tpu::LoadWeightOp>(opd.getDefiningOp())) {
      auto it = weightMapping.find(opd);
      if (it == weightMapping.end()) {
        llvm::errs() << "not find: " << opd.getDefiningOp()->getName() << "\n";
        llvm_unreachable("value mapping false");
      }
      opdTensors.emplace_back(it->second);
    } else {
      auto it = valueMapping.find(opd);
      if (it == valueMapping.end()) {
        llvm::errs() << "not find: " << opd.getDefiningOp()->getName() << "\n";
        llvm_unreachable("value mapping false");
      }
      opdTensors.emplace_back(it->second);
    }
  }
}

void CPUOpKernel::assignResultTensor(Operation &op, value_map_t &valueMapping) {
  auto result = op.getResult(0);
  auto size = getTensorSize(result);
  resTensor = std::make_shared<TensorData>(max_batch_size, size, false);
  valueMapping[result] = resTensor;
}

int32_t CPUOpKernel::max_batch_size = 1;

} // namespace mlir