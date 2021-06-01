
#ifndef INTERPRETER_CORE_H
#define INTERPRETER_CORE_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include <memory>
#include <vector>

namespace mlir {

using SyncedData = std::shared_ptr<std::vector<float>>;
using SyncedDataShape = std::vector<int64_t>;
using value_map_t = DenseMap<Value, SyncedData>;

enum class DataType { INT8, BF16, FP32 };

class OpKernel {

public:
  OpKernel(){};
  virtual ~OpKernel(){};
  virtual void invoke() = 0;
  virtual void set_tensor(const std::vector<float> &data) = 0;
  virtual std::vector<float> get_tensor() = 0;

  std::string get_name() { return this->name; }
  SyncedDataShape get_shape() { return this->shape; }
  std::string get_data_type() {
    if (this->datatype == DataType::BF16) {
      return "BF16";
    } else if (this->datatype == DataType::INT8) {
      return "INT8";
    }
    return "FP32";
  }
  void set_name(std::string name) { this->name = name; }
  void set_shape(SyncedDataShape &shape) { this->shape = shape; }
  void set_datatype(std::string type) {
    if (type == "NONE") {
      this->datatype = DataType::FP32;
    } else if (type == "BF16") {
      this->datatype = DataType::BF16;
    } else if (type == "INT8" || type == "UINT8") {
      this->datatype = DataType::INT8;
    } else {
      llvm_unreachable("Not support type");
    }
  }
  std::string get_op_type() { return this->op_type; }

  virtual void dump() {
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
  };

public:
  SyncedDataShape shape;
  std::string name;
  std::string op_type;

  size_t size;
  DataType datatype = DataType::FP32;
};

} // namespace mlir

#endif