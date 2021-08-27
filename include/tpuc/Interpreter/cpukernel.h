#ifndef INTERPRETER_CPU_CORE_H
#define INTERPRETER_CPU_CORE_H

#include <memory>
#include <vector>
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"


namespace mlir {

class TensorData {
public:
  explicit TensorData(int32_t batch, size_t size, bool dummy)
      : batch(batch), _size(size) {
    _data = std::make_shared<std::vector<float>>(size * batch);
    _ptr = _data->data();
  }

  explicit TensorData(size_t size) : TensorData(1, size, false) {}

  explicit TensorData(size_t size, float init_val) : batch(1), _size(size) {
    _data = std::make_shared<std::vector<float>>(size, init_val);
    _ptr = _data->data();
  }

  explicit TensorData(std::shared_ptr<std::vector<float>> vec) : batch(1) {
    _size = vec->size();
    _data = vec;
    _ptr = _data->data();
  }

  size_t size() { return _size; }
  float *data() { return _ptr; }
  float &at(size_t idx) { return *(_ptr + idx); }
  const float &at(size_t idx) const { return *(_ptr + idx); }
  float &operator[](size_t idx) { return *(_ptr + idx); }
  const float &operator[](size_t idx) const { return *(_ptr + idx); }
  std::vector<float>::iterator begin() {
    return std::vector<float>::iterator(_ptr);
  }
  std::vector<float>::iterator end() {
    return std::vector<float>::iterator(_ptr + size());
  }
  void assign(std::vector<float>::const_iterator first,
              std::vector<float>::const_iterator last) {
    size_t num = last - first;
    assert(num <= size());
    memcpy(_ptr, &(*first), num * sizeof(float));
  }
  void assign(std::vector<float>::const_iterator first,
              std::vector<float>::const_iterator last, int32_t bidx) {
    size_t num = last - first;
    assert(num <= size());
    assert(bidx <= batch);
    memcpy(_data->data() + bidx * size(), &(*first), num * sizeof(float));
  }
  void set_batch_idx(int32_t bidx) {
    assert(bidx <= batch);
    _ptr = _data->data() + bidx * size();
  }
  std::shared_ptr<std::vector<float>> tensor(int32_t bidx) {
    return std::make_shared<std::vector<float>>(
        std::vector<float>::iterator(_data->data() + bidx * size()),
        std::vector<float>::iterator(_data->data() + (bidx + 1) * size()));
  }

public:
  int32_t batch;

private:
  size_t _size;
  std::shared_ptr<std::vector<float>> _data;
  float *_ptr;
};

enum class DataType { INT8, BF16, FP32 };

using SyncedData = std::shared_ptr<TensorData>;
using SyncedDataShape = std::vector<int64_t>;
using value_map_t = DenseMap<Value, SyncedData>;
using weight_map_t = DenseMap<Value, SyncedData>;

class CPUOpKernel {

public:
  CPUOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping, bool hasOpds = true);

  CPUOpKernel() = delete;

  virtual ~CPUOpKernel() {}

  virtual void set_tensor(const std::vector<float> &data) {
    llvm_unreachable("NOT support set_tensor");
  }

  virtual void invoke() = 0;

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

  virtual void dump();

  std::vector<float> get_tensor() {
    // deep copy
    std::vector<float> ret(resTensor->begin(), resTensor->end());
    return ret;
  }

  static std::string generateSignature(Operation &op);

protected:
  void assignOperandTensors(Operation &op, const value_map_t &valueMapping,
                            const weight_map_t &weightMapping);
  void assignResultTensor(Operation &op, value_map_t &valueMapping);

public:
  SyncedDataShape shape;
  std::string name;
  std::string op_type;

  size_t size;
  DataType datatype = DataType::FP32;

  Operation *op;
  std::string signature;
  bool dirty = true;
  static int32_t max_batch_size;

protected:
  std::vector<SyncedData> opdTensors;
  SyncedData resTensor;
};

}; // namespace mlir

#endif