//===- FileUtilities.h - utilities for working with tensor files -------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Common utilities for working with tensor files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TENSORFILE_H_
#define MLIR_SUPPORT_TENSORFILE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

#include "cnpy.h"

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, std::error_code &EC, bool readOnly,
      bool newCreate)
      : filename(filename), readOnly(readOnly), newCreate(newCreate) {
    if (!newCreate) {
      auto ret = load();
      assert(succeeded(ret));
    } else {
      map.clear();
    }
  }

  ~TensorFile() {}

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const T* data,
      TensorType &type) {
    if (readOnly)
      return failure();
    auto it = map.find(name.str());
    if (it != map.end()) {
      llvm::errs() << "failed to add tensor " << name.str() << ", already exist\n";
      return failure();
    }
    std::vector<int64_t> shape = type.getShape();
    std::vector<size_t> shape_npz;
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      shape_npz.push_back((size_t)*it);
    }
    cnpy::npz_add_array(map, name.str(), &data[0], shape_npz);
    return success();
  }

  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
      TensorType &type) {
    return addTensor(name, data->data(), type);
  }

  /// read a tensor from file
  /// if the name is not found, return failure()
  /// type is provided for checking, return failure() if type does not match
  template<typename T>
  LogicalResult readTensor(llvm::StringRef name, T* data, size_t count) {
    auto it = map.find(name.str());
    if (it == map.end()) {
      llvm::errs() << "failed to find tensor " << name.str() << " to read\n";
      return failure();
    }
    auto arr = it->second;
    if (arr.num_bytes() != count * sizeof(T)) {
      llvm::errs() << "size does not match for tensor " << name.str() << "\n";
      return failure();
    }
    memcpy(data, arr.data_holder->data(), arr.num_bytes());
    return success();
  }

  template<typename T>
  std::unique_ptr<std::vector<T> > readTensor(llvm::StringRef name,
      TensorType &type) {
    std::vector<int64_t> shape = type.getShape();
    assert(shape.size() <= 4);
    auto count = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto data = std::make_unique<std::vector<T> >(count);
    auto ret = readTensor(name, (T*)data.get()->data(), count);
    assert(succeeded(ret));
    return data;
  }

  /// delete a tensor from file
  /// if the name is not found, return failure()
  template<typename T>
  LogicalResult deleteTensor(llvm::StringRef name) {
    if (readOnly)
      return failure();
    auto it = map.find(name.str());
    if (it == map.end()) {
      llvm::errs() << "failed to find tensor " << name.str() << " to delete\n";
      return failure();
    }
    map.erase(it);
    return success();
  }

  void keep(void) {
    // TODO: assuming all tensors are in float
    cnpy::npz_save_all<float>(filename.str(), map);
  }

private:
  /// load the file
  LogicalResult load(void) {
    map = cnpy::npz_load(filename.str());
    assert(map.size() > 0);
    return success();
  }

  llvm::StringRef filename;
  bool readOnly;
  bool newCreate;
  cnpy::npz_t map;
};

/// Open the file specified by its name for reading. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openInputTensorFile(llvm::StringRef filename,
              std::string *errorMessage = nullptr);

/// Create a new file specified by its name for writing. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openOutputTensorFile(llvm::StringRef outputFilename,
               std::string *errorMessage = nullptr);

/// Open a existing file specified by its name for updating. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openTensorFile(llvm::StringRef filename,
               std::string *errorMessage = nullptr);

} // namespace mlir

#endif // MLIR_SUPPORT_TENSORFILE_H_
