//===- CaffeToMlirTranslate.cpp - Caffe to MLIR module conversion ----===//
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
// This file implements a translation from caffe module to MLIR TPU ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/signal_handler.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

using namespace mlir;

// Importer that takes an Caffe model and imports it as an MLIR module in the TPU
// dialect.
class CaffeImporter {
 public:
  explicit CaffeImporter(mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* caffe_to_mlir_name_map)
      : module_(module),
        builder_(module.getContext()),
        function_name_map_(caffe_to_mlir_name_map) {}

  // Import the Caffe model file into the MLIR Module.
  LogicalResult Import(const llvm::StringRef inputFilename) {
    return failure();
  }

  // Import the Caffe Net into the MLIR Module.
  // Status Import(const caffe::Net& net);

 private:
  mlir::ModuleOp module_;
  mlir::Builder builder_;

  //std::unordered_map<std::string, mlir::FuncOp> function_map_;
  std::unordered_map<std::string, std::string>* function_name_map_;
};

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
                                  MLIRContext *context) {
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> caffe_to_mlir_name;
  CaffeImporter importer(module.get(), &caffe_to_mlir_name);
  auto status = importer.Import(inputFilename);
  if (failed(status)) {
    mlir::emitError(mlir::UnknownLoc::get(context));
  }
  assert(succeeded(status));
  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, context);
                 });
