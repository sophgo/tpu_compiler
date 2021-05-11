//===- buildCviModel.h - class for cvimodel ---------------------*- C++ -*-===//
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
// This header file defines class needed to generate cvimodel.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_DIALECT_TPU_MLIR_TO_CVIMODEL_H
#define MIR_DIALECT_TPU_MLIR_TO_CVIMODEL_H

#include <chrono>
#include <iomanip>
#include <ctime>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <fstream>
#include "cvibuilder/cvimodel_generated.h"
#include "cvibuilder/parameter_generated.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace cvi::model;

using FBStringVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>;
using FBWeightVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Weight>>>;
using FBSectionVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Section>>>;
using FBTensorVector =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>;
using FBProgram = flatbuffers::Offset<Program>;
using FBRoutine = flatbuffers::Offset<Routine>;
using FBSection = flatbuffers::Offset<Section>;
using FBModel = flatbuffers::Offset<Model>;
using FBWeight = flatbuffers::Offset<Weight>;
using FBPreProcessHints = flatbuffers::Offset<PreProcessHints>;

class CviTensor {
public:
  CviTensor(std::string name, TensorType &type, int64_t offset, bool isWeight);

  void setInt8SymQuantInfo(float qscale) {
    this->quant_type = QuantType_INT8_SYM;
    this->qscale = qscale;
  }

  void setInt8AsymQuantInfo(float qscale, int zero_point) {
    this->quant_type = QuantType_INT8_ASYM;
    this->qscale = qscale;
    this->zero_point = zero_point;
  }

  std::string name;
  size_t shape[4] = {1, 1, 1, 1};
  size_t stride[4] = {0, 0, 0, 0};
  size_t size;
  int64_t offset = -1;
  DType dtype;
  int dsize;

  QuantType quant_type = QuantType_NONE;
  int zero_point = 0;
  float qscale = 0.0f;
  std::vector<float> scale;
  std::vector<float> mean;
  std::string pixel_format;
  bool aligned = false;

  bool is_weight = false;
  bool overwritten = false;
};

class CviRoutine {
public:
  CviRoutine(flatbuffers::FlatBufferBuilder &fbb, bool isTpuRoutine)
    : isTpuRoutine(isTpuRoutine), fbb_(fbb) {}
  virtual ~CviRoutine() {}

  std::vector<Operation *> ops;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::string name;
  bool isTpuRoutine;

  virtual flatbuffers::Offset<Routine> build() = 0;

protected:
  flatbuffers::FlatBufferBuilder &fbb_;
};

class CviCpuRoutine : public CviRoutine {
public:
  CviCpuRoutine(flatbuffers::FlatBufferBuilder &fbb, FuncOp &fn, std::string &fnName);
  flatbuffers::Offset<Routine> build();

private:
  Operation *op_;
  void serializeFuncArgs(std::vector<uint8_t> &args);
};

class CviTpuRoutine : public CviRoutine {
public:
  CviTpuRoutine(flatbuffers::FlatBufferBuilder &fbb, FuncOp &fn, std::string &fnName);
  flatbuffers::Offset<Routine> build();

  std::vector<uint8_t> cmdbuf;

private:
  void codeGen();
};

class CviModelBuilder {
public:
  CviModelBuilder(ModuleOp &module);
  void storeModel(llvm::raw_ostream &output);

  ~CviModelBuilder() {
    for (auto &it : routines_) {
      delete it;
    }
  }

private:
  std::string modelName_;
  FuncOp mainFunc_;
  std::vector<CviRoutine *> routines_;
  std::vector<Operation *> ops_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  flatbuffers::FlatBufferBuilder fbb_;
  std::vector<uint8_t> binBuffer_;
  std::vector<std::shared_ptr<CviTensor>> tensorMaps_;
  int64_t privateGmemSize_ = 0;
  int64_t sharedGmemSize_ = 0;
  int batchNum_ = 0;

  void addRoutine(std::string funcName);
  void parseModule();
  FBModel build();
  FBWeightVector buildWeightMap();
  FBTensorVector buildNeuronMap();
  FBProgram buildProgram();
  FBSectionVector buildSections();
  FBSection buildSection(std::string name, cvi::model::SectionType type,
                         std::string fileName);
  FBSection buildSection(std::string name, cvi::model::SectionType type,
                         std::vector<uint8_t>& data);
};

namespace mlir {
void registerToCvimodelTranslation();
}

#endif
