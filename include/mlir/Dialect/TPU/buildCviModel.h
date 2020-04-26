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

#ifndef MIR_DIALECT_TPU_BUILD_CVIMODEL_H
#define MIR_DIALECT_TPU_BUILD_CVIMODEL_H

#include "cvibuilder/cvimodel_generated.h"
#include "cvibuilder/softmax_generated.h"
#include "cvibuilder/quantization_generated.h"
#include "cvibuilder/ssd_detection_generated.h"
#include "cvibuilder/parameter_generated.h"
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include <chrono>
#include <iomanip>
#include <ctime>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <openssl/md5.h>

#ifdef __unix__
#include <dirent.h>
#else
#include <io.h>
#endif

using namespace mlir;
using namespace cvi::model;
using FlatStrVecOffset = flatbuffers::Offset<flatbuffers::Vector<
                         flatbuffers::Offset<flatbuffers::String>>>;
//  Version shall be in schema
#define MAJOR_VER  1
#define MIN_VER  0
#define SUBMIN_VER 0

const std::string DFT_CHIP = "cv1835";
const std::string DFT_MODEL_FILENAME = "default.cm";

typedef struct {
  char magic[8];
  uint32_t body_size;
  char major;
  char minor;
  char md5[16];
  char padding[2];
} CviModelHeader;

typedef struct{
  QuantType type;
  float max_value;
  float min_value;
  float zero_point;
  float qscale;
} CviQuantInfo;

typedef struct {
  std::string name;
  long offset;
  size_t size;
  size_t shape[4];
  DType dtype;
} CviWeight;

typedef struct {
  size_t tensor_id;
  std::string name;
  long gaddr;
  DType dtype;
  size_t shape[4];
  size_t stride[4];
  CviQuantInfo quant;
  bool overWrote;
} CviTensor;

class CviMlirParser {
  public :
    CviMlirParser(ModuleOp &module) {
      init(module);
    }
    void init(ModuleOp &module);
    void collectedFuncInfo();
    void insertTensorMap(Operation *op);
    void insertWeightMap(Operation *op);

    size_t getBatchNum() { return batchNum_; }

    std::string getModelName() { return modelName_; }

    int getTpuNumRoutine() { return tpuOpsVec_.size(); }

    int getCpuNumRoutine() { return cpuInputTensorName_.size(); }

    std::string getCpuLibPath() { return cpuLibPath_;}

    std::string getWeightBinFileName() {return weightBinFileName_; }

    std::string getTpuFuncName() { return tpuFuncName_; }

    std::vector<uint8_t> getCmdBuf() { return cmdBuf_; }

    std::map<std::string, CviWeight>& getWeightMap() { return weightMap_; }

    std::vector<FuncOp> getCpuFunc() { return cpuFunc_; }

    std::vector<std::vector<Operation *>> getTpuOpsVec() {
      return tpuOpsVec_;
    }

    std::vector<std::pair<std::string, CviTensor>>& getTensorPairs() {
      return tensorPairs_;
    }

    std::vector<std::string>& getTpuInputTensorName(int tpuIndex = 0){
      return tpuInputTensorName_[tpuIndex];
    }

    std::vector<std::string>& getTpuOutputTensorName(int tpuIndex = 0) {
      return tpuOutputTensorName_[tpuIndex];
    }

    std::vector<std::string>& getCpuInputTensorName(int cpuIndex = 0) {
      return cpuInputTensorName_[cpuIndex];
    }

    std::vector<std::string>& getCpuOutputTensorName(int cpuIndex = 0) {
      return cpuOutputTensorName_[cpuIndex];
    }

    std::vector<std::string>& getProgramInTensorName() {
      return programInTensorName_;
    }

    std::vector<std::string>& getProgramOutTensorName() {
      return programOutTensorName_;
    }

    void setCmdBuf(std::vector<uint8_t> &cmdBuf) {
       cmdBuf_ = cmdBuf;
    }

  private :
    size_t batchNum_;
    std::string modelName_;
    std::string cpuLibPath_;
    std::string tpuFuncName_;
    std::vector<FuncOp> cpuFunc_;
    std::vector<uint8_t> cmdBuf_;
    std::string weightBinFileName_;
    std::map<std::string, CviWeight> weightMap_;
    std::vector<std::vector<Operation *>> tpuOpsVec_;
    std::vector<std::string> programInTensorName_;
    std::vector<std::string> programOutTensorName_;
    std::vector<std::vector<std::string>> tpuInputTensorName_;
    std::vector<std::vector<std::string>> tpuOutputTensorName_;
    std::vector<std::vector<std::string>> cpuInputTensorName_;
    std::vector<std::vector<std::string>> cpuOutputTensorName_;
    std::vector<std::pair<std::string, CviTensor>> tensorPairs_;
 };

class CviRoutine {
  public :
     CviRoutine(flatbuffers::FlatBufferBuilder *flatBuilder,
                CviMlirParser *parser)
                : flatBuilder_(flatBuilder), parser_(parser){}
    flatbuffers::FlatBufferBuilder *flatBuilder_;
    CviMlirParser *parser_;
};

class CviTpuRoutine : public CviRoutine {
  public :
    CviTpuRoutine(flatbuffers::FlatBufferBuilder *flatBuilder,
                  CviMlirParser *parser, int tpuIndex)
                  : CviRoutine(flatBuilder, parser), tpuIndex_(tpuIndex) {}
    flatbuffers::Offset<Routine> buildTpuRoutine();
  private :
    int tpuIndex_;
};

class CviCpuRoutine : public CviRoutine {
  public :
    CviCpuRoutine(flatbuffers::FlatBufferBuilder *flatBuilder,
                  CviMlirParser *parser, Operation *op)
                  : CviRoutine(flatBuilder, parser), op_(op) {}
    flatbuffers::Offset<Routine> buildCpuRoutine();
    void setInputOutputNames();
    std::vector<uint8_t> cpuOpSerialize();
  private :
    Operation *op_;
    std::vector<std::string> inputTensorNames_;
    std::vector<std::string> outputTensorNames_;
};

class CviProgram {
  using FlatTensorVecOffset = flatbuffers::Offset<flatbuffers::Vector<
                              flatbuffers::Offset<Tensor>>>;
  public :
    CviProgram(CviMlirParser *parser,
               flatbuffers::FlatBufferBuilder *flatBuilder)
               : parser_(parser), flatBuilder_(flatBuilder) {}
    void buildNeuronMap(FlatTensorVecOffset &flatTensorMap,
                        long &allocatedGmem);
    void buildInputsOutputs(FlatStrVecOffset& flatInputTensors,
                            FlatStrVecOffset &flatOutputTensors);
    flatbuffers::Offset<Program> build();
    void buildRoutines();
    void splitCpuRoutines(FuncOp &fn);

  private :
    CviMlirParser *parser_;
    std::vector<flatbuffers::Offset<Routine>> routines_;
    flatbuffers::FlatBufferBuilder *flatBuilder_;
};

class CviModel {
  using FlatWeightVecOffset = flatbuffers::Offset<flatbuffers::Vector<
                                            flatbuffers::Offset<Weight>>>;
  using FlatSectionVecOffset = flatbuffers::Offset<flatbuffers::Vector<
                                             flatbuffers::Offset<Section>>>;
  public :
    CviModel(CviMlirParser *parser,
             flatbuffers::FlatBufferBuilder *flatBuilder)
             : program_(parser, flatBuilder), parser_(parser),
               flatBuilder_(flatBuilder) {}
    void storeModel(llvm::raw_ostream &output);
    void buildWeightMap(FlatWeightVecOffset &flatWeightMap);
    FlatSectionVecOffset buildSections();
    flatbuffers::Offset<Model> build();
    void dataEncrypt(std::vector<uint8_t> &totalBin, uint8_t* resData);

  private :
    CviProgram program_;
    CviMlirParser *parser_;
    flatbuffers::FlatBufferBuilder *flatBuilder_;
};

#endif // MIR_DIALECT_TPU_BUILD_CVIMODEL_
