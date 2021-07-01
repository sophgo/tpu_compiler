//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
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
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//
#include <set>
#include <memory>
#include <sstream>
#include <fstream>
#include <regex>
#include <map>
#include <elf.h>
#include <openssl/md5.h>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "mlir/Support/FileUtilities.h"
#include "tpuc/Support/TensorFile.h"
#include "cvibuilder/cvimodel_generated.h"
#include "tpuc/MlirToCviModelTranslate.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"
#include "lz4.h"

#define DEBUG_TYPE "mlir-to-cvimodel"

static llvm::cl::opt<std::string>
    clCustomRuntimeLibraries("custom-runtime-lib",
                             llvm::cl::desc("Specify a comma-delimited list of custom op runtime lib"));

static llvm::cl::opt<std::string>
    clWeightBinFileName("weight-file", llvm::cl::desc("saved weight bin filename"));

extern llvm::cl::opt<std::string> clRunChipType;

static llvm::cl::opt<bool>
    clCompressCmdbuf("z", llvm::cl::desc("Enable compress cmdbuf"),
                  llvm::cl::init(false));

static llvm::cl::opt<bool>
    clUsingDmabuf("using-dmabuf", llvm::cl::desc("using dmabuf"),
                  llvm::cl::init(false));


typedef struct {
  char magic[8];
  uint32_t body_size;
  char major;
  char minor;
  char md5[16];
  char chip[16];
  char padding[2];
} CviModelHeader;

static void getFnInputsOutputs(FuncOp &fn, std::vector<std::string> &inputs,
                               std::vector<std::string> &outputs) {
  fn.walk([&](Operation *op) {
    if (isa<tpu::InputOp>(op)) {
      auto opName = getOpName(op);
      inputs.push_back(opName.str());
    } else if (isa<ReturnOp>(op)) {
      for (int i = 0; i < (int)op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        auto opdName = getOpName(opd);
        outputs.push_back(opdName.str());
      }
    }
  });
}

static void getOpGroupInputsOutputs(std::vector<Operation *> &group,
                                    std::vector<std::string> &inputs,
                                    std::vector<std::string> &outputs) {
  std::set<Operation *> producers;
  std::set<Operation *> consumers;
  std::set<Operation *> inputSet;
  std::set<Operation *> outputSet;
  std::vector<Operation *> output_candicates;

  for (auto op : group) {
    producers.insert(op);
    output_candicates.push_back(op);
    if (!llvm::isa<tpu::LoadWeightOp>(op) &&
        !llvm::isa<tpu::TL_LG_LoadCoeffOp>(op) &&
        !llvm::isa<tpu::NoneOp>(op)) {
      for (int i = 0; i < (int)op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        consumers.insert(opd);
      }
    }
  }
  std::set_difference(consumers.begin(), consumers.end(), producers.begin(),
                      producers.end(), std::inserter(inputSet, inputSet.begin()));
  std::set_difference(producers.begin(), producers.end(), consumers.begin(),
                      consumers.end(), std::inserter(outputSet, outputSet.begin()));

  std::set<std::string> inputsNameSet;
  for (auto op : inputSet) {
    inputs.push_back(op->getAttr("name").cast<StringAttr>().getValue().str());
    inputsNameSet.insert(op->getAttr("name").cast<StringAttr>().getValue().str());
  }

  // set diff not keep originial input order
  // re-order it for cpu op
  for (auto op : output_candicates) {
    if (outputSet.find(op) != outputSet.end()) {
      // check its inputs order by mlir
      if (auto castOp = llvm::dyn_cast<tpu::GenericCpuOp>(op)) {
        // get org order

        std::set<std::string> orgInputsSet;
        std::vector<std::string> orgInputs;
        for (int i = 0; i < (int)castOp.getNumOperands(); i++) {
          auto opd = castOp.getOperand(i).getDefiningOp();
          auto opdName = opd->getAttr("name").cast<StringAttr>().getValue();
          orgInputs.push_back(opdName.str());
          orgInputsSet.insert(opdName.str());
        }

        // check input is equal and non-order
        std::vector<std::string> diff;
        std::set_difference(
            orgInputsSet.begin(), orgInputsSet.end(),
            inputsNameSet.begin(), inputsNameSet.end(),
            std::inserter(diff, diff.begin()));

        // order not eq, reset to mlir order
        if (diff.size() == 0) {
          LLVM_DEBUG({
              std::stringstream msg;
              msg << "output name:" <<
                castOp->getAttr("name").cast<StringAttr>().getValue().str() << "\n";
              msg << " set diff inputs:\n";
              for (auto n : inputs) {
                msg << n << ",";
              }
              msg << "\norg order:\n";
              for (auto n : orgInputs) {
                msg << n << ",";
              }
              llvm::dbgs() << msg.str();
          });

          std::copy(orgInputs.begin(), orgInputs.end(), inputs.begin());
        }
      }
    }
  }

  // should keep output tensors in an inherent order.
  for (auto op : output_candicates) {
    if (outputSet.find(op) != outputSet.end()) {
      outputs.push_back(op->getAttr("name").cast<StringAttr>().getValue().str());
    }
  }
}

static void buildInputsOutputs(flatbuffers::FlatBufferBuilder &fbb,
                               std::vector<std::string> &inputs,
                               std::vector<std::string> &outputs,
                               FBStringVector &fbInputs, FBStringVector &fbOutputs) {

  std::vector<flatbuffers::Offset<flatbuffers::String>> fbStrVec;
  for (auto &name : inputs) {
    fbStrVec.push_back(fbb.CreateString(name));
  }
  fbInputs = fbb.CreateVector(fbStrVec);
  fbStrVec.clear();
  for (auto &name : outputs) {
    fbStrVec.push_back(fbb.CreateString(name));
  }
  fbOutputs = fbb.CreateVector(fbStrVec);
}

static void genMD5Hash(std::vector<uint8_t> &totalBin, uint8_t *resData) {
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, totalBin.data(), totalBin.size());
  MD5_Final(resData, &ctx);
}

static std::string getStrOfCurrentTime() {
  std::stringstream ssTime;
  auto clockNow = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(clockNow);
  ssTime << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
  return ssTime.str();
}

CviTensor::CviTensor(std::string name, TensorType &type, int64_t offset, bool isWeight)
    : name(name), offset(offset), is_weight(isWeight) {
  std::vector<int64_t> tensorShape = type.getShape();
  auto elementType = type.getElementType();

  if (elementType.isF32()) {
    dtype = DType::DType_FP32;
    dsize = 4;
  } else if (elementType.isBF16()) {
    dtype = DType::DType_BF16;
    dsize = 2;
  } else if (elementType.isInteger(8)) {
    dtype = elementType.isUnsignedInteger() ?
            DType::DType_UINT8 : DType::DType_INT8;
    dsize = 1;
  } else if (elementType.isInteger(16)) {
    dtype = elementType.isUnsignedInteger() ?
            DType::DType_UINT16 : DType::DType_INT16;
    dsize = 2;
  } else if (elementType.isInteger(32)) {
    dtype = DType::DType_INT32;
    dsize = 4;
  } else {
    llvm_unreachable("unsupported data type");
  }

  for (int i = 0; i < std::min((int)tensorShape.size(), 4); i++) {
    shape[i] = tensorShape[i];
  }
  if (tensorShape.size() > 4) {
    for (int i = 4; i < (int)tensorShape.size(); i++) {
      shape[3] *= tensorShape[i];
    }
  }
  size = dsize;
  for (int i = 0; i < 4; i++) {
    size *= shape[i];
  }
}

CviCpuRoutine::CviCpuRoutine(flatbuffers::FlatBufferBuilder &fbb, FuncOp &fn,
                             std::string &fnName)
    : CviRoutine(fbb, false) {
  fn.walk([&](Operation *op) {
    if (op->getName().getDialect()->getNamespace() != "tpu" ||
        llvm::isa<tpu::InputOp>(op) ||
        llvm::isa<tpu::WeightFileOp>(op) ||
        llvm::isa<ReturnOp>(op)) {
    } else if (op->getAttr("fn")) {
      auto belong = op->getAttr("fn").cast<StringAttr>().getValue();
      if (belong == fnName) {
        if (isa<tpu::ReshapeOp>(op)) {
          ops.push_back(op);
        } else if (auto castOp = llvm::dyn_cast<tpu::GenericCpuOp>(op)) {
          ops.push_back(op);
          op_ = op;
          SmallVector<StringRef, 2> sub_strs;
          auto opName = castOp.operation_name();
          opName.split(sub_strs, ".");
          name = sub_strs.size() > 1 ? sub_strs[1].str() : sub_strs[0].str();
        }
      }
    }
  });
  getOpGroupInputsOutputs(ops, inputs, outputs);
}

void CviCpuRoutine::serializeFuncArgs(std::vector<uint8_t> &args) {
  flatbuffers::FlatBufferBuilder fbb(1024);
  flatbuffers::Offset<cvi::cpu_op::Attribute> attr;
  std::vector<flatbuffers::Offset<cvi::cpu_op::Attribute>> param;
  auto paramDictAttr = op_->getAttr("param").cast<DictionaryAttr>();
  for (auto &iter : paramDictAttr) {
    auto key = iter.first.data();
    auto flatKey = fbb.CreateString(key);
    if (iter.second.isa<StringAttr>()) {
      auto value = iter.second.cast<StringAttr>().getValue();
      std::string strValue = std::string(value.data(), value.size());
      auto flatValue = fbb.CreateString(strValue);
      auto strAttr = cvi::cpu_op::CreateStrAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, strAttr, 0, 0);
    } else if (iter.second.isa<BoolAttr>()) {
      auto value = iter.second.cast<BoolAttr>().getValue();
      auto boolAttr = cvi::cpu_op::CreateBoolAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, boolAttr, 0, 0, 0, 0);
    } else if (iter.second.isa<IntegerAttr>()) {
      auto value = iter.second.cast<IntegerAttr>().getInt();
      auto intAttr = cvi::cpu_op::CreateIntAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, intAttr, 0, 0, 0);
    } else if (iter.second.isa<FloatAttr>()) {
      auto value = iter.second.cast<FloatAttr>().getValueAsDouble();
      auto floatAttr = cvi::cpu_op::CreateFloatAttr(fbb, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(fbb, floatAttr, 0, 0, 0, 0, 0);
    } else if (iter.second.isa<DenseFPElementsAttr>()) {
      std::vector<float> fpArray;
      auto value = iter.second.cast<DenseFPElementsAttr>();
      for (APFloat realVal : value) {
        fpArray.push_back(realVal.convertToFloat());
      }
      auto flatValue = fbb.CreateVector(fpArray);
      auto fpArrayAttr = cvi::cpu_op::CreateFloatArrayAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, fpArrayAttr, 0);
    } else if (iter.second.isa<DenseIntElementsAttr>()) {
      std::vector<int> intArray;
      auto value = iter.second.cast<DenseIntElementsAttr>();
      for (APInt intVal : value) {
        intArray.push_back(intVal.getZExtValue());
      }
      auto flatValue = fbb.CreateVector(intArray);
      auto intArrayAttr = cvi::cpu_op::CreateIntArrayAttr(fbb, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, 0, intArrayAttr);
    } else if (iter.second.isa<ArrayAttr>()) {
      auto value = iter.second.cast<ArrayAttr>();
      if ((*value.begin()).dyn_cast_or_null<IntegerAttr>()) {
        std::vector<int> intArray;

        for (auto& intVal : value) {
          intArray.push_back(intVal.cast<IntegerAttr>().getInt());
        }
        auto flatValue = fbb.CreateVector(intArray);
        auto intArrayAttr = cvi::cpu_op::CreateIntArrayAttr(fbb, flatKey, flatValue);
        attr = cvi::cpu_op::CreateAttribute(fbb, 0, 0, 0, 0, 0, intArrayAttr);
      }
      else {
        llvm_unreachable("unsupported type, only support i32 array parsing");
      }
    } else {
      llvm_unreachable("unsupported type");
    }
    param.push_back(attr);
  }

  auto fbParam = cvi::cpu_op::CreateParameterDirect(fbb, &param);
  fbb.Finish(fbParam);

  uint8_t *ptr = fbb.GetBufferPointer();
  for (uint32_t i = 0; i < fbb.GetSize(); i++) {
    args.push_back(*ptr++);
  }
}

flatbuffers::Offset<Routine> CviCpuRoutine::build() {
  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);

  std::vector<uint8_t> args;
  serializeFuncArgs(args);
  auto fbRoutine = CreateCpuRoutineDirect(fbb_, name.c_str(), &args);
  return CreateRoutine(fbb_, RoutineType_CPU, fbInputs, fbOutputs, 0, fbRoutine);
}

CviTpuRoutine::CviTpuRoutine(flatbuffers::FlatBufferBuilder &fbb, FuncOp &fn,
                             std::string &fnName)
    : CviRoutine(fbb, true) {

  name = fnName;

  fn.walk([&](Operation *op) {
    if (op->getName().getDialect()->getNamespace() != "tpu" || llvm::isa<tpu::InputOp>(op) ||
        llvm::isa<tpu::WeightFileOp>(op) || llvm::isa<ReturnOp>(op)) {
    } else if (op->getAttr("fn")) {
      auto belong = op->getAttr("fn").cast<StringAttr>().getValue();
      if (name == belong) {
        ops.push_back(op);
      }
    }
  });

  getOpGroupInputsOutputs(ops, inputs, outputs);

  codeGen();
}

void CviTpuRoutine::codeGen() {
  auto backend_ctx = cvi_backend_create_context(clRunChipType.c_str());

  for (auto op : ops) {
    if (auto tgOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
      int layer_id = mlir::getOpLayerId(op);
      cvi_backend_set_layer_id(backend_ctx, layer_id);
      tgOp.codegen((void *)backend_ctx);
    } else if (auto tlOp = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op)) {
      int layer_id = mlir::getOpLayerId(op);
      cvi_backend_set_layer_id(backend_ctx, layer_id);
      // enable parallel
      if (tlOp.getEnableParallel() &&
          !isa<tpu::TL_LG_INT8_LrnOp>(op) &&
          !isa<tpu::TL_LG_BF16_LrnOp>(op)) {
        cvi_backend_parallel_enable(backend_ctx);
      }
      // tl codegen
      tlOp.codegen((void *)backend_ctx);
      // disable parallel
      if (tlOp.getDisableParallel() &&
          !isa<tpu::TL_LG_INT8_LrnOp>(op) &&
          !isa<tpu::TL_LG_BF16_LrnOp>(op)) {
        cvi_backend_parallel_disable(backend_ctx);
      }
    }
  }

  cvi_backend_submit(backend_ctx);
  cvi_backend_get_cmdbuf(backend_ctx, cmdbuf);
  if (clUsingDmabuf) {
    cvi_backend_dmabuf_convert(backend_ctx, dmabuf);
  }
  cvi_backend_delete_context(backend_ctx);
}

flatbuffers::Offset<Routine> CviTpuRoutine::build() {
  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs, outputs, fbInputs, fbOutputs);
  auto fbName = fbb_.CreateString(name);
  if (clUsingDmabuf) {
    auto fbRoutine = CreateTpuRoutine(fbb_, 0, fbName);
    return CreateRoutine(fbb_, RoutineType_TPU, fbInputs, fbOutputs, fbRoutine,
                         0);
  } else {
    auto fbRoutine = CreateTpuRoutine(fbb_, fbName);
    return CreateRoutine(fbb_, RoutineType_TPU, fbInputs, fbOutputs, fbRoutine,
                         0);
  }
}

CviModelBuilder::CviModelBuilder(ModuleOp &module) : fbb_(1024) {
  std::vector<std::string> subFuncNames;
  for (auto fn : module.getOps<FuncOp>()) {
    if (fn.getName() == "tpu_func") {
      mainFunc_ = fn;
      if (fn->getAttr("chipname")) {
        clRunChipType = fn->getAttr("chipname").cast<StringAttr>().getValue().str();
      }
      privateGmemSize_ = fn->getAttr("private_gmem").cast<IntegerAttr>().getInt();
      sharedGmemSize_ = fn->getAttr("shared_gmem").cast<IntegerAttr>().getInt();
      continue;
    }

    addRoutine(fn.getName().str());
  }

  parseModule();
}

void CviModelBuilder::addRoutine(std::string funcName) {
  bool tpu = (funcName.substr(0, 3) == "tpu");
  CviRoutine *rt = nullptr;
  if (tpu) {
    rt = new CviTpuRoutine(fbb_, mainFunc_, funcName);
  } else {
    rt = new CviCpuRoutine(fbb_, mainFunc_, funcName);
  }
  routines_.push_back(rt);
}

static void loadQScaleTable(FuncOp &fn, std::map<std::string, float> &qscaleMap,
                            std::map<std::string, int> &zpMap) {
  auto tableName = fn->getAttr("qscale_table").cast<StringAttr>().getValue().str();
  std::ifstream infile(tableName);

  std::string line;
  std::regex pattern("[a-zA-Z0-9.:;_\\/-]+ [-+0-9.e]+ [-+0-9]+");
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string name;
    if (std::regex_match(line, pattern)) {
      float qscale;
      int32_t zero_point;
      if (!(iss >> name >> qscale >> zero_point)) {
        break;
      }
      qscaleMap[name] = qscale;
      zpMap[name] = zero_point;
    } else {
      llvm::errs() << line;
      llvm::errs() << "\n  => not match required format\n";
      assert(false);
    }
  }
}

void CviModelBuilder::parseModule() {

  std::map<std::string, float> qscaleMap;
  std::map<std::string, int> zpMap;
  loadQScaleTable(mainFunc_, qscaleMap, zpMap);
  getFnInputsOutputs(mainFunc_, inputs_, outputs_);

  mainFunc_.walk([&](Operation *op) {
    if (op->getName().getDialect()->getNamespace() != "tpu" || isa<tpu::NoneOp>(op) ||
        isa<ReturnOp>(op)) {
      // continue
    } else if (llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op) &&
               !isa<tpu::TL_LG_JoinOp>(op)) {
      // continue
    } else if (auto castOp = llvm::dyn_cast<tpu::WeightFileOp>(op)) {
      modelName_ = castOp.filename().str();
      int pos = modelName_.find("_");
      if (pos != -1) {
        modelName_ = modelName_.substr(0, pos);
      }
    } else if (auto castOp = llvm::dyn_cast<tpu::LoadWeightOp>(op)) {
      auto type = castOp.getResult().getType().template cast<TensorType>();
      auto tensor = std::make_shared<CviTensor>(
          castOp.name().str(), type,
          castOp.offset().getValue(),
          true);
      tensorMaps_.push_back(tensor);
    } else if (auto castOp = llvm::dyn_cast<tpu::ReshapeOp>(op)) {
      auto findTensor = [this](std::string name) {
        for (auto &tensor : this->tensorMaps_) {
          if (tensor->name == name) {
            return tensor;
          }
        }
        llvm_unreachable(("cannot find tensor:" + name).c_str());
      };
      auto name = castOp.name().str();
      auto type = castOp.getResult().getType().template cast<TensorType>();
      auto opd = op->getOperand(0).getDefiningOp();
      auto opdName = opd->getAttr("name").cast<StringAttr>().getValue();
      auto opdTensor = findTensor(opdName.str());
      auto tensor = std::make_shared<CviTensor>(name, type, opdTensor->offset, false);
      tensor->overwritten = opdTensor->overwritten;
      tensorMaps_.push_back(tensor);
    } else if (auto castOp = llvm::dyn_cast<tpu::InputOp>(op)) {
      auto name = castOp.name().str();
      auto type = op->getOperand(0).getType().template cast<TensorType>();
      int64_t offset = op->getAttr("gaddr") ?
                       op->getAttr("gaddr").cast<IntegerAttr>().getInt() : -1;
      auto tensor = std::make_shared<CviTensor>(name, type, offset, false);
      if (qscaleMap.find(name) != qscaleMap.end()) {
        if (zpMap[name] == 0){
          tensor->setInt8SymQuantInfo(qscaleMap[name]);
        } else {
          tensor->setInt8AsymQuantInfo(qscaleMap[name], zpMap[name]);
        }
      }
      if (castOp.preprocessAttr()) {
        auto preprocess = castOp.preprocessAttr();
        for (auto s : preprocess.scale().getAsValueRange<FloatAttr>()) {
          tensor->scale.push_back(s.convertToFloat());
        }
        for (auto m : preprocess.mean().getAsValueRange<FloatAttr>()) {
          tensor->mean.push_back(m.convertToFloat());
        }
        tensor->pixel_format = preprocess.pixel_format().getValue().str();
        tensor->aligned = preprocess.aligned().getValue();
      }
      tensorMaps_.push_back(tensor);
    } else {
      auto name = op->getAttr("name").cast<StringAttr>().getValue().str();
      auto type = op->getResult(0).getType().template cast<TensorType>();
      int64_t offset =
          op->getAttr("gaddr") ? op->getAttr("gaddr").cast<IntegerAttr>().getInt() : -1;
      auto tensor = std::make_shared<CviTensor>(name, type, offset, false);

      if (qscaleMap.find(name) != qscaleMap.end()) {
        if (zpMap[name] == 0){
          tensor->setInt8SymQuantInfo(qscaleMap[name]);
        } else {
          tensor->setInt8AsymQuantInfo(qscaleMap[name], zpMap[name]);
        }
      }

      if (!batchNum_) {
        batchNum_ = tensor->shape[0];
      }
      bool overwritten = false;
      if (op->getAttr("tl_store_flag") &&
          !op->getAttr("tl_store_flag").cast<BoolAttr>().getValue()) {
        overwritten = true;
      }
      if (op->getAttr("buffer_reused") &&
          op->getAttr("buffer_reused").cast<BoolAttr>().getValue()) {
        overwritten = true;
      }
      if (op->getAttr("store_compr_act") &&
          op->getAttr("store_compr_act").cast<BoolAttr>().getValue()) {
        overwritten = true;
      }
      if (op->getAttr("do_early_stride") &&
          op->getAttr("do_early_stride").cast<BoolAttr>().getValue()) {
        overwritten = true;
      }

      if (llvm::dyn_cast<tpu::TL_LG_JoinOp>(op)) {
        auto tpuOp =
            llvm::dyn_cast<tpu::TL_LG_StoreOp>(op->getOperand(0).getDefiningOp());
        if (tpuOp.store_compr_act().hasValue())
          overwritten = tpuOp.store_compr_act().getValue();
      }

      if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_SliceOp>(op)) {
        if (tpuOp.load_compr_act().hasValue())
          overwritten = tpuOp.load_compr_act().getValue();
      }

      tensor->overwritten = overwritten;
      tensorMaps_.push_back(tensor);
    }
  });
}

FBSection CviModelBuilder::buildSection(std::string name, cvi::model::SectionType type,
                                        std::string fileName) {
  auto fbName = fbb_.CreateString(name);
  uint32_t size = 0;
  uint32_t offset = 0;
  if (!fileName.empty()) {
    std::string errorMessage;
    auto file = openInputFile(fileName, &errorMessage);
    size = (uint32_t)file->getBufferSize();
    offset = (uint32_t)binBuffer_.size();
    auto ptr = (const uint8_t *)file->getBufferStart();
    for (uint32_t i = 0; i < size; i++) {
      binBuffer_.push_back(ptr[i]);
    }
  }
  return CreateSection(fbb_, type, fbName, size, offset);
}

FBSection CviModelBuilder::buildSection(std::string name, cvi::model::SectionType type,
                                        std::vector<uint8_t>& data) {
  auto fbName = fbb_.CreateString(name);
  uint32_t size = 0;
  uint32_t offset = 0;

  size = (uint32_t)data.size();
  offset = (uint32_t)binBuffer_.size();

  // if need compress data
  do {
    if (!clCompressCmdbuf || !size) {
      break;
    } else if (type != SectionType_CMDBUF &&
               type != SectionType_DMABUF) {
      break;
    }

    size_t out_bufsize = LZ4_compressBound(size);
    std::vector<uint8_t> out_buf(out_bufsize);

    size_t out_size = LZ4_compress_default(
        reinterpret_cast<char *>(data.data()),
        reinterpret_cast<char *>(out_buf.data()), size, out_bufsize);
    if (out_size < 1) {
      llvm::errs() << "compress buf failed!\n";
      break;
    }

    llvm::errs() << "compress buf [" << name.c_str()
                 << " ] " << size << " => "
                 << out_size << "\n";

    if (out_size > size) {
      llvm::errs() << "compressed size large than decompressed size don't need compress!\n";
      break;
    }
    binBuffer_.insert(binBuffer_.end(), out_buf.begin(), out_buf.begin() + out_size);
    return CreateSection(fbb_, type, fbName, out_size, offset, false, true, size);
  } while (false);

  // don't need compress data
  if (data.size()) {
    binBuffer_.insert(binBuffer_.end(), data.begin(), data.end());
  }
  return CreateSection(fbb_, type, fbName, size, offset);
}

FBSectionVector CviModelBuilder::buildSections() {
  std::vector<FBSection> sectionVec;
  // build weight section
  assert(!clWeightBinFileName.empty());
  auto weightSec = buildSection("weight", SectionType_WEIGHT, clWeightBinFileName);
  sectionVec.push_back(weightSec);

  // build tpu cmdbuf section
  for (auto rt : routines_) {
    if (rt->isTpuRoutine) {
      auto tpuRt = (CviTpuRoutine *)rt;
      if (clUsingDmabuf) {
        auto dmabufSec =
            buildSection(tpuRt->name, SectionType_DMABUF, tpuRt->dmabuf);
        sectionVec.push_back(dmabufSec);
      } else {
        auto cmdbufSec =
            buildSection(tpuRt->name, SectionType_CMDBUF, tpuRt->cmdbuf);
        sectionVec.push_back(cmdbufSec);
      }
    }
  }

  // build custom cpu functions section
  if (!clCustomRuntimeLibraries.empty()) {
    auto isFileExist = [](StringRef &file) {
      std::ifstream ifs(file.str().c_str());
      return ifs.good();
    };
    auto getElfMachineField = [](StringRef &file) {
      std::ifstream ifs(file.str().c_str());
      Elf64_Ehdr hdr;
      ifs.read((char *)(&hdr), sizeof(hdr));
      return (uint16_t)hdr.e_machine;
    };

    SmallVector<StringRef, 2> paths;
    StringRef(clCustomRuntimeLibraries).split(paths, ",");

    for (auto &path : paths) {
      if (isFileExist(path)) {
        auto machine = getElfMachineField(path);
        SectionType type = SectionType_FUNC_X86;
        if (machine == 0x3E) { // 'amd64'
          type = SectionType_FUNC_X86;
          llvm::errs() << "find x86_64 custom plugin:" << path << "\n";
        } else if (machine == 0xB7) { // 'aarch64'
          type = SectionType_FUNC_AARCH64;
          llvm::errs() << "find aarch64 custom plugin:" << path << "\n";
        } else {
          llvm::errs() << "unsupported plugin format\n";
          assert(0);
        }
        auto customSec = buildSection("custom", type, path.str());
        sectionVec.push_back(customSec);
      }
    }
  }
  return fbb_.CreateVector(sectionVec);
}

FBModel CviModelBuilder::build() {
  Version modelVersion =
      Version(MajorVersion_value, MinorVersion_value, SubMinorVersion_value);
  auto fbModelName = fbb_.CreateString(modelName_);
  auto fbBuildTime = fbb_.CreateString(getStrOfCurrentTime());
  auto fbTarget = fbb_.CreateString(clRunChipType);
  auto fbMlirVersion = fbb_.CreateString(MLIR_VERSION);
  auto fbWeightMap = buildWeightMap();
  auto fbSections = buildSections();
  auto fbProgram = buildProgram();
  std::vector<FBProgram> programVec;
  programVec.push_back(fbProgram);
  auto fbProgramVec = fbb_.CreateVector(programVec);
  return CreateModel(fbb_, &modelVersion, fbModelName, fbBuildTime, 0, 0,
                     fbWeightMap, fbProgramVec, fbSections, fbTarget, fbMlirVersion);
}

FBWeightVector CviModelBuilder::buildWeightMap() {
  std::vector<FBWeight> fbWeightVec;

  for (auto &tensor : tensorMaps_) {
    if (!tensor->is_weight)
      continue;

    std::vector<int64_t> shape;
    for (int i = 0; i < 4; i++) {
      shape.push_back(tensor->shape[i]);
    }
    auto fbName = fbb_.CreateString(tensor->name);
    auto fbShape = CreateShapeDirect(fbb_, &shape);
    auto fbWeight =
        CreateWeight(fbb_, fbName, tensor->offset, tensor->size, fbShape, tensor->dtype);
    fbWeightVec.push_back(fbWeight);
  }
  return fbb_.CreateVector(fbWeightVec);
}

FBTensorVector CviModelBuilder::buildNeuronMap() {
  std::vector<flatbuffers::Offset<Tensor>> tensorVec;

  for (auto &tensor : tensorMaps_) {
    if (tensor->is_weight)
      continue;

    std::vector<int64_t> shape;
    for (int i = 0; i < 4; i++) {
      shape.push_back(tensor->shape[i]);
    }
    auto fbShapeVec = fbb_.CreateVector(shape);
    auto fbShape = CreateShape(fbb_, fbShapeVec);
    auto fbQuant = CreateQuantInfo(fbb_, tensor->quant_type, 0, 0,
                                   tensor->zero_point, tensor->qscale);
    auto fbTensor = CreateTensorDirect(
        fbb_, 0, tensor->name.c_str(), tensor->offset, tensor->dtype,
        fbShape, 0, fbQuant, tensor->overwritten,
        tensor->scale.size() ? &tensor->scale : nullptr,
        tensor->mean.size() ? &tensor->mean : nullptr,
        tensor->pixel_format.length() ?
            tensor->pixel_format.c_str() : nullptr,
        tensor->aligned);
    tensorVec.push_back(fbTensor);
  }
  return fbb_.CreateVector(tensorVec);
}

FBProgram CviModelBuilder::buildProgram() {
  auto fbNeuronMap = buildNeuronMap();

  FBStringVector fbInputs, fbOutputs;
  buildInputsOutputs(fbb_, inputs_, outputs_, fbInputs, fbOutputs);

  std::vector<FBRoutine> fbRoutineVec;
  for (auto rt : routines_) {
    fbRoutineVec.push_back(rt->build());
  }
  auto fbRoutines = fbb_.CreateVector(fbRoutineVec);
  return CreateProgram(fbb_, batchNum_, 0, fbInputs, fbOutputs, fbNeuronMap, fbRoutines,
                       (uint32_t)sharedGmemSize_, (uint32_t)privateGmemSize_);
}

void CviModelBuilder::storeModel(llvm::raw_ostream &output) {
  FBModel fbModel = build();
  fbb_.Finish(fbModel);

  std::vector<uint8_t> modelData;
  modelData.resize(fbb_.GetSize() + binBuffer_.size());
  uint8_t *dst = modelData.data();
  uint8_t *src = fbb_.GetBufferPointer();
  for (uint32_t i = 0; i < fbb_.GetSize(); i++) {
    *dst++ = *src++;
  }
  src = binBuffer_.data();
  for (uint32_t i = 0; i < binBuffer_.size(); i++) {
    *dst++ = *src++;
  }
  binBuffer_.clear();

  CviModelHeader header;
  genMD5Hash(modelData, (uint8_t *)header.md5);
  std::string magic = u8"CviModel";
  std::string padding = u8"AA";
  memcpy(header.magic, magic.c_str(), 8);
  memcpy(header.padding, magic.c_str(), 2);
  memset(header.chip, 0, sizeof(header.chip));
  strncpy(header.chip, clRunChipType.c_str(), clRunChipType.length());
  header.body_size = fbb_.GetSize();
  header.major = MajorVersion_value; // defined in cvimodel.fbs
  header.minor = MinorVersion_value; // defined in cvimodel.fbs

  output.write(reinterpret_cast<char *>(&header), sizeof(CviModelHeader));
  output.write(reinterpret_cast<char *>(modelData.data()), modelData.size());
}

static LogicalResult translateModule(ModuleOp module, llvm::raw_ostream &output) {
  CviModelBuilder builder(module);
  builder.storeModel(output);
  return success();
}

void mlir::registerToCvimodelTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-cvimodel",
      [](ModuleOp module, raw_ostream &output) {
        return translateModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<tpu::TPUDialect, mlir::StandardOpsDialect>();
      });
}
