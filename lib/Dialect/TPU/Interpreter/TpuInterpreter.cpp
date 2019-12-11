//===- TpuInterpreter.cpp - Implementation of TPU Op Interpreter ---------===//
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
// This file implements the TPU dialect Interpreter.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"

#include <numeric>
#include <functional>

namespace mlir {

LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  // #include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
  if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(opInst)) {
    llvm::errs() << "LoadFileOp" << "\n";
    auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
    llvm::errs() << "  filename " << filename << "\n";
    weight_is = std::make_unique<std::ifstream>(filename.str(),
        std::ios::in | std::ios::binary);

    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    llvm::errs() << "LoadWeightOp" << "\n";
    auto offset = loadWeightOp.offset().getLimitedValue();
    llvm::errs() << "  offset " << offset << "\n";
    auto result = loadWeightOp.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto weight_data = std::make_unique<std::vector<float> >(size);

    weight_is.get()->seekg(offset, std::ios::beg);
    weight_is.get()->read((char*)weight_data.get()->data(), size * sizeof(float));

    valueMapping[result] = std::move(weight_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    llvm::errs() << "Conv2DOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::AveragePool2DOp>(opInst)) {
    llvm::errs() << "AveragePool2DOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::MaxPool2DOp>(opInst)) {
    llvm::errs() << "MaxPool2DOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    llvm::errs() << "FullyConnectedOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::ReluOp>(opInst)) {
    llvm::errs() << "ReluOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::BatchNormOp>(opInst)) {
    llvm::errs() << "BatchNormOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::ScaleOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::EltwiseOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::ReshapeOp>(opInst)) {
    llvm::errs() << "ReshapeOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here

    valueMapping[result] = std::move(result_data);

    return success();
  }

  if (auto op = dyn_cast<ConstantOp>(opInst)) {
    llvm::errs() << "ConstantOp" << "\n";
    //op.dump();
    // TODO: use specific Op for null operand
    // only support zero constant for now
    // TODO: check isZero

    // it it safe to ignore, put null pointer to the valueMapping
    auto result = op.getResult();
    valueMapping[result] = std::move(nullptr);

    return success();
  }

  if (auto op = dyn_cast<ReturnOp>(opInst)) {
    llvm::errs() << "ReturnOp" << "\n";
    //op.dump();
    std::vector<float> *return_vec;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        return_vec = it->second.get();
        llvm::errs() << "      vec size = " << return_vec->size() << "\n";
      }
      operandIdx++;
    }

    //copy the value into outputs
    assert(outputs.size() == 1);
    outputs[0]->swap(*return_vec);

    return success();
  }

  return opInst.emitError("unsupported operation: ")
         << opInst.getName();
}

LogicalResult ModuleInterpreter::runBlock(Block &bb) {
  // Traverse operations.
  for (auto &op : bb) {
    if (failed(runOperation(op)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runOneFunction(FuncOp func) {
  llvm::errs() << "func " << func.getName() << "\n";
  // Clear the value mappings, it is only relevant within one function.
  valueMapping.clear();

  // Add function arguments to the value remapping table.
  unsigned int argIdx = 0;
  assert(inputs.size() == 1);
  for (auto arg : func.getArguments()) {
    llvm::errs() << "arg " << argIdx << ": ";
    arg->getType().dump();
    llvm::errs() << "\n";

    // copy the inputs[0] into a unique_ptr pointed vector
    // TODO: pass input as unique_ptr directly
    auto input = std::make_unique<std::vector<float> >();
    input.get()->swap(*inputs[0]);
    valueMapping[arg] = std::move(input);
    argIdx++;
  }
  assert(argIdx == 1);

  // Then, convert blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    llvm::errs() << "run " << function.getName() << "\n";

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult runTpuModule(ModuleOp m,
    std::vector<std::vector<float> *> &inputs,
    std::vector<std::vector<float> *> &outputs) {
  return ModuleInterpreter::runModule<>(m, inputs, outputs);
}

} // namespace mlir
