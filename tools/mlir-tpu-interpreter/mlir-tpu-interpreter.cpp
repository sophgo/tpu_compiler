//===- mlir-tpu-interpreter.cpp - MLIR TPU Dialect Interpreter Driver---------------------===//
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
// Main entry point to a command line utility that executes an MLIR file on the
// TPU by interpreting MLIR to certain function calls and executing later.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static void dump_data_float_abs(const char * const desc, const void * const addr,
    int n, int c, int h, int w)
{
#define ABS_LEN 4
#define ABS_COUNT 1
  int ni, ci, hi, wi;
  int off;
  const float *data = (const float *)addr;

  /* Output description if given. */
  if (desc != NULL)
    printf("%s: abs, n=%d, c=%d, h=%d, w=%d\n", desc, n, c, h, w);

  /* Process first and last 4 col and 4 raw in the data. */
  for (ni = 0; ni < n; ni++) {
    for (ci = 0; ci < c; ci++) {
      if ((ni * c + ci) == ABS_COUNT)
        printf("\n .\n .\n .\n");
      if ((ni * c + ci) >= ABS_COUNT && (ni * c + ci) <= (n * c - ABS_COUNT - 1))
        continue;
      printf("=== n = %02d, c = %02d ===\n", ni, ci);
      for (hi = 0; hi < h; hi++) {
        if (hi == ABS_LEN)
          printf(" ... \n");
        if (hi >= ABS_LEN && hi <= h - ABS_LEN - 1)
          continue;
        printf("[ ");
        for (wi = 0; wi < w; wi++) {
          if (wi == ABS_LEN)
            printf(" ... ");
          if (wi >= ABS_LEN && wi <= w - ABS_LEN - 1)
            continue;
          off = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
          if (data[off] >= 0)
            printf(" ");
          printf("%2.2f ", data[off]);
        }
        printf("]\n");
      }
    }
  }
}

static size_t read_bianry_file(std::string filename, std::vector<float> &v,
    size_t bytes = 0) {
  std::ifstream is;
  is.open(filename.c_str(), std::ios::in | std::ios::binary);
  if (!is) {
    llvm::errs() << "Open " << filename << " failed.\n";
    assert(0);
  }
  // use size in argument first
  if (bytes == 0) {
    // if vector is pre-allocated, use the vector size
    if (v.size() != 0) {
      bytes = v.size() * sizeof(float);
    } else {
      // finally, use the file total size
      is.seekg(0, is.end);
      bytes = is.tellg();
      is.seekg(0, is.beg);
    }
  }
  if (v.size() * sizeof(float) < bytes) {
    v.resize(bytes/sizeof(float));
  }
  llvm::errs() << "read " << bytes << " bytes from " << filename << "\n";
  is.read(reinterpret_cast<char*>(v.data()), bytes);
  is.close();
  return bytes;
}

static size_t write_bianry_file(std::string filename, std::vector<float> &v,
    size_t size = 0) {
  std::ofstream os;
  os.open(filename.c_str(), std::ios::out | std::ios::binary);
  if (size == 0) {
    size = v.size() * sizeof(float);
  }
  llvm::errs() << "write " << size << " bytes to " << filename << "\n";
  os.write(reinterpret_cast<const char*>(v.data()), size);
  os.close();
  return size;
}

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> inputTensorFilename("tensor-in",
    llvm::cl::desc("Input Tensor Filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputTensorFilename("tensor-out",
    llvm::cl::desc("Output Tensor Filename"),
    llvm::cl::init("-"));

static OwningModuleRef parseMLIRInput(StringRef inputFilename,
                                      MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningModuleRef(parseSourceFile(sourceMgr, context));
}

namespace mlir {

int TpuInterpreterMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TPU interpreter driver\n");

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  std::vector<int64_t> input_shape({1,3,224,224});
  std::vector<float> input_vec(1*3*224*224);
  read_bianry_file(inputTensorFilename, input_vec);
  std::map<std::string, std::vector<float> > results;

  if (failed(runTpuModule(m.get(), input_shape, input_vec, &results, nullptr)))
    return EXIT_FAILURE;

  for ( auto it = results.begin(); it != results.end(); it++ ) {
    if (outputTensorFilename == "-") {
      dump_data_float_abs("output", it->second.data(), 1, 1, 10, 100);
    } else {
      write_bianry_file(outputTensorFilename, it->second);
    }
  }

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}

} // namespace mlir

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(createPrintTpuOpStatsPass());

  if (failed(pm.run(m)))
    return failure();

  if (failed(m.verify()))
    return failure();

  return success();
}

// TODO: merge with JitRunnerMain()
int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  return mlir::TpuInterpreterMain(argc, argv, &runMLIRPasses);
}
