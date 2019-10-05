#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>


// -------------
// pure C++ code
// -------------

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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

#if 0
int TpuInterpreterMain(std::string filename,
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

  std::vector<float> input(1*3*224*224);
  std::vector<float> output(1*1000);
  //std::fill (std::begin(input), std::end(input), 1.0f);
  read_bianry_file(inputTensorFilename, input);

  std::vector<std::vector<float> *> inputs({&input});
  std::vector<std::vector<float> *> outputs({&output});

  if (failed(runTpuModule(m.get(), inputs, outputs)))
    return EXIT_FAILURE;

  if (outputTensorFilename == "-") {
    dump_data_float_abs("output", outputs[0]->data(), 1, 1, 10, 100);
  } else {
    write_bianry_file(outputTensorFilename, output);
  }

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}
#endif

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// Static initialization for standard op dialect registration.
static DialectRegistration<StandardOpsDialect> StandardOps;

class py_module {
public:
  py_module() {}

  void load(std::string filename) {
    //const char *argv= "-";
    //llvm::cl::ParseCommandLineOptions(1, &argv, "PY MLIR\n");
    module = parseMLIRInput(filename, &context);
    if (!module) {
      llvm::errs() << "could not parse the input IR\n";
      exit(-1);
    }
  }

  void dump() {
    module->dump();
  }

private:
  MLIRContext context;
  OwningModuleRef module;
};

// wrap as Python module
PYBIND11_MODULE(pymlir,m)
{
  m.doc() = "pybind11 for mlir";

  py::class_<py_module>(m, "module", "MLIR Module")
    .def(py::init<>())
    .def("load", &py_module::load,
         "load module from IR")
    .def("dump", &py_module::dump,
         "dump module");
}
