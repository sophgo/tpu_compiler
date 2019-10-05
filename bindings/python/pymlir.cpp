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

static int runTpuInterpreter(OwningModuleRef &m,
    std::vector<float> &input, std::vector<float> &output,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  //std::vector<float> input(1*3*224*224);
  //std::vector<float> output(1*1000);

  std::vector<std::vector<float> *> inputs({&input});
  std::vector<std::vector<float> *> outputs({&output});

  if (failed(runTpuModule(m.get(), inputs, outputs)))
    return EXIT_FAILURE;

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}

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

  // wrap C++ function with NumPy array IO
  py::array run(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    std::vector<float> input(array.size());
    // copy py::array -> std::vector
    std::memcpy(input.data(), array.data(), array.size() * sizeof(float));

    // run intererence
    std::vector<float> output(1*1000);
    int status = runTpuInterpreter(module, input, output, nullptr);
    assert(status == EXIT_SUCCESS);

    // return NumPy array
    //ssize_t ndim = array.ndim();
    //std::vector<ssize_t> shape(ndim);
    //shape.assign(array.shape(), array.shape() + ndim);
    //std::vector<ssize_t> strides(ndim);
    //strides.assign(array.strides(), array.strides() + ndim);

    //ssize_t ndim = 2;
    //std::vector<ssize_t> shape(1, 1000);
    //std::vector<ssize_t> strides(1000 * sizeof(float), sizeof(float));

    return py::array(py::buffer_info(
      output.data(),                           /* data as contiguous array  */
      sizeof(float),                           /* size of one scalar        */
      py::format_descriptor<float>::format(),  /* data type                 */
      2, //ndim,                                    /* number of dimensions      */
      {1, 1000}, //shape,                                   /* shape of the matrix       */
      {1000 * sizeof(float), sizeof(float)} //strides                                  /* strides for each axis     */
    ));
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
         "dump module")
    .def("run", &py_module::run,
         "run module inference with input array, and return output array");
}
