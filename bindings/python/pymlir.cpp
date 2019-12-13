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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
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

#define OP_TYPE "type"
#define OP_QUANT "quant"

typedef std::map<std::string, std::vector<float>> tensor_map_t;

static bool isValidOp(Operation &op)
{
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::LoadFileOp>(op) &&
          op.getName().getDialect().str() == "tpu");
}

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

static int runTpuInterpreter(OwningModuleRef &m, std::vector<float> &input, std::vector<float> &output,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer, tensor_map_t &tensorMap) {
  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  //std::vector<float> input(1*3*224*224);
  //std::vector<float> output(1*1000);

  std::vector<std::vector<float> *> inputs({&input});
  std::vector<std::vector<float> *> outputs({&output});

  if (failed(ModuleInterpreter::runModuleAndGetValueMap<>(m.get(), inputs, outputs, tensorMap)))
    return EXIT_FAILURE;

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

template <typename Dtype>
static py::array getPythonArray(std::vector<Dtype> &vec, const std::vector<int64_t> &shape)
{
  std::vector<unsigned> stride_v(shape.size(), sizeof(Dtype));
  for (int i = shape.size()-1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }

  return py::array(py::buffer_info(
    vec.data(),                           /* data as contiguous array  */
    sizeof(Dtype),                           /* size of one scalar        */
    py::format_descriptor<Dtype>::format(),  /* data type                 */
    shape.size(), //ndim,                                    /* number of dimensions      */
    shape, //shape,                                   /* shape of the matrix       */
    stride_v //strides                                  /* strides for each axis     */
  ));
}

template
static py::array getPythonArray(std::vector<float> &vec, const std::vector<int64_t> &shape);
template
static py::array getPythonArray(std::vector<int64_t> &vec, const std::vector<int64_t> &shape);

// Static initialization for standard op dialect registration.
static DialectRegistration<StandardOpsDialect> StandardOps;

class py_module {
public:
  py_module() {}

  void load(std::string filename) {
    module = parseMLIRInput(filename, &context);
    if (!module) {
      llvm::errs() << "could not parse the input IR\n";
      exit(-1);
    }

    parseMLIRInfo();
  }

  void dump() {
    module->dump();
  }

  void parseMLIRInfo() {
    ModuleOp m = module.get();

    for (FuncOp function : m.getOps<FuncOp>()) {
      for (Block &bb : function.getBlocks()) {
        for (auto &op : bb) {
          if (!isValidOp(op)) {
            if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(op)) {
              weightFilePath_ = loadFileOp.getAttrOfType<StringAttr>("filename").getValue().str();
            }

            continue;
          }

          // TODO: Only support one output tesor for now.
          auto result = op.getResult(0);
          std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();

          py::dict py_temp;
          py_temp[OP_TYPE] = op.getName().getStringRef().str();
          py_temp[OP_QUANT] = getOpQuant(&op);
          py::str py_s(getOpName(&op).str());
          opInfo_[py_s] = py_temp;

          shapeMap_[getOpName(&op).str()] = shape;
        }
      }
    }
  }

  py::dict getAllTensor() {
    py::dict py_ret;
    for (auto it = tensorMap_.begin(); it != tensorMap_.end(); it++) {
      auto op = it->first;
      auto data = it->second;
      py::str py_s(op);

      assert(shapeMap_.end() != shapeMap_.find(op));
      py_ret[py_s] = getPythonArray(data, shapeMap_[op]);

    }

    return py_ret;
  }

  py::array getTensor(std::string op_name) {
    py::array py_ret;

    for (auto it = tensorMap_.begin(); it != tensorMap_.end(); it++ ) {
      auto op = it->first;

      if (op == op_name) {
        auto data = it->second;

        assert(shapeMap_.end() != shapeMap_.find(op));
        py_ret = getPythonArray(data, shapeMap_[op]);
        break;
      }
    }

    return py_ret;
  }

  py::str getWeightFilePath() {
    py::str py_s(weightFilePath_);

    return py_s;
  }

  // wrap C++ function with NumPy array IO
  py::array run(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    std::vector<float> input(array.size());
    // copy py::array -> std::vector
    std::memcpy(input.data(), array.data(), array.size() * sizeof(float));

    // run intererence
    std::vector<float> output(1*1000);
    int status = runTpuInterpreter(module, input, output, nullptr, tensorMap_);
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

public:
  py::dict opInfo_;

private:
  MLIRContext context;
  OwningModuleRef module;
  tensor_map_t tensorMap_;
  std::string weightFilePath_;
  std::map<std::string, std::vector<int64_t>> shapeMap_;
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
    .def("get_all_tensor", &py_module::getAllTensor,
         "dump all tensor data")
    .def("get_tensor", &py_module::getTensor,
         "get one tensor data")
    .def_readwrite("op_info", &py_module::opInfo_)
    .def("get_weight_file_path", &py_module::getWeightFilePath,
         "get weight file path")
    .def("run", &py_module::run,
         "run module inference with input array, and return output array");
}
