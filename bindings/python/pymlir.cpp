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

#define OP_NAME "name"
#define OP_TYPE "type"
#define OP_QUANT "quant"

typedef std::map<std::string, std::vector<float> > tensor_map_t;
typedef std::map<std::string, std::vector<int64_t> > shape_map_t;

static bool isValidOp(Operation &op)
{
  return (op.getName().getDialect().str() == "tpu"
          && !isa<tpu::WeightFileOp>(op)
          && !isa<tpu::LoadWeightOp>(op)
          && !isa<tpu::NoneOp>(op)
          );
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

static py::dict getTensorDict(tensor_map_t &tensorMap, shape_map_t &shapeMap) {
  py::dict py_ret;
  for (auto it = tensorMap.begin(); it != tensorMap.end(); it++) {
    auto op = it->first;
    auto data = it->second;
    py::str py_s(op);

    assert(shapeMap.end() != shapeMap.find(op));
    py_ret[py_s] = getPythonArray(data, shapeMap[op]);
  }

  return py_ret;
}

// Static initialization for standard op dialect registration.
static DialectRegistration<StandardOpsDialect> StandardOps;
static mlir::DialectRegistration<mlir::tpu::TPUDialect> TPUOps;

class py_module {
public:
  py_module() {}
  ~py_module() {
    if (interpreter_)
      delete interpreter_;
  }

  void load(std::string filename) {
    module = parseMLIRInput(filename, &context);
    if (!module) {
      llvm::errs() << "could not parse the input IR\n";
      exit(-1);
    }

    interpreter_ = new ModuleInterpreter(module.get());

    parseMLIRInfo();
  }

  void dump() {
    module->dump();
  }

  void setDeivce(std::string d) {
    if (interpreter_ == nullptr){
      llvm::errs() << "Not initialize model.\n";
      exit(-1);
    }else{
      interpreter_->setDevice(d);
    }
  }

  void parseMLIRInfo() {
    ModuleOp m = module.get();

    for (FuncOp function : m.getOps<FuncOp>()) {
      for (Block &bb : function.getBlocks()) {
        for (auto &op : bb) {
          if (!isValidOp(op)) {
            if (auto weightFileOp = dyn_cast<tpu::WeightFileOp>(op)) {
              weightFilePath_ = weightFileOp.getAttrOfType<StringAttr>("filename").getValue().str();
            }
            continue;
          }

          py::dict py_temp;
          py_temp[OP_NAME] = getOpName(&op).str();
          py_temp[OP_TYPE] = op.getName().getStringRef().str();
          // py_temp[OP_QUANT] = getOpQuant(&op).str();
          opInfo_.append(py_temp);
        }
      }
    }
  }

  py::dict getAllTensor() {
    return getTensorDict(tensorMap_, shapeMap_);
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
  py::dict run(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    std::vector<float> input_vec(array.size());
    std::memcpy(input_vec.data(), array.data(), array.size() * sizeof(float));
    std::vector<int64_t> input_shape;
    for (ssize_t i = 0; i < array.ndim(); ++i) {
      input_shape.push_back((int64_t)array.shape()[i]);
    }
    tensor_map_t results;
    if (failed(runTpuModule(module.get(), interpreter_, input_shape, input_vec,
                            &results, &shapeMap_, &tensorMap_))) {
      assert(false);
    }

    return getTensorDict(results, shapeMap_);
  }

public:
  py::list opInfo_;

private:
  MLIRContext context;
  OwningModuleRef module;
  std::string weightFilePath_;
  tensor_map_t tensorMap_;
  shape_map_t shapeMap_;
  ModuleInterpreter *interpreter_;
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
    .def("setDevice", &py_module::setDeivce, "set inference device, cpu or gpu")
    .def("run", &py_module::run,
         "run module inference with input array, and return output array");
}
