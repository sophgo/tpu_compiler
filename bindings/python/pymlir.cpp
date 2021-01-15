#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// -------------
// pure C++ code
// -------------

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"

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

typedef std::map<std::string, std::vector<float>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

static bool isValidOp(Operation &op) {
  return (op.getName().getDialect().str() == "tpu" &&
          !isa<tpu::WeightFileOp>(op) && !isa<tpu::LoadWeightOp>(op) &&
          !isa<tpu::NoneOp>(op));
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
static py::array getPythonArray(std::vector<Dtype> &vec,
                                const std::vector<int64_t> &shape) {
  std::vector<unsigned> stride_v(shape.size(), sizeof(Dtype));
  for (int i = shape.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }

  return py::array(
      py::buffer_info(vec.data(),    /* data as contiguous array  */
                      sizeof(Dtype), /* size of one scalar        */
                      py::format_descriptor<Dtype>::format(), /* data type */
                      shape.size(),                           // ndim/
                      shape,                                  // shape
                      stride_v                                // strides
                      ));
}

template static py::array getPythonArray(std::vector<float> &vec,
                                         const std::vector<int64_t> &shape);
template static py::array getPythonArray(std::vector<int64_t> &vec,
                                         const std::vector<int64_t> &shape);

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

class py_module {
public:
  py_module() {
    auto &registry = context.getDialectRegistry();
    registry.insert<tpu::TPUDialect, StandardOpsDialect>();
  }
  ~py_module() {
    if (interpreter_)
      interpreter_.reset();
  }

  void load(std::string filename) {
    module_ = parseMLIRInput(filename, &context);
    if (!module_) {
      llvm_unreachable("could not parse the input IR\n");
    }
    if (interpreter_) {
      interpreter_.reset();
      interpreter_ = std::make_unique<ModuleInterpreter>(module_.get());
    } else {
      interpreter_ = std::make_unique<ModuleInterpreter>(module_.get());
    }
    parseMLIRInfo();
  }

  void dump(std::string name) { interpreter_->dump(name); }

  void parseMLIRInfo() {
    ModuleOp m = module_.get();

    for (FuncOp function : m.getOps<FuncOp>()) {
      for (Block &bb : function.getBlocks()) {
        for (auto &op : bb) {
          if (!isValidOp(op)) {
            if (auto weightFileOp = dyn_cast<tpu::WeightFileOp>(op)) {
              weightFilePath_ =
                  weightFileOp.getAttrOfType<StringAttr>("filename")
                      .getValue()
                      .str();
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

  py::dict getAllTensor() { return getTensorDict(tensorMap_, shapeMap_); }
  py::dict get_input_details() {
    py::dict ret;
    std::vector<std::pair<std::string, size_t>> inputs =
        interpreter_->get_input_details();
    for (auto &i : inputs) {
      ret[i.first.c_str()] = i.second;
    }
    return ret;
  }
  py::list get_output_details() {
    py::list ret;
    std::vector<std::string> outputs = interpreter_->get_output_details();
    for (auto &i : outputs) {
      ret.append(i);
    }
    return ret;
  }
  py::dict get_tensor_info() {
    std::vector<std::pair<std::string, std::string>> op_infos =
        interpreter_->get_tensor_info();
    py::dict ret;
    for (auto &i : op_infos) {
      ret[i.first.c_str()] = i.second;
    }
    return ret;
  }

  py::array getTensor(std::string op_name) {
    py::array py_ret;

    for (auto it = tensorMap_.begin(); it != tensorMap_.end(); it++) {
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

  void setPluginFilePath(std::string path) { pluginFilePath_ = path; }
  void allocate_tensors() { interpreter_->allocate_tensors(); }
  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    std::vector<float> input_data(data.size());
    std::memcpy(input_data.data(), data.data(), data.size() * sizeof(float));
    interpreter_->set_tensor(name, input_data);
  }
  py::array get_tensor(std::string name) {
    std::vector<float> tensor = interpreter_->get_tensor(name);
    std::vector<int64_t> shape = interpreter_->get_tensor_shape(name);
    return getPythonArray(tensor, shape);
  }
  void invoke(const std::string name) { interpreter_->invoke(name); }
  void invoke() { interpreter_->invoke(); }

  // wrap C++ function with NumPy array IO
  py::dict
  run(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    if (!interpreter_) {
      throw std::runtime_error("Not load mlir Model");
    }
    std::vector<float> input_vec(array.size());
    std::memcpy(input_vec.data(), array.data(), array.size() * sizeof(float));
    std::vector<int64_t> input_shape;
    for (ssize_t i = 0; i < array.ndim(); ++i) {
      input_shape.push_back((int64_t)array.shape()[i]);
    }
    tensor_map_t results;

    if (failed(interpreter_->doRun(input_shape, input_vec, &results,
                                   &tensorMap_))) {
      assert(false);
    }
    interpreter_->getShape(&shapeMap_);

    return getTensorDict(results, shapeMap_);
  }

public:
  py::list opInfo_;

private:
  MLIRContext context;
  OwningModuleRef module_;
  std::string weightFilePath_;
  tensor_map_t tensorMap_;
  shape_map_t shapeMap_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
  std::string pluginFilePath_ = "";
};

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";

  py::class_<py_module>(m, "module", "MLIR Module")
      .def(py::init<>())
      .def("load", &py_module::load, "load module from IR")
      .def("set_plugin", &py_module::setPluginFilePath,
           "set file path of custom op plugin")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("get_tensor", &py_module::getTensor, "get one tensor data")
      .def_readwrite("op_info", &py_module::opInfo_)
      .def("get_weight_file_path", &py_module::getWeightFilePath,
           "get weight file path")
      .def("run", &py_module::run,
           "run module inference with input array, and return output array")
      .def("allocate_tensors", &py_module::allocate_tensors)
      .def("set_tensor", &py_module::set_tensor)
      .def("get_tensors", &py_module::get_tensor)
      .def("get_tensors_info", &py_module::get_tensor_info)
      .def("dump", &py_module::dump)
      .def("invoke", py::overload_cast<>(&py_module::invoke))
      .def("invoke", py::overload_cast<const std::string>(&py_module::invoke))
      .def("get_input_details", &py_module::get_input_details)
      .def("get_output_details", &py_module::get_output_details);
}
