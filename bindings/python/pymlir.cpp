#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// -------------
// pure C++ code
// -------------

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
  return (op.getName().getDialect()->getNamespace() == "tpu" &&
          !isa<tpu::WeightFileOp>(op) && !isa<tpu::LoadWeightOp>(op) &&
          !isa<tpu::NoneOp>(op));
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
  py_module() {}
  ~py_module() {
    interpreter_.reset();
    context.reset();
  }

  void load(std::string filename) {
    if (context) {
      context.reset();
    }

    DialectRegistry registry;
    registry.insert<tpu::TPUDialect, StandardOpsDialect>();
    context = std::make_unique<MLIRContext>(registry);

    module_ = parseMLIRInput(filename);
    if (!module_) {
      llvm_unreachable("could not parse the input IR\n");
    }
    if (interpreter_) {
      interpreter_.reset();
    }
    interpreter_ = std::make_unique<ModuleInterpreter>(module_.get());
    interpreter_->allocate_tensors();
    parseMLIRInfo();
  }

  OwningModuleRef parseMLIRInput(StringRef inputFilename) {
    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      llvm_unreachable("read find failed");
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningModuleRef(parseSourceFile(sourceMgr, context.get()));
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
                  weightFileOp->getAttrOfType<StringAttr>("filename")
                      .getValue()
                      .str();
            }
            continue;
          }
          py::dict py_temp;
          py_temp[OP_NAME] = getOpName(&op).str();
          py_temp[OP_TYPE] = op.getName().getStringRef().str();
          if (auto quantableOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
            py_temp[OP_QUANT] = quantableOp.getOpQuant().str();
          } else {
            py_temp[OP_QUANT] = "NONE";
          }
          opInfo_.append(py_temp);
        }
      }
    }
  }

  py::dict getAllTensor() {
    tensor_map_t tensorMap_;
    shape_map_t shapeMap_;
    auto all_tensor_names = interpreter_->get_all_tensor_name();
    for (auto &tensor_name : all_tensor_names) {
      tensorMap_[tensor_name] = interpreter_->get_tensor(tensor_name);
      shapeMap_[tensor_name] = interpreter_->get_tensor_shape(tensor_name);
    }

    return getTensorDict(tensorMap_, shapeMap_);
  }
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

  py::dict getWeightData() {
    auto weight_map = interpreter_->getWeightData();
    tensor_map_t tensorMap_;
    shape_map_t shapeMap_;
    for (auto &weight_tensor : weight_map) {
      std::string name = weight_tensor.first;
      std::vector<float> tensor_data = weight_tensor.second.first;
      std::vector<int64_t> tensor_shape = weight_tensor.second.second;
      tensorMap_[name] = tensor_data;
      shapeMap_[name] = tensor_shape;
    }
    return getTensorDict(tensorMap_, shapeMap_);
  }
  void setWeightData(
      std::map<std::string,
               py::array_t<float, py::array::c_style | py::array::forcecast>>
          weight_map) {

    for (auto &weight_tensor : weight_map) {
      std::string name = weight_tensor.first;
      auto weight_data = weight_tensor.second;
      std::vector<float> input_vec(weight_data.size());
      std::memcpy(input_vec.data(), weight_data.data(),
                  weight_data.size() * sizeof(float));
      interpreter_->setWeightData(name, input_vec);
    }
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

    size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                        1, std::multiplies<int64_t>());
    auto input_details = interpreter_->get_input_details();
    size_t all_need_data_size = 0;
    for (auto &i : input_details) {
      all_need_data_size += i.second;
    }
    if (input_size != all_need_data_size) {
      llvm::errs() << "input data size: " << input_size << "\n";
      for (auto &i : input_details) {
        llvm::errs() << i.first << " needed data size: " << i.second << "\n";
      }
      llvm::errs() << "all input needed size: " << all_need_data_size << "\n";
      llvm_unreachable("input data size not same with all input needed size");
    }

    // set tensor
    size_t slice_idx = 0;
    for (auto &i : input_details) {
      std::vector<float> input_data(input_vec.begin() + slice_idx,
                                    input_vec.begin() + slice_idx + i.second);
      slice_idx += i.second;
      interpreter_->set_tensor(i.first, input_data);
    }
    assert(slice_idx == input_vec.size());
    interpreter_->invoke();

    tensor_map_t results;
    shape_map_t shapeMap_;

    auto output_details = interpreter_->get_output_details();
    for (auto &output_name : output_details) {
      results[output_name] = interpreter_->get_tensor(output_name);
      shapeMap_[output_name] = interpreter_->get_tensor_shape(output_name);
    }

    return getTensorDict(results, shapeMap_);
  }

public:
  py::list opInfo_;

private:
  std::unique_ptr<MLIRContext> context;
  OwningModuleRef module_;
  std::string weightFilePath_;
  // tensor_map_t tensorMap_;
  // shape_map_t shapeMap_;

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
      .def("set_tensor", &py_module::set_tensor)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def_readwrite("op_info", &py_module::opInfo_)
      .def("get_weight_file_path", &py_module::getWeightFilePath,
           "get weight file path")
      .def("getWeightData", &py_module::getWeightData, "get weight data")
      .def("setWeightData", &py_module::setWeightData, "set weight data")
      .def("run", &py_module::run,
           "run module inference with input array, and return output array")
      .def("allocate_tensors", &py_module::allocate_tensors)
      .def("get_tensors_info", &py_module::get_tensor_info)
      .def("dump", &py_module::dump)
      .def("invoke", py::overload_cast<>(&py_module::invoke))
      .def("invoke", py::overload_cast<const std::string>(&py_module::invoke))
      .def("get_input_details", &py_module::get_input_details)
      .def("get_output_details", &py_module::get_output_details);
}
