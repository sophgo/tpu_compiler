#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <chrono>

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
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
namespace py = pybind11;

using tensor_map_t = std::map<std::string, std::vector<float>>;
using shape_map_t = std::map<std::string, std::vector<int64_t>>;
using np_data_t = py::array_t<float, py::array::c_style | py::array::forcecast>;

static py::array genNumpyArray(std::shared_ptr<std::vector<float>> &vec,
                               const std::vector<int64_t> &shape) {
  std::vector<unsigned> stride_v(shape.size(), sizeof(float));
  for (int i = shape.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }
  return py::array(py::buffer_info(vec->data(), sizeof(float),
                      py::format_descriptor<float>::format(),
                      shape.size(), shape, stride_v));
}


class Timer {
public:
  Timer() {
    restart();
  }

  void restart() {
    start = std::chrono::steady_clock::now();
  }

  void stopAndPrint(std::string tag) {
    auto duration = std::chrono::steady_clock::now() - start;
    auto mills = std::chrono::duration_cast<
                      std::chrono::milliseconds>(duration);
    printf("%s => %d ms\n", tag.c_str(), (int)mills.count());
  }
private:
  std::chrono::steady_clock::time_point start;
};

class PyTuner {
public:
  PyTuner(int batch);
  ~PyTuner() {}

  void load(const std::string &mlir_file);
  void quantize(const std::string &calib_table, const std::string &mix_table);
  void buildInterpreter(std::string &target_op);
  void setData(const std::string &name, np_data_t &array, int bidx);
  void setData(np_data_t &array, int bidx);
  void invokeTo(std::string &name);
  py::list getOpInfo();
  py::array getTensor(std::string &name, int bidx);
  std::string getTensorType(std::string &name);
  py::dict getAllTensors(int bidx);
  py::list getOutputDetails();

private:
  bool isValidOp(Operation *op);

public:
  static std::string version;

private:
  std::unique_ptr<MLIRContext> context;
  OwningModuleRef module_;
  int batch = 0;
  std::vector<std::unique_ptr<MlirModuleInterpreter>> interpreters;
  std::string pluginFilePath_ = "";
};

PyTuner::PyTuner(int batch) : batch(batch) {
  for (int i = 0;  i < batch; i++) {
    interpreters.emplace_back(std::make_unique<MlirModuleInterpreter>());
  }
}

void PyTuner::load(const std::string &mlir_file) {
  // Timer timer;
  if (context) {
    context.reset();
  }

  DialectRegistry registry;
  registry.insert<tpu::TPUDialect, StandardOpsDialect>();
  context = std::make_unique<MLIRContext>(registry);

  std::string errorMessage;
  auto file = openInputFile(mlir_file, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    llvm_unreachable("read find failed");
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // delete previous module
  auto module = this->module_.release();
  if (module) {
    module.erase();
  }

  this->module_ = OwningModuleRef(parseSourceFile(sourceMgr, context.get()));
  if (!this->module_) {
    llvm_unreachable("could not parse the input IR\n");
  }
  // timer.stopAndPrint("load");
}

void PyTuner::quantize(const std::string &calib_table, const std::string &mix_table) {
  // Timer timer;

  std::vector<const char*> cmdline = {
    "tpuc-opt",
    "--chipname", "cv183x",
    "--calibration-table", calib_table.c_str(),
    "--quant-int8-mix-bf16-layers-from-file",
    mix_table.c_str()
  };

  registerDoAssignChipNamePass();
  registerDoImportCalibrationTablePass();
  registerDoTpuQuantPass();

  llvm::cl::ResetAllOptionOccurrences();
  llvm::cl::ParseCommandLineOptions(cmdline.size(), cmdline.data());

  mlir::PassManager pm(context.get());
  pm.addNestedPass<FuncOp>(mlir::createAssignChipNamePass());
  pm.addNestedPass<FuncOp>(mlir::createImportCalibrationTablePass());
  pm.addNestedPass<FuncOp>(mlir::createTpuQuantPass());
  mlir::applyPassManagerCLOptions(pm);
  if (mlir::failed(pm.run(*this->module_))) {
    assert(0);
  }
  /*
  std::string errorMessage;
  auto output = openOutputFile("quantized_xxxx.mlir", &errorMessage);
  this->module_->print(output->os());
  output->keep();
  */
  // timer.stopAndPrint("quantize");
}

static int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

void PyTuner::buildInterpreter(std::string &target_op) {
  // Timer timer;
  MlirModuleInterpreter::updateWeightMap(module_);
  // timer.stopAndPrint("update weight list");
  // timer.restart();

  #pragma omp parallel for schedule(static, omp_schedule(batch))
  for (int i = 0; i < batch; i++) {
    interpreters[i]->loadModule(module_, target_op);
  }
  // timer.stopAndPrint("update kernel list");
}

bool PyTuner::isValidOp(Operation *op) {
  return (op->getName().getDialect()->getNamespace() == "tpu" &&
          !isa<tpu::WeightFileOp>(op) &&
          !isa<tpu::LoadWeightOp>(op) &&
          !isa<tpu::NoneOp>(op));
}

py::list PyTuner::getOpInfo() {
  py::list op_list;
  ModuleOp m = module_.get();
  for (FuncOp func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (!isValidOp(op)) {
        return;
      } else {
        py::dict d;
        d["name"] = getOpName(op).str();
        d["type"] = op->getName().getStringRef().str();
        if (auto quantableOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
          d["quant"] = quantableOp.getOpQuant().str();
        } else {
          d["quant"] = "FP32";
        }
        op_list.append(d);
      }
    });
    break;
  }
  return op_list;
}

void PyTuner::setData(const std::string &name,
                      np_data_t &array, int bidx) {
  int idx = 0;
  auto &details = interpreters[bidx]->input_details;
  if (!name.empty()) {
    bool found = false;
    for (; idx < (int)details.size(); idx++) {
      if (name == details[idx].first) {
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::errs() << "cannot find input tensor with name:"
                  << name << "\n";
      llvm_unreachable("Please check..");
    }
  }
  if (array.size() != details[idx].second) {
    llvm::errs() << "input tensor size not same, needed is "
                << details[idx].second << ", get "
                << array.size() << "\n";
    llvm_unreachable("please check..");
  }
  std::vector<float> input_data(array.size());
  std::memcpy(input_data.data(), array.data(),
              array.size() * sizeof(float));
  interpreters[bidx]->setTensor(details[idx].first, input_data);
}

void PyTuner::setData(np_data_t &array, int bidx) {
  setData("", array, bidx);
}

py::array PyTuner::getTensor(std::string &name, int bidx) {
  std::shared_ptr<std::vector<float>> tensor = interpreters[bidx]->getTensor(name);
  std::vector<int64_t> shape = interpreters[bidx]->getTensorShape(name);
  return genNumpyArray(tensor, shape);
}

std::string PyTuner::getTensorType(std::string &name) {
  return interpreters[0]->getDataType(name);
}

py::list PyTuner::getOutputDetails() {
  py::list list;
  auto outputs = interpreters[0]->outputDetails();
  for (auto &i : outputs) {
    list.append(i);
  }
  return list;
}

py::dict PyTuner::getAllTensors(int bidx) {
  py::dict dict;
  for (auto &kv : interpreters[bidx]->activationMapping) {
    py::str py_s(kv.first);
    dict[py_s] = genNumpyArray(kv.second, {(int64_t)kv.second->size()});
  }
  return dict;
}

void PyTuner::invokeTo(std::string &targetOp) {
  // Timer timer;
  for (int i = 0; i < batch; i++) {
    interpreters[i]->invokeTo(targetOp);
  }
  // timer.stopAndPrint("invokeTo");
}

std::string PyTuner::version = MLIR_VERSION;

// wrap as Python module
PYBIND11_MODULE(pytuner, m) {
  m.doc() = "pytuner for mlir";

  py::class_<PyTuner>(m, "tuner", "MLIR Tuner")
      .def(py::init<int>())
      .def("load", &PyTuner::load, "load module from fp32 mlir file")
      .def("quantize", &PyTuner::quantize, py::arg("calib_table"),
           py::arg("mix_table") = "",
           "quantization with calib_table and mix_table")
      .def("build", &PyTuner::buildInterpreter, py::arg("target_op") = "",
           "rebuild interpreter to target layer if set")
      .def("op_info", &PyTuner::getOpInfo)
      .def("set_data", py::overload_cast<np_data_t&, int>(&PyTuner::setData))
      .def("set_data", py::overload_cast<const std::string&, np_data_t&, int>(&PyTuner::setData))
      .def("get_tensor", &PyTuner::getTensor)
      .def("get_tensor_type", &PyTuner::getTensorType)
      .def("get_all_tensors", &PyTuner::getAllTensors)
      .def("get_output_details", &PyTuner::getOutputDetails)
      .def("invoke", &PyTuner::invokeTo, py::arg("target_op") = "",
           "run forward to taget layer if set")
      .def_readonly_static("version", &PyTuner::version);

  py::class_<Timer>(m, "timer", "MLIR Timer")
      .def(py::init<>())
      .def("restart", &Timer::restart)
      .def("stop_and_show", &Timer::stopAndPrint);
}