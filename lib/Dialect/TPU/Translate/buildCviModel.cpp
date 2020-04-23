#include "mlir/Dialect/TPU/buildCviModel.h"

static llvm::cl::opt<std::string> clCpuLibPath(
    "cpu-lib-path",
    llvm::cl::desc("indicated cpu lib path"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> clInputWeightBin(
    "tpu-input-weight-bin",
    llvm::cl::desc("saved weight bin filename"),
    llvm::cl::init("-"));

static DType convert2DType(Type type) {
  switch (type.getKind()) {
  case StandardTypes::BF16:
    return DType::DType_BF16;
  case StandardTypes::F32:
    return DType::DType_FP32;
  case StandardTypes::Integer: {
    auto integer = type.cast<IntegerType>();
    switch (integer.getWidth()) {
    case 8:
      return DType::DType_INT8;
    case 16:
      return DType::DType_INT16;
    case 32:
      return DType::DType_INT32;
    default :
      llvm_unreachable("unsupported type");
    }
  }
  case StandardTypes::RankedTensor: {
    auto v = type.cast<RankedTensorType>();
    auto tensorType = v.getElementType();
    return convert2DType(tensorType);
  }
  default :
    llvm_unreachable("unsupported type");
  }
}

#ifdef __unix__
static bool findCpuLib(std::string path, std::string libName) {
  DIR *dir;
  struct dirent *ptr;
  if ((dir = opendir(path.c_str())) == NULL) {
    return false;
  }
  while ((ptr = readdir(dir)) != nullptr) {
    if (strcmp(ptr->d_name, libName.c_str()) == 0) {
      return true;
    }
  }
  closedir(dir);
  return false;
}
#else
static bool findCpuLib(std::string path, std::string libName) {
  struct _finddata_t fileinfo;
  long handle = _findfirst(path.c_str(), &fileinfo);
  if( -1 == handle) {
    return false;
  }
  do {
    if (strcmp(fileinfo.name, libName.c_str()) == 0)
      return true;$
  } while(_findnext(hand, &fileinfo) == 0);

  return false;
}
#endif

static size_t getDTypeByteWidths(DType dtype) {
  switch (dtype) {
  case DType::DType_FP32 :
  case DType::DType_INT32 :
  case DType::DType_UINT32 :
    return 4;
  case DType::DType_BF16 :
  case DType::DType_INT16 :
    return 2;
  case DType::DType_INT8 :
  case DType::DType_UINT8 :
    return 1;
  default :
    llvm_unreachable("unsupported type");
  }
}

void CviMlirParser::insertWeightMap(Operation *op) {
  std::string weightName;
  if (!isa<tpu::LoadWeightOp>(op))
    return;
  if (op->getAttr("name") == nullptr)
    return;
  else
    weightName = op->getAttr("name").cast<StringAttr>().getValue();

  std::pair<std::string, CviWeight> weightPair;
  weightPair.first = weightName;
  long &offset = weightPair.second.offset;
  if (op->getAttr("offset")) {
    offset = op->getAttr("offset").cast<IntegerAttr>().getInt();
  }

  DType &dtype = weightPair.second.dtype;
  size_t *shape = weightPair.second.shape;
  auto resType = op->getResult(0)->getType();
  dtype = convert2DType(resType);
  auto resTensorType = resType.cast<RankedTensorType>();
  for (unsigned int i = 0; i < resTensorType.getShape().size(); i++) {
    shape[i] = resTensorType.getShape()[i];
  }
  for (int i = resTensorType.getShape().size(); i < 4; i++) {
    shape[i] = 1;
  }

  weightPair.second.name = weightName;
  weightPair.second.size = 0;
  weightMap_.insert(weightPair);
}

void CviMlirParser::insertTensorMap(Operation *op) {
  std::string tensorName;
  if (isa<tpu::WeightFileOp>(op) || isa<tpu::LoadWeightOp>(op) ||
      isa<ReturnOp>(op) || isa<tpu::TG_CallOp>(op) ||
      isa<tpu::ReshapeOp>(op) || isa<tpu::TG_INT8_SliceOp>(op) ||
      isa<tpu::TG_BF16_SliceOp>(op))
    return;
  if (op->getAttr("name") == nullptr)
    return;
  else
   tensorName = op->getAttr("name").cast<StringAttr>().getValue();

  std::pair<std::string, CviTensor> tensorPair;
  tensorPair.first = tensorName;

  DType &dtype = tensorPair.second.dtype;
  size_t *shape = tensorPair.second.shape;
  size_t *stride = tensorPair.second.stride;
  CviQuantInfo &quant = tensorPair.second.quant;
  bool &overWrote = tensorPair.second.overWrote;
  long &gaddr = tensorPair.second.gaddr;
  gaddr = -1;
  if (op->getAttr("gaddr")) {
    gaddr = op->getAttr("gaddr").cast<IntegerAttr>().getInt();
  }
  if (op->getAttr("offset")) {
    gaddr = op->getAttr("offset").cast<IntegerAttr>().getInt();
  }

  auto resType = op->getResult(0)->getType();
  dtype = convert2DType(resType);
  auto resTensorType = resType.cast<RankedTensorType>();
  for (unsigned int i = 0; i < resTensorType.getShape().size(); i++) {
    shape[i] = resTensorType.getShape()[i];
  }
  for (int i = resTensorType.getShape().size(); i < 4; i++) {
    shape[i] = 1;
  }

  stride[0] = 1;
  stride[1] = stride[0] * shape[0];
  stride[2] = stride[1] * shape[1];
  stride[3] = stride[2] * shape[2];

  quant.type = QuantType::QuantType_NONE;
  quant.max_value = 1.0;
  quant.min_value = 0.0;
  quant.zero_point = 0.0;
  quant.qscale = 1.0;

  overWrote = op->getAttr("fuse_next") ? true : false;
  if (!overWrote) {
    if (op->getAttr("tl_store_flag")) {
     auto storeFlag  = op->getAttr("tl_store_flag").cast<BoolAttr>().getValue();
     if (!storeFlag)
       overWrote = true;
    }

    if (op->getAttr("buffer_reused")) {
     auto reusedFlag  =
         op->getAttr("buffer_reused").cast<BoolAttr>().getValue();
     if (reusedFlag)
       overWrote = true;
    }
  }

  tensorPairs_.push_back(tensorPair);
  sort(tensorPairs_.begin(), tensorPairs_.end(),
      [](std::pair<std::string, CviTensor> left,
      std::pair<std::string, CviTensor> right) {
        return left.second.gaddr < right.second.gaddr;
      });
}

void CviMlirParser::collectedFuncInfo() {
  std::vector<Operation *> defOps;
  std::vector<Operation *> useOps;
  for (auto &tpuOps : tpuOpsVec_) {
    for (auto &op : tpuOps) {
      for (unsigned int i = 0; i < op->getNumOperands(); i++) {
        auto def = op->getOperand(i)->getDefiningOp();
        if (isa<tpu::LoadWeightOp>(def))
          continue;
        useOps.push_back(def);
      }
      defOps.push_back(op);
    }
    std::sort(useOps.begin(), useOps.end());
    useOps.erase(unique(useOps.begin(), useOps.end()), useOps.end());
    for (auto defIt = defOps.begin(); defIt != defOps.end();) {
      auto useIt = find(useOps.begin(), useOps.end(), *defIt);
      if (useIt != useOps.end()) {
        useOps.erase(useIt);
        defIt = defOps.erase(defIt);
      } else
        ++defIt;
    }

    std::string attrName = "name";
    std::vector<std::string> inputTensorName;
    std::vector<std::string> outputTensorName;
    for (auto &defOp : defOps) {
      auto name = defOp->getAttr(attrName).cast<StringAttr>().getValue();
      outputTensorName.push_back(name);
    }
    for (auto &useOp : useOps) {
      auto name = useOp->getAttr(attrName).cast<StringAttr>().getValue();
      inputTensorName.push_back(name);
    }
    tpuInputTensorName_.push_back(inputTensorName);
    tpuOutputTensorName_.push_back(outputTensorName);
  }

  std::vector<std::string> cpuInputTensorName;
  std::vector<std::string> cpuOutputTensorName;
  for (auto &fn : cpuFunc_) {
    if (fn == 0)
      continue;
    for (unsigned int i = 0; i < fn.getNumArguments(); i++) {
      auto name =
          fn.getArgAttr(i, "tpu.tensor_name").cast<StringAttr>().getValue();
      cpuInputTensorName.push_back(name);
    }
    fn.walk([&](Operation *op) {
      if (isa<ReturnOp>(op)) {
        for (size_t i = 0; i < op->getNumOperands(); i++) {
          int resultNo = -1;
          auto value = op->getOperand(i);
          auto defOp = value->getDefiningOp();
          if (OpResult *result = dyn_cast<OpResult>(value))
            resultNo = result->getResultNumber();
          std::string attrName = "name";
          if (defOp->getNumResults() > 1) {
            std::string nameSuffix =
                std::string("_") + std::to_string(resultNo);
            attrName = "name" + nameSuffix;
          }
          auto name = defOp->getAttr(attrName).cast<StringAttr>().getValue();
          cpuOutputTensorName.push_back(name);
        }
      }
    });

    cpuInputTensorName_.push_back(cpuInputTensorName);
    cpuOutputTensorName_.push_back(cpuOutputTensorName);
    cpuInputTensorName.clear();
    cpuOutputTensorName.clear();
  }

  for (int i = 0; i < getTpuNumRoutine(); i++) {
    programInTensorName_.insert(programInTensorName_.end(),
                                tpuInputTensorName_[i].begin(),
                                tpuInputTensorName_[i].end());
    programOutTensorName_.insert(programOutTensorName_.end(),
                                 tpuOutputTensorName_[i].begin(),
                                 tpuOutputTensorName_[i].end());
  }

  for (int i = 0; i < getCpuNumRoutine(); i++) {
    programInTensorName_.insert(programInTensorName_.end(),
                                cpuInputTensorName_[i].begin(),
                                cpuInputTensorName_[i].end());
    programOutTensorName_.insert(programOutTensorName_.end(),
                                 cpuOutputTensorName_[i].begin(),
                                 cpuOutputTensorName_[i].end());
  }

  // input like use, output like def
  std::sort(programInTensorName_.begin(), programInTensorName_.end());
  programInTensorName_.erase(unique(programInTensorName_.begin(),
                                    programInTensorName_.end()),
                                    programInTensorName_.end());

  for (auto outIt = programOutTensorName_.begin();
            outIt != programOutTensorName_.end();) {
    auto inIt = find(programInTensorName_.begin(),
                     programInTensorName_.end(), *outIt);
    if (inIt != programInTensorName_.end()) {
      programInTensorName_.erase(inIt);
      outIt = programOutTensorName_.erase(outIt);
    } else {
      ++outIt;
    }
  }
}

void CviMlirParser::init(ModuleOp &module) {
  for (FuncOp fn : module.getOps<FuncOp>()) {
    std::string fnName = fn.getName();
    if (fnName.compare(0, 8, "tpu_func") == 0) {
      tpuFuncName_ = fnName;
      auto argType = fn.getArgument(0)->getType();
      auto argTensorType = argType.cast<RankedTensorType>();
      batchNum_ = argTensorType.getShape()[0];
    } else {
      cpuFunc_.push_back(fn);
    }
  }

  for (FuncOp fn : module.getOps<FuncOp>()) {
    std::string fnName = fn.getName();
    bool tpuFunc = true;
    if (fnName.compare(0, 8, "cpu_func") == 0) {
      tpuFunc = false;
    }

    int index = 0;
    std::vector<Operation *> tpuOps;
    fn.walk([&](Operation *op) {
      if (!tpuFunc) {
        insertTensorMap(op);
        return;
      }
      if (isa<tpu::InputOp>(op)) {
        insertTensorMap(op);
        return;
      }
      if (isa<tpu::LoadWeightOp>(op)) {
        insertWeightMap(op);
        return;
      }
      if (isa<tpu::WeightFileOp>(op)) {
        modelName_ = op->getAttr("filename").cast<StringAttr>().getValue().str();
        int pos = modelName_.find("_");
        if (pos != -1) {
          modelName_ = modelName_.substr(0, pos);
        }
        return;
      }
      if (isa<tpu::TG_CallOp>(op)) {
        if (!tpuOps.empty()) {
          if (tpuOps.size() == 1) {
            tpuOps.clear();
            return;
          }
          tpuOpsVec_.push_back(tpuOps);
          cpuFunc_.insert(cpuFunc_.begin() + index, nullptr);
          tpuOps.clear();
          index++;
        } else {
          index++;
        }
      } else {
        tpuOps.push_back(op);
      }
      insertTensorMap(op);
    });
  }
  collectedFuncInfo();
  weightBinFileName_ = clInputWeightBin;
  cpuLibPath_ = clCpuLibPath;
}

static std::string getCpuName(Operation *op) {
  if (isa<tpu::DetectionOutputOp>(op)) {
    return "detection_output";
  } else if (isa<tpu::SoftmaxOp>(op)){
    return "softmax";
  } else if (isa<tpu::QuantOp>(op)) {
    return "quant";
  } else if (isa<tpu::GenericCpuOp>(op)){
    auto cpuOp =
        op->getAttr("operation_name").cast<StringAttr>().getValue().str();
    std::string dialect = "tpu.";
    int pos = cpuOp.find(dialect);
    cpuOp = cpuOp.erase(pos, dialect.size());
    return cpuOp;
  } else {
    llvm_unreachable("unsupported cpu op");
  }
}

std::vector<uint8_t> CviCpuRoutine::cpuOpSerialize() {
  flatbuffers::FlatBufferBuilder builder(1024);
  flatbuffers::Offset<cvi::cpu_op::Attribute> attr;
  std::vector<flatbuffers::Offset<cvi::cpu_op::Attribute>> param;
  auto paramDictAttr = op_->getAttr("param").cast<DictionaryAttr>();
  for (auto &iter : paramDictAttr) {
    auto key = iter.first.data();
    auto flatKey = builder.CreateString(key);
    if (iter.second.isa<StringAttr>()) {
      auto value = iter.second.cast<StringAttr>().getValue();
      std::string strValue = std::string(value.data(), value.size());
      auto flatValue = builder.CreateString(strValue);
      auto strAttr = cvi::cpu_op::CreateStrAttr(builder, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(builder, 0, 0, 0, strAttr, 0, 0);
    } else if (iter.second.isa<IntegerAttr>()) {
      auto value = iter.second.cast<IntegerAttr>().getInt();
      auto intAttr = cvi::cpu_op::CreateIntAttr(builder, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(builder, 0, 0, intAttr, 0, 0, 0);
    } else if (iter.second.isa<FloatAttr>()) {
      auto value = iter.second.cast<FloatAttr>().getValueAsDouble();
      auto floatAttr = cvi::cpu_op::CreateFloatAttr(builder, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(builder, floatAttr, 0, 0, 0, 0, 0);
    } else if (iter.second.isa<BoolAttr>()) {
      auto value = iter.second.cast<BoolAttr>().getValue();
      auto boolAttr = cvi::cpu_op::CreateBoolAttr(builder, flatKey, value);
      attr = cvi::cpu_op::CreateAttribute(builder, 0, boolAttr, 0, 0, 0, 0);
    } else if (iter.second.isa<DenseFPElementsAttr>()) {
      std::vector<float> fpArray;
      auto value = iter.second.cast<DenseFPElementsAttr>();
      for (APFloat realVal : value) {
        fpArray.push_back(realVal.convertToFloat());
      }
      auto flatValue = builder.CreateVector(fpArray);
      auto fpArrayAttr =
          cvi::cpu_op::CreateFloatArrayAttr(builder, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(builder, 0, 0, 0, 0, fpArrayAttr, 0);
    } else if (iter.second.isa<DenseIntElementsAttr>()) {
      std::vector<int> intArray;
      auto value = iter.second.cast<DenseIntElementsAttr>();
      for (APInt intVal : value) {
        intArray.push_back(intVal.getZExtValue());
      }
      auto flatValue = builder.CreateVector(intArray);
      auto intArrayAttr =
          cvi::cpu_op::CreateIntArrayAttr(builder, flatKey, flatValue);
      attr = cvi::cpu_op::CreateAttribute(builder, 0, 0, 0, 0, 0, intArrayAttr);
    } else {
      llvm_unreachable("unsupported type");
    }
    param.push_back(attr);
  }

  auto flatParam = cvi::cpu_op::CreateParameterDirect(builder, &param);
  builder.Finish(flatParam);

  std::vector<uint8_t> serializeData;
  uint8_t* data = builder.GetBufferPointer();
  for (unsigned int i = 0; i < builder.GetSize(); i++) {
    serializeData.push_back(*data);
    data++;
  }
  return serializeData;
}

void CviCpuRoutine::setInputOutputNames() {
  std::string outputName = op_->getAttr("name").cast<StringAttr>().getValue();
  // get function from op
  auto block = op_->getBlock();
  while (!llvm::isa<FuncOp>(block->getParentOp())) {
    block = block->getParentOp()->getBlock();
  }

  const FuncOp &fnOp = cast<FuncOp>(*(block->getParentOp()));
  for (unsigned int i = 0; i < op_->getNumOperands(); i++) {
    auto opd = op_->getOperand(i);
    auto opdDef = opd->getDefiningOp();
    std::string inputName;

    // It comes from the input of the function
    if (opdDef == nullptr) {
      for (unsigned int argIndex = 0; argIndex < block->getNumArguments();
           argIndex++) {
        auto arg = block->getArgument(argIndex);
        if (arg == opd) {
          auto fn = const_cast<FuncOp*>(&fnOp);
          auto nameAttr = fn->getArgAttr(argIndex, "tpu.tensor_name");
          inputName = nameAttr.cast<StringAttr>().getValue();
          break;
        }
      }
    }

    while ((opdDef != nullptr) && isa<tpu::ReshapeOp>(opdDef)) {
      auto opdReshape = opdDef->getOperand(0);
      opdDef = opdReshape->getDefiningOp();
      if (opdDef == nullptr) {
        for (unsigned int argIndex = 0; argIndex < block->getNumArguments();
             argIndex++) {
          auto arg = block->getArgument(argIndex);
          if (arg == opdReshape) {
            auto fn = const_cast<FuncOp*>(&fnOp);
            auto nameAttr = fn->getArgAttr(argIndex, "tpu.tensor_name");
            inputName = nameAttr.cast<StringAttr>().getValue();
            break;
          }
        }
      }
    }

    if (opdDef) {
      inputName = opdDef->getAttr("name").cast<StringAttr>().getValue();
    }

    inputTensorNames_.push_back(inputName);
  }
  outputTensorNames_.push_back(outputName);
}

flatbuffers::Offset<Routine> CviCpuRoutine::buildCpuRoutine() {
  setInputOutputNames();
  FlatStrVecOffset flatInputTensorNames;
  FlatStrVecOffset flatOutputTensorNames;
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatTensorNameVec;

  for (auto &tensorName : inputTensorNames_) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatInputTensorNames = flatBuilder_->CreateVector(flatTensorNameVec);

  flatTensorNameVec.clear();
  for (auto &tensorName : outputTensorNames_) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatOutputTensorNames = flatBuilder_->CreateVector(flatTensorNameVec);
  auto funcName = getCpuName(op_);
  std::vector<uint8_t> serializeData = cpuOpSerialize();
  auto flatCpuRoutine = CreateCpuRoutineDirect(*flatBuilder_, funcName.data(),
                                               &serializeData);
  return CreateRoutine(*flatBuilder_, RoutineType_CPU, flatInputTensorNames,
                       flatOutputTensorNames, 0, flatCpuRoutine);

}

flatbuffers::Offset<Routine> CviTpuRoutine::buildTpuRoutine() {
  FlatStrVecOffset flatInputTensorNames;
  FlatStrVecOffset flatOutputTensorNames;

  std::vector<std::string> inputTensorName =
      parser_->getTpuInputTensorName(tpuIndex_);
  std::vector<flatbuffers::Offset<flatbuffers::String>> flatTensorNameVec;

  for (auto &tensorName : inputTensorName) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatInputTensorNames = flatBuilder_->CreateVector(flatTensorNameVec);
  flatTensorNameVec.clear();

  std::vector<std::string> outputTensorName =
      parser_->getTpuOutputTensorName(tpuIndex_);

  for (auto &tensorName : outputTensorName) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatOutputTensorNames = flatBuilder_->CreateVector(flatTensorNameVec);
  auto funcName = parser_->getTpuFuncName();
  auto flatTpuRoutine = CreateTpuRoutineDirect(*flatBuilder_, funcName.data());
  return CreateRoutine(*flatBuilder_, RoutineType_TPU, flatInputTensorNames,
                       flatOutputTensorNames, flatTpuRoutine, 0);
}

void CviProgram::splitCpuRoutines(FuncOp &fn) {
  fn.walk([&](Operation *op) {
    if (isa<tpu::ReshapeOp>(op) || isa<ReturnOp>(op) || llvm::isa<FuncOp>(op)) {
      return;
    }
    CviCpuRoutine cpuRoutine(flatBuilder_, parser_, op);
    auto flatCpuRoutine = cpuRoutine.buildCpuRoutine();
    routines_.push_back(flatCpuRoutine);
  });
}

void CviProgram::buildRoutines() {
  int tpuIndex = 0;
  std::vector<FuncOp> funcOp = parser_->getCpuFunc();
  for (unsigned int i = 0; i < funcOp.size(); i++) {
    if (funcOp[i] == 0) {
      CviTpuRoutine tpuRoutine(flatBuilder_, parser_, tpuIndex);
      auto flatTpuRoutine = tpuRoutine.buildTpuRoutine();
      routines_.push_back(flatTpuRoutine);
      tpuIndex++;
    } else {
      splitCpuRoutines(funcOp[i]);
    }
  }
}

void CviProgram::buildNeuronMap(FlatTensorVecOffset &flatTensorMap,
                                long &allocatedGmem) {
  size_t tensorIndex = 1;
  long maxNeuronSize = 0;
  std::vector<flatbuffers::Offset<Tensor>> tensorVec;
  std::vector<std::pair<std::string, CviTensor>> tensorPairs =
      parser_->getTensorPairs();

  for (auto &tensorPair : tensorPairs) {
    auto tensorName = flatBuilder_->CreateString(tensorPair.first);
    auto tensor = tensorPair.second;
    auto gaddr = tensor.gaddr;
    std::vector<long> shape;
    std::vector<long> stride;

    size_t tensorSize = 1;
    for (int i = 0; i < 4; i++) {
      tensorSize *= tensor.shape[i];
      shape.push_back(tensor.shape[i]);
    }
    if (tensorPair.second.gaddr == -1)
      tensorSize = 0;

    auto byteWidths = getDTypeByteWidths(tensor.dtype);
    maxNeuronSize += tensorSize * byteWidths;
    if (gaddr > allocatedGmem)
      allocatedGmem = gaddr + tensorSize * byteWidths;

    for (int i = 0; i < 4; i++)
      stride.push_back(tensor.stride[i]);

    auto flatTensorType = tensor.dtype;
    auto flatQuantType = tensor.quant.type;
    auto flatShapeVec = flatBuilder_->CreateVector(shape);
    auto flatStrideVec = flatBuilder_->CreateVector(stride);
    auto flatShape = CreateShape(*flatBuilder_, flatShapeVec);
    auto flatStride = CreateShape(*flatBuilder_, flatStrideVec);
    auto flatQuantInfo = CreateQuantInfo(*flatBuilder_, flatQuantType,
                                tensor.quant.max_value, tensor.quant.min_value,
                                tensor.quant.zero_point, tensor.quant.qscale);
    auto flatTensor = CreateTensor(*flatBuilder_, tensorIndex, tensorName,
                                   tensor.gaddr, flatTensorType, flatShape,
                                   flatStride, flatQuantInfo, tensor.overWrote);

    tensorVec.push_back(flatTensor);
    tensorIndex++;
    allocatedGmem = gaddr + tensorSize * byteWidths;
  }
  flatTensorMap = flatBuilder_->CreateVector(tensorVec);
}

void CviProgram::buildInputsOutputs(FlatStrVecOffset &flatInputTensors,
                                    FlatStrVecOffset &flatOutputTensors) {
  std::vector<std::string> inputTensorName = parser_->getProgramInTensorName();
  std::vector<std::string> outputTensorName =
      parser_->getProgramOutTensorName();

  std::vector<flatbuffers::Offset<flatbuffers::String>> flatTensorNameVec;
  for (auto &tensorName : inputTensorName) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatInputTensors = flatBuilder_->CreateVector(flatTensorNameVec);
  flatTensorNameVec.clear();

  for (auto &tensorName : outputTensorName) {
    auto flatTensorName = flatBuilder_->CreateString(tensorName);
    flatTensorNameVec.push_back(flatTensorName);
  }
  flatOutputTensors = flatBuilder_->CreateVector(flatTensorNameVec);
}

flatbuffers::Offset<Program> CviProgram::build(){
  FlatStrVecOffset flatInputTensors;
  FlatStrVecOffset flatOutputTensors;
  FlatTensorVecOffset flatTensorMap;
  size_t batchNum = parser_->getBatchNum();
  long allocatedGmem = 0;

  buildNeuronMap(flatTensorMap, allocatedGmem);
  buildInputsOutputs(flatInputTensors, flatOutputTensors);
  buildRoutines();

  auto flatRoutines = flatBuilder_->CreateVector(routines_);
  return CreateProgram(*flatBuilder_, batchNum, allocatedGmem, flatInputTensors,
                       flatOutputTensors, flatTensorMap, flatRoutines);
}

CviModel::FlatSectionVecOffset CviModel::buildSections() {
  // build weight section
  std::string errorMessage;
  std::vector<flatbuffers::Offset<Section>> sectionVec;
  std::string fileName = parser_->getWeightBinFileName();
  auto weightBinFile = openInputFile(fileName, &errorMessage);
  size_t binBufOffset = 0;
  auto size = weightBinFile->getBufferSize();
  auto flatName = flatBuilder_->CreateString("weight");
  auto weightSection = CreateSection(*flatBuilder_, SectionType_WEIGHT,
                                     flatName, size, binBufOffset);
  sectionVec.push_back(weightSection);

  // build tpu cmdbuf section
  binBufOffset += size;
  auto funcName = parser_->getTpuFuncName();
  size = parser_->getCmdBuf().size();
  flatName = flatBuilder_->CreateString(funcName);
  auto tpuSection = CreateSection(*flatBuilder_, SectionType_CMDBUF, flatName,
                                  size, binBufOffset);
  sectionVec.push_back(tpuSection);

  // build cpu section
  binBufOffset += size;
  std::vector<FuncOp> funcOp = parser_->getCpuFunc();
  for (auto &fn : funcOp) {
    if (fn == 0)
      continue;
    fn.walk([&](Operation *op) {
      if (isa<tpu::ReshapeOp>(op) || isa<ReturnOp>(op) || llvm::isa<FuncOp>(op))
        return;
      long offset = binBufOffset;
      auto opName = getCpuName(op);
      auto cpuLibName = opName + std::string(".so");
      auto type = SectionType_FUNC_X86;
      auto cpuLibPath = parser_->getCpuLibPath();
      if (findCpuLib(cpuLibPath, cpuLibName)) {
        auto cpuLibFile = openInputFile(cpuLibName, &errorMessage);
        size = cpuLibFile->getBufferSize();
        binBufOffset += size;
      } else {
        size = 0;
        offset = 0;
      }
      auto flatName = flatBuilder_->CreateString(opName);
      auto cpuSection = CreateSection(*flatBuilder_, type, flatName,
                                      size, offset);
      sectionVec.push_back(cpuSection);
    });
  }
  return flatBuilder_->CreateVector(sectionVec);
}

flatbuffers::Offset<Model> CviModel::build() {
  std::stringstream ssTime;
  auto clockNow = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(clockNow);
  ssTime << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S");
  std::string strTime = ssTime.str();
  std::string modelName = parser_->getModelName();
  Version modelVersion = Version(MAJOR_VER, MIN_VER, SUBMIN_VER);
  auto flatModelName = flatBuilder_->CreateString(modelName);
  auto flatTime = flatBuilder_->CreateString(strTime);
  FlatWeightVecOffset flatWeightMap;

  buildWeightMap(flatWeightMap);
  auto flatSections = buildSections();
  auto flatProgram = program_.build();
  std::vector<flatbuffers::Offset<Program>> programVec;
  programVec.push_back(flatProgram);
  auto flatProgramVec = flatBuilder_->CreateVector(programVec);
  return CreateModel(*flatBuilder_, &modelVersion, flatModelName, flatTime, 0,
                     0, flatWeightMap, flatProgramVec, flatSections);
}

void CviModel::buildWeightMap(FlatWeightVecOffset &flatWeightMap) {
  std::vector<flatbuffers::Offset<Weight> > flatWeightVec;
  std::string tensorName;
  std::map<std::string, CviWeight> weightMap = parser_->getWeightMap();

  for (auto &weightPair : weightMap) {
    size_t size = 1;
    size_t offset = 0;
    std::vector<long> shape;
    auto weightName = weightPair.first;
    offset = weightPair.second.offset;

    for (int i = 0; i < 4; i++) {
      int value = weightPair.second.shape[i];
      size *= value;
      shape.push_back(value);
    }

    auto flatShape = CreateShapeDirect(*flatBuilder_, &shape);
    auto dtype = weightPair.second.dtype;
    auto flatWeightPair = CreateWeightDirect(*flatBuilder_, weightName.data(),
                                  offset, size, flatShape, dtype);
    flatWeightVec.push_back(flatWeightPair);
  }
  flatWeightMap = flatBuilder_->CreateVector(flatWeightVec);
}

void CviModel::dataEncrypt(std::vector<uint8_t> &totalBin, uint8_t *resData) {
  MD5_CTX ctx;
  MD5_Init(&ctx);
  MD5_Update(&ctx, totalBin.data(), totalBin.size());
  MD5_Final(resData, &ctx);
}

void CviModel::storeModel(llvm::raw_ostream &output) {
  std::string fileName = parser_->getWeightBinFileName();
  std::string errorMessage;
  auto weightBinFile = openInputFile(fileName, &errorMessage);
  auto cmdbuf = parser_->getCmdBuf();

  flatbuffers::Offset<Model> flatModel = build();
  flatBuilder_->Finish(flatModel);
  size_t modelSize = flatBuilder_->GetSize();
  auto count = modelSize + weightBinFile->getBufferSize() + cmdbuf.size();
  std::vector<uint8_t> totalBin;
  totalBin.reserve(count);

  uint8_t *flatModelData = flatBuilder_->GetBufferPointer();
  for (unsigned i = 0; i < modelSize; i++) {
    auto elem = *(flatModelData + i);
    totalBin.push_back(elem);
  }
  auto bufferStart = weightBinFile->getBufferStart();
  for (unsigned int i = 0; i < weightBinFile->getBufferSize(); i++) {
    auto elem = *(bufferStart + i);
    totalBin.push_back(elem);
  }
  totalBin.insert(totalBin.end(), cmdbuf.begin(), cmdbuf.end());
  CviModelHeader header;
  dataEncrypt(totalBin, (uint8_t*)header.md5);
  std::string magic = u8"CviModel";
  std::string padding = u8"AA";
  strncpy(header.magic, magic.c_str(), 8);
  strncpy(header.padding, padding.c_str(), 2);
  header.body_size = modelSize;
  header.major = MAJOR_VER;
  header.minor = MIN_VER;

  output.write(reinterpret_cast<char *>(&header), sizeof(CviModelHeader));
  output.write(reinterpret_cast<char *>(totalBin.data()), totalBin.size());
}
