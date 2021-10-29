
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Interpreter/cpu/argmax.hpp"
#include "tpuc/Interpreter/cpu/batchnorm.hpp"
#include "tpuc/Interpreter/cpu/broadcast.hpp"
#include "tpuc/Interpreter/cpu/clip.hpp"
#include "tpuc/Interpreter/cpu/concat.hpp"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/Interpreter/cpu/conv3d.hpp"
#include "tpuc/Interpreter/cpu/convfc.hpp"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Interpreter/cpu/csc.hpp"
#include "tpuc/Interpreter/cpu/customop.hpp"
#include "tpuc/Interpreter/cpu/deconv.hpp"
#include "tpuc/Interpreter/cpu/depthtospace.hpp"
#include "tpuc/Interpreter/cpu/detection_output.hpp"
#include "tpuc/Interpreter/cpu/dilate.hpp"
#include "tpuc/Interpreter/cpu/eltwise.hpp"
#include "tpuc/Interpreter/cpu/fullyconnected.hpp"
#include "tpuc/Interpreter/cpu/instancenorm.hpp"
#include "tpuc/Interpreter/cpu/interpolation.hpp"
#include "tpuc/Interpreter/cpu/input.hpp"
#include "tpuc/Interpreter/cpu/gru.hpp"
#include "tpuc/Interpreter/cpu/lstm.hpp"
#include "tpuc/Interpreter/cpu/lrn.hpp"
#include "tpuc/Interpreter/cpu/matmul.hpp"
#include "tpuc/Interpreter/cpu/normalize.hpp"
#include "tpuc/Interpreter/cpu/layernorm.hpp"
#include "tpuc/Interpreter/cpu/pad.hpp"
#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/Interpreter/cpu/pooling.hpp"
#include "tpuc/Interpreter/cpu/pool_mask.hpp"
#include "tpuc/Interpreter/cpu/priorbox.hpp"
#include "tpuc/Interpreter/cpu/proposal.hpp"
#include "tpuc/Interpreter/cpu/quadraticSum.hpp"
#include "tpuc/Interpreter/cpu/quant.hpp"
#include "tpuc/Interpreter/cpu/reflectionpad.hpp"
#include "tpuc/Interpreter/cpu/reduce.hpp"
#include "tpuc/Interpreter/cpu/reflectionpad.hpp"
#include "tpuc/Interpreter/cpu/reorg.hpp"
#include "tpuc/Interpreter/cpu/reverse.hpp"
#include "tpuc/Interpreter/cpu/roi_pooling.hpp"
#include "tpuc/Interpreter/cpu/scale.hpp"
#include "tpuc/Interpreter/cpu/scale_lut.hpp"
#include "tpuc/Interpreter/cpu/shuffle_channel.hpp"
#include "tpuc/Interpreter/cpu/slice.hpp"
#include "tpuc/Interpreter/cpu/softmax.hpp"
#include "tpuc/Interpreter/cpu/std.hpp"
#include "tpuc/Interpreter/cpu/swap_channel.hpp"
#include "tpuc/Interpreter/cpu/tile.hpp"
#include "tpuc/Interpreter/cpu/upsample.hpp"
#include "tpuc/Interpreter/cpu/embedding.hpp"
#include "tpuc/Interpreter/cpu/matmul.hpp"
#include "tpuc/Interpreter/cpu/zero_mask.hpp"
#include "tpuc/Interpreter/cpukernel.h"

#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>

namespace mlir {

std::string MlirModuleInterpreter::chipname = "cv183x";
void MlirModuleInterpreter::loadModule(OwningModuleRef &module_op, std::string target_op) {
  for (auto func : module_op->getOps<FuncOp>()) {
    MInfo Machineinfo;
    if (func->getAttr("chipname")) {
      chipname = func->getAttr("chipname").cast<StringAttr>().getValue().str();
      Machineinfo.getChipInfo(func);
    }
    updateKernelList(func, target_op);
    break;
  }
}

bool MlirModuleInterpreter::isKernelDirty(std::shared_ptr<CPUOpKernel> &krnl,
                                          Operation *op) {
  // always return false if op is inputOp.
  if (isa<tpu::InputOp>(op)) {
    return false;
  }
  if (krnl->dirty) {
    return true;
  }
  // check if signature updated.
  auto new_signature = CPUOpKernel::generateSignature(*op);
  bool dirty = (krnl->signature != new_signature) ? true : false;
  if (!dirty) {
    // check if all opd are all not dirty
    for (auto opd : op->getOperands()) {
      auto parentOp = opd.getDefiningOp();
      if (!parentOp || isa<tpu::NoneOp>(parentOp)) {
        continue;
      }
      auto it = kernel_map_.find(getOpName(parentOp).str());
      if (it == kernel_map_.end()) {
        continue;
      }
      auto &opd_krnl = it->second;
      if (opd_krnl->dirty) {
        dirty = true;
        break;
      }
    }
  }
  return dirty;
}

void MlirModuleInterpreter::updateKernelList(FuncOp &func, std::string &target_op) {
  kernel_list_.clear();
  valueMapping.clear();
  bool completed = false;
  func.walk([&](Operation *op) {
    if (completed) {
      return;
    }
    if (isa<tpu::WeightFileOp>(op) ||
        isa<tpu::NoneOp>(op)) {
      return;
    } else if (isa<ReturnOp>(op)) {
      for (auto opd : op->getOperands()) {
        output_details.push_back(getOpName(opd.getDefiningOp()).str());
      }
      return;
    } else if (isa<tpu::RetinaFaceDetectionOp>(op)) {
      llvm::errs() << "no support " << op->getName().getStringRef()
                  << " op in interpreter_v2\n";
      return;
    } else if (op->getName().getDialect()->getNamespace() != "tpu") {
      return;
    } else if (isa<tpu::LoadWeightOp>(op)) {
      return;
    }

    auto name = getOpName(op).str();
    if (!target_op.empty() && target_op == name) {
      completed = true;
    }

    auto it = kernel_map_.find(name);
    if (it != kernel_map_.end()) {
      auto &krnl = it->second;
      bool dirty = isKernelDirty(krnl, op);
      if (!dirty) {
        krnl->dirty = false;
        krnl->op = op;
        valueMapping[op->getResult(0)] = activationMapping[name];
        kernel_list_.emplace_back(krnl);
        // llvm::errs() << name << " is clean\n";
        return;
      } else {
        activationMapping.erase(name);
      }
    }

    std::shared_ptr<CPUOpKernel> krnl;
    if (isa<tpu::Conv2DOp>(op)) {
      krnl = std::make_shared<Conv2DOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EltwiseAddOp>(op)) {
      krnl = std::make_shared<EltwiseAddOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EluOp>(op)) {
      krnl = std::make_shared<EluOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::QuantOp>(op)) {
      krnl = std::make_shared<QuantOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PoolAvg2DOp>(op) || isa<tpu::PoolMax2DOp>(op)) {
      krnl = std::make_shared<PoolingOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ScaleOp>(op)) {
      krnl = std::make_shared<ScaleOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ScaleLutOp>(op)) {
      krnl = std::make_shared<ScaleLutOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::InputOp>(op)) {
      krnl = std::make_shared<InputOpKernel>(*op, valueMapping, weightMapping, input_details);
    } else if (isa<tpu::SoftmaxOp>(op)) {
      krnl = std::make_shared<SoftmaxOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SigmoidOp>(op)) {
      krnl = std::make_shared<SigmoidOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SwishOp>(op)) {
      krnl = std::make_shared<SwishOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SliceOp>(op)) {
      krnl = std::make_shared<SliceOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ConcatOp>(op)) {
      krnl = std::make_shared<ConcatOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ConvFcOp>(op)) {
      krnl = std::make_shared<ConvFcOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::DeConv2DOp>(op)) {
      krnl = std::make_shared<DeConv2DOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::AbsOp>(op)) {
      krnl = std::make_shared<AbsOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ArgMaxOp>(op)) {
      krnl = std::make_shared<ArgMaxOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::BatchNormOp>(op)) {
      krnl = std::make_shared<BatchNormOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::BroadcastAddOp>(op)) {
      krnl = std::make_shared<BroadcastAddOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::BroadcastMulOp>(op)) {
      krnl = std::make_shared<BroadcastMulOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::BroadcastSubOp>(op)) {
      krnl = std::make_shared<BroadcastSubOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ClipOp>(op)) {
      krnl = std::make_shared<ClipOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::Conv3DOp>(op)) {
      krnl = std::make_shared<Conv3DOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::CropOp>(op)) {
      krnl = std::make_shared<CropOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::CscOp>(op)) {
      krnl = std::make_shared<CscOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::CustomOp>(op)) {
      krnl = std::make_shared<CustomOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::DetectionOutputOp>(op)) {
      krnl = std::make_shared<DetectionOutputOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::DilateOp>(op)) {
      krnl = std::make_shared<DilateOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EltwiseMaxOp>(op)) {
      krnl = std::make_shared<EltwiseMaxOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EltwiseMinOp>(op)) {
      krnl = std::make_shared<EltwiseMinOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EltwiseMulOp>(op)) {
      krnl = std::make_shared<EltwiseMulOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ExpOp>(op)) {
      krnl = std::make_shared<ExpOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::FrcnDetectionOp>(op)) {
      krnl = std::make_shared<FrcnDetectionOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::FullyConnectedOp>(op)) {
      krnl = std::make_shared<FullyConnectedOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::GruOp>(op)) {
      krnl = std::make_shared<GruOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::InstanceNormOp>(op)) {
      krnl = std::make_shared<InstanceNormOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::InterpOp>(op)) {
      krnl = std::make_shared<InterpolationOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LayerNormOp>(op)) {
      krnl = std::make_shared<LayerNormOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LeakyReluOp>(op)) {
      krnl = std::make_shared<LeakyReluOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LogOp>(op)) {
      krnl = std::make_shared<LogOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LrnOp>(op)) {
      krnl = std::make_shared<LrnOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LrnOneOp>(op)) {
      krnl = std::make_shared<LrnOneOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LrnTwoOp>(op)) {
      krnl = std::make_shared<LrnTwoOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LrnThreeOp>(op)) {
      krnl = std::make_shared<LrnThreeOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::LstmOp>(op)) {
      krnl = std::make_shared<LstmOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::MatMulOp>(op)) {
      krnl = std::make_shared<MatMulOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::MishOp>(op)) {
      krnl = std::make_shared<MishOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::MatMulOp>(op)) {
      krnl = std::make_shared<MatMulOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::NormalizeOp>(op)) {
      krnl = std::make_shared<NormalizeOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PoolMaskOp>(op)) {
      krnl = std::make_shared<PoolMaskOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PadOp>(op)) {
      krnl = std::make_shared<PadOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PermuteOp>(op)) {
      krnl = std::make_shared<PermuteOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PixelShuffleOp>(op)) {
      krnl = std::make_shared<DepthToSpaceOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PReluOp>(op)) {
      krnl = std::make_shared<PReluOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::PriorBoxOp>(op)) {
      krnl = std::make_shared<PriorBoxOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ProposalOp>(op)) {
      krnl = std::make_shared<ProposalOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::QuadraticSumOp>(op)) {
      krnl = std::make_shared<QuadraticSumOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReduceL2Op>(op)) {
      krnl = std::make_shared<ReduceL2OpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReduceMaxOp>(op)) {
      krnl = std::make_shared<ReduceMaxOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReduceMinOp>(op)) {
      krnl = std::make_shared<ReduceMinOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReduceSumOp>(op)) {
      krnl = std::make_shared<ReduceSumOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReduceMeanOp>(op)) {
      krnl = std::make_shared<ReduceMeanOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReflectionPadOp>(op)) {
      krnl = std::make_shared<ReflectionPadOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReluOp>(op)) {
      krnl = std::make_shared<ReluOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReciprocalOp>(op)) {
      krnl = std::make_shared<ReciprocalOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReorgOp>(op)) {
      krnl = std::make_shared<ReorgOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ReverseOp>(op)) {
      krnl = std::make_shared<ReverseOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ROIPoolingOp>(op)) {
      krnl = std::make_shared<ROIPoolingOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SqrtOp>(op)) {
      krnl = std::make_shared<SqrtOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SquareOp>(op)) {
      krnl = std::make_shared<SquareOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ShuffleChannelOp>(op)) {
      krnl = std::make_shared<ShuffleChannelOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SoftmaxCpuOp>(op)) {
      krnl = std::make_shared<SoftmaxOpKernel>(*op, valueMapping, weightMapping, true);
    } else if (isa<tpu::SoftPlusOp>(op)) {
      krnl = std::make_shared<SoftPlusOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::SwapChannelOp>(op)) {
      krnl = std::make_shared<SwapChannelOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::StdOp>(op)) {
      krnl = std::make_shared<StdOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::TanHOp>(op)) {
      krnl = std::make_shared<TanHOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::TileOp>(op)) {
      krnl = std::make_shared<TileOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::UpsampleOp>(op)) {
      krnl = std::make_shared<UpsampleOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::YoloDetectionOp>(op)) {
      krnl = std::make_shared<YoloDetectionOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::EmbeddingOp>(op)) {
      krnl = std::make_shared<EmbeddingOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::ZeroMaskOp>(op)) {
      krnl = std::make_shared<ZeroMaskOpKernel>(*op, valueMapping, weightMapping);
    } else if (isa<tpu::RetinaFaceDetectionOp>(op)) {
      llvm::errs() << "no support " << op->getName().getStringRef()
                  << " op in interpreter_v2\n";
    } else if (isa<tpu::ReshapeOp>(op)) {
      auto resTensor = valueMapping[op->getOperand(0)];
      activationMapping[name] = resTensor;
      valueMapping[op->getResult(0)] = resTensor;
      auto opd_name = getOpName(op->getOperand(0).getDefiningOp()).str();
      auto krnl = kernel_map_[opd_name];
      kernel_map_[name] = krnl;
      return;
    } else {
      std::stringstream err_msg;
      llvm::errs() << "no support " << op->getName().getStringRef()
                  << " op in interpreter_v2\n";
      llvm_unreachable("TODO");
    }
    auto resTensor = valueMapping[op->getResult(0)];
    activationMapping[name] = resTensor;
    kernel_map_[name] = krnl;
    kernel_list_.emplace_back(krnl);
  });
}

weight_map_t MlirModuleInterpreter::weightMapping;
void MlirModuleInterpreter::updateWeightMap(OwningModuleRef &module_op) {
  for (auto func : module_op->getOps<FuncOp>()) {
    TensorFile *wfile = nullptr;
    func.walk([&](tpu::WeightFileOp castOp) {
      wfile = castOp.get();
    });
    func.walk([&](tpu::LoadWeightOp castOp) {
      auto result = castOp.getResult();
      auto name = getOpName(castOp.getOperation()).str();
      auto type = result.getType().cast<TensorType>();
      assert(type.getElementType().isF32());
      auto array = wfile->readTensor<float>(name, type);
      weightMapping[castOp.getResult()] = std::make_shared<TensorData>(std::move(array));
    });
    break;
  }
}

void MlirModuleInterpreter::invokeTo(std::string name, int32_t bidx) {
  for (auto &kv : valueMapping) {
    kv.second->set_batch_idx(bidx);
  }
  for (auto &krnl : kernel_list_) {
    krnl->invoke();
    if (!name.empty() && name == krnl->name) {
      break;
    }
  }
}

void MlirModuleInterpreter::fastInvokeAllBatch(std::string name, int32_t batch) {
  bool last_batch = false;
  for (int i = 0; i < batch; i++) {
    if (i == batch - 1) {
      last_batch = true;
    }
    for (auto &kv : valueMapping) {
      kv.second->set_batch_idx(i);
    }
    for (auto &krnl : kernel_list_) {
      if (krnl->dirty) {
        // llvm::errs() << " dirty:" << krnl->name << "\n";
        krnl->invoke();
        if (last_batch) { // last batch, change the dirty flag
          krnl->dirty = false;
        }
      }
      if (!name.empty() && name == krnl->name) {
        break;
      }
    }
  }
}

void MlirModuleInterpreter::setTensor(std::string &name,
                                       const std::vector<float> &data,
                                       int32_t bidx) {
  auto it = activationMapping.find(name);
  if (it == activationMapping.end()) {
    llvm::errs() << " Not Find Op name: " << name << " tensor \n";
    llvm_unreachable("Error");
  }
  auto &activation = it->second;
  activation->assign(data.begin(), data.end(), bidx);
}

std::shared_ptr<std::vector<float>>
MlirModuleInterpreter::getTensor(std::string &name, int32_t bidx) {
  auto it = activationMapping.find(name);
  if (it == activationMapping.end()) {
    llvm::errs() << " Not Find Op name: " << name << " tensor \n";
    llvm_unreachable("Error");
  }
  return it->second->tensor(bidx);
}

std::vector<int64_t>
MlirModuleInterpreter::getTensorShape(std::string &name) {
  auto it = kernel_map_.find(name);
  if (it == kernel_map_.end()) {
    llvm::errs() << " Not Find Op name: " << name << " tensor \n";
    llvm_unreachable("Error");
  }
  return it->second->get_shape();
}

std::string
MlirModuleInterpreter::getDataType(std::string &name) {
  auto it = kernel_map_.find(name);
  if (it == kernel_map_.end()) {
    llvm::errs() << " Not Find Op name: " << name << " tensor \n";
    llvm_unreachable("Error");
  }
  return it->second->get_data_type();
}

void MlirModuleInterpreter::dump(std::string &name) {
  for (auto &krnl : kernel_list_) {
    if (krnl->get_name() == name) {
      return krnl->dump();
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " tensor \n";
}

std::string MlirModuleInterpreter::customOpPluginFile_ = "";

} // namespace mlir
