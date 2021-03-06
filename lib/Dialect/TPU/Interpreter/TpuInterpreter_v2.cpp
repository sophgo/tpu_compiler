//===- TpuInterpreter.cpp - Implementation of TPU Op Interpreter ---------===//
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
// This file implements the TPU dialect Interpreter.
//
//===----------------------------------------------------------------------===//

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
#include "tpuc/Interpreter/cpu/copy.hpp"
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
#include "tpuc/Interpreter/cpu/reorg.hpp"
#include "tpuc/Interpreter/cpu/reverse.hpp"
#include "tpuc/Interpreter/cpu/roi_pooling.hpp"
#include "tpuc/Interpreter/cpu/scale.hpp"
#include "tpuc/Interpreter/cpu/scale_lut.hpp"
#include "tpuc/Interpreter/cpu/shuffle_channel.hpp"
#include "tpuc/Interpreter/cpu/softmax.hpp"
#include "tpuc/Interpreter/cpu/std.hpp"
#include "tpuc/Interpreter/cpu/swap_channel.hpp"
#include "tpuc/Interpreter/cpu/tile.hpp"
#include "tpuc/Interpreter/cpu/upsample.hpp"
#include "tpuc/Interpreter/cpu/embedding.hpp"
#include "tpuc/Interpreter/cpu/matchtemplate.hpp"
#include "tpuc/Interpreter/cpu/matmul.hpp"
#include "tpuc/Interpreter/cpu/zero_mask.hpp"
#include "tpuc/Interpreter/cpukernel.h"

#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>

namespace mlir {

void ModuleInterpreter::prepareOperation(Operation &op) {
  if (isa<tpu::InputOp>(op)) {
    auto input_kernel_op =
        std::make_unique<InputOpKernel>(op, valueMapping, weightMapping, input_details);
    oplist.push_back(std::move(input_kernel_op));
    return;
  }
  if (isa<ReturnOp>(op)) {
    // collect resultsList
    for (auto opd : op.getOperands()) {
      output_details.push_back(getOpName(opd.getDefiningOp()).str());
    }
    return;
  }
  if (isa<tpu::WeightFileOp>(op)) {
    auto weightFileOp = dyn_cast<tpu::WeightFileOp>(op);
    weightFile_ = weightFileOp.get();
    return;
  }
  if (isa<tpu::LoadWeightOp>(op)) {
    auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op);
    LLVM_DEBUG(llvm::errs() << "LoadWeightOp"
                            << "\n";);

    auto result = loadWeightOp.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result.getType().dump();
               llvm::errs() << "\n";);

    auto tensor_name = loadWeightOp.name().str();
    LLVM_DEBUG(llvm::errs() << "  tensor_name " << tensor_name << "\n";);

    auto type = result.getType().cast<TensorType>();
    std::unique_ptr<std::vector<float>> tensor = nullptr;
    if (type.getElementType().isF32()) {
      tensor = std::move(weightFile_->readTensor<float>(tensor_name, type));
    } else {
      llvm_unreachable("only support fp32 weight");
    }
    std::string weight_name = loadWeightOp.name().str();
    std::vector<float> weight_data(tensor->begin(), tensor->end());
    std::vector<int64_t> weight_shape(type.getShape().begin(),
                                      type.getShape().end());
    weightMapping[result] = std::make_shared<TensorData>(std::move(tensor));
    return;
  }
  if (isa<tpu::AbsOp>(op)) {
    auto abs_kernel_op = std::make_unique<AbsOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(abs_kernel_op));
    return;
  }
  if (isa<tpu::ArgMaxOp>(op)) {
    auto argmax_kernel_op = std::make_unique<ArgMaxOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(argmax_kernel_op));
    return;
  }
  if (isa<tpu::BatchNormOp>(op)) {
    auto bn_kernel_op = std::make_unique<BatchNormOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(bn_kernel_op));
    return;
  }
  if (isa<tpu::BroadcastAddOp>(op)) {
    auto broadcastadd_kernel_op =
        std::make_unique<BroadcastAddOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(broadcastadd_kernel_op));
    return;
  }
  if (isa<tpu::BroadcastMulOp>(op)) {
    auto broadcastmul_kernel_op =
        std::make_unique<BroadcastMulOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(broadcastmul_kernel_op));
    return;
  }
  if (isa<tpu::BroadcastSubOp>(op)) {
    auto broadcastsub_kernel_op =
        std::make_unique<BroadcastSubOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(broadcastsub_kernel_op));
    return;
  }
  if (isa<tpu::ClipOp>(op)) {
    auto clip_kernel_op = std::make_unique<ClipOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(clip_kernel_op));
    return;
  }
  if (isa<tpu::ConcatOp>(op)) {
    auto concat_kernel_op = std::make_unique<ConcatOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(concat_kernel_op));
    return;
  }
  if (isa<tpu::Conv2DOp>(op)) {
    auto conv_kernel_op = std::make_unique<Conv2DOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(conv_kernel_op));
    return;
  }
  if (isa<tpu::Conv3DOp>(op)) {
    auto conv_kernel_op = std::make_unique<Conv3DOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(conv_kernel_op));
    return;
  }
  if (isa<tpu::ConvFcOp>(op)) {
    auto new_op = std::make_unique<ConvFcOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(new_op));
    return;
  }
  if (isa<tpu::CropOp>(op)) {
    auto crop_kernel_op = std::make_unique<CropOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(crop_kernel_op));
    return;
  }
  if (isa<tpu::CopyOp>(op)) {
    auto cast_op = std::make_unique<CopyOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(cast_op));
    return;
  }
  if (isa<tpu::CscOp>(op)) {
    auto csc_kernel_op = std::make_unique<CscOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(csc_kernel_op));
    return;
  }
  if (isa<tpu::CustomOp>(op)) {
    auto custom_kernel_op = std::make_unique<CustomOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(custom_kernel_op));
    return;
  }
  if (isa<tpu::DeConv2DOp>(op)) {
    auto deconv_kernel_op =
        std::make_unique<DeConv2DOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(deconv_kernel_op));
    return;
  }
  if (isa<tpu::DetectionOutputOp>(op)) {
    auto do_kernel_op =
        std::make_unique<DetectionOutputOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(do_kernel_op));
    return;
  }
  if (isa<tpu::DilateOp>(op)) {
    auto d_kernel_op = std::make_unique<DilateOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(d_kernel_op));
    return;
  }
  if (isa<tpu::EltwiseAddOp>(op)) {
    auto elt_add_kernel_op =
        std::make_unique<EltwiseAddOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(elt_add_kernel_op));
    return;
  }
  if (isa<tpu::MulConstOp>(op)) {
    auto mul_const_kernel_op = std::make_unique<MulConstOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(mul_const_kernel_op));
    return;
  }
  if (isa<tpu::EltwiseMaxOp>(op)) {
    auto elt_max_kernel_op =
        std::make_unique<EltwiseMaxOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(elt_max_kernel_op));
    return;
  }
  if (isa<tpu::EltwiseMinOp>(op)) {
    auto elt_min_kernel_op =
        std::make_unique<EltwiseMinOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(elt_min_kernel_op));
    return;
  }
  if (isa<tpu::EltwiseMulOp>(op)) {
    auto elt_mul_kernel_op =
        std::make_unique<EltwiseMulOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(elt_mul_kernel_op));
    return;
  }
  if (isa<tpu::ExpOp>(op)) {
    auto exp_kernel_op = std::make_unique<ExpOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(exp_kernel_op));
    return;
  }
  if (isa<tpu::EluOp>(op)) {
    auto elu_kernel_op = std::make_unique<EluOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(elu_kernel_op));
    return;
  }
  if (isa<tpu::FrcnDetectionOp>(op)) {
    auto f_kernel_op =
        std::make_unique<FrcnDetectionOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(f_kernel_op));
    return;
  }
  if (isa<tpu::FullyConnectedOp>(op)) {
    auto fc_kernel_op =
        std::make_unique<FullyConnectedOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(fc_kernel_op));
    return;
  }
  if (isa<tpu::GruOp>(op)) {
    auto kernel_op = std::make_unique<GruOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(kernel_op));
    return;
  }
  if (isa<tpu::InstanceNormOp>(op)) {
    auto instanceOp = std::make_unique<InstanceNormOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(instanceOp));
    return;
  }
  if (isa<tpu::InterpOp>(op)) {
    auto i_kernel_op =
        std::make_unique<InterpolationOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(i_kernel_op));
    return;
  }
  if (isa<tpu::LayerNormOp>(op)) {
    auto kernel_op = std::make_unique<LayerNormOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(kernel_op));
    return;
  }
  if (isa<tpu::LeakyReluOp>(op)) {
    auto lr_kernel_op = std::make_unique<LeakyReluOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(lr_kernel_op));
    return;
  }
  if (isa<tpu::LogOp>(op)) {
    auto t_kernel_op = std::make_unique<LogOpKernel>(op, valueMapping);
    oplist.push_back(std::move(t_kernel_op));
    return;
  }
  if (isa<tpu::LrnOp>(op)) {
    auto lrn_kernel_op = std::make_unique<LrnOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(lrn_kernel_op));
    return;
  }
  if (isa<tpu::LrnOneOp>(op)) {
    auto lrn_kernel_op = std::make_unique<LrnOneOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(lrn_kernel_op));
    return;
  }
  if (isa<tpu::LrnTwoOp>(op)) {
    auto lrn_kernel_op = std::make_unique<LrnTwoOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(lrn_kernel_op));
    return;
  }
  if (isa<tpu::LrnThreeOp>(op)) {
    auto lrn_kernel_op = std::make_unique<LrnThreeOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(lrn_kernel_op));
    return;
  }
  if (isa<tpu::LstmOp>(op)) {
    auto kernel_op = std::make_unique<LstmOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(kernel_op));
    return;
  }
  if (isa<tpu::MatchTemplateOp>(op)) {
    auto match_t_kernel_op = std::make_unique<MatchTemplateOpOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(match_t_kernel_op));
    return;
  }
  if (isa<tpu::MatMulOp>(op)) {
    auto matmul_kernel_op = std::make_unique<MatMulOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(matmul_kernel_op));
    return;
  }
  if (isa<tpu::MishOp>(op)) {
    auto mish_kernel_op = std::make_unique<MishOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(mish_kernel_op));
    return;
  }
  if (isa<tpu::MatMulOp>(op)) {
    auto kernel_op = std::make_unique<MatMulOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(kernel_op));
    return;
  }
  if (isa<tpu::NoneOp>(op)) {
    return;
  }
  if (isa<tpu::NormalizeOp>(op)) {
    auto norm_kernel_op = std::make_unique<NormalizeOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(norm_kernel_op));
    return;
  }
  if (isa<tpu::PadOp>(op)) {
    auto pad_kernel_op = std::make_unique<PadOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(pad_kernel_op));
    return;
  }
  if (isa<tpu::PermuteOp>(op)) {
    auto permute_kernel_op =
        std::make_unique<PermuteOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(permute_kernel_op));
    return;
  }
  if (isa<tpu::PixelShuffleOp>(op)) {
    auto ps_kernel_op =
        std::make_unique<DepthToSpaceOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(ps_kernel_op));
    return;
  }
  if (isa<tpu::PoolAvg2DOp>(op) || isa<tpu::PoolMax2DOp>(op)) {
    auto pool_kernel_op = std::make_unique<PoolingOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(pool_kernel_op));
    return;
  }
  if (isa<tpu::PoolMaskOp>(op)) {
    auto kernel_op = std::make_unique<PoolMaskOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(kernel_op));
    return;
  }
  if (isa<tpu::PReluOp>(op)) {
    auto prelu_kernel_op = std::make_unique<PReluOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(prelu_kernel_op));
    return;
  }
  if (isa<tpu::PriorBoxOp>(op)) {
    auto priorbox_kernel_op =
        std::make_unique<PriorBoxOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(priorbox_kernel_op));
    return;
  }
  if (isa<tpu::ProposalOp>(op)) {
    auto p_kernel_op = std::make_unique<ProposalOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(p_kernel_op));
    return;
  }
  if (isa<tpu::QuadraticSumOp>(op)) {
    auto q_kernel_op = std::make_unique<QuadraticSumOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(q_kernel_op));
    return;
  }
  if (isa<tpu::QuantOp>(op)) {
    auto quant_kernel_op = std::make_unique<QuantOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(quant_kernel_op));
    return;
  }
  if (isa<tpu::ReflectionPadOp>(op)) {
    auto reflect_op = std::make_unique<ReflectionPadOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(reflect_op));
    return;
  }
  if (isa<tpu::ReduceL2Op>(op)) {
    auto r_kernel_op = std::make_unique<ReduceL2OpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReduceMaxOp>(op)) {
    auto r_kernel_op = std::make_unique<ReduceMaxOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReduceMinOp>(op)) {
    auto r_kernel_op = std::make_unique<ReduceMinOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReduceSumOp>(op)) {
    auto r_kernel_op = std::make_unique<ReduceSumOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReduceMeanOp>(op)) {
    auto r_kernel_op = std::make_unique<ReduceMeanOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReluOp>(op)) {
    auto relu_kernel_op = std::make_unique<ReluOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(relu_kernel_op));
    return;
  }
  if (isa<tpu::ReshapeOp>(op)) {
    auto reshape_kernel_op =
        std::make_unique<ReshapeOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(reshape_kernel_op));
    return;
  }
  if (isa<tpu::ReorgOp>(op)) {
    auto r_kernel_op = std::make_unique<ReorgOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ReverseOp>(op)) {
    auto r_kernel_op = std::make_unique<ReverseOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ROIPoolingOp>(op)) {
    auto r_kernel_op = std::make_unique<ROIPoolingOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(r_kernel_op));
    return;
  }
  if (isa<tpu::ScaleOp>(op)) {
    auto scale_kernel_op = std::make_unique<ScaleOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(scale_kernel_op));
    return;
  }
  if (isa<tpu::ScaleLutOp>(op)) {
    auto scale_lut_kernel_op =
        std::make_unique<ScaleLutOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(scale_lut_kernel_op));
    return;
  }
  if (isa<tpu::PowOp>(op)) {
    auto cast_op = std::make_unique<PowOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(cast_op));
    return;
  }
  if (isa<tpu::ShuffleChannelOp>(op)) {
    auto sc_kernel_op =
        std::make_unique<ShuffleChannelOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(sc_kernel_op));
    return;
  }
  if (isa<tpu::SigmoidOp>(op)) {
    auto sig_kernel_op = std::make_unique<SigmoidOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(sig_kernel_op));
    return;
  }
  if (isa<tpu::SwishOp>(op)) {
    auto sws_kernel_op = std::make_unique<SwishOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(sws_kernel_op));
    return;
  }
  if (isa<tpu::StdOp>(op)) {
    auto std_kernel_op = std::make_unique<StdOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(std_kernel_op));
    return;
  }
  if (isa<tpu::SoftmaxOp>(op)) {
    auto softmax_kernel_op =
        std::make_unique<SoftmaxOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(softmax_kernel_op));
    return;
  }
  if (isa<tpu::SoftmaxCpuOp>(op)) {
    auto softmax_kernel_op =
        std::make_unique<SoftmaxOpKernel>(op, valueMapping, weightMapping, true);
    oplist.push_back(std::move(softmax_kernel_op));
    return;
  }
  if (isa<tpu::SoftPlusOp>(op)) {
    auto s_kernel_op = std::make_unique<SoftPlusOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(s_kernel_op));
    return;
  }
  if (isa<tpu::SwapChannelOp>(op)) {
    auto sc_kernel_op = std::make_unique<SwapChannelOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(sc_kernel_op));
    return;
  }
  if (isa<tpu::TanHOp>(op)) {
    auto t_kernel_op = std::make_unique<TanHOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(t_kernel_op));
    return;
  }
  if (isa<tpu::TileOp>(op)) {
    auto t_kernel_op = std::make_unique<TileOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(t_kernel_op));
    return;
  }
  if (isa<tpu::UpsampleOp>(op)) {
    auto up_kernel_op = std::make_unique<UpsampleOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(up_kernel_op));
    return;
  }
  if (isa<tpu::YoloDetectionOp>(op)) {
    auto yo_kernel_op =
        std::make_unique<YoloDetectionOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(yo_kernel_op));
    return;
  }
  if (isa<tpu::ZeroMaskOp>(op)) {
    auto std_kernel_op = std::make_unique<ZeroMaskOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(std_kernel_op));
    return;
  }
  if (isa<tpu::EmbeddingOp>(op)) {
    auto embedding_kernel_op = std::make_unique<EmbeddingOpKernel>(op, valueMapping, weightMapping);
    oplist.push_back(std::move(embedding_kernel_op));
    return;
  }
  if (isa<tpu::RetinaFaceDetectionOp>(op)) {
    llvm::errs() << "no support " << op.getName().getStringRef()
                << " op in interpreter_v2\n";
    return;
  }
  std::stringstream err_msg;
  llvm::errs() << "no support " << op.getName().getStringRef()
               << " op in interpreter_v2\n";
  llvm_unreachable("TODO");
}

void ModuleInterpreter::invoke() {
  std::lock_guard<std::mutex> lock(invoke_lock);
  for (auto &node : oplist) {
    LLVM_DEBUG(node->dump());
    node->invoke();
  }
}
void ModuleInterpreter::invoke(std::string name) {
  std::lock_guard<std::mutex> lock(invoke_lock);
  for (auto &node : oplist) {
    if (node->get_name() == name) {
      node->invoke();
      return;
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " \n";
}

void ModuleInterpreter::invoke_to(const std::string& name) {
  std::lock_guard<std::mutex> lock(invoke_lock);
  for (auto &node : oplist) {
    node->invoke();
    if (node->get_name() == name) {
      return;
    }
  }
  std::string errorMsg = "Not Find target Op:" + name;
  llvm_unreachable(errorMsg.c_str());
}

bool ModuleInterpreter::set_tensor(std::string name,
                                   const std::vector<float> &data) {
  for (auto &node : oplist) {
    if (node->get_name() == name) {
      node->set_tensor(data);
      return true;
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " tensor \n";
  return false;
}

std::vector<float> ModuleInterpreter::get_tensor(std::string name) {
  std::lock_guard<std::mutex> lock(invoke_lock);
  for (auto &node : oplist) {
    if (node->get_name() == name) {
      return node->get_tensor();
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " tensor \n";
  return std::vector<float>();
}

std::vector<int64_t> ModuleInterpreter::get_tensor_shape(std::string name) {
  for (auto &node : oplist) {
    if (node->get_name() == name) {
      return node->get_shape();
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " tensor \n";
  return std::vector<int64_t>();
}

void ModuleInterpreter::dump(std::string name) {
  for (auto &node : oplist) {
    if (node->get_name() == name) {
      return node->dump();
    }
  }
  llvm::errs() << " Not Find Op name: " << name << " tensor \n";
}

void ModuleInterpreter::allocate_tensors() {

  for (FuncOp func : mlirModule.getOps<FuncOp>()) {
    MInfo Machineinfo;
    if (func->getAttr("chipname")) {
      Machineinfo.getChipInfo(func);
    }
    for (Block &bb : func.getBlocks()) {
      for (auto &op : bb) {
        prepareOperation(op);
      }
    }
  }
};


std::string ModuleInterpreter::customOpPluginFile_ = "";

} // namespace mlir
