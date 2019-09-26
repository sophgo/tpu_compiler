//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - (_d_) * ((_k_) - 1) - 1) / (_s_) + 1)

static int64_t findPadForSamePadding(int64_t i, int64_t o, int64_t k, int64_t s, int64_t d) {
  //llvm::errs() << "i: " << i << ", o: " << o << ", k: " << k << ", s: " << s << ", d: " << d << "\n";
  if (k == 1) {
    return 0;
  }
  for (int64_t p = 1; p <= k - 1; ++p) {
    if (calcConv2DSpatialOutput(i, k, s, p, d) == o) {
      return p;
    }
  }
  assert(false);
  return 0;
}

static void dumpConv2DOpParam(tpu::Conv2DOp &op, const bool verbose = false) {
  llvm::raw_ostream &os = (verbose == true) ? llvm::errs() : llvm::nulls();

  auto dh = op.dilation_h_factor().getLimitedValue();  // APInt, use .getLimitedValue(); to get uint65_t
  auto dw = op.dilation_w_factor().getLimitedValue();
  auto sh = op.stride_h().getLimitedValue();
  auto sw = op.stride_w().getLimitedValue();
  os << "  >> " << "sh: " << sh << ", sw: " << sw << ", dh : " << dh << ", dw: " << dw << "\n";

  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = op.filter()->getType().cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());

  os << "  >> " << "input shape  : "
      << i_s[0] << "," << i_s[1] << "," << i_s[2] << "," << i_s[3] << "\n";
  os << "  >> " << "output shape : "
      << o_s[0] << "," << o_s[1] << "," << o_s[2] << "," << o_s[3] << "\n";
  os << "  >> " << "filter shape : "
      << f_s[0] << "," << f_s[1] << "," << f_s[2] << "," << f_s[3] << "\n";

  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  auto n = i_s[0];
  if (n == -1) {
    os << "  >> " << "No determined N, use batch size 1" << "\n";
    n = 1;
  }

  auto ih = i_s[2];
  auto iw = i_s[3];
  auto oc = f_s[0];
  auto ic = f_s[1];
  auto kh = f_s[2];
  auto kw = f_s[3];
  auto oh = o_s[2];
  auto ow = o_s[3];

  int64_t ph, pw;
  auto padding_attr = op.getAttrOfType<StringAttr>("padding");
  if (padding_attr.getValue() == "SAME") {
    ph = findPadForSamePadding(ih, oh, kh, sh, dh);
    pw = findPadForSamePadding(iw, ow, kw, sw, dw);
  } else if (padding_attr.getValue() == "VALID") {
    ph = 0;
    pw = 0;
  } else {
    assert(false);
  }

  llvm::errs()
      << ih << ", " << iw << ", " << ic << ", " << oc << ", "
      << kh << ", " << kw << ", " << sh << ", " << sw << ", "
      << dh << ", " << dw << ", " << ph << ", " << pw << ", "
      <<"\n";
}

static void dumpFullyConnectedOpParam(tpu::FullyConnectedOp &op, const bool verbose = false) {
  llvm::raw_ostream &os = (verbose == true) ? llvm::errs() : llvm::nulls();

  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = op.filter()->getType().cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());

  os << "  >> " << "input shape  : " << i_s[0] << "," << i_s[1] << "\n";
  os << "  >> " << "output shape : " << o_s[0] << "," << o_s[1] << "\n";
  os << "  >> " << "filter shape : " << f_s[0] << "," << f_s[1] << "\n";

  assert((i_s[0] == o_s[0]) && "input M not equal to output M");
  auto M = i_s[0];
  if (M == -1) {
    os << "  >> " << "No determined N, use batch size 1" << "\n";
    M = 1;
  }
  assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
  auto K = i_s[1];
  assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
  auto N = o_s[1];

  llvm::errs() << M << ", " << K << ", " << N <<"\n";
}

namespace {

class PrintTpuOpStatsPass_v0 : public ModulePass<PrintTpuOpStatsPass_v0> {
public:
  explicit PrintTpuOpStatsPass_v0(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnModule() override {
    ModuleManager moduleManager(getModule());

    os << "Funcs walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto func : getModule().getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk<mlir::tpu::Conv2DOp>([&](mlir::tpu::Conv2DOp op) {
        dumpConv2DOpParam(op);
      });
      func.walk<mlir::tpu::FullyConnectedOp>([&](mlir::tpu::FullyConnectedOp op) {
        dumpFullyConnectedOpParam(op);
      });
    }
    os << "\n";
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<ModulePassBase> mlir::createPrintTpuOpStatsPass_v0() {
  return std::make_unique<PrintTpuOpStatsPass_v0>();
}

/// TPU OP Stats v0 is just print all conv layer shapes
/// so we can import the shapes into spreadsheet for analysis
/// use google spreadsheet, "Data" => "Split text to columns"
static PassRegistration<PrintTpuOpStatsPass_v0>
    pass("print-tpu-op-stats-v0",
         "Print statistics of TPU operations.");
