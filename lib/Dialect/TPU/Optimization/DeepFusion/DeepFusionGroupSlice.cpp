//=======- DeepFusionGroupSlice.cpp - Implementation of TPU df Slice ------===//
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
// This file implements the TPU dialect deep fusion group Slice pass.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MathExtras.h"
#include "tpuc/SimpleAnalysis.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "deep-fusion-group-slice"

using namespace mlir;

namespace {

typedef struct {
  Operation *begin;
  Operation *end;
  int nSec;
  bool included;
} SubGroup;

static inline int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

static uint32_t getOpLine(Operation *op) {
  auto loc = op->getLoc().cast<FileLineColLoc>();
  return loc.getLine();
}

class DeepFusionGroupSlice : public mlir::PassWrapper<DeepFusionGroupSlice, FunctionPass> {
public:
  explicit DeepFusionGroupSlice() {}

  void runOnFunction() override {
    MInfo::getChipInfo(getFunction());
    assert(MInfo::version && "refer to set-chip");
    init();
    deepFusionGroupOpt();
  }

private:
  MLIRContext * context_;
  Operation * insertionPoint_;
  int batchSize_;
  int nSecs_;
  std::vector<Operation *> skippedOps_;
  std::vector<int> allNSecs_;

private:
  void init();
  void setAllNSecs();
  void deepFusionGroupOpt();
  void doGroup(FuncOp &fn,
               std::vector<std::vector<Operation *>> &fusionGroups,
               int batchSize);

  template<typename opTy>
  bool canEltwiseFused(Operation *opInst);

  void findBestSubGroups(std::vector<SubGroup> &subGroups,
                       std::vector<Operation *> group,
                       std::vector<SubGroup> &bestSubGroups);

  void mergeSubGroups(std::vector<SubGroup> &subGroups,
                      unsigned int searchIdx = 0);
  void doSlice(std::vector<Operation *> &group);
  void tg2TL(std::vector<Operation *> &group);

  bool isFusionOp(Operation *opInst, int batchSize = -1);
  void findInOutOps(std::vector<Operation *> group,
                    std::vector<Operation *> &groupInOps,
                    std::vector<Operation *> &groupOutOps);

  Operation * getInsertionPoint() { return insertionPoint_; }

  void setInsertionPoint(Operation *insertionPoint) {
    insertionPoint_ = insertionPoint;
  }

  int computeCost(SubGroup subGroup, std::vector<Operation*> group);

  void insertSliceOp(std::pair<Operation *, Operation *> &sliceOpPair,
                     int sliceIdx, int curN);

  template<typename srcOpTy, typename dstOpTy>
  void genTLEltwiseOp(Operation *srcOp, std::vector<Value> opds,
                      Operation *&dstOp, int loopIdx, int curN);
  void genTLPoolAvgOp(Operation *srcOp, std::vector<Value> opds,
                      Operation *&dstOp, int loopIdx, int curN);
  void genTLConvOp(Operation *srcOp, std::vector<Value> opds,
                   Operation *&dstOp, int loopIdx, int curN);
  void concatOps(std::vector<std::vector<Operation *>> outOps,
                 std::vector<Operation *> &orderedGroupOutOps);
  void genTLLutOp(Operation *srcOp, std::vector<Value> opds,
                  Operation *&dstOp, int loopIdx, int curN);
  void genTLScaleOp(Operation *srcOp, std::vector<Value> opds,
                           Operation *&dstOp, int loopIdx, int curN);
  void genTLPixelShuffleOp(Operation *srcOp, std::vector<Value> opds,
                           Operation *&dstOp, int loopIdx, int curN);
  void genTLPReluOp(Operation *srcOp, std::vector<Value> opds,
                    Operation *&dstOp, int loopIdx, int curN);
  void genTLOp(Operation *srcOp, std::vector<Value> opds,
               Operation *&dstOp, int loopIdx, int curN);
};


void DeepFusionGroupSlice::init() {
  auto fn = getFunction();
  auto fnResultType = fn.getType().getInput(0);
  auto fnResultShape = fnResultType.cast<RankedTensorType>().getShape();
  batchSize_ = fnResultShape[0];
  nSecs_ = 1;
  context_ = &getContext();
  setAllNSecs();
}

void DeepFusionGroupSlice::setAllNSecs() {
  std::set<int> allNSlices;
  for (int i = 1; i <= batchSize_; i++) {
    auto nSlice = ceiling_func(batchSize_, i);
    if (allNSlices.find(nSlice) == allNSlices.end()) {
      allNSlices.insert(nSlice);
      allNSecs_.push_back(i);
    }
  }
}

template<typename opTy>
bool DeepFusionGroupSlice::canEltwiseFused(Operation *opInst) {
  auto eltwiseOp = cast<opTy>(opInst);
  auto opd0Inst = opInst->getOperand(0).getDefiningOp();
  auto opd1Inst = opInst->getOperand(1).getDefiningOp();
  auto getPrevOpInst = [](Operation *curOp) -> Operation * {
    auto prevOp = curOp->getPrevNode();
    while(isa<tpu::LoadWeightOp>(prevOp)) {
      prevOp = prevOp->getPrevNode();
    }
    return prevOp;
  };
  auto prevOp = getPrevOpInst(opInst);
  if (prevOp == opd0Inst) {
    return true;
  }
  if (prevOp == opd1Inst) {
    if(eltwiseOp.m_i8_inputs().hasValue()){
      std::vector<int32_t> m_i8_inputs_array;
      arrayAttrToVector(eltwiseOp.m_i8_inputs().getValue(), m_i8_inputs_array);
      assert(m_i8_inputs_array.size() == 2);

      int32_t m_i8_inputs[2];
      m_i8_inputs[0] = m_i8_inputs_array[1];
      m_i8_inputs[1] = m_i8_inputs_array[0];
      Builder builder(context_);
      eltwiseOp->setAttr("m_i8_inputs",
                  builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs})));
    }
    opInst->setOperand(0, opd1Inst->getResult(0));
    opInst->setOperand(1, opd0Inst->getResult(0));
    return true;
  }
  return false;
}

// This pass mainly optimizes deep fusion.
// Find the most suitable combination by dividing batch number
// to do deep fusion.
void DeepFusionGroupSlice::deepFusionGroupOpt() {
  auto fn = getFunction();
  std::vector<std::vector<Operation *>> fusionGroups;
  doGroup(fn, fusionGroups, 1);

  for (auto &group : fusionGroups) {
    std::vector<SubGroup> subGroups;
    std::vector<std::pair<Operation *, int>> cutPoints;
    if (group.size() == 1 && isFusionOp(group[0], batchSize_)) {
      tg2TL(group);
      continue;
    }
    if (group.size() > 1) {
      std::vector<Operation *> traversedOps;
      for (auto beginIter = allNSecs_.rbegin(), endIter = allNSecs_.rend();
           beginIter != endIter; ++beginIter) {
        auto nSecs = *beginIter;
        if (nSecs == batchSize_)
          continue;
        for (auto opInst : group) {
          if (std::find(traversedOps.begin(), traversedOps.end(), opInst) !=
                        traversedOps.end())
            continue;
          if (!isFusionOp(opInst, ceiling_func(batchSize_, nSecs))) {
            cutPoints.push_back({opInst, nSecs});
            traversedOps.push_back(opInst);
          }
        }
      }
      auto cmp = [](std::pair<Operation *, int> a,
                    std::pair<Operation *, int> b) -> bool {
        return getOpLine(a.first) < getOpLine(b.first);
      };
      std::sort(cutPoints.begin(), cutPoints.end(), cmp);

      auto getNextOp = [&group](Operation * op) -> Operation * {
        auto iter = std::find(group.begin(), group.end(), op);
        ++iter;
        if (iter == group.end()) {
          return nullptr;
        }
        return *iter;
      };

      auto getNextSec = [this](int sec) -> int {
        auto iter = std::find(this->allNSecs_.begin(),
                              this->allNSecs_.end(), sec);
        if (++iter != this->allNSecs_.end()) {
          return *iter;
        } else {
          llvm_unreachable("wrong input sec");
        }
      };
      auto begin = group[0];
      if (cutPoints.empty()) {
        SubGroup subGroup;
        subGroup.begin = begin;
        subGroup.end = group.back();
        subGroup.nSec = 1;
        subGroup.included = true;
        subGroups.push_back(subGroup);
      }
      for (auto cutPoint : cutPoints) {
        SubGroup subGroup;
        if (begin != cutPoint.first) {
          subGroup.begin = begin;
          subGroup.end = cutPoint.first;
          subGroup.nSec = 1;
          subGroup.included = false;
          subGroups.push_back(subGroup);
        }
        subGroup.begin = cutPoint.first;

        if (cutPoint.first == group.back()) {
          subGroup.end = group.back();
          subGroup.nSec = getNextSec(cutPoint.second);
          subGroup.included = true;
        } else {
          if (cutPoint == cutPoints.back()) {
            subGroup.end = getNextOp(cutPoint.first);
            subGroup.nSec = getNextSec(cutPoint.second);
            subGroup.included = false;
            subGroups.push_back(subGroup);

            subGroup.begin = getNextOp(cutPoint.first);
            subGroup.end = group.back();
            subGroup.nSec = 1;
            subGroup.included = true;
          } else {
            subGroup.end = getNextOp(cutPoint.first);
            subGroup.nSec = getNextSec(cutPoint.second);
            subGroup.included = false;
          }
        }

        begin = getNextOp(cutPoint.first);
        subGroups.push_back(subGroup);
      }
      LLVM_DEBUG(
        llvm::dbgs() << "\nall ops in sub group: " << "\n";
        for (auto op : group) {
          llvm::dbgs() << "  " << mlir::getOpName(op) << ",";
        }
        llvm::dbgs() << "\n" << "before merge sub groups: " << "\n";
        for (auto &subGroup : subGroups) {
          llvm::dbgs() << "  subGroup[" <<
                          mlir::getOpName(subGroup.begin) << ", " <<
                          mlir::getOpName(subGroup.end) <<
                          (subGroup.included ? "]" : ")") << " nSec: " <<
                          subGroup.nSec << "\n";
        });

      std::vector<SubGroup> bestCombinedGroup;
      mergeSubGroups(subGroups);

      LLVM_DEBUG(
        llvm::dbgs() << "after merge sub groups: " << "\n";
        for (auto &subGroup : subGroups) {
          llvm::dbgs() << "  subGroup[" <<
                       mlir::getOpName(subGroup.begin) << ", " <<
                       mlir::getOpName(subGroup.end) <<
                       (subGroup.included ? "]" : ")") << " nSec: " <<
                       subGroup.nSec << "\n";
        }
        llvm::dbgs() << "\n";);

      findBestSubGroups(subGroups, group, bestCombinedGroup);

      LLVM_DEBUG(
        llvm::dbgs() << "****the best sub groups****" << "\n";
        for (auto &subGroup : bestCombinedGroup) {
          llvm::dbgs() << "subGroup[" <<
                       mlir::getOpName(subGroup.begin) << ", " <<
                       mlir::getOpName(subGroup.end) <<
                       (subGroup.included ? "]" : ")") << " nSec: " <<
                       subGroup.nSec << "\n";
        }
        if (bestCombinedGroup.empty())
          llvm::dbgs() << "  NULL" << "\n" << "\n";);

      for (auto &subGroup : bestCombinedGroup) {
        std::vector<Operation *> subGroupOps;
        auto beginIter = std::find(group.begin(), group.end(), subGroup.begin);
        auto endIter = std::find(group.begin(), group.end(), subGroup.end);
        if (subGroup.included) {
          endIter = group.end();
        }
        subGroupOps.insert(subGroupOps.end(), beginIter, endIter);
        nSecs_ = subGroup.nSec;
        if (subGroupOps.size() == 1)
          continue;
        if (nSecs_ == 1) {
          tg2TL(subGroupOps);
          continue;
        }
        doSlice(subGroupOps);
      }
    }
  }
}

void DeepFusionGroupSlice::findBestSubGroups(std::vector<SubGroup> &subGroups,
                                        std::vector<Operation *> group,
                                        std::vector<SubGroup> &bestSubGroups) {
  if (subGroups.size() == 1) {
    bestSubGroups = subGroups;
    bestSubGroups.back().end = group.back();
    if (computeCost(subGroups[0], group) < 0) {
      bestSubGroups.clear();
    }
    return;
  }
  std::vector<std::vector<SubGroup>> allCombinedGroups;
  std::vector<int> groupSecs;
  std::vector<std::vector<int>> allCombinedSecs;
  for (auto &subGroup : subGroups) {
    groupSecs.push_back(subGroup.nSec);
  }

  allCombinedSecs.push_back(groupSecs);
  std::vector<int> tmpCombinedSecs;
  // O(1 + n - 1 + n - 2) = O(2n)
  for (unsigned int i = 0; i < groupSecs.size(); i++) {
    for (unsigned int j = i + 1; j < groupSecs.size(); j++) {
      tmpCombinedSecs.insert(tmpCombinedSecs.end(), groupSecs.data(),
                             groupSecs.data() + i);
      auto max = *std::max_element(groupSecs.data() + i,
                                   groupSecs.data() + j + 1);
      tmpCombinedSecs.insert(tmpCombinedSecs.end(), j - i + 1, max);
      tmpCombinedSecs.insert(tmpCombinedSecs.end(),
                             groupSecs.data() + j + 1,
                             groupSecs.data() + groupSecs.size());
      allCombinedSecs.push_back(tmpCombinedSecs);
      tmpCombinedSecs.clear();
    }
  }

  for (auto &combinedSecs : allCombinedSecs) {
    std::vector<SubGroup> tmpGroups;
    assert(combinedSecs.size() == subGroups.size() && "must be the same size");
    for (unsigned int i = 0; i < subGroups.size(); i++) {
      auto subGroup = subGroups[i];
      subGroup.nSec = combinedSecs[i];
      tmpGroups.push_back(subGroup);
    }
    allCombinedGroups.push_back(tmpGroups);
  }

  LLVM_DEBUG(llvm::dbgs() << "all combined groups: " << "\n";);
  for (auto &combinedGroups: allCombinedGroups) {
    mergeSubGroups(combinedGroups);

    LLVM_DEBUG(
      llvm::dbgs() << "##### combined group #####" << "\n";
      for (auto &subGroup : combinedGroups) {
        llvm::dbgs() << "subGroup[" <<
                     mlir::getOpName(subGroup.begin) << ", " <<
                     mlir::getOpName(subGroup.end) <<
                     (subGroup.included ? "]" : ")") << " nSec: " <<
                     subGroup.nSec << "\n";
    });
  }

  int maxProfit = 0x80000000;
  int bestIdx = 0;
  for (unsigned int i = 0; i < allCombinedGroups.size(); i++) {
    int tmpProfit = 0;
    for (auto &subGroup : allCombinedGroups[i])
      tmpProfit += computeCost(subGroup, group);

    LLVM_DEBUG(
      llvm::dbgs() << "group profit :" << "\n" << "  " << tmpProfit << "\n";
    );

    if (tmpProfit > maxProfit) {
      maxProfit = tmpProfit;
      bestIdx = i;
    }
  }

  if (maxProfit < 0) {
    bestSubGroups.clear();
  } else {
    bestSubGroups = allCombinedGroups[bestIdx];
  }
}

// Combine the group with the same segmentation size.
void DeepFusionGroupSlice::mergeSubGroups(std::vector<SubGroup> &subGroups,
                                          unsigned int searchIdx) {
  unsigned int curIdx = searchIdx + 1;
  for (; curIdx < subGroups.size(); curIdx++) {
    if (subGroups[searchIdx].nSec != subGroups[curIdx].nSec) {
      searchIdx = curIdx;
      break;
    }
    auto lastEndOp = subGroups[searchIdx].end;
    auto curBeginOp = subGroups[curIdx].begin;
    if (lastEndOp == curBeginOp) {
      subGroups[searchIdx].end = subGroups[curIdx].end;
      subGroups[searchIdx].included = subGroups[curIdx].included;
      subGroups[curIdx].nSec = -1;
    }
  }
  if (curIdx == subGroups.size()) {
    auto removeCmp = [](SubGroup val) -> bool {
      return val.nSec == -1;
    };
    subGroups.erase(std::remove_if(subGroups.begin(),
                         subGroups.end(), removeCmp), subGroups.end());
    return;
  }
  mergeSubGroups(subGroups, searchIdx);
}

// Find the group that can do deep fusion
void DeepFusionGroupSlice::doGroup(FuncOp &fn,
                      std::vector<std::vector<Operation *>> &fusionGroups,
                      int batchSize) {
  std::vector<Operation *> fusionGroup;
  fn.walk([&](mlir::Operation *opInst) {
    if (isa<tpu::LoadWeightOp>(opInst))
      return;
    if (isa<tpu::InputOp>(opInst)) {
      for (auto &use : opInst->getResult(0).getUses()) {
        skippedOps_.push_back(use.getOwner());
      }
      return;
    }

    auto iter = std::find(skippedOps_.begin(), skippedOps_.end(), opInst);
    if (iter != skippedOps_.end()) {
      if (fusionGroup.size())
        fusionGroups.push_back(fusionGroup);
      fusionGroup.clear();
      return;
    }
    if (isFusionOp(opInst, batchSize)) {
      if (isa<tpu::TG_INT8_EltwiseAddOp>(opInst) &&
          !canEltwiseFused<tpu::TG_INT8_EltwiseAddOp>(opInst)) {
        fusionGroup.clear();
        return;
      }
      if (isa<tpu::TG_INT8_EltwiseMulOp>(opInst) &&
          !canEltwiseFused<tpu::TG_INT8_EltwiseMulOp>(opInst)) {
        fusionGroup.clear();
        return;
      }

      fusionGroup.push_back(opInst);
    } else {
      if (fusionGroup.size())
        fusionGroups.push_back(fusionGroup);
      fusionGroup.clear();
    }
  });
}

// Calculate the profit after deep fusion
int DeepFusionGroupSlice::computeCost(
                    SubGroup subGroup, std::vector<Operation*> group) {
  if (group.size() < 1)
    return 0;

  int reducedLmem = 0;
  int addedLmem = 0;
  int profitableLmem = 0;
  auto beginIter = std::find(group.begin(), group.end(), subGroup.begin);
  auto endIter = std::find(group.begin(), group.end(), subGroup.end);
  auto nSec = subGroup.nSec;

  if (subGroup.end == group.back() && subGroup.included == true) {
    endIter = group.end();
  }

  if (std::distance(beginIter, endIter) == 1)
    return 0;

  std::vector<Operation *> subGroupOps;
  std::vector<Operation *> subGroupInOps;
  std::vector<Operation *> subGroupOutOps;
  subGroupOps.insert(subGroupOps.begin(), beginIter, endIter);
  findInOutOps(subGroupOps, subGroupInOps, subGroupOutOps);

  for (auto iter = beginIter; iter != endIter; ++iter) {
    int64_t n, c, h, w;
    auto shape = getTensorShape((*iter)->getResult(0));
    getNCHW(shape, n, c, h, w);
    if (std::find(subGroupOutOps.begin(), subGroupOutOps.end(), *iter) ==
                  subGroupOutOps.end())
      reducedLmem += MInfo::getSizePerLane(n, c, h, w, true);

    if (isa<tpu::TG_INT8_Conv2DOp>(*iter)) {
      auto convOp = cast<tpu::TG_INT8_Conv2DOp>(*iter);
      bool is_dw, with_bias, do_relu;
      int n, ic, ih, iw, oc, oh, ow, g;
      int kh, kw, sh, sw, ins_h, ins_w, pt, pb, pl, pr, dh, dw, pad_value;
      bool is_deconv = isa<tpu::TG_INT8_DeConv2DOp>(convOp.getOperation());
      parseConvParam(convOp.param(), is_deconv, convOp.input(), convOp.output(),
                     n, ic, ih, iw, oc, oh, ow, g, kh, kw,
                     ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                     with_bias, do_relu, pad_value);
      addedLmem += MInfo::getSizePerLane(1, oc, kh * kw, ic / g, false);
      addedLmem += MInfo::getSizePerLane(1, oc, 1, with_bias ? 9 : 5, false);
    }
    profitableLmem += reducedLmem - addedLmem * (nSec - 1);
  }
  return profitableLmem;
}

void DeepFusionGroupSlice::tg2TL(std::vector<Operation *> &group) {
  Operation * dstOp = nullptr;
  for (auto op : group) {
    setInsertionPoint(op);
    std::vector<Value> opds;
    opds.push_back(op->getOperand(0));
    genTLOp(op, opds, dstOp, 0, batchSize_);
  }
}

void DeepFusionGroupSlice::doSlice(std::vector<Operation *> &group) {
  std::vector<Operation *> groupInOps;
  std::vector<Operation *> groupOutOps;

  setInsertionPoint((*(--group.end()))->getNextNode());
  findInOutOps(group, groupInOps, groupOutOps);

  std::vector<Operation *> lastOps;
  std::map<Operation *, Operation *> opsMap;
  std::vector<std::vector<Operation *>> outOps(groupOutOps.size());
  std::vector<Operation *> orderedGroupOutOps;

  for (int i = 0; i < nSecs_; i++) {
    auto remainedN = batchSize_ - i * ceiling_func(batchSize_, nSecs_);
    auto nSlice = remainedN < 0 ? batchSize_ - (i - 1) *
                              ceiling_func(batchSize_, nSecs_) :
                              ceiling_func(batchSize_, nSecs_);
    auto curN = (i == nSecs_ - 1) ? (batchSize_ - i * nSlice) : nSlice;
    for (unsigned int j = 0; j < groupInOps.size(); j++) {
      std::pair<Operation *, Operation *> sliceOpPair;
      sliceOpPair.first = groupInOps[j];
      sliceOpPair.second = nullptr;
      insertSliceOp(sliceOpPair, i, curN);
      opsMap.insert(sliceOpPair);
      skippedOps_.push_back(sliceOpPair.second);
      skippedOps_.push_back(sliceOpPair.first);
    }

    auto outOpsIter = outOps.begin();
    for (unsigned int j = 0; j < group.size(); j++) {
      auto groupOp = group[j];
      std::vector<Value> opds;
      for (auto opd : groupOp->getOperands()) {
        if (isa<tpu::LoadWeightOp>(opd.getDefiningOp()))
          continue;
        auto op = opsMap.find(opd.getDefiningOp());
        if (op != opsMap.end())
          opds.push_back(op->second->getResult(0));
      }

      Operation *dstOp = nullptr;
      genTLOp(groupOp, opds, dstOp, i, curN);
      opsMap[groupOp] = dstOp;
      skippedOps_.push_back(dstOp);
      skippedOps_.push_back(groupOp);

      for (auto &use : groupOp->getResult(0).getUses()) {
        auto iter = find(groupOutOps.begin(), groupOutOps.end(),
                                              use.getOwner());
        if (iter != groupOutOps.end()) {
          outOpsIter->push_back(dstOp);
          if (i == 0)
            orderedGroupOutOps.push_back(groupOp);
          ++outOpsIter;
        }
      }
    }
    opsMap.clear();
  }
  concatOps(outOps, orderedGroupOutOps);
}

bool DeepFusionGroupSlice::isFusionOp(Operation *opInst, int batchSize) {
  uint64_t totalPerLane = -1;
  if (isa<tpu::TpuTGOpCodegenInterface>(opInst)) {
    auto shape = getTensorShape(opInst->getResult(0));
    if (shape.size() != 4)
      return false;
  }
  if (isa<tpu::TG_INT8_Conv2DOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_Conv2DOp>(opInst);
    if (op.input().hasOneUse() == false) {
      return false;
    }
    totalPerLane = SimpleConv2DMemoryUsageAnalysis(op, nullptr, batchSize);

  } else if (isa<tpu::TG_INT8_EltwiseAddOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_EltwiseAddOp>(opInst);
    totalPerLane = SimpleEltwiseMemoryUsageAnalysis(op, nullptr, batchSize);

  } else if (isa<tpu::TG_INT8_EltwiseMulOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_EltwiseMulOp>(opInst);
    totalPerLane = SimpleEltwiseMemoryUsageAnalysis(op, nullptr, batchSize);

  } else if (isa<tpu::TG_INT8_LutOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_LutOp>(opInst);
    totalPerLane = SimpleLutMemoryUsageAnalysis(op, nullptr, batchSize);

  } else if (isa<tpu::TG_INT8_PoolAvg2DOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_PoolAvg2DOp>(opInst);
    totalPerLane = SimpleIOMemoryUsageAnalysis(op, nullptr,batchSize);

  }
  // else if (isa<tpu::TG_INT8_ScaleOp>(opInst)) {
  //   auto op = cast<tpu::TG_INT8_ScaleOp>(opInst);
  //   totalPerLane = SimpleScaleMemoryUsageAnalysis(op,nullptr, batchSize);
  // }
  else if (isa<tpu::TG_INT8_PixelShuffleOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_PixelShuffleOp>(opInst);
    totalPerLane = SimplePixelShuffleMemoryUsageAnalysis(op,
                                                         nullptr, batchSize);

  } else if (isa<tpu::TG_INT8_PReluOp>(opInst)) {
    auto op = cast<tpu::TG_INT8_PReluOp>(opInst);
    totalPerLane = SimplePReluMemoryUsageAnalysis(op, nullptr, batchSize);

  } else {
    return false;
  }
  if (totalPerLane < MInfo::lmem_per_lane) {
    return true;
  }
  return false;
}

void DeepFusionGroupSlice::findInOutOps(std::vector<Operation *> group,
                                        std::vector<Operation *> &groupInOps,
                                        std::vector<Operation *> &groupOutOps) {
  std::vector<Operation *> inOps;
  std::vector<Operation *> outOps;
  for (auto op : group) {
    for (auto opd : op->getOperands()) {
      if (isa<tpu::LoadWeightOp>(opd.getDefiningOp()))
        continue;
      auto iter = std::find(inOps.begin(), inOps.end(), opd.getDefiningOp());
      if (iter == inOps.end())
        inOps.push_back(opd.getDefiningOp());
    }
    for (auto &use : op->getResult(0).getUses()) {
      auto iter = std::find(group.begin(), group.end(), use.getOwner());
      if (iter == group.end())
        outOps.push_back(use.getOwner());
    }
  }
  std::sort(inOps.begin(), inOps.end());
  std::sort(outOps.begin(), outOps.end());
  std::sort(group.begin(), group.end());
  std::set_difference(inOps.begin(), inOps.end(), group.begin(), group.end(),
                      std::inserter(groupInOps, groupInOps.begin()));
  std::set_difference(outOps.begin(), outOps.end(), group.begin(), group.end(),
                      std::inserter(groupOutOps, groupOutOps.begin()));
}

void DeepFusionGroupSlice::insertSliceOp(
                           std::pair<Operation *, Operation *> &sliceOpPair,
                           int sliceIdx, int curN) {
  Builder builder(context_);
  auto op = sliceOpPair.first;
  Operation *sliceOp = nullptr;
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  int offset = sliceIdx * ceiling_func(batchSize_, nSecs_);
  std::vector<Value> sliceOperands;
  sliceOperands.push_back(op->getResult(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("axis",
                                       builder.getI32IntegerAttr(0)));
  attrs.push_back(builder.getNamedAttr("offset",
                                       builder.getI32IntegerAttr(offset)));

  std::string sliceOpName = mlir::getOpName(op).str() + std::string("_slice_") +
                                                std::to_string(sliceIdx);
  attrs.push_back(builder.getNamedAttr("name",
                                       builder.getStringAttr(sliceOpName)));

  sliceOp = OpBuilder(getInsertionPoint()).create<tpu::TG_INT8_SliceOp>(
                      op->getLoc(),
                      resultType, ArrayRef<Value>{sliceOperands},
                      ArrayRef<NamedAttribute>{attrs});
  sliceOpPair.second = sliceOp;
}

void DeepFusionGroupSlice::genTLOp(Operation *srcOp,
                                  std::vector<Value> opds,
                                  Operation *&dstOp, int loopIdx, int curN) {
  if (isa<tpu::TG_INT8_Conv2DOp>(srcOp)) {
    genTLConvOp(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_EltwiseAddOp>(srcOp)) {
    genTLEltwiseOp<tpu::TG_INT8_EltwiseAddOp, tpu::TL_EltwiseAddOp>(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_EltwiseMulOp>(srcOp)) {
    genTLEltwiseOp<tpu::TG_INT8_EltwiseMulOp, tpu::TL_EltwiseMulOp>(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_LutOp>(srcOp)) {
    genTLLutOp(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_PoolAvg2DOp>(srcOp)) {
    genTLPoolAvgOp(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_ScaleOp>(srcOp)) {
    genTLScaleOp(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_PixelShuffleOp>(srcOp)) {
    genTLPixelShuffleOp(srcOp, opds, dstOp, loopIdx, curN);
  } else if (isa<tpu::TG_INT8_PReluOp>(srcOp)) {
    genTLPReluOp(srcOp, opds, dstOp, loopIdx, curN);
  } else {
    llvm_unreachable("unsupported op");
  }
}

void DeepFusionGroupSlice::genTLLutOp(Operation *srcOp,
                                     std::vector<Value> opds,
                                     Operation *&dstOp, int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_LutOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  operands.push_back(opds[0]);
  operands.push_back(op.getOperand(1));
  operands.push_back(op.getOperand(2));

  std::vector<NamedAttribute> attrs;
  uint32_t la_invalid = 0xffffffff;
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  std::string lutName = op.getOpName().str();
  if (bSlice)
    lutName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(lutName)));

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_LutOp>(
                      srcOp->getLoc(), resultType,
                      ArrayRef<Value>{operands},
                      ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

template<typename srcOpTy, typename dstOpTy>
void DeepFusionGroupSlice::genTLEltwiseOp(Operation *srcOp,
                                         std::vector<Value> opds,
                                         Operation *&dstOp,
                                         int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<srcOpTy>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  if (opds.size() == 1) {
    operands.push_back(opds[0]);
    operands.push_back(op.getOperand(1));
  } else if (opds.size() == 2) {
    operands.push_back(opds[0]);
    operands.push_back(opds[1]);
  } else {
    assert(0 && "error input size");
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("rshift", op.rshiftAttr()));
  if(op.m_i8_inputs().hasValue()) {
    attrs.push_back(builder.getNamedAttr("m_i8_inputs", op.m_i8_inputsAttr()));
  }
  if(op.m_i32_output().hasValue())
    attrs.push_back(builder.getNamedAttr("m_i32_output",
                                          op.m_i32_outputAttr()));

  attrs.push_back(builder.getNamedAttr("do_relu",
                  builder.getBoolAttr(op.do_relu())));
  if (op.do_early_stride()) {
    attrs.push_back(builder.getNamedAttr("do_early_stride",
                                         builder.getBoolAttr(true)));
    attrs.push_back(builder.getNamedAttr("early_stride_h",
                                         op.early_stride_hAttr()));
    attrs.push_back(builder.getNamedAttr("early_stride_w",
                                         op.early_stride_wAttr()));
  }
  uint32_t la_invalid = 0xffffffff;
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  std::string eltwiseName = op.getOpName().str();
  if (bSlice)
    eltwiseName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("name",
                                     builder.getStringAttr(eltwiseName)));

  dstOp = OpBuilder(getInsertionPoint()).create<dstOpTy>(
                      srcOp->getLoc(), resultType,
                      ArrayRef<Value>{operands},
                      ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::genTLPoolAvgOp(Operation *srcOp,
                                         std::vector<Value> opds,
                                         Operation *&dstOp,
                                         int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_PoolAvg2DOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  operands.push_back(opds[0]);

  std::vector<NamedAttribute> attrs;
  std::string poolAvgName = op.getOpName().str();
  if (bSlice)
    poolAvgName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("param", op.paramAttr()));
  attrs.push_back(builder.getNamedAttr("name",
                                     builder.getStringAttr(poolAvgName)));

  if(op.rshift().hasValue()) {
    attrs.push_back(builder.getNamedAttr("rshift", op.rshiftAttr()));
  }
  if(op.m_i8().hasValue()) {
    attrs.push_back(builder.getNamedAttr("m_i8", op.m_i8Attr()));
  }

  uint32_t la_invalid = 0xffffffff;
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_PoolAvg2DOp>(
                      srcOp->getLoc(), resultType,
                      ArrayRef<Value>{operands},
                      ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::genTLConvOp(Operation *srcOp,
                                      std::vector<Value> opds,
                                      Operation *&dstOp,
                                      int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_Conv2DOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  operands.push_back(opds[0]);
  operands.push_back(op.getOperand(1));
  operands.push_back(op.getOperand(2));

  std::vector<NamedAttribute> attrs;
  std::string convName = op.getOpName().str();
  if (bSlice)
    convName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("param", op.paramAttr()));
  attrs.push_back(builder.getNamedAttr("name",
                                      builder.getStringAttr(convName)));
  if(op.do_ic_alignment().hasValue()){
    attrs.push_back(builder.getNamedAttr("do_ic_alignment",
                    builder.getBoolAttr(op.do_ic_alignment().getValue())));
  }

  if (op.do_leaky_relu()) {
    attrs.push_back(builder.getNamedAttr("do_leaky_relu",
                                         op.do_leaky_reluAttr()));
    if (op.rshift_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_pos",
                                           op.rshift_posAttr()));
    if (op.m_i8_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_pos", op.m_i8_posAttr()));
    if (op.rshift_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_neg", op.rshift_negAttr()));
    if (op.m_i8_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_neg", op.m_i8_negAttr()));
  }

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_LA_Conv2DOp>(
                      srcOp->getLoc(), resultType,
                      ArrayRef<Value>{operands},
                      ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::genTLScaleOp(Operation *srcOp,
                                              std::vector<Value> opds,
                                              Operation *&dstOp,
                                              int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_ScaleOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  operands.push_back(opds[0]);
  if (opds.size() == 2) {
    operands.push_back(opds[1]);
  } else {
    operands.push_back(op.getOperand(1));
  }
  operands.push_back(op.getOperand(2));

  std::vector<NamedAttribute> attrs;

  std::string bdcastName = op.getOpName().str();
  if (bSlice)
    bdcastName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("name",
                                       builder.getStringAttr(bdcastName)));

  uint32_t la_invalid = 0xffffffff;
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  attrs.push_back(builder.getNamedAttr("param",
    tpu::ConvParam::get(
        builder.getI32IntegerAttr(1),
        builder.getI32IntegerAttr(1),
        builder.getI32IntegerAttr(1),
        builder.getI32IntegerAttr(1),
        builder.getStringAttr("VALID"),
        builder.getI32IntegerAttr(1),
        builder.getI32IntegerAttr(1),
        builder.getI32IntegerAttr(0), // pd_t
        builder.getI32IntegerAttr(0), // pd_b
        builder.getI32IntegerAttr(0), // pd_l
        builder.getI32IntegerAttr(0), // pd_r
        builder.getI32IntegerAttr(1),
        builder.getBoolAttr(true),    // is_dw
        builder.getBoolAttr(false),   // with_bias
        builder.getBoolAttr(false),   // do_relu
        builder.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
        builder.getI32IntegerAttr(0), // pad_value
        builder.getContext())));

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_ScaleOp>(
                    srcOp->getLoc(), resultType,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::genTLPixelShuffleOp(Operation *srcOp,
                                              std::vector<Value> opds,
                                              Operation *&dstOp,
                                              int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_PixelShuffleOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());

  std::vector<Value> operands;
  operands.push_back(opds[0]);
  std::vector<NamedAttribute> attrs;

  std::string psName = op.getOpName().str();
  if (bSlice)
    psName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("name",
                                       builder.getStringAttr(psName)));

  uint32_t la_invalid = 0xffffffff;
  attrs.push_back(builder.getNamedAttr("factor", op.upscale_factorAttr()));
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_PixelShuffleOp>(
                    srcOp->getLoc(), resultType,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::genTLPReluOp(Operation *srcOp,
                                              std::vector<Value> opds,
                                              Operation *&dstOp,
                                              int loopIdx, int curN) {
  Builder builder(context_);
  bool bSlice = (curN == batchSize_) ? false : true;
  auto op = cast<tpu::TG_INT8_PReluOp>(srcOp);
  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(srcOp->getResult(0));
  getNCHW(shape, n, c, h, w);
  auto tensorType = srcOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
                                   {curN, c, h, w},
                                   tensorType.getElementType());
  std::vector<Value> operands;
  operands.push_back(opds[0]);
  operands.push_back(op.getOperand(1));
  std::vector<NamedAttribute> attrs;

  std::string psName = op.getOpName().str();
  if (bSlice)
    psName += std::string("_") + std::to_string(loopIdx);

  attrs.push_back(builder.getNamedAttr("name",
                                       builder.getStringAttr(psName)));

  uint32_t la_invalid = 0xffffffff;
  assert(op.rshift_pos().hasValue());
  assert(op.m_i8_pos().hasValue());
  assert(op.rshift_neg().hasValue());

  attrs.push_back(builder.getNamedAttr("rshift_pos", op.rshift_posAttr()));
  attrs.push_back(builder.getNamedAttr("rshift_neg", op.rshift_negAttr()));
  attrs.push_back(builder.getNamedAttr("m_i8_pos", op.m_i8_posAttr()));
  attrs.push_back(builder.getNamedAttr("lm_layout", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("la_input", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_working", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("la_output", builder.getI32IntegerAttr(la_invalid)));
  attrs.push_back(builder.getNamedAttr("tl_load_flag", builder.getBoolAttr(true)));
  attrs.push_back(builder.getNamedAttr("tl_store_flag", builder.getBoolAttr(true)));

  dstOp = OpBuilder(getInsertionPoint()).create<tpu::TL_PReluOp>(
                    srcOp->getLoc(), resultType,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});
  if (!bSlice) {
    srcOp->replaceAllUsesWith(dstOp);
  }
}

void DeepFusionGroupSlice::concatOps(
                           std::vector<std::vector<Operation *>> outOps,
                           std::vector<Operation *> &orderedGroupOutOps) {
  Builder builder(context_);
  for (unsigned int i = 0; i < outOps.size(); i++) {
    auto outOpVec = outOps[i];
    std::vector<Value> operands;
    for (auto op: outOpVec)
      operands.push_back(op->getResult(0));

    std::vector<NamedAttribute> attrs;
    std::string concatOpName = mlir::getOpName(orderedGroupOutOps[i]).str();
    attrs.push_back(builder.getNamedAttr("name",
                                         builder.getStringAttr(concatOpName)));
    auto insertionPoint = getInsertionPoint();
    auto concatOp = OpBuilder(insertionPoint).create<tpu::TG_ConcatNOp>(
                    insertionPoint->getLoc(),
                    orderedGroupOutOps[i]->getResult(0).getType(),
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});

    std::vector<Operation *> useOps;
    for (auto &use : orderedGroupOutOps[i]->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      useOps.push_back(useOp);
    }
    for (auto useOp : useOps) {
      if (getOpLine(useOp) < getOpLine(concatOp.getOperation()))
        continue;
      useOp->replaceUsesOfWith(orderedGroupOutOps[i]->getResult(0),
                               concatOp.getOperation()->getResult(0));
    }
  }
}
} // namespace

std::unique_ptr<mlir::Pass> mlir::createDeepFusionGroupSlice() {
  return std::make_unique<DeepFusionGroupSlice>();
}
