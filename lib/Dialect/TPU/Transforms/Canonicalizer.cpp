//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tpuc/Passes.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "tpuc/Dialect/TPU/Passes.h.inc"
}

using namespace mlir;

namespace {
/// Canonicalize operations in nested regions.
struct TpucCanonicalizer : public TpucCanonicalizerBase<TpucCanonicalizer> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;

    // TODO: Instead of adding all known patterns from the whole system lazily
    // add and cache the canonicalization patterns for ops we see in practice
    // when building the worklist.  For now, we just grab everything.
    auto *context = &getContext();
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(patterns, context);

    Operation *op = getOperation();
    applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};
} // end anonymous namespace

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> mlir::createTpucCanonicalizerPass() {
  return std::make_unique<TpucCanonicalizer>();
}
    