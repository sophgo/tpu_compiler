/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "LMemManager.hpp"
#include "LMemAllocMethod.h"

namespace mlir {

static llvm::cl::OptionCategory clOptionsCategory("Layer Group LocalMem Options");

llvm::cl::opt<bool> PGLmemMethod(
    "lmem-method-pg",
    llvm::cl::desc("local memory alloc method:ProfileGuided"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

LmemManager::LmemManager(NetGraph* net_graph) : net_graph_(net_graph) {

}

bmerr_t LmemManager::assign_local_memory(Group* group, net_timestep* time_step, bool one_shoot) {
  if (PGLmemMethod) {
    LMemAllocProfileGuided lmem_alloc;
    return lmem_alloc.assign_local_memory(group, net_graph_, time_step,
                                          one_shoot);
  } else {
    LMemAllocFitFirst lmem_alloc;
    return lmem_alloc.assign_local_memory(group, net_graph_, time_step,
                                          one_shoot);
  }
}
}
