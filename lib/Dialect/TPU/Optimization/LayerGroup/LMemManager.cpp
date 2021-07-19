/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "LMemManager.hpp"
#include "LMemAllocMethod.h"

namespace mlir {

llvm::cl::opt<bool> PGLmemMethod(
    "PG-lmem-method",
    llvm::cl::desc("ProfileGuided-localmem-method"),
    llvm::cl::init(false));

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
