/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: BM1880v2BackendContext.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#define DEBUG_TYPE "backend_context"

CviBackendContext::CviBackendContext(const char *runchip, int nodechip_num,
                                               std::vector<int8_t> &weight)
    : BM188xBackendContext(weight) {
  // New kernel API
  cvk_reg_info_t req_info;
  strncpy(req_info.chip_ver_str, runchip, sizeof(req_info.chip_ver_str)-1);
  req_info.cmdbuf_size = 0x10000000;
  req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
  cvk_ctx_ = cvikernel_register(&req_info);

  // Default mapping between tdma base selection and global memory region.
  tdmaBaseSelects[NEURON_MEMORY] = 0;
  tdmaBaseSelects[WEIGHT_MEMORY] = 1;
  tdmaBaseSelects[INPUT_MEMORY] = 2;  // the same as neuron
  tdmaBaseSelects[OUTPUT_MEMORY] = 2; // the same as neuron

  LLVM_DEBUG(llvm::errs() << "register " << runchip << " done\n";);
}

CviBackendContext::~CviBackendContext() {
  if (cvk_ctx_)
    cvk_ctx_->ops->cleanup(cvk_ctx_);
}

void CviBackendContext::submit() {
  uint32_t size;

  if (cvk_ctx_) {
    const uint8_t *cmdbuf = cvk_ctx_->ops->acquire_cmdbuf(cvk_ctx_, &size);
    write_cmdbuf(cmdbuf, size);
    cvk_ctx_->ops->reset(cvk_ctx_);
  }
}

uint8_t CviBackendContext::getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const {
  // we store memory region value in bits (40 ~ 41) of gaddr;
  uint32_t memoryRegion = ((((uint64_t)gaddr) >> 40) & 0x03);
  if (memoryRegion < MAX_GLOBAL_MEMORY_REGION) {
    return tdmaBaseSelects[memoryRegion];
  }
  return 0;
}


CviBackendContext *cvi_backend_create_context(
    std::vector<int8_t> &weight_data) {
  std::string runchip = CVI_TPU_VERSION_183X;
  CviBackendContext *ctx = new CviBackendContext(runchip.c_str(), 1, weight_data);
  return ctx;
}

CviBackendContext *cvi_backend_create_context_chip(
    std::vector<int8_t> &weight_data, const char *runchip) {
  CviBackendContext *ctx = new CviBackendContext(runchip, 1, weight_data);
  return ctx;
}

void cvi_backend_submit(CviBackendContext *ctx) {
  ctx->submit();
}

void cvi_backend_get_cmdbuf(CviBackendContext *ctx,
    std::vector<uint8_t> &cmdbuf) {
  ctx->read_cmdbuf(cmdbuf);
}

void cvi_backend_parallel_enable(CviBackendContext *ctx) {
  ctx->parallel_enable();
}

void cvi_backend_parallel_disable(CviBackendContext *ctx) {
  ctx->parallel_disable();
}

int cvi_backend_chip_context(CviBackendContext *ctx, CVI_CHIP_INFO_E cvi_chip_info_e) {
  return ctx->cvi_chip_info_context(cvi_chip_info_e);
}

void cvi_backend_set_layer_id(CviBackendContext *ctx, int layer_id) {
  ctx->set_layer_id(layer_id);
}
