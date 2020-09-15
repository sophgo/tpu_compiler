/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */

#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include "CviBackendContext.h"

#define DEBUG_TYPE "CviBackendContext"

CviBackendContext::CviBackendContext(const char *runchip) {
  // New kernel API
  cvk_reg_info_t req_info;
  strncpy(req_info.chip_ver_str, runchip, sizeof(req_info.chip_ver_str) - 1);
  req_info.cmdbuf_size = 0x10000000;
  req_info.cmdbuf = static_cast<uint8_t *>(malloc(req_info.cmdbuf_size));
  cvk_ctx_ = cvikernel_register(&req_info);

  // Default mapping between tdma base selection
  // and global memory region.
  tdmaBaseSelects[NEURON_MEMORY] = 0;
  tdmaBaseSelects[WEIGHT_MEMORY] = 1;
  tdmaBaseSelects[INPUT_MEMORY] = 2;
  tdmaBaseSelects[OUTPUT_MEMORY] = 2;

  LLVM_DEBUG(llvm::errs() << "register " << runchip << " done\n";);
}

CviBackendContext::~CviBackendContext() {
  cvk_ctx_->ops->cleanup(cvk_ctx_);
}

void CviBackendContext::write_cmdbuf(const void* cmdbuf, uint32_t size) {
  cmdbuf_.resize(size);
  memcpy(&cmdbuf_[0], cmdbuf, size);
}

void CviBackendContext::read_cmdbuf(std::vector<uint8_t>& out_cmdbuf) {
  out_cmdbuf.assign(cmdbuf_.begin(), cmdbuf_.end());
}

void CviBackendContext::submit() {
  uint32_t size;
  uint8_t *cmdbuf = cvk_ctx_->ops->acquire_cmdbuf(cvk_ctx_, &size);
  write_cmdbuf(cmdbuf, size);
  cvk_ctx_->ops->reset(cvk_ctx_);
}

int CviBackendContext::cvi_chip_info_context(CVI_CHIP_INFO_E cvi_chip_info_e) const {
  if (cvi_chip_info_e == CVI_CHIP_VERSION)
    return cvk_ctx_->info.version;
  else if (cvi_chip_info_e == CVI_CHIP_NODE_SHIFT)
    return cvk_ctx_->info.node_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_NUM)
    return cvk_ctx_->info.npu_num;
  else if (cvi_chip_info_e == CVI_CHIP_LANE_SHIFT)
    return cvk_ctx_->info.npu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_EU_NUM)
    return cvk_ctx_->info.eu_num;
  else if (cvi_chip_info_e == CVI_CHIP_EU_SHIFT)
    return cvk_ctx_->info.eu_shift;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_SIZE)
    return cvk_ctx_->info.lmem_size;
  else if (cvi_chip_info_e == CVI_CHIP_LMEM_BANK)
    return cvk_ctx_->info.lmem_banks;
  else
    assert(0);
}

uint8_t CviBackendContext::getTdmaBaseSelectIndexFromGaddr(gaddr_t gaddr) const {
  // we store memory region value in bits (40 ~ 41) of gaddr;
  uint32_t memoryRegion = ((((uint64_t)gaddr) >> 40) & 0x03);
  if (memoryRegion < MAX_GLOBAL_MEMORY_REGION) {
    return tdmaBaseSelects[memoryRegion];
  }
  return 0;
}

CviBackendContext *cvi_backend_create_context(const char *runchip) {
  CviBackendContext *ctx = new CviBackendContext(runchip);
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
