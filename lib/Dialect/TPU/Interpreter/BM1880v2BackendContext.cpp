#include "BM1880v2BackendContext.h"
#include <iostream>


BM1880v2BackendContext::BM1880v2BackendContext(int chip_ver, int nodechip_num,
                                               std::vector<int8_t> &weight)
    : BM188xBackendContext(weight) {
  hw.chip_version = chip_ver;
  hw.local_mem_shift = 15;
  hw.local_mem_banks = 8;
  hw.nodechip_shift = get_num_shift(nodechip_num);
  hw.npu_shift = 5;
  hw.eu_shift = 4;
  hw.global_mem_size = 1ULL << 32;
  hw.local_mem_size = 1 << hw.local_mem_shift;
  hw.nodechip_num = 1 << hw.nodechip_shift;
  hw.npu_num = 1 << hw.npu_shift;
  hw.eu_num = 1 << hw.eu_shift;
  hw.unit_size = INT8_SIZE;

  std::cout << "BM1880v2BackendContext param:" << std::endl
            << "  chip_version " << hw.chip_version << std::endl
            << "  nodechip_shift " << hw.nodechip_shift << std::endl
            << "  npu_shift " << hw.npu_shift << std::endl
            << "  eu_shift " << hw.eu_shift << std::endl
            << "  local_mem_shift " << hw.local_mem_shift << std::endl
            << "  local_mem_banks " << hw.local_mem_banks << std::endl
            << "  global_mem_size 0x" << std::hex << hw.global_mem_size << std::endl
            << "  nodechip_num " << hw.nodechip_num << std::endl
            << "  npu_num " << hw.npu_num << std::endl
            << "  eu_num " << hw.eu_num << std::endl
            << "  local_mem_size 0x" << std::hex << hw.local_mem_size << std::endl;

  assert(hw.chip_version == BM_CHIP_BM1880v2);
  bmk_info_.chip_version = 18802;
  bmk_info_.cmdbuf_size = 0x10000000;
  bmk_info_.cmdbuf = static_cast<u8 *>(malloc(bmk_info_.cmdbuf_size));
  assert(bmk_info_.cmdbuf);

  bmk_ = bmk1880v2_register(&bmk_info_);
  std::cout << "bmk1880v2_register done";
}

BM1880v2BackendContext::~BM1880v2BackendContext() {
  if (bmk_) {
    std::cout << "bmk1880v2_cleanup";
    bmk1880v2_cleanup(bmk_);
  }
}

void BM1880v2BackendContext::submit() {
  u32 size;
  const u8 *cmdbuf = bmk1880v2_acquire_cmdbuf(bmk_, &size);
  write_cmdbuf(cmdbuf, size);
  bmk1880v2_reset(bmk_);
}
