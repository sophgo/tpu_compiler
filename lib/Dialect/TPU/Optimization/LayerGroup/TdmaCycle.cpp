#include "TdmaCycle.hpp"
#include <algorithm>
#include <cmath>
#include "NetGraph.hpp"
#include "Group.hpp"

#define DEBUG_TYPE "tmda_cycle"

namespace mlir {

void TdmaCycle::setup_hw_config() {
    FuncOp * fn = net_graph_->getFn();
    std::string chipname = "cx1835";
    if (fn->getAttr("chipname")) {
      chipname = fn->getAttr("chipname").cast<StringAttr>().getValue().str();
    }
    if (chipname == "cv183x") {
      // 1835 ddr config
      // frequency: 1886Mhz
      uint64_t dram_frequency = 1886;
      // ddr bytewidth: 32bit/8 = 4bytes
      uint64_t dram_byte_width = (32 / 8);
      dram_bw_ = dram_frequency * dram_byte_width;

      // SRAM config
      // frequency: 650Mhz
      uint64_t sram_frequency = 650;
      // sram bytewidth: LOCAL_MEM_WIDTH
      uint64_t sram_byte_width = LOCAL_MEM_WIDTH;
      sram_bw_ = sram_frequency * sram_byte_width;
    } else if (chipname == "cv182x") {
      // 1822 ddr config
      // frequency: 1886Mhz
      uint64_t dram_frequency = 1886;
      // ddr bytewidth: 32bit/8 = 4bytes
      uint64_t dram_byte_width = (32 / 8);
      dram_bw_ = dram_frequency * dram_byte_width;

      // SRAM config
      // frequency: 650Mhz
      uint64_t sram_frequency = 650;
      // sram bytewidth: LOCAL_MEM_WIDTH
      uint64_t sram_byte_width = LOCAL_MEM_WIDTH;
      sram_bw_ = sram_frequency * sram_byte_width;
    } else {
      assert(!"chip setting not configed.\n");
    }
}

int TdmaCycle::get_cycle(const TENSOR_STEP& step) {
  init(step);

  if (step.second == TIMESTEP_LOAD) {
    return get_cycle_load();
  } else {
    return get_cycle_store();
  }
}

int TdmaCycle::init(const TENSOR_STEP& step) {
  transpose = false;
  aligned = false;
  tensor_id = step.first;
  tensor = net_graph_->get_tensor_by_id(tensor_id);

  const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);

  net_graph_->get_tensor_dim(tensor_id, tensor_dim);
  memcpy(&local_shape, &tensor_dim, sizeof(int) * 4);

  if (tensor_type == TENSOR_COEFF) {
    local_shape[0] = 1;
    local_shape[1] = tensor_dim[1];
    local_shape[2] = tensor_dim[3] * tensor_dim[2];
    local_shape[3] = tensor_dim[0];
  } else if (tensor_type == TENSOR_BIAS || tensor_type == TENSOR_COEFF_LUT) {
  } else if (tensor_type == TENSOR_DEPTHCONV_OPD1) {
    aligned = (true);
  } else {
    int n_slice = tensor->n_slice;
    int h_idx = tensor->h_idx;
    int h_slice = tensor->h_slice;
    int h_end = h_idx + h_slice;
    h_idx = h_idx > 0 ? h_idx : 0;
    h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

    local_shape[0] = (n_slice);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (h_slice);
    local_shape[3] = (tensor_dim[3]);

    if (tensor_type == TENSOR_NEURON || tensor_type == TENSOR_NEURON_WINOGRAD) {
      aligned = (true);
    }
  }
  return 0;
}

int TdmaCycle::get_cycle_load() {
  inst_->sys_dtype = transpose;
  inst_->src_n = local_shape[0];
  inst_->src_c = local_shape[1];
  inst_->src_h = local_shape[2];
  inst_->src_w = local_shape[3];

  // fill descriptor
  dim src_stride = tg_default_stride(inst_->src_n, inst_->src_c,
      inst_->src_h, inst_->src_w, tensor->unit_size());

  inst_->src_n_stride = src_stride.n;
  inst_->src_c_stride = src_stride.c;
  inst_->src_h_stride = src_stride.h;

  // local memory
  inst_->dst_c = inst_->src_c;
  inst_->dst_h = inst_->src_h;
  inst_->dst_w = inst_->src_w;
  dim dst_stride = tl_default_stride(inst_->src_n, inst_->dst_c,
      inst_->dst_h, inst_->dst_w, tensor->unit_size(), aligned);
  inst_->dst_n_stride = dst_stride.n;
  inst_->dst_c_stride = dst_stride.c;
  inst_->dst_h_stride = dst_stride.h;

  inst_->src_fmt = tensor->unit_size();
  inst_->dst_fmt = tensor->unit_size();
  inst_->spec_func = 0;
  inst_->transpose_md = 0;
  inst_->trans_fmt = 0;
  inst_->src_base_addr_high = 0;
  inst_->src_base_addr_low = 0;

  // emit
  inst_->dram_byte_count = 0;
  inst_->sram_byte_count = 0;

  cal_load_cycle();

  // time(ns)
  uint64_t dram_time = inst_->dram_byte_count * 1000 / dram_bw_;
  uint64_t sram_time = inst_->sram_byte_count * 1000 / sram_bw_;
  uint64_t total_time = dram_time > sram_time ? dram_time : sram_time;

  // LLVM_DEBUG(llvm::errs() << llvm::format( "  [Balance Layer] load %s cycle is %lu ns\n",
  //       tensor->name().c_str(), total_time););

  return total_time;
}

int TdmaCycle::get_cycle_store() {
  inst_->sys_dtype = transpose;
  inst_->src_n = local_shape[0];
  inst_->src_c = local_shape[1];
  inst_->src_h = local_shape[2];
  inst_->src_w = local_shape[3];

  // fill descriptor
  // local memory
  dim src_stride = tl_default_stride(inst_->src_n, inst_->src_c,
      inst_->src_h, inst_->src_w, tensor->unit_size(), aligned);

  inst_->src_n_stride = src_stride.n;
  inst_->src_c_stride = src_stride.c;
  inst_->src_h_stride = src_stride.h;

  // global memory
  dim dst_stride = tg_default_stride(inst_->src_n, inst_->dst_c,
      inst_->dst_h, inst_->dst_w, tensor->unit_size());
  inst_->dst_c = inst_->src_c;
  inst_->dst_h = inst_->src_h;
  inst_->dst_w = inst_->src_w;
  inst_->dst_n_stride = dst_stride.n;
  inst_->dst_c_stride = dst_stride.c;
  inst_->dst_h_stride = dst_stride.h;

  inst_->src_fmt = tensor->unit_size();
  inst_->dst_fmt = tensor->unit_size();
  inst_->spec_func = 0;
  inst_->transpose_md = 0;
  inst_->trans_fmt = 0;
  inst_->dst_base_addr_high = (((uint64_t)2) << 40) >> 32;
  inst_->dst_base_addr_low = 0;

  inst_->dram_byte_count = 0;
  inst_->sram_byte_count = 0;

  cal_store_cycle();

  uint64_t dram_time = inst_->dram_byte_count * 1000 / dram_bw_;
  uint64_t sram_time = inst_->sram_byte_count * 1000 / sram_bw_;
  uint64_t total_time = dram_time > sram_time ? dram_time : sram_time;

  // LLVM_DEBUG(llvm::errs() << llvm::format( "  [Balance Layer] store %s cycle is %lu us\n",
  //       tensor->name().c_str(), total_time););

  return total_time;
}

void TdmaCycle::get_tdma_cycle(uint64_t baseAddr, uint64_t data_size, bool isStore) {
  uint64_t dram_byte_count = 0;
  bool isCwTranspose = (inst_->transpose_md == 3);
  int max_burst_length = MAX_BURST_LENGTH;
  if (isCwTranspose && isStore)
    max_burst_length = SPECIAL_FUNCTION_BURST_LENGTH;
  int max_burst_size = max_burst_length * AXI_BUS_WIDTH;

  bool isCross4KBoundary =
      ((baseAddr & FOUR_KB_MASK) != (( ((baseAddr + data_size) ) ) & FOUR_KB_MASK));
  if(isCross4KBoundary) {
    int nearest4KBBoundary = align_up(baseAddr, FOUR_KB);
    if(baseAddr != (uint64_t)nearest4KBBoundary) {
      int headOffsetTo4K = ((nearest4KBBoundary) )  - ((baseAddr ) );
      int headBurstNumber = headOffsetTo4K / (max_burst_size);
      int remainSizeForBurst =
          align_up(headOffsetTo4K % max_burst_size, AXI_BUS_WIDTH);
      for(int packetNum = 0; packetNum < headBurstNumber; packetNum++) {
        uint64_t tempBasedAddr = baseAddr + packetNum * max_burst_size;
        tempBasedAddr = (packetNum == 0) ?
                        tempBasedAddr : align_down(tempBasedAddr, AXI_BUS_WIDTH);
        uint64_t tempByteCnt = calByteCnt(tempBasedAddr, max_burst_size);
        dram_byte_count += tempByteCnt;
      }
      if(remainSizeForBurst > 0){
        uint64_t tempBasedAddr = baseAddr + headBurstNumber * max_burst_size;
        tempBasedAddr = (headBurstNumber == 0) ?
                        tempBasedAddr :
                        align_down(tempBasedAddr, AXI_BUS_WIDTH);
        uint64_t tempByteCnt =
                    calByteCnt(tempBasedAddr, remainSizeForBurst);
        dram_byte_count += tempByteCnt;
      }
    }
    int tailOffsetTo4K = ((baseAddr + data_size) )  - ((nearest4KBBoundary ) );
    int tailBurstNumber = tailOffsetTo4K / (max_burst_size);
    int tailRemainSizeForBurst =
        align_up(tailOffsetTo4K % (max_burst_size), AXI_BUS_WIDTH);
    for(int packetNum = 0; packetNum < tailBurstNumber; packetNum++) {
      uint64_t tempBasedAddr = nearest4KBBoundary + packetNum * max_burst_size;
      tempBasedAddr = (packetNum == 0) ?
                      tempBasedAddr :
                      align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt =
                calByteCnt(tempBasedAddr, max_burst_size);
      dram_byte_count += tempByteCnt;
    }
    if (tailRemainSizeForBurst > 0) {
      uint64_t tempBasedAddr =
               nearest4KBBoundary + tailBurstNumber * max_burst_size;
      tempBasedAddr = align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt =
               calByteCnt(tempBasedAddr, tailRemainSizeForBurst);
      dram_byte_count += tempByteCnt;
    }
  } else {
    int addressRangePeriod = baseAddr + data_size - align_down(baseAddr, 16);
    int prevMaxBurstNumber =
          addressRangePeriod / (max_burst_size);
    int prevRemainSizeForBurst =
          align_up(addressRangePeriod % (max_burst_size), AXI_BUS_WIDTH);
    for(int packetNum = 0; packetNum < prevMaxBurstNumber; packetNum++) {
      uint64_t tempBasedAddr = baseAddr + packetNum * max_burst_size;
      tempBasedAddr = (packetNum == 0) ?
                      tempBasedAddr :
                      align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt =
                calByteCnt(tempBasedAddr, max_burst_size);
      dram_byte_count += tempByteCnt;
    }
    if (prevRemainSizeForBurst) {
      uint64_t tempBasedAddr = baseAddr + prevMaxBurstNumber * max_burst_size;
      tempBasedAddr = (prevMaxBurstNumber == 0) ?
                      tempBasedAddr :
                      align_down(tempBasedAddr, AXI_BUS_WIDTH);
      uint64_t tempByteCnt =
                calByteCnt(tempBasedAddr, prevRemainSizeForBurst);
      dram_byte_count += tempByteCnt;
    }
  }

  inst_->dram_byte_count = dram_byte_count;
}

static uint64_t inline get_src_address(tdma_inst_t * r) {
  uint64_t addr = (((uint64_t)r->src_base_addr_high << SRC_BASE_ADDR_HIGH_SHIFT)
                   | r->src_base_addr_low);
  return addr;
}

static uint64_t inline get_dst_address(tdma_inst_t* r) {
  uint64_t addr =  (((uint64_t)r->dst_base_addr_high << SRC_BASE_ADDR_HIGH_SHIFT)
                   | r->dst_base_addr_low);
  return addr;
}

// copy from TdmaLoader.cc
void TdmaCycle::cal_load_cycle() {
  if(inst_->sys_dtype == 1) { //matrix
    //Force src_n to 1
    inst_->src_h = 1;
    inst_->src_h_stride = inst_->src_c_stride;
    inst_->src_n = 1;
  }
  bool isMatrix = (inst_->sys_dtype == 1);
  bool isTranspose = isMatrix ?
                      ((inst_->spec_func == 1) && (inst_->transpose_md == 0)):
                      false;
  bool isHwcMode = (inst_->spec_func == 1) &&
                    ((inst_->transpose_md == 1) || (inst_->transpose_md == 2));
  if(isHwcMode) {
    int realC = inst_->src_c;
    int realW = inst_->src_w;
    inst_->src_w = realC;
    inst_->src_h_stride = realC;
    inst_->src_c = realW;
    inst_->src_c_stride = inst_->src_h_stride * inst_->src_h;
  }
  int c_stride = inst_->src_c_stride;
  bool isHContinuous = isTranspose ?
                        false :
                      (inst_->src_h_stride - inst_->src_w <= DATA_MAX_DISTANCE);
  bool isCContinuous = isHContinuous &&
                      (c_stride - inst_->src_w * inst_->src_h <= DATA_MAX_DISTANCE) ;
  bool isSrcBf16 = (inst_->src_fmt == 2);
  int srcDataSize = isSrcBf16 ? 2 : 1;
  int h_last_valid = (inst_->src_h_stride) * (inst_->src_h - 1) +
                      inst_->src_w * srcDataSize;
  int c_last_valid = c_stride * (inst_->src_c - 1) + h_last_valid;
  int generalCopySize = inst_->src_n_stride;
  bool isGeneralMove = inst_->trans_fmt;

  if(isGeneralMove) {
    uint64_t baseAddr = get_src_address(inst_);
    get_tdma_cycle(baseAddr, generalCopySize, false);
  } else {
    if(isCContinuous) {
      uint64_t baseAddr = get_src_address(inst_);
      for(int n = 0; n < (int)inst_->src_n; n++) {
        uint64_t addr = baseAddr + (inst_->src_n_stride * n);
        get_tdma_cycle(addr, c_last_valid, false);
      }
    } else if (isHContinuous) {
      uint64_t baseAddr = get_src_address(inst_);
      for(int n = 0; n < (int)inst_->src_n; n++) {
        for(int c = 0; c < (int)inst_->src_c; c++) {
          uint64_t addr = baseAddr
                          + inst_->src_n_stride * n
                          + c_stride * c;
          get_tdma_cycle(addr, h_last_valid, false);
        }
      }
    } else {
      if(isTranspose) {
        const int localSramRowSize = 8;
        const int localSramColSize = 8 * 16;
        uint64_t baseAddr = get_src_address(inst_);
        for(int c = 0; c < (int)inst_->src_c; c+= localSramRowSize) {
          for(int w = 0; w < (int)inst_->src_w; w+= localSramColSize) {
            int loopRowSize = (inst_->src_c - c >= localSramRowSize) ?
                              localSramRowSize : inst_->src_c - c;
            for(int k = 0; k < loopRowSize; k++){
              uint64_t addr = baseAddr
                              + c_stride * (c + k)
                              + w;
              int size = (inst_->src_w - w >= localSramColSize) ?
                          localSramColSize : inst_->src_w - w;
              get_tdma_cycle(addr, size, false);
            }
          }
        }
      }else {
        uint64_t baseAddr = get_src_address(inst_);
        for(int n = 0; n < (int)inst_->src_n; n++) {
          for(int c = 0; c < (int)inst_->src_c; c++) {
            for(int h = 0; h < (int)inst_->src_h; h++) {
              uint64_t addr = baseAddr
                              + inst_->src_n_stride * n
                              + c_stride * c
                              + inst_->src_h_stride * h;
              uint64_t data_size = inst_->src_w * srcDataSize;
              get_tdma_cycle(addr, data_size, false);
            }
          }
        }
      }
    }
  }
  inst_->sram_byte_count = calSramCycle(inst_);
}


uint64_t TdmaCycle::calByteCnt(uint64_t baseAddr, uint64_t size) {
  uint64_t tempBaseAddrAlign16Byte = align_down(baseAddr, AXI_BUS_WIDTH);
  uint64_t tempEndAddrAlign16Byte = tempBaseAddrAlign16Byte + size;
  uint64_t nearestTempBasedAddr64BBoundary = align_up(tempBaseAddrAlign16Byte, BYTE64);
  uint64_t nearestTempEndAddr64BBoundary = align_up(tempEndAddrAlign16Byte, BYTE64);
  uint64_t tempByteCnt = !(tempBaseAddrAlign16Byte == nearestTempBasedAddr64BBoundary)
                           + (nearestTempEndAddr64BBoundary - nearestTempBasedAddr64BBoundary) / BYTE64;
  tempByteCnt = tempByteCnt * BYTE64;
  return tempByteCnt;
}

uint64_t TdmaCycle::calSramCycle(tdma_inst_t* _inst) {
  bool isDstBf16 = (inst_->dst_fmt == 2);
  uint64_t dataSize = isDstBf16 ? 2 : 1;

  bool isStoreHContinuous = (inst_->dst_h_stride - inst_->dst_w == 0);
  int store_h_last_valid = (inst_->dst_h_stride) * (inst_->dst_h - 1) + inst_->dst_w * dataSize;
  uint64_t storeCycleTime = (isStoreHContinuous) ?
    inst_->src_n * inst_->dst_c * ceiling_func(store_h_last_valid, LOCAL_MEM_WIDTH) :
    inst_->src_n * inst_->dst_c * inst_->dst_h * ceiling_func(inst_->dst_w * dataSize, LOCAL_MEM_WIDTH);
  return storeCycleTime;
}

// copy from TdmaStorer.cc
void TdmaCycle::cal_store_cycle() {
  int dst_n = inst_->src_n;
  bool isNcTranspose = (inst_->transpose_md == 0);
  bool isSpecialFunction = (inst_->spec_func == 1);
  if(inst_->sys_dtype == 1) {
    //Force src_n to 1
    inst_->dst_h = 1;
    inst_->dst_h_stride = inst_->dst_c_stride;
    dst_n = 1;
  }
  if(isSpecialFunction && isNcTranspose) {
    dst_n = inst_->src_c;
  }
  // cout << tdmaDesTskTypeName[inst_->funcName] << endl;
  int c_stride = inst_->dst_c_stride;
  bool isDstBf16 = (inst_->dst_fmt == 2);
  int dataSize = isDstBf16 ? 2 : 1;
  bool isCwTranspose = (inst_->transpose_md == 3);
  bool isHContinuous = isCwTranspose ? false :
                      (inst_->dst_h_stride - inst_->dst_w * dataSize <= STORE_DATA_MAX_DISTANCE);
  bool isCContinuous = (c_stride - inst_->dst_w * inst_->dst_h * dataSize <= STORE_DATA_MAX_DISTANCE) && isHContinuous;
  int h_last_valid = (inst_->dst_h_stride) * (inst_->dst_h - 1) + inst_->dst_w * dataSize;
  int c_last_valid = c_stride * (inst_->dst_c - 1) + h_last_valid;
  int generalCopySize = inst_->src_n_stride;
  bool isGeneralMove = inst_->trans_fmt;

  if(isGeneralMove) {
    uint64_t baseAddr = get_dst_address(inst_);
    get_tdma_cycle(baseAddr, generalCopySize, true);
  } else {
    if(isCContinuous) {
      for(int i = 0; i < dst_n; i++) {
        uint64_t baseAddr = get_dst_address(inst_) + inst_->dst_n_stride * i;
        get_tdma_cycle(baseAddr, c_last_valid, true);
      }
    } else if (isHContinuous) {
      for(int i = 0; i < (int)dst_n; i++) {
        for(int j = 0; j < (int)inst_->dst_c; j++) {
          uint64_t baseAddr =get_dst_address(inst_) + inst_->dst_n_stride * i + c_stride * j;
          get_tdma_cycle(baseAddr, h_last_valid, true);
        }
      }
    } else {
      for(int i = 0; i < (int)dst_n; i++) {
        for(int j = 0; j < (int)inst_->dst_c; j++) {
          for(int k = 0; k < (int)inst_->dst_h; k++) {
            uint64_t baseAddr = get_dst_address(inst_) + inst_->dst_n_stride * i
              + c_stride * j
              + inst_->dst_h_stride * k;
            uint64_t data_size = inst_->dst_w * dataSize;
            get_tdma_cycle(baseAddr, data_size, true);
          }
        }
      }
    }
  }
  inst_->sram_byte_count = calSramCycle(inst_);
}


}