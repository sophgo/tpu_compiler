/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgConvKernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <map>

#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "cvi_backend_conv_kernel"

#define ASSERT(x) assert(x)

#define RELU   0
#define PRELU  1

#define SPLIT_FAILED 0xFFFF


//
// Dimension indices for 2D tensor.
//
struct NCHW {
  enum dim { N = 0, C = 1, H = 2, W = 3 };
};

//
// dimension indices for 3D tensor.
//
struct NGCHW {
  enum dim { N = 0, G = 1, C = 2, H = 3, W = 4};
};

struct NCDHW {
  enum dim { N = 0, C = 1, D = 2, H = 3, W = 4 };
};

// Concept from StridedLinearLayoutMap of affine map.
//
//         BufferMap          =>         TiledBufferMap
//
//  |<-     stride    ->|            |<-     stride    ->|
//    offset
//  |<-->|
//  ---------------------            .....................
//  |                   |            .     tiled offset  .
//  |    |<- shape ->|  |            .    |<-->|         .
//  |    -------------  |            .    .............  .
//  |    |           |  |            .    .  tiled shape .
//  |    |           |  |            .    .    |<-  ->|  .
//  |    |...........|  |            .    .....|------|  .
//  |    |    .OOOOOO|  |            .    .    |OOOOOO|  .
//  |    |    .OOOOOO|  |            .    .    |OOOOOO|  .
//  |    -------------  |            .    .....--------  .
//  |-------------------|            .....................
//
struct BufferMap {
  uint64_t address;
  uint32_t offset;
  uint32_t mem_region;
  std::vector<uint32_t> shapes;  // [n, c, h, w]
  std::vector<uint32_t> strides; // [n, c, h, w]
};

// We may need to explicitly express tiled formation for
// tdma load, tiu computation and tdma store.
struct TiledBufferMap {
  BufferMap *parent;
  uint64_t offset;
  std::vector<uint32_t> shapes;  // [n, c, h, w]
};

class MemoryDescriptor {
public:
  MemoryDescriptor() {};

  void setShapes(std::vector<uint32_t> shapes) {
    shapes_ = shapes;
  }

  std::vector<uint32_t> getShapes() {
    return shapes_;
  }

  void setDataFormat(cvk_fmt_t fmt) {
    fmt_ = fmt;
  }

  cvk_fmt_t getDataFormat() {
    return fmt_;
  }

  void setStrides(std::vector<uint32_t> strides) {
    strides_ = strides;
  }

  std::vector<uint32_t> getStrides() {
    if (strides_.size())
      return strides_;
  }

  void setAddress(uint64_t address) {
    address_ = address;
  }

  uint64_t getAddress() {
    return address_;
  }

  uint32_t getDataFormatSize() {
    switch (fmt_) {
      case CVK_FMT_F32:
        return 4;
      case CVK_FMT_BF16:
        return 2;
      default:
        return 1;
    }
  }

  void setLayerId(uint32_t layer_id) {
    layerId_ = layer_id;
  }

  uint32_t getLayerId() {
    return layerId_;
  }

  void setMemRegion(uint32_t mem_region) {
    memRegion_ = mem_region;
  }

  uint32_t getMemRegion() {
    return memRegion_;
  }

  // Calculate offset using current positions
  uint64_t getCurrentOffset(std::vector<uint32_t> cur_poss) {
    assert(cur_poss.size() == shapes_.size() &&
           "Expect current positions and shapes have same dims");
    assert(strides_.size() == shapes_.size() &&
           "Expect stride and shapes have same dims");

    uint64_t offset = 0;
    for (uint32_t i = 0; i < cur_poss.size(); ++i) {
      offset += cur_poss[i] * strides_[i];
    }

    return offset;
  }

  void setCompressed(bool compressed) {
    compressed_ = compressed;
  }

  bool getCompressed() {
    return compressed_;
  }

  // Expect physical shape, but it is very difficult in our system.
  std::vector<uint32_t> shapes_;

  // Default: int8 data type
  cvk_fmt_t fmt_ = {CVK_FMT_I8};

  // Physical layout
  // Not use logical shape to derive physical layout (e.g. do_ic_alignment)
  std::vector<uint32_t> strides_;

  uint64_t address_ = {0};
  uint32_t layerId_ = {0};
  uint32_t memRegion_ = {0};

  // TDMA load needs to enable decompression if true.
  // TDMA store needs to enable compression if true.
  bool compressed_ = {false};
};

class LocalMemoryDescriptor: public MemoryDescriptor {
public:
  LocalMemoryDescriptor(const CviBackendContext &ctx,
      std::vector<uint32_t> shapes, cvk_fmt_t fmt, uint8_t eu_align)
        : ctx_(ctx) {
    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;
  }

  LocalMemoryDescriptor(const CviBackendContext &ctx) : ctx_(ctx) {}

  ~LocalMemoryDescriptor() {
    // Kernel release resource in reverse order.
    assert(!cvk_tl_ && "Expect cvk freed");
  }

  std::vector<uint32_t> getStrides() {
    if (strides_.size())
      return strides_;

    assert(shapes_.size() == 5 && "Expect 5D tensor now");
    assert(shapes_[NGCHW::G] == 1 && "Expect 1 group");
    cvk_tl_shape_t tl_shapes = {
        shapes_[NGCHW::N], shapes_[NGCHW::C], shapes_[NGCHW::H],
        shapes_[NGCHW::W]};
    cvk_tl_stride_t tl_strides =
        ctx_.tl_default_stride(tl_shapes, fmt_, eu_align_);

    strides_ = {tl_strides.n, tl_strides.c, tl_strides.h, tl_strides.w};

    return strides_;
  }

  void initialize(std::vector<uint32_t> shapes, cvk_fmt_t fmt, uint8_t eu_align)
  {
    // Group should not appear in local memory descriptor since H/W does not
    // support grouped convolution.
    // And we do not support conv3d yet.
    assert(shapes.size() == 5 && "Expect 5D tensor");
    assert(shapes_[NGCHW::G] == 1 && "Expect 1 group");

    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;

    cvk_tl_shape_t tl_shapes = {
        shapes_[NCHW::N], shapes_[NCHW::C], shapes_[NCHW::H], shapes_[NCHW::W]};
    cvk_tl_stride_t tl_strides =
        ctx_.tl_default_stride(tl_shapes, fmt_, eu_align_);

    strides_ = {tl_strides.n, tl_strides.c, tl_strides.h, tl_strides.w};
  }

  void setEuAlign(uint8_t eu_align) {
    eu_align_ = eu_align;
  }

  uint8_t getEuAlign() {
    return eu_align_;
  }

  void allocate(std::vector<uint32_t> shapes, cvk_fmt_t fmt, uint8_t eu_align) {
    shapes_ = shapes;
    fmt_ = fmt;
    eu_align_ = eu_align;

    allocate();
  }

  void allocate() {
    assert(!cvk_tl_ && "Expect no allocated before");
    assert(shapes_.size() == 5 && "Expect 5D tensor");
    assert(shapes_[NGCHW::G] == 1 && "Expect 1 group");

    cvk_tl_shape_t tl_shape = {
        shapes_[NGCHW::N], shapes_[NGCHW::C], shapes_[NGCHW::H],
        shapes_[NGCHW::W]};
    cvk_tl_ = ctx_.lmem_alloc_tensor(tl_shape, fmt_, eu_align_);
    assert(cvk_tl_ && "Expect allocated");

    address_ = cvk_tl_->start_address;
    strides_ = {cvk_tl_->stride.n, cvk_tl_->stride.c, cvk_tl_->stride.h,
                cvk_tl_->stride.w};
  }

  // Return previously allocated kernel local memory information
  // DO NOT use it for tmda/tiu operation directly !
  // It is not always that each tile size equals to it.
  // And tdma load/store and tiu op may use different shape.
  cvk_tl_t *getAllocated() {
    assert(cvk_tl_ && "Expected allocated");
    return cvk_tl_;
  }

  void free() {
    if (cvk_tl_) {
      ctx_.lmem_free_tensor(cvk_tl_);
      cvk_tl_ = nullptr;
    }
  }

  cvk_tl_shape_t getCvkShape() {
    assert(shapes_.size() && "Expect shape assigned");
    return {shapes_[NCHW::N], shapes_[NCHW::C], shapes_[NCHW::H],
            shapes_[NCHW::W]};
  }

  uint32_t getSizePerLane() {
    if (shapes_.size() && strides_.size()) {
      assert(shapes_.size() == strides_.size() &&
             "Expect shape and strid have same size");
      return shapes_[NCHW::N] * strides_[NCHW::N];
    }

    assert(shapes_.size() && "Expect shape assigned");

    return ctx_.lmem_tensor_to_size(getCvkShape(), fmt_, eu_align_);
  }

private:
  const CviBackendContext &ctx_;
  uint8_t eu_align_ = {0};
  cvk_tl_t *cvk_tl_ = {nullptr};
};

class GlobalMemoryDescriptor : public MemoryDescriptor {
public:
  GlobalMemoryDescriptor(const CviBackendContext &ctx,
      std::vector<uint32_t> shapes, cvk_fmt_t fmt) : ctx_(ctx) {
    shapes_ = shapes;
    fmt_ = fmt;

    setDefaultStrides();
  }

  void setDefaultStrides() {
    assert(((shapes_.size() == 4) || (shapes_.size() == 5)) &&
           "Expect 4D or 5D tensor");
    if (((shapes_.size() != 4) && (shapes_.size() != 5)))
      return;

    strides_.resize(shapes_.size());
    strides_[strides_.size() - 1]  = getDataFormatSize();
    for (int i = (int)strides_.size() - 2; i >= 0; --i)
      strides_[i] = shapes_[i + 1] * strides_[i + 1];
  }

  std::vector<uint32_t> getStrides() {
    if (strides_.size())
      return strides_;

    setDefaultStrides();
    return strides_;
  }

private:
  const CviBackendContext &ctx_;
};

class Tdma {
public:
  Tdma(MemoryDescriptor *dst, MemoryDescriptor *src) : dst_(dst), src_(src) {

  }

  void transfer() {
    if (static_cast<GlobalMemoryDescriptor*>(dst_) &&
        static_cast<LocalMemoryDescriptor*>(src_))
      load();
  }

private:
  void load() {

  }

  MemoryDescriptor *dst_;
  MemoryDescriptor *src_;
};

class CmdDescriptor {
public:
  enum CmdTypeEnum {
    LoadBiasCmdType,
    LoadInputCmdType,
    LoadWeightCmdType,
    ComputCmdType,
    StoreOutputCmdType,
    ParallelCmdType
  };

  CmdDescriptor(CmdTypeEnum cmdType, bool parallelEnabled)
      : cmdType_(cmdType), parallelEnabled_(parallelEnabled) {}

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
      uint32_t lmIndex) : cmdType_(cmdType), gmOutputPoss_(gmOutputPoss) {
    lmIndexes_.push_back(lmIndex);
  }

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
      uint32_t lmIndex, uint32_t icPos) : cmdType_(cmdType),
      gmOutputPoss_(gmOutputPoss) {
    lmIndexes_.push_back(lmIndex);
    icPos_ = icPos;
  }

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
      std::vector<uint32_t> lmIndexes) : cmdType_(cmdType),
      gmOutputPoss_(gmOutputPoss), lmIndexes_(lmIndexes) {}

  CmdDescriptor(CmdTypeEnum cmdType, std::vector<uint32_t> gmOutputPoss,
      std::vector<uint32_t> lmIndexes, uint32_t icPos) : cmdType_(cmdType),
      gmOutputPoss_(gmOutputPoss), lmIndexes_(lmIndexes), icPos_(icPos) {}

  static std::string getCmdTypeStr(CmdTypeEnum cmdType);

  CmdTypeEnum getCmdType() {
    return cmdType_;
  }

  std::vector<uint32_t> getGmOutputPoss() {
    return gmOutputPoss_;
  }

  std::vector<uint32_t> getLmIndexes() {
    return lmIndexes_;
  }

  bool isParallelEnabled() {
    return parallelEnabled_;
  }

  void setIntraCmdParalEnabled(bool enabled) {
    intraCmdParalEnabled_ = true;
  }

  bool isIntraCmdParalEnabled() {
    return intraCmdParalEnabled_;
  }

  uint32_t getIcPos() {
    return icPos_;
  }

private:
  CmdTypeEnum cmdType_;
  std::vector<uint32_t> gmOutputPoss_;
  std::vector<uint32_t> lmIndexes_;
  bool parallelEnabled_ = {false};
  bool intraCmdParalEnabled_ = {false};
  uint32_t icPos_ = {0};
};

std::string CmdDescriptor::getCmdTypeStr(CmdTypeEnum cmdType) {
  switch (cmdType) {
    case LoadBiasCmdType:
      return "LoadBias";

    case LoadInputCmdType:
      return "LoadInput";

    case LoadWeightCmdType:
      return "LoadWeight";

    case ComputCmdType:
      return "Compute";

    case StoreOutputCmdType:
      return "StoreOutput";

    case ParallelCmdType:
      return "Parallel";

    default:
      assert(0 && "Unexpected cmd type");
  }

  return " ";
}

// Command sequence pattern for 1822 intra command parallism:
//   cmd            local memory access state
//   Load bias      (Write)
//   Load input     (Write)
//   Load weight    (Write)
//   TPU compute    (Output Write)
//   Store output   (ReadAfterWrite)
//
//
// Case 1: reuse activation
//                     bias      weight      input      output
//   LD input0                                W|
//   LD bias0           W|
//   LD weight0                    W|                              (O)
//   TIU0             RAW|       RAW|       RAW|          W|       (O)
//   ST output0                                         RAW|       (O)
//   LD bias1            |R
//   LD weight1                     |R
//   TIU1                |RAW       |RAW    RAR|           |W      (X)
//   ST output1                                            |RAW
//
//
// Case 2: reuse weight
//                     bias      weight      input      output
//   LD bias0          W|
//   LD weight0                    W|                              => swap
//   LD input0                                W|                   => swap
//   TIU0            RAW|        RAW|       RAW|           W|
//   ST output0                                          RAW|
//   LD input1                                 |W
//   TIU1            RAR|        RAR|          |RAW         |W     (X)
//   ST output1                                             |RAW
//
class IntraCmdParallelAnalysis {
public:
  IntraCmdParallelAnalysis(
      std::vector<std::unique_ptr<CmdDescriptor>> &cmdQueue)
      : cmdQueue_(cmdQueue) {

    // Double buffer
    for (uint32_t i = 0; i < 2; ++i) {
      lmBiasAccessStates_.push_back(UnknownState);
      lmWeightAccessStates_.push_back(UnknownState);
      lmInputAccessStates_.push_back(UnknownState);
      lmOutputAccessStates_.push_back(UnknownState);
    }

    assignLmAccessState();
  }

  void assignLmAccessState();

  // Simplified data dependency process based on hand-crafted double-buffer
  // assignment.
  enum AccessEvent {
    WriteEvent,
    ReadEvent
  };
  enum AccessState {
    UnknownState,
    WriteState,
    ReadAfterWriteState,
    ReadAfterReadState,
    WriteAfterWriteState,
  };

  struct CmdLmState {
    CmdLmState(CmdDescriptor::CmdTypeEnum cmdType,
        std::vector<AccessState> biass, std::vector<AccessState> weights,
        std::vector<AccessState> inputs, std::vector<AccessState> outputs) :
        cmdType_(cmdType), biass_(biass), weights_(weights), inputs_(inputs),
        outputs_(outputs) {}

    CmdDescriptor::CmdTypeEnum cmdType_;
    std::vector<AccessState> biass_;
    std::vector<AccessState> weights_;
    std::vector<AccessState> inputs_;
    std::vector<AccessState> outputs_;
    bool isIntraCmdParal_ = {false};
  };

  static std::string getAccessEventStr(AccessEvent event);
  static std::string getAccessStateStr(AccessState state);

  void receiveAccessEvent(AccessState *state, AccessEvent event);

  uint32_t reverseSearchBiasOrWeight(AccessState state, uint32_t lmIndex,
      uint32_t endQueueIndex);
  uint32_t searchStoreOutput(AccessState state, uint32_t lmIndex,
      uint32_t startQueueIndex);

  bool isIntrCmdParalTiu(uint32_t index);
  bool isIntrCmdParalLoadWeight(uint32_t index, uint32_t lmWeightIndex);
  bool isIntrCmdParalStoreOutput(uint32_t index, uint32_t lmOutputIndex);
  void tryEnableIntraCmdParal(uint32_t index);

  void analyze();

  void dumpStates();

private:
  const std::vector<std::unique_ptr<CmdDescriptor>> &cmdQueue_;

  std::vector<AccessState> lmBiasAccessStates_;
  std::vector<AccessState> lmWeightAccessStates_;
  std::vector<AccessState> lmInputAccessStates_;
  std::vector<AccessState> lmOutputAccessStates_;

  // Record local memory status of each command
  std::vector<std::unique_ptr<CmdLmState>> cmdLmStates_;
};

std::string IntraCmdParallelAnalysis::getAccessEventStr(AccessEvent event) {
  switch(event) {
    case WriteEvent:
      return "W";

    case ReadEvent:
      return "R";

    default:
      assert(0 && "Unexpected event");
      break;
  }

  return " ";
}


std::string IntraCmdParallelAnalysis::getAccessStateStr(AccessState state) {
  switch(state) {
    case UnknownState:
      return "UNK";

    case WriteState:
      return "W";

    case ReadAfterWriteState:
      return "RAW";

    case ReadAfterReadState:
      return "RAR";

    case WriteAfterWriteState:
      return "WAW";

    default:
      assert(0 && "Unexpected state");
  }

  return " ";
}

void IntraCmdParallelAnalysis::assignLmAccessState() {
  for (const auto &it : cmdQueue_) {
    std::vector<uint32_t> lmIndexes = it->getLmIndexes();

    if (it->getCmdType() == CmdDescriptor::LoadBiasCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadBiasCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmBiasAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::LoadInputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadInputCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmInputAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::LoadWeightCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  LoadWeightCmdDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmWeightAccessStates_[lmIndexes[0]], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::ComputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  ComputeCmdDesc\n");

      uint32_t lmInputIndex = lmIndexes[0];  // input
      uint32_t lmWeightIndex = lmIndexes[1];  // weight
      uint32_t lmOutputIndex = lmIndexes[2];  // output

      receiveAccessEvent(&lmBiasAccessStates_[lmWeightIndex], ReadEvent);
      receiveAccessEvent(&lmWeightAccessStates_[lmWeightIndex], ReadEvent);
      receiveAccessEvent(&lmInputAccessStates_[lmInputIndex], ReadEvent);
      receiveAccessEvent(&lmOutputAccessStates_[lmOutputIndex], WriteEvent);

    } else if (it->getCmdType() == CmdDescriptor::StoreOutputCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  StoreOutputDesc\n");

      assert(lmIndexes[0] < 2);
      receiveAccessEvent(&lmOutputAccessStates_[lmIndexes[0]], ReadEvent);

    } else if (it->getCmdType() == CmdDescriptor::ParallelCmdType) {
      // LLVM_DEBUG(llvm::dbgs() << "  ParallelCmdDesc\n");
    } else {
      assert(0 && "Unexpected cmd desc\n");
    }

    cmdLmStates_.push_back(
      std::make_unique<CmdLmState>(
        it->getCmdType(), lmBiasAccessStates_,
        lmWeightAccessStates_, lmInputAccessStates_, lmOutputAccessStates_));
  }
}

void IntraCmdParallelAnalysis::receiveAccessEvent(AccessState *state,
    AccessEvent event) {
  switch (*state) {
    case UnknownState:
      assert(event == WriteEvent && "Expect write event in UNK state");
      // UNK -> W
      if (event == WriteEvent)
        *state = WriteState;
    break;

    case WriteState:
      // assert(event == ReadEvent && "Expect read event in W state");
      if (event == ReadEvent) {
        // W -> RAW
        *state = ReadAfterWriteState;
      } else if (event == WriteEvent) {
        // W -> WAW
        // Only the output state allowed in ps32 mode.
        *state = WriteAfterWriteState;
      }
    break;

    case ReadAfterWriteState:
      if (event == ReadEvent) {
        // RAW -> RAR
        *state = ReadAfterReadState;
      } else if (event == WriteEvent) {
        // RAW -> W, next tpu operation
        *state = WriteState;
      }
    break;

    case ReadAfterReadState:
      // RAR -> W, next tpu operation
      if (event == WriteEvent)
        *state = WriteState;
    break;

    case WriteAfterWriteState:
      if (event == ReadEvent) {
        // WAW -> WAR
        *state = ReadAfterWriteState;
      } else if (event == WriteEvent) {
        // WAW -> WAW
      } else {
        assert(0 && "Unexpected event in WAW state");
      }
      break;

    default:
    assert(0 && "Unexpected event");
    break;
  }
}

// Backward search of LoadBias or LoadWeight.
// Caller check the required command type.
uint32_t IntraCmdParallelAnalysis::reverseSearchBiasOrWeight(
    AccessState state, uint32_t lmIndex, uint32_t endQueueIndex) {
  assert(endQueueIndex < cmdQueue_.size() && "Expect valid range");
  assert(lmIndex < cmdLmStates_[lmIndex]->weights_.size());

  for (int i = endQueueIndex; i >= 0; --i) {
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::LoadWeightCmdType &&
        cmdLmStates_[i]->weights_[lmIndex] == state)
      return static_cast<uint32_t>(i);
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::LoadBiasCmdType &&
        cmdLmStates_[i]->biass_[lmIndex] == state)
      return static_cast<uint32_t>(i);
  }

  assert(0 && "Expect valid index of bias/weight");

  return endQueueIndex;
}

// Forward search of StoreOutput
uint32_t IntraCmdParallelAnalysis::searchStoreOutput(AccessState state,
    uint32_t lmIndex, uint32_t startQueueIndex) {
  assert(startQueueIndex < cmdQueue_.size() && "Expect valid range");
  assert(lmIndex < cmdLmStates_[lmIndex]->outputs_.size());

  for (uint32_t i = startQueueIndex; i < cmdQueue_.size(); ++i) {
    if (cmdLmStates_[i]->cmdType_ == CmdDescriptor::StoreOutputCmdType &&
        cmdLmStates_[i]->outputs_[lmIndex] == state)
      return i;
  }

  assert(0 && "Expect valid index of output");

  return startQueueIndex;
}

// All of bias, weight and input are at RAW (read after write) state
bool IntraCmdParallelAnalysis::isIntrCmdParalTiu(uint32_t index) {
  assert(index < cmdQueue_.size() && "Expect valid range");
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::ComputCmdType &&
         "Expect compute cmd");

  std::vector<uint32_t> lmIndexes = cmdQueue_[index]->getLmIndexes();
  uint32_t lmInputIndex = lmIndexes[0];  // input
  uint32_t lmWeightIndex = lmIndexes[1];  // weight

  AccessState biasState = cmdLmStates_[index]->biass_[lmWeightIndex];
  AccessState weightState = cmdLmStates_[index]->weights_[lmWeightIndex];
  AccessState inputState = cmdLmStates_[index]->inputs_[lmInputIndex];

  if (biasState == ReadAfterWriteState && weightState == ReadAfterWriteState &&
      inputState == ReadAfterWriteState)
    return true;

  return false;
}

// LoadWeight at W (write) state
bool IntraCmdParallelAnalysis::isIntrCmdParalLoadWeight(uint32_t index,
    uint32_t lmWeightIndex) {
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadWeightCmdType &&
         "Expect load weight cmd");

  if (cmdLmStates_[index]->weights_[lmWeightIndex] == WriteState)
    return true;

  return false;
}

// StoreOutput at RAW(read after writer) state
bool IntraCmdParallelAnalysis::isIntrCmdParalStoreOutput(uint32_t index,
    uint32_t lmOutputIndex) {
  assert(cmdLmStates_[index]->cmdType_ == CmdDescriptor::StoreOutputCmdType &&
         "Expect load weight cmd");

  if (cmdLmStates_[index]->outputs_[lmOutputIndex] == ReadAfterWriteState)
    return true;

  return false;
}

void IntraCmdParallelAnalysis::tryEnableIntraCmdParal(uint32_t index) {
  assert(index < cmdQueue_.size() && "Expect valid index");

  // Find compute command first
  if (cmdLmStates_[index]->cmdType_ != CmdDescriptor::ComputCmdType)
    return;

  std::vector<uint32_t> lmIndexes = cmdQueue_[index]->getLmIndexes();
  uint32_t lmWeightIndex = lmIndexes[1];  // weight
  uint32_t lmOutputIndex = lmIndexes[2];  // output

  // Check compute command
  if (!isIntrCmdParalTiu(index))
    return;

  // Check loadWeight and StoreOutput
  uint32_t firstArgIndex =
      reverseSearchBiasOrWeight(WriteState, lmWeightIndex, index);
  uint32_t firstOutputIndex =
      searchStoreOutput(ReadAfterWriteState, lmOutputIndex, index);

  if (isIntrCmdParalLoadWeight(firstArgIndex, lmWeightIndex) &&
      isIntrCmdParalStoreOutput(firstOutputIndex, lmOutputIndex)) {

    // tuple of tdma load, tiu and tdma store
    cmdQueue_[index]->setIntraCmdParalEnabled(true);
    cmdQueue_[firstArgIndex]->setIntraCmdParalEnabled(true);
    cmdQueue_[firstOutputIndex]->setIntraCmdParalEnabled(true);

    cmdLmStates_[index]->isIntraCmdParal_ = true;
    cmdLmStates_[firstArgIndex]->isIntraCmdParal_ = true;
    cmdLmStates_[firstOutputIndex]->isIntraCmdParal_ = true;
  }
}

void IntraCmdParallelAnalysis::analyze() {
  for (uint32_t i = 0; i < cmdQueue_.size(); ++i) {
    assert(cmdQueue_[i]->getCmdType() == cmdLmStates_[i]->cmdType_);

    tryEnableIntraCmdParal(i);
  }
}

void IntraCmdParallelAnalysis::dumpStates() {
  LLVM_DEBUG(llvm::dbgs() << "\n  IntraCmdParallelAnalysis::dumpStates:\n");

  // For input, bias, weight, store
  auto dumpLmIndex = [=](uint32_t index) {
    LLVM_DEBUG(llvm::dbgs()
        << "[" << cmdQueue_[index]->getLmIndexes()[0] << "]");
  };

  // For compute
  auto dumpLmIndexes = [=](uint32_t index) {
    LLVM_DEBUG(llvm::dbgs()
        << "[" << cmdQueue_[index]->getLmIndexes()[2] << "]");
  };

  // For parallel
  auto dumpLmParal = [=](uint32_t index) {
    if (cmdQueue_[index]->isParallelEnabled())
      LLVM_DEBUG(llvm::dbgs() << "(E)");
    else
      LLVM_DEBUG(llvm::dbgs() << "(D)");
  };

  auto dumpCmdLmIndex = [=](uint32_t index) {
    if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadInputCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadBiasCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::LoadWeightCmdType ||
        cmdLmStates_[index]->cmdType_ == CmdDescriptor::StoreOutputCmdType)
      dumpLmIndex(index);
    else if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::ComputCmdType)
      dumpLmIndexes(index);
    else if (cmdLmStates_[index]->cmdType_ == CmdDescriptor::ParallelCmdType)
      dumpLmParal(index);
  };

  for (uint32_t i = 0; i < cmdQueue_.size(); ++i) {
    assert(cmdQueue_[i]->getCmdType() == cmdLmStates_[i]->cmdType_);

    LLVM_DEBUG(llvm::dbgs()
        << "    [" << i << "] cmd "
        << CmdDescriptor::getCmdTypeStr(cmdLmStates_[i]->cmdType_));

    dumpCmdLmIndex(i);

    LLVM_DEBUG(llvm::dbgs()
        <<  ": bias " << getAccessStateStr(cmdLmStates_[i]->biass_[0])
        << "|" << getAccessStateStr(cmdLmStates_[i]->biass_[1])
        << ", weight " << getAccessStateStr(cmdLmStates_[i]->weights_[0])
        << "|" << getAccessStateStr(cmdLmStates_[i]->weights_[1])
        << ", input " << getAccessStateStr(cmdLmStates_[i]->inputs_[0])
        << "|" << getAccessStateStr(cmdLmStates_[i]->inputs_[1])
        << ", ouput " << getAccessStateStr(cmdLmStates_[i]->outputs_[0])
        << "|" << getAccessStateStr(cmdLmStates_[i]->outputs_[1]));

    if (cmdQueue_[i]->isIntraCmdParalEnabled())
      LLVM_DEBUG(llvm::dbgs() << "    => O");

    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "\n");
}

// Manual CMODEL Debug:
//   Backend:
//     Assign layer id, output position.
//     Record input, output, weight bias information.
//     Change layer id to ((1 << 15) | layer_id) in TIU command buffer.
//
//   CMODEL:
//     Detect altered layer_id.
//     Convolution store input, weight, bias and output.
//
//   HOST:
//     Extract input/output/weight from npz used in mlir.
//     Compare data from host and CMODEL.
//
//   E.g.
//     Output positions:
//       [ig=0][oc_pos=736][n_pos=6][oh_pos=31][ow_pos=0][ic_pos=0]
//
//     MLIR:
//       tl_lw_memopt_func_tg_Conv2d_int8.mlir
//
//     TPU output:
//       data1 = np.load('WZC-0_cmdbuf_out_bs8.npz')
//       data11 = data1['6fac0227e623ad2e7a08b330d5a6ffe3']
//
//     Tiled TPU output::
//       oc_pos=736, oc_step=32, oh_pos=31, oh_step=25,  (1, 32, 56, 56)
//       data12 = data11[6:7, 736:736+32, 31:31+25, :]
//       np.savetxt('tpu_conv_oc_pos_736_oh_pos_31.txt',
//                  np.reshape(data12, (np.size(data12), 1)))
//
//     CPU output:
//       data2 = np.load('WZC-0_tensor_all_int8.npz')
//       data21 = data2['6fac0227e623ad2e7a08b330d5a6ffe3']
//
//     Tiled CPU output:
//       oc_pos=736, oc_step=32, oh_pos=31, oh_step=25,  (1, 32, 56, 56)
//       data22 = data21[6:7, 736:736+32, 31:31+25, :]
//       np.savetxt('cpu_conv_oc_pos_736_oh_pos_31.txt',
//                  np.reshape(data22, (np.size(data22), 1)))
//
//     Weight:
//       Weight (2048, 256, 1, 1)
//       data3 = np.load('WZC-0_4_558b2e062f9d.npz')
//       weight = data3['6fac0227e623ad2e7a08b330d5a6ffe3_0_quant_lowered']
//
//     Tiled weight:
//       Weight oc_pos=736, oc_step=32, (32, 256, 1, 1)
//       weight1 = weight[736:736+32, :, :, :]
//       np.savetxt('weight_pos_736_step_32.txt',
//                  np.reshape(weight1, (np.size(weight1), 1)))
//
//     Bias:
//       bias = data3['6fac0227e623ad2e7a08b330d5a6ffe3_1_quant_pack']
//
//     Tiled bias:
//       oc_pos=736, oc_step=32, (32, 1, 9)
//       bias1 = bias[736:736+32,:,:]
//       np.savetxt('bias_pos_736_step_32.txt',
//                  np.reshape(bias1, (np.size(bias1), 1)))
//
struct CModelDebug {
  bool enabled_;
  bool found_;
  uint16_t layerId_;

  struct GmInfo {
    uint64_t addr;
    uint64_t addrOffset;
    std::vector<uint32_t> shapes;
    std::vector<uint32_t> poss;
  };

  void assignOutput(uint32_t layerId, std::vector<uint32_t> poss) {
    enabled_ = true;
    found_ = false;
    layerId_ = layerId;
    output_.poss = poss;
  }

  bool isOutputMatched(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
      bool isWeightOrBias = false) {
    if (!enabled_ || (layerId_ != layerId))
      return false;

    if (isWeightOrBias) {
      assert(!gmOutputPoss[NGCHW::N] && !gmOutputPoss[NGCHW::H] &&
             !gmOutputPoss[NGCHW::W]);
      gmOutputPoss[NGCHW::N] = output_.poss[NGCHW::N];
      gmOutputPoss[NGCHW::H] = output_.poss[NGCHW::H];
      gmOutputPoss[NGCHW::W] = output_.poss[NGCHW::W];
    }

    if ((output_.poss[NGCHW::N] == gmOutputPoss[NGCHW::N]) &&
        (output_.poss[NGCHW::G] == gmOutputPoss[NGCHW::G]) &&
        (output_.poss[NGCHW::C] == gmOutputPoss[NGCHW::C]) &&
        (output_.poss[NGCHW::H] == gmOutputPoss[NGCHW::H]) &&
        (output_.poss[NGCHW::W] == gmOutputPoss[NGCHW::W]))
      return true;
    return false;
  }

  void updateLayerId(uint32_t &layerId,
      std::vector<uint32_t> gmOutputPoss) {

    if (isOutputMatched(layerId, gmOutputPoss)) {
      layerId = (1 << 15) | layerId;
    }
  }

  void recordGmInfo(GmInfo &entity, uint64_t addr, uint64_t addrOffset,
      std::vector<uint32_t> poss, std::vector<uint32_t> shapes) {
    entity.addr = addr;
    entity.addrOffset = addrOffset;
    entity.poss = poss;
    entity.shapes = shapes;
  }

  void recordOutput(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
      uint64_t addr, uint64_t addrOffset, std::vector<uint32_t> gmOutputShapes)
  {
    if (isOutputMatched(layerId, gmOutputPoss)) {
      recordGmInfo(output_, addr, addrOffset, gmOutputPoss, gmOutputShapes);
      found_ = true;
    }
  }

  void recordInput(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
      uint64_t addr, uint64_t addrOffset, std::vector<uint32_t> gmInputPoss,
      std::vector<uint32_t> gmInputShapes, bool ignoreOutputChannel) {

    if (!enabled_)
      return;

    if (ignoreOutputChannel) {
      assert(!gmOutputPoss[NGCHW::C]);
      gmOutputPoss[NGCHW::C] = output_.poss[NGCHW::C];
    }

    if (isOutputMatched(layerId, gmOutputPoss))
      recordGmInfo(input_, addr, addrOffset, gmInputPoss, gmInputShapes);
  }

  void recordWeight(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
      uint64_t addr, uint64_t addrOffset, std::vector<uint32_t> gmWeightPoss,
      std::vector<uint32_t> gmWeightShapes) {
    if (isOutputMatched(layerId, gmOutputPoss, true))
      recordGmInfo(weight_, addr, addrOffset, gmWeightPoss, gmWeightShapes);
  }

  void recordBias(uint32_t layerId, std::vector<uint32_t> gmOutputPoss,
      uint64_t addr, uint64_t addrOffset, std::vector<uint32_t> gmBiasPoss,
      std::vector<uint32_t> gmBiasShapes) {
    if (isOutputMatched(layerId, gmOutputPoss, true))
      recordGmInfo(bias_, addr, addrOffset, gmBiasPoss, gmBiasShapes);
  }

  void dumpDims(std::vector<uint32_t> &dims) {
    if (dims.size() < 5)
      return;

    LLVM_DEBUG(llvm::dbgs()
        << "(" << dims[NGCHW::N]
        << ", " << dims[NGCHW::G]
        << ", " << dims[NGCHW::C]
        << ", " << dims[NGCHW::H]
        << ", " << dims[NGCHW::W]
        << ")");
  }

  void dump() {
    if (!enabled_ || !found_)
      return;

    // Replace with raw_ostream
    LLVM_DEBUG(llvm::dbgs()
        << "CMODEL Debug:\n"
        << "  enabled " << enabled_
        << ", layer_id " << layerId_ << "\n"
        << "  output addr " << llvm::format_hex(output_.addr, 10)
        << "(offset=" << llvm::format_hex(output_.addrOffset, 10)
        << "), poss");
    dumpDims(output_.poss);
    LLVM_DEBUG(llvm::dbgs() << ", shapes");
    dumpDims(output_.shapes);

    LLVM_DEBUG(llvm::dbgs()
        << "\n  input addr " << llvm::format_hex(input_.addr, 10)
        << "(offset=" << llvm::format_hex(input_.addrOffset, 10)
        << "), poss");
    dumpDims(input_.poss);
    LLVM_DEBUG(llvm::dbgs() << ", shapes");
    dumpDims(input_.shapes);
    LLVM_DEBUG(llvm::dbgs()
        << "\n  weight addr " << llvm::format_hex(weight_.addr, 10)
        << "(offset=" << llvm::format_hex(weight_.addrOffset, 10)
        << ", poss");
    dumpDims(weight_.poss);
    LLVM_DEBUG(llvm::dbgs() << ", shape");
    dumpDims(weight_.shapes);
    LLVM_DEBUG(llvm::dbgs()
        << "\n  bias addr " << llvm::format_hex(bias_.addr, 10)
        << "(offset=" << llvm::format_hex(bias_.addrOffset, 10)
        << "), poss");
    dumpDims(bias_.poss);
    LLVM_DEBUG(llvm::dbgs() << ", shapes");
    dumpDims(bias_.shapes);
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  GmInfo output_;
  GmInfo input_;
  GmInfo weight_;
  GmInfo bias_;
};

struct Conv_ARGS {
  gaddr_t ga_ifmap;
  gaddr_t ga_ofmap;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int groups;
  int output_c;
  uint16_t kh;
  uint16_t kw;
  uint16_t dilation_h;
  uint16_t dilation_w;
  uint8_t pad_top;
  uint8_t pad_bottom;
  uint8_t pad_left;
  uint8_t pad_right;
  uint8_t insert_h;
  uint8_t insert_w;
  uint8_t stride_h;
  uint8_t stride_w;
  bool do_bias;
  bool do_activation;
  float *activation_arg;
  int activation_gt_scale;
  int activation_gt_rshift;
  int activation_le_scale;  // slope; TODO
  int activation_le_rshift;
  int right_shift_width;
  bool do_chl_quan;
  uint32_t layer_id;
  bool do_ic_alignment;
  bool store_compr_act;
  bool load_compr_act;
  bool fused_conv_relu;
  bool do_leaky_relu;
  cvk_fmt_t input_fmt;
  cvk_fmt_t output_fmt;
  cvk_fmt_t tiu_fmt;
  uint8_t gm_input_region;
  uint8_t gm_output_region;
  uint8_t gm_activation_region;
  uint8_t gm_weight_region;
  bool ps32_output;
  int pad_value;
};

typedef struct {
  int n;
  int oc;
  int ic;
  int h;
  int w;
  uint32_t n_step;
  uint32_t oc_step;
  uint32_t oh_step;
  uint32_t ow_step;
  uint32_t ih_step;
  uint32_t iw_step;
  uint32_t ic_step;
  uint32_t total_needed;
} SLICES;

//
// We use the local memory to determine the tiled size in both global and local
// memory.
// Then we split the output in global memory and use it to derive:
//   1. tiled ouput size, position in global memory for tmda load
//   2. tiled output size, position in local memory for tdma load
//   3. tiled input size, position in global memory for tdma load
//   4. tiled input size, position in local memory for tdma load
//   5. tiled weight size, position in global memory for tdma load
//   6. tiled weight size, position in local memory for tdma load
//   5. tiled output size, position in local for tpu computation
//   6. tiled input size, position in local for tpu computation
//   7. tiled ouput size, position in global memory for tmda store
//   8. tiled output size, position in local memory for tdma store
//
//  It is really painful that shape/stride for tdma load, tpu compute, tdma
//  store are not always the same.
//
// 1. Double convolution for odd input channels:
//  input channel:        3
//  weight:               4 (offline modified)
//
//  ifmap lmem alloation: 4
//  ifmap tdma load:      3
//  ifmap tiu:            4
//
//
// 2. Align width for tdma efficiency:
//  input width:            28
//  kernel stride:          2
//
//  ifmap lmem allocation: 28
//  ifmap tdma load:       28
//  ifmap tiu:             27
//
class Conv {
public:
  Conv(const CviBackendContext &ctx) : ctx_(ctx) {
    memset(&args_, 0, sizeof(args_));
    memset(&slices_, 0, sizeof(slices_));
    use_double_buffer_ = false;
  }

  bool determineTileSize(bool useDoubleBuffer);
  bool determinePs32TileSize(bool useDoubleBuffer);
  bool determineDwTileSize(bool useDoubleBuffer);

  void convReuseWeight();
  void convReuseActivation();
  void dwConv();

  bool canNoTile();
  void convNoTile();

  void convNaive();

  void initializeGlobalMemInput();
  void initializeGlobalMemOutput();
  void initializeGlobalMemWeight();
  void initializeGlobalBias();
  void initializeGlobalMem();

  void initializeFusedActivation();
  void initializeTile();
  void determineTilePolicy();
  void doConvByTilePolicy();

  uint32_t getElementTypeSize(cvk_fmt_t fmt);

  void allocateTiledLocalMem(
      std::vector<std::unique_ptr<LocalMemoryDescriptor>> &lmDescs,
      uint32_t count, std::vector<uint32_t> shapes, uint32_t eu_align);
  void allocateLocalMemOfInput();
  void deallocateLocalMemOfInput();
  void allocateLocalMemOfOutput();
  void deallocateLocalMemOfOutput();
  void allocateLocalMemOfWeight();
  void deallocateLocalMemOfWeight();
  void allocateLocalMemOfBias();
  void deallocateLocalMemOfBias();
  void allocateLocalMemOfFusedActivation();
  void deallocateLocalMemOfFusedActivation();
  void allocateAllLocalMem();
  void deallocateAllLocalMem();

  std::vector<uint32_t> getTiledShapesForLmAllocationOfInput();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfOuput();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfWeight();
  std::vector<uint32_t> getTiledShapesForLmAllocationOfBias();
  uint32_t getTiledEuAlignForLmAllocationOfInput();
  uint32_t getTiledEuAlignForLmAllocationOfOutput();
  uint32_t getTiledEuAlignForLmAllocationOfWeight();
  uint32_t getTiledEuAlignForLmAllocationOfBias();

  std::vector<uint32_t> getTiledGmShapesOfWeightForTdmaLoad(
      std::vector<uint32_t> gmOutputPoss, uint32_t icPos);
  std::vector<uint32_t> getTiledLmShapesOfWeightForTiu(
      std::vector<uint32_t> gmOutputPoss, uint32_t icPos);

  void getTiledGmPossAndShapesOfInputForTiu(
      std::vector<uint32_t> gmOutputPoss,
      std::vector<uint32_t> gmOutputPossShapes,
      std::vector<uint32_t> &cur_gm_input_poss,
      std::vector<uint32_t> &cur_gm_input_shapes,
      std::vector<uint32_t> &cur_gm_input_paddings,
      uint32_t ic_pos);

  std::vector<uint32_t> getTiledGmShapesOfBiasForTdmaLoad(
      std::vector<uint32_t> gmOutputPoss);
  std::vector<uint32_t> getTiledLmShapesOfBiasForTiu(
      std::vector<uint32_t> gmOutputPoss);

  std::vector<uint32_t> getTiledGmShapesOfOutputForTiu(
      std::vector<uint32_t> gmOutputPoss);

  void fillConstantLmInput(cvk_tl_t *lmLoad,
      std::vector<uint32_t> &cur_gm_input_paddings);
  void adjustComputeForPadOnlyInput(cvk_tl_t *lmInput,
      std::vector<uint32_t> &cur_gm_input_paddings);

  void loadBias(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
      uint32_t cmdQueueIndex);
  void loadWeight(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
      uint32_t cmdQueueIndex, uint32_t icPos = 0);
  void loadInput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
      uint32_t cmdQueueIndex, uint32_t ic_pos = 0);
  void computeConv(
      cvk_tl_t *tl_output, cvk_tl_t *tl_input, cvk_tl_t *tl_weight,
      cvk_tl_t *tl_bias, std::vector<uint32_t> &cur_gm_input_paddings,
      uint8_t intraCmdParal, uint32_t icPos = 0);
  void computePerTensorConv(
      cvk_tl_t *tl_output, cvk_tl_t *tl_input, cvk_tl_t *tl_weight,
      cvk_tl_t *tl_bias, std::vector<uint32_t> &cur_gm_input_paddings,
      uint8_t intraCmdParal, uint32_t icPos = 0);
  void computeLeakyRelu(cvk_tl_t *tl_output);
  void compute(std::vector<uint32_t> gmOutputPoss,
      std::vector<uint32_t> lmIndexes, uint32_t cmdQueueIndex,
      uint32_t icPos = 0);
  void storeOutput(std::vector<uint32_t> gmOutputPoss, uint32_t lmIndex,
      uint32_t cmdQueueIndex);

  uint32_t getPs32Mode(uint32_t icPos);
  bool getReluAllowed(uint32_t icPos);
  bool getBiasAllowed(uint32_t icPos);
  bool getRshiftAllowed(uint32_t icPos);

  bool isDwConv();
  bool isConvPs32();

  void generateCmd();

  // CMODEL Debug
  void configCModelDebug();

  uint32_t batch_size() {
    return args_.input_n;
  }

  uint32_t input_height() {
    return args_.input_h;
  }

  uint32_t input_width() {
    return args_.input_w;
  }

  uint32_t insert_height() {
    return args_.insert_h;
  }

  uint32_t insert_width() {
    return args_.insert_w;
  }

  uint32_t inserted_input_height() {
    return args_.input_h + (args_.input_h - 1) * args_.insert_h;
  }

  uint32_t inserted_input_width() {
    return args_.input_w + (args_.input_w - 1) * args_.insert_w;
  }

  uint32_t groups() {
    return args_.groups;
  }

  uint32_t group_input_channels() {
    return args_.input_c / args_.groups;
  }

  uint32_t group_output_channels() {
    return args_.output_c / args_.groups;
  }

  uint32_t kernel_height() {
    return args_.kh;
  }

  uint32_t kernel_width() {
    return args_.kw;
  }

  uint32_t dilation_height() {
    return args_.dilation_h;
  }

  uint32_t dilation_width() {
    return args_.dilation_w;
  }

  uint32_t padding_top() {
    return args_.pad_top;
  }

  uint32_t padding_bottom() {
    return args_.pad_bottom;
  }

  uint32_t padding_left() {
    return args_.pad_left;
  }

  uint32_t padding_right() {
    return args_.pad_right;
  }

  int pad_value(){
    return args_.pad_value;
  }

  uint32_t subsampling_height() {
    assert(args_.stride_h >= 1);
    return args_.stride_h;
  }

  uint32_t subsampling_width() {
    assert(args_.stride_w >= 1);
    return args_.stride_w;
  }

  uint32_t dilated_kernel_height() {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  uint32_t dilated_kernel_width() {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  uint32_t output_height() {
    uint32_t padded_input_height = padding_top() + inserted_input_height() +
                                   padding_bottom();
    return (padded_input_height - dilated_kernel_height()) /
              subsampling_height() + 1;
  }

  uint32_t output_width() {
    uint32_t padded_input_width = padding_left() + inserted_input_width() +
                                  padding_right();
    return (padded_input_width - dilated_kernel_width()) / subsampling_width()
              + 1;
  }

  uint32_t getNpuNum() {
    return ctx_.cvi_chip_info_context(CVI_CHIP_LANE_NUM);
  }

  uint32_t getEuNum() {
    return static_cast<uint32_t>(ctx_.cvi_chip_info_context(CVI_CHIP_EU_NUM));
  }

  uint32_t getLmSizePerLane() {
    return static_cast<uint32_t>(
        ctx_.cvi_chip_info_context(CVI_CHIP_LMEM_SIZE));
  }

  uint32_t getMaxBatchOfHardware() {
    // 12bit, max 4095-32(lanes)
    return 4095 - 32;
  }

  uint32_t getMaxHeightOfHardware() {
    // 12bit, max 4095-32(lanes)
    return 4095 - 32;
  }

  uint32_t getMaxWidthOfHardware() {
    // 12bit, max 4095-32(lanes)
    return 4095 - 32;
  }

  uint32_t getMaxChannelOfHardware() {
    // 12bit, max 4095-32(lanes)
    return 4095 - 32;
  }

  // Arguments from dialect
  Conv_ARGS args_;

private:
  const CviBackendContext &ctx_;

  SLICES slices_;
  bool use_double_buffer_;

  enum TilePolicy {
    NoTilePolicyType,
    SingleBufferPolicyType,
    SingleBufferPs32PolicyType,
    ReuseWeightPolicyType,
    ReuseActivationPolicyType,
    MaxTilePolicyType,
  };
  TilePolicy tilePolicy_;

  // Global memory descriptor
  std::unique_ptr<GlobalMemoryDescriptor> gmInputDesc_;
  std::unique_ptr<GlobalMemoryDescriptor> gmOutputDesc_;
  std::unique_ptr<GlobalMemoryDescriptor> gmWeightDesc_;
  std::unique_ptr<GlobalMemoryDescriptor> gmBiasDesc_;

  // Local memory descriptor
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmInputDescs_;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmOutputDescs_;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmWeightDescs_;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmBiasDescs_;
  std::vector<std::unique_ptr<LocalMemoryDescriptor>> lmFusedActDescs_;

  // Collection of tiled commands
  std::vector<std::unique_ptr<CmdDescriptor>> cmdQueue_;

  CModelDebug cModelDebug_ = {0};
};

// Input data layout (N, C, H, W) => (N, G, C, H, W)
void Conv::initializeGlobalMemInput() {
  // Actual physical layout
  // not logical layout (e.g. do_ic_alignment)
  std::vector<uint32_t> shapes = {
      batch_size(), groups(), group_input_channels(), input_height(),
      input_width()};
  gmInputDesc_ = std::make_unique<GlobalMemoryDescriptor>(ctx_, shapes,
                                                          args_.input_fmt);
  gmInputDesc_->setLayerId(args_.layer_id);
  gmInputDesc_->setMemRegion(args_.gm_input_region);
  gmInputDesc_->setAddress(args_.ga_ifmap);

  if (args_.load_compr_act)
    gmInputDesc_->setCompressed(true);
}

// Output data layout (N, C, H, W) => (N, G, C, H, W)
void Conv::initializeGlobalMemOutput() {
  std::vector<uint32_t> shapes = {
      batch_size(), groups(), group_output_channels(), output_height(),
      output_width()};
  gmOutputDesc_ = std::make_unique<GlobalMemoryDescriptor>(ctx_, shapes,
                                                           args_.output_fmt);
  gmOutputDesc_->setLayerId(args_.layer_id);
  gmOutputDesc_->setMemRegion(args_.gm_output_region);
  gmOutputDesc_->setAddress(args_.ga_ofmap);

  if (args_.store_compr_act)
    gmOutputDesc_->setCompressed(true);
}

// Weight data layout (Og, Oc, Kh*Kw, Ic) => (1, Og, Oc, Kh*Kw, Ic)
void Conv::initializeGlobalMemWeight() {
  uint32_t input_c = group_input_channels();

  // Physical layout
  // weight is already altered for do_ic_alignment
  // do_ic_alignment is not applied in depthwise convolution.
  input_c =
      !isDwConv() && args_.do_ic_alignment ? align_up(input_c, 2) : input_c;

  std::vector<uint32_t> shapes = {
      1, groups(), group_output_channels(), kernel_height() * kernel_width(),
      input_c};
  gmWeightDesc_ = std::make_unique<GlobalMemoryDescriptor>(ctx_, shapes,
                                                           args_.tiu_fmt);
  gmWeightDesc_->setLayerId(args_.layer_id);
  gmWeightDesc_->setMemRegion(args_.gm_weight_region);
  gmWeightDesc_->setAddress(args_.ga_weight);
}

// Bias data layout
//   Per-channel: (1, Og, Oc, 1, [9/5])
//   Per-tensor:  (2, Og, Oc, 1, 1)
void Conv::initializeGlobalBias() {
  if (!args_.do_chl_quan && !args_.do_bias)
    return;

  std::vector<uint32_t> shapes;
  if (args_.do_chl_quan) {
    uint32_t perchannel_size = ctx_.chan_quan_param_size(args_.do_bias);
    shapes = {1, groups(), group_output_channels(), 1, perchannel_size};
  } else {
    shapes = {2, groups(), group_output_channels(), 1, 1};
  }

  gmBiasDesc_ = std::make_unique<GlobalMemoryDescriptor>(ctx_, shapes,
                                                         args_.tiu_fmt);
  gmBiasDesc_->setLayerId(args_.layer_id);
  gmBiasDesc_->setMemRegion(args_.gm_weight_region);
  gmBiasDesc_->setAddress(args_.ga_bias);
}

void Conv::initializeGlobalMem() {
  initializeGlobalMemInput();
  initializeGlobalMemOutput();
  initializeGlobalMemWeight();
  initializeGlobalBias();
}

void Conv::initializeTile() {
  slices_.n = 1;
  slices_.oc = 1;
  slices_.ic = 1;
  slices_.h = 1;
  slices_.w = 1;
  slices_.n_step = batch_size();
  slices_.oc_step = group_output_channels();
  slices_.oh_step = output_height();
  slices_.ow_step = output_width();
  slices_.ih_step = input_height();
  slices_.iw_step = input_width();
  slices_.ic_step = group_input_channels();

  use_double_buffer_ = true;
}

void Conv::initializeFusedActivation() {
  // Check conv+relu or conv+leaky relu
  args_.fused_conv_relu = false;
  args_.do_leaky_relu = false;
  if (args_.do_activation) {
    if (!args_.activation_arg || args_.activation_arg[0] == 0.0f)
      args_.fused_conv_relu = true;
    else
      args_.do_leaky_relu = true;
  }
}

uint32_t Conv::getElementTypeSize(cvk_fmt_t fmt) {
  switch (fmt) {
    case CVK_FMT_F32:
      return 4;
    case CVK_FMT_BF16:
      return 2;
    default:
      return 1;
  }
}

void Conv::allocateTiledLocalMem(
    std::vector<std::unique_ptr<LocalMemoryDescriptor>> &lmDescs,
    uint32_t count, std::vector<uint32_t> shapes, uint32_t eu_align) {
  assert(shapes.size() == 5 && "Expect 5D tensor");

  for (uint32_t i = 0; i < count; ++i) {
    lmDescs.push_back(
        std::make_unique<LocalMemoryDescriptor>(ctx_, shapes, args_.tiu_fmt,
                                                eu_align));
    lmDescs.back()->setLayerId(args_.layer_id);
    lmDescs.back()->allocate();
  }

  assert(lmDescs.size() == count && "Expect all allocated");
}

// Shape (tiledN, 1, IC/g, tiledIH, tiledIW)
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfInput() {
  std::vector<uint32_t> shapes = {
    slices_.n_step, 1, slices_.ic_step, slices_.ih_step,
    slices_.iw_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfInput() {
  return 1; // aligned
}

void Conv::allocateLocalMemOfInput() {
  uint32_t count = use_double_buffer_ ? 2 : 1;

  allocateTiledLocalMem(lmInputDescs_, count,
                        getTiledShapesForLmAllocationOfInput(),
                        getTiledEuAlignForLmAllocationOfInput());
}

void Conv::deallocateLocalMemOfInput() {
  if (use_double_buffer_)
    lmInputDescs_[1]->free();

  lmInputDescs_[0]->free();
}

// Shape (tiledN, 1, Oc/g, tiledOH, tiledOW)
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfOuput() {
  uint32_t ofmapSizeMultiplier
      = (slices_.ic_step < group_input_channels()) ? 4 : 1;

  std::vector<uint32_t> shapes = {
      slices_.n_step * ofmapSizeMultiplier, 1, slices_.oc_step,
      slices_.oh_step, slices_.ow_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfOutput() {
  return 1; // aligned
}

void Conv::allocateLocalMemOfOutput() {
  uint32_t count = use_double_buffer_ ? 2 : 1;
  allocateTiledLocalMem(lmOutputDescs_, count,
      getTiledShapesForLmAllocationOfOuput(),
      getTiledEuAlignForLmAllocationOfOutput());
}

void Conv::deallocateLocalMemOfOutput() {
  if (use_double_buffer_)
    lmOutputDescs_[1]->free();

  lmOutputDescs_[0]->free();
}

// H/W does not support group convolution.
// S/W handles one group at once.
//
// Shape (1, 1, tiledOc, kh*kw, ic)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t> Conv::getTiledGmShapesOfWeightForTdmaLoad(
    std::vector<uint32_t> gmOutputPoss, uint32_t icPos) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t cur_oc = std::min(group_output_channels() - oc_pos, slices_.oc_step);
  uint32_t cur_ic = std::min(group_input_channels() - icPos, slices_.ic_step);

  std::vector<uint32_t> tiledShapes = {
      1, 1, cur_oc, kernel_height() * kernel_width(), cur_ic};

  return tiledShapes;
}

// TIU shape != tdma shape
//   TDMA shapes (1, 1, tiledOc, kh*kw, ic)
//   TIU shapes  (1, ic, tiledOc, kh, kw)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t> Conv::getTiledLmShapesOfWeightForTiu(
    std::vector<uint32_t> gmOutputPoss, uint32_t icPos) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t cur_oc = std::min(group_output_channels() - oc_pos, slices_.oc_step);
  uint32_t cur_ic = std::min(group_input_channels() - icPos, slices_.ic_step);

  std::vector<uint32_t> shapes = {
      1, cur_ic, cur_oc, kernel_height(), kernel_width()};

  return shapes;
}

// Shape(1, 1, tiledOc, Kh*Kw, Ic)
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfWeight() {
  std::vector<uint32_t> shapes = {
      1, 1, slices_.oc_step, kernel_height() * kernel_width(),
      slices_.ic_step};

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfWeight() {
  return 0; // Not aligned
}

void Conv::allocateLocalMemOfWeight() {
  uint32_t count = use_double_buffer_ ? 2 : 1;

  allocateTiledLocalMem(lmWeightDescs_, count,
                        getTiledShapesForLmAllocationOfWeight(),
                        getTiledEuAlignForLmAllocationOfWeight());
}

void Conv::deallocateLocalMemOfWeight() {
  if (use_double_buffer_)
    lmWeightDescs_[1]->free();

  lmWeightDescs_[0]->free();
}

// Per-channel: (1, 1, tiled_oc, 1, [5/9])
//   w/  bias: bias(4) + multiplier(4) + shift(1)
//   w/o bias: multiplier(4) + shift(1)
// Per-tensor:  (2, 1, tiled_oc, 1, 1)
std::vector<uint32_t> Conv::getTiledShapesForLmAllocationOfBias() {
  std::vector<uint32_t> shapes;

  if (args_.do_chl_quan) {
    uint32_t perchannel_size = ctx_.chan_quan_param_size(args_.do_bias);
    shapes = {1, 1, slices_.oc_step, 1, perchannel_size};
  } else if (args_.do_bias) {
    shapes = {2, 1, slices_.oc_step, 1, 1};
  }

  return shapes;
}

uint32_t Conv::getTiledEuAlignForLmAllocationOfBias() {
  return 0; // Not aligned
}

void Conv::allocateLocalMemOfBias() {
  if (args_.do_chl_quan || args_.do_bias) {
    uint32_t count = use_double_buffer_ ? 2 : 1;
    allocateTiledLocalMem(lmBiasDescs_, count,
                          getTiledShapesForLmAllocationOfBias(),
                          getTiledEuAlignForLmAllocationOfBias());
  }
}

void Conv::deallocateLocalMemOfBias() {
  if (args_.do_chl_quan || args_.do_bias) {
    if (use_double_buffer_)
      lmBiasDescs_[1]->free();

    lmBiasDescs_[0]->free();
  }
}

void Conv::allocateLocalMemOfFusedActivation() {
  if (args_.do_leaky_relu) {
    // Leaky relu needs two local memory for tl_reg, tl_relu
    // Same setting as output
    allocateTiledLocalMem(lmFusedActDescs_, 2,
        getTiledShapesForLmAllocationOfOuput(),
        getTiledEuAlignForLmAllocationOfOutput());
  }
}

void Conv::deallocateLocalMemOfFusedActivation() {
  if (args_.do_leaky_relu) {
    // tl_reg, tl_relu, not double buffer
    lmFusedActDescs_[1]->free();
    lmFusedActDescs_[0]->free();
  }
}

void Conv::allocateAllLocalMem() {
  allocateLocalMemOfWeight();
  allocateLocalMemOfInput();
  allocateLocalMemOfOutput();
  allocateLocalMemOfBias();
  allocateLocalMemOfFusedActivation();
}

void Conv::deallocateAllLocalMem() {
  //
  // Release resource in reverse order
  //
  deallocateLocalMemOfFusedActivation();
  deallocateLocalMemOfBias();
  deallocateLocalMemOfOutput();
  deallocateLocalMemOfInput();
  deallocateLocalMemOfWeight();
}

// H/W does not support group convolution.
// S/W handles one group at once.
//
// Bias data layout
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t> Conv::getTiledGmShapesOfBiasForTdmaLoad(
    std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t cur_oc = std::min(group_output_channels() - oc_pos, slices_.oc_step);

  // TDMA shapes same as allocation except group fixed to 1
  std::vector<uint32_t> shapes = {
    gmBiasDesc_->getShapes()[NGCHW::N], /*og=*/1, cur_oc, 1,
    gmBiasDesc_->getShapes()[NGCHW::W]};

  return shapes;
}

// Bias data layout
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// But
//   TIU per-channel: (1, 1, tiledOc, 1, 1)
//   TIU per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t> Conv::getTiledLmShapesOfBiasForTiu(
    std::vector<uint32_t> gmOutputPoss) {
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  uint32_t cur_oc = std::min(group_output_channels() - oc_pos, slices_.oc_step);

  std::vector<uint32_t> shapes = {
    gmBiasDesc_->getShapes()[NGCHW::N], 1, cur_oc, 1, 1};

  return shapes;
}

// Bias shape
//   Per-channel: (1, 1, tiledOc, 1, [9/5])
//   Per-tensor:  (2, 1, tiledOc, 1, 1)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
void Conv::loadBias(std::vector<uint32_t> gmOutputPoss,
    uint32_t lmIndex, uint32_t cmdQueueIndex) {
  if (!args_.do_chl_quan && !args_.do_bias)
    return;

  uint32_t ig_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];

  // Global memory
  std::vector<uint32_t> gm_shapes =
      getTiledGmShapesOfBiasForTdmaLoad(gmOutputPoss);
  std::vector<uint32_t> tiled_cur_poss = {0, ig_pos, oc_pos, 0, 0};

  uint64_t ga_offset = gmBiasDesc_->getCurrentOffset(tiled_cur_poss);
  uint64_t ga_load = gmBiasDesc_->getAddress() + ga_offset;
  cvk_tg_stride_t gm_stride = {
      gmBiasDesc_->getStrides()[NGCHW::N], gmBiasDesc_->getStrides()[NGCHW::C],
      gmBiasDesc_->getStrides()[NGCHW::H]};

  // Local memory
  cvk_tl_t tl_bias;
  cvk_tl_shape_t tl_bias_shape = {
      gm_shapes[NGCHW::N], gm_shapes[NGCHW::C], gm_shapes[NGCHW::H],
      gm_shapes[NGCHW::W]};
  ctx_.lmem_init_tensor(&tl_bias, tl_bias_shape,
                        lmBiasDescs_[lmIndex]->getDataFormat(),
                        lmBiasDescs_[lmIndex]->getEuAlign());
  tl_bias.start_address = lmBiasDescs_[lmIndex]->getAddress();

  LLVM_DEBUG(llvm::dbgs()
      << "\n  [ig=" << ig_pos << "][oc_pos=" << oc_pos
      << "] tdma_load_stride:\n"
      << "    new tl_bias gaddr " << llvm::format_hex(ga_load, 10)
      << "(offset=" << llvm::format_hex(ga_offset, 10)
      << "), laddr "
      << llvm::format_hex(tl_bias.start_address, 10)
      << ", shape (" << tl_bias.shape.n
      << ", " << tl_bias.shape.c
      << ", " << tl_bias.shape.h
      << ", " << tl_bias.shape.w
      << "), gstride (" << gm_stride.n << ", " << gm_stride.c
      << ", " << gm_stride.h
      << ")\n\n");

  if (args_.tiu_fmt == CVK_FMT_I8)
    ctx_.tdma_load_stride(&tl_bias, ga_load, gm_stride);
  else if (args_.tiu_fmt == CVK_FMT_BF16)
    ctx_.tdma_load_stride(&tl_bias, ga_load, gm_stride);
  else {
    assert(0 && "Bias only supports i8/bf16");
  }

  cModelDebug_.recordBias(args_.layer_id, gmOutputPoss, ga_load, ga_offset,
      tiled_cur_poss, gm_shapes);
}

// Weight shape (1, 1, tiledOc, kh*kw, ic)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
void Conv::loadWeight(std::vector<uint32_t> gmOutputPoss,
    uint32_t lmIndex, uint32_t cmdQueueIndex, uint32_t icPos) {
  uint32_t ig_pos = gmOutputPoss[NGCHW::G];
  uint32_t oc_pos = gmOutputPoss[NGCHW::C];
  assert(group_output_channels() > oc_pos && "Expect valid tiled weight");
  assert(group_output_channels() >= slices_.oc_step &&
         "Expect valid tiled weight");

  std::vector<uint32_t> tiled_shapes =
      getTiledGmShapesOfWeightForTdmaLoad(gmOutputPoss, icPos);
  std::vector<uint32_t> tiled_cur_poss = {0, ig_pos, oc_pos, 0, icPos};

  // Need abstraction for tiled local/global memory
  // Need abstraction for tdma load

  // Global memory
  uint64_t ga_offset = gmWeightDesc_->getCurrentOffset(tiled_cur_poss);
  cvk_tg_t ts_data = {0};
  ts_data.base_reg_index =
      ctx_.getTdmaBaseSelectIndexFromGaddr(gmWeightDesc_->getAddress());
  ts_data.start_address = gmWeightDesc_->getAddress() + ga_offset;
  ts_data.fmt = gmWeightDesc_->getDataFormat();
  ts_data.shape = {
      tiled_shapes[NGCHW::N], tiled_shapes[NGCHW::C], tiled_shapes[NGCHW::H],
      tiled_shapes[NGCHW::W]};
  ts_data.stride = {
      gmWeightDesc_->getStrides()[NGCHW::N],
      gmWeightDesc_->getStrides()[NGCHW::C],
      gmWeightDesc_->getStrides()[NGCHW::H]};

  // Local memory
  cvk_tl_t *tl_allocated_weight = lmWeightDescs_[lmIndex]->getAllocated();

  cvk_tl_shape_t tl_load_shape = {
      tiled_shapes[NGCHW::N], tiled_shapes[NGCHW::C], tiled_shapes[NGCHW::H],
      tiled_shapes[NGCHW::W]};
  cvk_tl_t tl_load_weight;
  ctx_.lmem_init_tensor(&tl_load_weight, tl_load_shape,
      tl_allocated_weight->fmt, tl_allocated_weight->eu_align);
  tl_load_weight.start_address = tl_allocated_weight->start_address;

  uint8_t intraCmdParal = 0;
  if (ctx_.has_cmd_pre_exe() && cmdQueueIndex < cmdQueue_.size())
    intraCmdParal = cmdQueue_[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
  p1.src = &ts_data;
  p1.dst = &tl_load_weight;
  p1.layer_id = args_.layer_id;
  p1.intra_cmd_paral = intraCmdParal;

  LLVM_DEBUG(llvm::errs()
      << "  [ig=" << ig_pos << "][oc_pos=" << oc_pos << "] tdma_load_stride:\n"
      << "    loadWeight tl_weight gaddr "
      << llvm::format_hex(ts_data.start_address, 10)
      << "(offset=" << ga_offset
      << "), laddr " << llvm::format_hex(tl_load_weight.start_address, 10)
      << ", shape (" << tl_load_weight.shape.n
      << ", " << tl_load_weight.shape.c
      << ", " << tl_load_weight.shape.h
      << ", " << tl_load_weight.shape.w
      << "), weight_gstride (" << ts_data.stride.n
      << ", " << ts_data.stride.c
      << ", " << ts_data.stride.h << ")\n"
      << "    intraCmdParal " << (int)intraCmdParal << "\n");

  ctx_.tdma_g2l_tensor_copy(&p1);

  cModelDebug_.recordWeight(args_.layer_id, gmOutputPoss, ts_data.start_address,
      ga_offset, tiled_cur_poss, tiled_shapes);
}

// Calculate the position, shape, padding of tiled input from the tiled output.
// For do_ic_alignment, cur_gm_input_shapes is not physical layout.
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
// padding (top, bottom, left, right)
void Conv::getTiledGmPossAndShapesOfInputForTiu(
    std::vector<uint32_t> gmOutputPoss,
    std::vector<uint32_t> gmOutputPossShapes,
    std::vector<uint32_t> &cur_gm_input_poss,
    std::vector<uint32_t> &cur_gm_input_shapes,
    std::vector<uint32_t> &cur_gm_input_paddings,
    uint32_t icPos) {

  uint32_t g_pos = gmOutputPoss[NGCHW::G];
  uint32_t oh_pos = gmOutputPoss[NGCHW::H];
  uint32_t cur_oh = gmOutputPossShapes[NGCHW::H];
  uint32_t oh_top = oh_pos;
  uint32_t oh_bot = oh_top + cur_oh;
  uint32_t ih_top =
      std::max(int(oh_top * subsampling_height()) - int(padding_top()), 0);
  uint32_t ih_bot =
      std::min((oh_bot - 1) * subsampling_height() + dilated_kernel_height() -
               padding_top(), inserted_input_height());

  ih_top = ceiling_func(ih_top, 1 + insert_height());
  ih_bot = ceiling_func(ih_bot, 1 + insert_height());
  uint32_t cur_ih = ih_bot - ih_top;

  uint32_t ph_top = 0;
  if (ih_top == 0) {
    ph_top = padding_top() - oh_top * subsampling_height();
  } else {
    int gap = (oh_top * subsampling_height() - padding_top()) % (1 + insert_height());
    ph_top = (gap == 0) ? 0 : (1 + insert_height() - gap);
  }

  uint32_t ph_bot = 0;
  if (ih_bot == input_height()) {
    ph_bot = (oh_bot - 1) * subsampling_height() + dilated_kernel_height() -
             padding_top() - inserted_input_height();
  } else {
    ph_bot = (oh_bot - 1) * subsampling_height() + dilated_kernel_height() -
             padding_top() - (ih_bot + (ih_bot - 1) * insert_height());
  }

  uint32_t ow_pos = gmOutputPoss[NGCHW::W]; // NCHW
  uint32_t cur_ow = gmOutputPossShapes[NGCHW::W];  // NCHW
  uint32_t ow_left = ow_pos;
  uint32_t ow_right = ow_left + cur_ow;
  uint32_t iw_left =
      std::max(int(ow_left * subsampling_width()) - int(padding_left()), 0);
  uint32_t iw_right =
      std::min((ow_right - 1) * subsampling_width() + dilated_kernel_width()
               - padding_left(), inserted_input_width());
  iw_left = ceiling_func(iw_left, 1 + insert_width());
  iw_right = ceiling_func(iw_right, 1 + insert_width());
  uint32_t cur_iw = iw_right - iw_left;

  // For better DMA transfer efficiency, use whole width.
  //   E.g.
  //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
  //
  //     input (27, 27) needed, but (27, 28) is better
  if (insert_width() == 0 && cur_iw < slices_.iw_step && subsampling_width() > 1) {
    assert((iw_left + slices_.iw_step) > iw_right);
    //cur_iw = slices_.iw_step;
  }

  uint32_t pw_left = 0;
  if (iw_left == 0) {
    pw_left = padding_left() - ow_left * subsampling_width();
  } else {
    int gap = (ow_left * subsampling_width() - padding_left()) % (1 + insert_width());
    pw_left = (gap == 0) ? 0 : (1 + insert_width() - gap);
  }

  uint32_t pw_right = 0;
  if (iw_right == input_width()) {
    pw_right = (ow_right - 1) * subsampling_width() + dilated_kernel_width() -
               padding_left() - inserted_input_width();
  } else {
    pw_right = (ow_right - 1) * subsampling_width() + dilated_kernel_width() -
               padding_left() - (iw_right + (iw_right - 1) * insert_width());
  }

  uint32_t n_pos = gmOutputPoss[NGCHW::N];
  uint32_t cur_n = gmOutputPossShapes[NGCHW::N];
  uint32_t cur_ic = std::min(group_input_channels() - icPos, slices_.ic_step);
  cur_gm_input_shapes = {cur_n, 1, cur_ic, cur_ih, cur_iw};
  cur_gm_input_poss = {n_pos, g_pos, icPos, ih_top, iw_left};

  // {top, bottom, left, right}
  cur_gm_input_paddings = {ph_top, ph_bot, pw_left, pw_right};

  LLVM_DEBUG(llvm::dbgs()
      << "\n  [ig=" << gmOutputPoss[NGCHW::G]
      << "][oc_pos=" << gmOutputPoss[NGCHW::C]
      << "][n_pos=" << gmOutputPoss[NGCHW::N]
      << "][oh_pos=" << gmOutputPoss[NGCHW::H]
      << "][ow_pos=" << gmOutputPoss[NGCHW::W]
      << "][ic_pos=" << icPos
      << "] cur_oh " << cur_oh << ", cur_ih " << cur_ih
      << ", ih_top " << ih_top << ", ih_bot " << ih_bot
      << ", cur_ow " << cur_ow << ", cur_iw " << cur_iw
      << ", iw_left " << iw_left << ", iw_right " << iw_right
      << ", ph_top " << ph_top << ", ph_bot " << ph_bot
      << ", pw_left " << pw_left << ", pw_right " << pw_right
      << ", oh_top " << oh_top << ", oh_bot " << oh_bot
      << ", ow_left " << ow_left << ", ow_right " << ow_right
      << ", ins_h " << insert_height() << ", ins_w " << insert_width()
      << "\n");
}

// H/W does not support group convolution.
// S/W handles one group at once.
//
// Shape (tiledN, 1, tiledOc, tiledOh, tiledOw)
//
// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
std::vector<uint32_t> Conv::getTiledGmShapesOfOutputForTiu(
    std::vector<uint32_t> gmOutputPoss) {
  std::vector<uint32_t> outputShapes = gmOutputDesc_->getShapes();
  std::vector<uint32_t> tiledOutputSteps = {
    slices_.n_step, 1, slices_.oc_step, slices_.oh_step, slices_.ow_step};

  std::vector<uint32_t> tiledOutputShapes;
  for (uint32_t i = 0; i < tiledOutputSteps.size(); ++i)
    tiledOutputShapes.push_back(
        std::min(outputShapes[i] - gmOutputPoss[i], tiledOutputSteps[i]));

  return tiledOutputShapes;
}

// It is possible that last tile of input only includes padding.
// Fill constant instead of load from global memory.
void Conv::fillConstantLmInput(cvk_tl_t *lmLoad,
    std::vector<uint32_t> &cur_gm_input_paddings) {
  // Use pad bottom as height
  lmLoad->shape.h = cur_gm_input_paddings[1];  // bottom
  lmLoad->stride = ctx_.tl_default_stride(lmLoad->shape, lmLoad->fmt,
                                          lmLoad->eu_align);

  cvk_tdma_g2l_tensor_fill_constant_param_t param = {0};
  param.dst = lmLoad;
  param.layer_id = args_.layer_id;

  if (args_.tiu_fmt == CVK_FMT_I8) {
    param.constant = 0;
  } else {
    param.constant = ctx_.convert_fp32_to_bf16(0.0);
  }
  ctx_.tdma_g2l_tensor_fill_constant(&param);
}

// Adjust input and padding for pad-only input.
void Conv::adjustComputeForPadOnlyInput(cvk_tl_t *lmInput,
    std::vector<uint32_t> &cur_gm_input_paddings) {

  // No need to change if height is non-zero
  if (lmInput->shape.h)
    return;

  // Use pad bottom as height
  // Clear pad bottom.
  lmInput->shape.h = cur_gm_input_paddings[1];
  lmInput->stride = ctx_.tl_default_stride(lmInput->shape, lmInput->fmt,
                                           lmInput->eu_align);
  cur_gm_input_paddings[1] = 0;
}

// Input shape (tiledN, 1, tiledOc, tiledIh, tiledIw)
//
// Calculate input shape from output
// gmOutputPoss shape (n_pos, (ig_pos, oc_pos), oh_pos, ow_pos)
void Conv::loadInput(std::vector<uint32_t> gmOutputPoss,
    uint32_t lmIndex, uint32_t cmdQueueIndex, uint32_t icPos) {
  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);

  std::vector<uint32_t> cur_gm_input_poss;
  std::vector<uint32_t> cur_gm_input_shapes;
  std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
  getTiledGmPossAndShapesOfInputForTiu(
      gmOutputPoss, gmOutputPossShapes, cur_gm_input_poss,
      cur_gm_input_shapes, cur_gm_input_paddings, icPos);

  uint64_t ga_input_offset = gmInputDesc_->getCurrentOffset(cur_gm_input_poss);
  uint64_t ga_input_load = gmInputDesc_->getAddress() + ga_input_offset;

  std::vector<uint32_t> gm_input_strides = gmInputDesc_->getStrides();
  cvk_tg_stride_t cvk_gm_input_stride = {
      gm_input_strides[NGCHW::N], gm_input_strides[NGCHW::C],
      gm_input_strides[NGCHW::H]};

  if (gmInputDesc_->getCompressed()) {
    // load compressed input
    assert(ga_input_load == gmInputDesc_->getAddress() &&
           cur_gm_input_shapes[0] == gmInputDesc_->getShapes()[0] &&
           cur_gm_input_shapes[1] == gmInputDesc_->getShapes()[1] &&
           cur_gm_input_shapes[2] == gmInputDesc_->getShapes()[2] &&
           cur_gm_input_shapes[3] == gmInputDesc_->getShapes()[3] &&
           cur_gm_input_shapes[4] == gmInputDesc_->getShapes()[4] &&
           "Expect no tiling for compressed input");

    cvk_cmpr_tg_t ts_data = {0};
    ts_data.t.base_reg_index =
        ctx_.getTdmaBaseSelectIndexFromGaddr(gmInputDesc_->getAddress());
    ts_data.t.start_address = ga_input_load;
    ts_data.t.fmt = gmInputDesc_->getDataFormat();
    ts_data.t.shape = {
        cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
        cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
    ts_data.t.stride = cvk_gm_input_stride;

    cvk_tdma_g2l_tensor_copy_decompressed_param_t param = {0};
    param.src = &ts_data;
    param.dst = lmInputDescs_[lmIndex]->getAllocated();

    if (args_.tiu_fmt == CVK_FMT_I8)
      ctx_.tdma_g2l_tensor_copy_decompressed(&param);
    else {
      assert(0 && "compressed input only supports i8");
    }
  } else {
    // Load uncompressed input
    cvk_tl_shape_t tl_shape = {
        cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
        cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
    cvk_tl_t tl_load;
    ctx_.lmem_init_tensor(&tl_load, tl_shape,
                          lmInputDescs_[lmIndex]->getDataFormat(),
                          lmInputDescs_[lmIndex]->getEuAlign());
    tl_load.start_address = lmInputDescs_[lmIndex]->getAddress();

    // Input is not altered, use actual shape/stride.
    if (args_.do_ic_alignment) {
      tl_load.shape.c -= 1;
      tl_load.stride = ctx_.tl_default_stride(tl_load.shape, tl_load.fmt,
                                              tl_load.eu_align);
    }

  LLVM_DEBUG(llvm::dbgs()
      << "\n  [ig=" << gmOutputPoss[NGCHW::G]
      << "][oc_pos=" << gmOutputPoss[NGCHW::C]
      << "][n_pos=" << gmOutputPoss[NGCHW::N]
      << "][oh_pos=" << gmOutputPoss[NGCHW::H]
      << "][ow_pos=" << gmOutputPoss[NGCHW::W]
      << "][ic_pos=" << icPos
      << "] new tdma_load_stride:\n"
      << "    tl_ifmap gaddr " << llvm::format_hex(ga_input_load, 10)
      << "(offset=" << llvm::format_hex(ga_input_offset, 10)
      << "), laddr "<< llvm::format_hex(tl_load.start_address, 10)
      << ", shape (" << tl_load.shape.n
      << ", " << tl_load.shape.c
      << ", " << tl_load.shape.h
      << ", " << tl_load.shape.w
      << "), gm_input_stride (" << cvk_gm_input_stride.n
      << ", " << cvk_gm_input_stride.c
      << ", " << cvk_gm_input_stride.h
      << ")\n\n");

    if ((args_.input_fmt == CVK_FMT_I8) && (args_.tiu_fmt == CVK_FMT_I8)) {
      if (tl_load.shape.h)
        ctx_.tdma_load_stride(&tl_load, ga_input_load, cvk_gm_input_stride);
      else
        fillConstantLmInput(&tl_load, cur_gm_input_paddings);
    } else if ((args_.input_fmt == CVK_FMT_BF16) &&
            (args_.tiu_fmt == CVK_FMT_BF16)) {
      if (tl_load.shape.h)
        ctx_.tdma_load_stride(&tl_load, ga_input_load,
                                   cvk_gm_input_stride);
      else
        fillConstantLmInput(&tl_load, cur_gm_input_paddings);
    } else {
      assert(0 && "Input only supports i8/bf16");
    }
  }

  bool ignoreOutputChannel = false;
  if ((tilePolicy_ == ReuseActivationPolicyType) ||
      (tilePolicy_ == SingleBufferPolicyType))
    ignoreOutputChannel = true;

  cModelDebug_.recordInput(args_.layer_id, gmOutputPoss, ga_input_load,
      ga_input_offset, cur_gm_input_poss, cur_gm_input_shapes,
      ignoreOutputChannel);
}

uint32_t Conv::getPs32Mode(uint32_t icPos) {
  // Normal mode
  if (slices_.ic_step == group_input_channels())
    return 0;

  // write 32b result at the first time
  if (icPos == 0)
    return 2;

  // load previous 32b result
  if ((icPos + slices_.ic_step) >= group_input_channels())
    return 1;

  // init & write 32bits partial sum
  return 3;
}

bool Conv::getReluAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  bool reluAllowed = ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
  return (args_.fused_conv_relu && reluAllowed);
}

bool Conv::getBiasAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  bool biasAllowed = ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
  return ((args_.do_bias || args_.do_chl_quan) && biasAllowed);
}

bool Conv::getRshiftAllowed(uint32_t icPos) {
  uint32_t ps32Mode = getPs32Mode(icPos);
  return ((ps32Mode == 0) || (ps32Mode == 1)) ? true : false;
}

void Conv::computeConv(
  cvk_tl_t *tl_output, cvk_tl_t *tl_input, cvk_tl_t *tl_weight,
  cvk_tl_t *tl_bias, std::vector<uint32_t> &cur_gm_input_paddings,
  uint8_t intraCmdParal, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);

  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.chl_quan_param = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.has_bias = getBiasAllowed(icPos) ? args_.do_bias : 0;
  param.relu_enable = getReluAllowed(icPos);
  param.ps32_mode = getPs32Mode(icPos);
  param.w_is_const = 0;
  param.layer_id = args_.layer_id;
  param.cmd_pre_exe_typ = intraCmdParal ? 1 : 0;  // wait weight
  param.cmd_pre_exe = intraCmdParal ? 3 : 0;      // load and store
                                                  // pre exec
  param.ins_val = pad_value();                      // symmetric quantization
  param.ins_fp = ctx_.convert_fp32_to_bf16((float)pad_value()); // symmetric quantization
  ctx_.tiu_convolution(&param);
}

void Conv::computePerTensorConv(
    cvk_tl_t *tl_output, cvk_tl_t *tl_input, cvk_tl_t *tl_weight,
    cvk_tl_t *tl_bias, std::vector<uint32_t> &cur_gm_input_paddings,
    uint8_t intraCmdParal, uint32_t icPos) {

  adjustComputeForPadOnlyInput(tl_input, cur_gm_input_paddings);

  // Both relu and bias used in no ps32 mode or last stage of ps32 mode.
  cvk_tiu_pt_convolution_param_t param = {0};
  param.ofmap = tl_output;
  param.ifmap = tl_input;
  param.weight = tl_weight;
  param.bias = getBiasAllowed(icPos) ? tl_bias : nullptr;
  param.ins_h = (tl_input->shape.h > 1) ? insert_height() : 0;
  param.ins_w = (tl_input->shape.w > 1) ? insert_width() : 0;
  param.ins_last_h = 0;
  param.ins_last_w = 0;
  param.pad_top = cur_gm_input_paddings[0];
  param.pad_bottom = cur_gm_input_paddings[1];
  param.pad_left = cur_gm_input_paddings[2];
  param.pad_right = cur_gm_input_paddings[3];
  param.stride_h = subsampling_height();
  param.stride_w = subsampling_width();
  param.dilation_h = dilation_height();
  param.dilation_w = dilation_width();
  param.relu_enable = getReluAllowed(icPos) ? 1 : 0;
  param.rshift_bits = getRshiftAllowed(icPos) ? args_.right_shift_width : 0;
  param.ps32_mode = getPs32Mode(icPos);
  param.w_is_const = 0;
  param.layer_id = args_.layer_id;
  param.cmd_pre_exe_typ = intraCmdParal ? 1 : 0;  // wait weight
  param.cmd_pre_exe = intraCmdParal ? 3 : 0;      // load and store
                                                  // pre exec
  param.ins_val = pad_value();                      // symmetric quantization
  param.ins_fp = ctx_.convert_fp32_to_bf16(float(pad_value())); // symmetric quantization

  ctx_.tiu_pt_convolution(&param);
}

void Conv::computeLeakyRelu(cvk_tl_t *tl_output) {
  cvk_tl_t tl_neg;
  ctx_.lmem_init_tensor(&tl_neg, tl_output->shape, tl_output->fmt,
                        tl_output->eu_align);
  tl_neg.start_address = lmFusedActDescs_[0]->getAddress();

  cvk_tl_t tl_relu;
  ctx_.lmem_init_tensor(&tl_relu, tl_output->shape, tl_output->fmt,
                        tl_output->eu_align);
  tl_relu.start_address = lmFusedActDescs_[1]->getAddress();

  bool isIgnorePosPart = (args_.activation_gt_scale == 0);
  bool isSlopeSmallerThanOne =
      ((args_.activation_le_scale >> args_.activation_le_rshift) == 0);

  if (isIgnorePosPart) {
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_relu;
    p4.a = tl_output;
    p4.b_const.val = args_.activation_le_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = args_.activation_le_rshift;
    p4.layer_id = args_.layer_id;
    p4.relu_enable = 0;
    ctx_.tiu_mul(&p4);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p1 = {0};
      p1.max = tl_output;
      p1.a = tl_output;
      p1.b = &tl_relu;
      p1.b_is_const = 0;
      p1.layer_id = args_.layer_id;
      ctx_.tiu_max(&p1);
    } else {
      cvk_tiu_min_param_t p1 = {0};
      p1.min = tl_output;
      p1.a = tl_output;
      p1.b = &tl_relu;
      p1.b_is_const = 0;
      p1.layer_id = args_.layer_id;
      ctx_.tiu_min(&p1);
    }
  } else {
    cvk_tiu_max_param_t p1 = {0};
    p1.max = &tl_relu;
    p1.a = tl_output;
    p1.b_is_const = 1;
    p1.b_const.is_signed = 1;
    p1.b_const.val = 0;
    p1.layer_id = args_.layer_id;
    ctx_.tiu_max(&p1);

    cvk_tiu_mul_param_t p2 = {0};
    p2.res_high = nullptr;
    p2.res_low = &tl_relu;
    p2.a = &tl_relu;
    p2.b_const.val = args_.activation_gt_scale;
    p2.b_const.is_signed = true;
    p2.b_is_const = 1;
    p2.rshift_bits = args_.activation_gt_rshift;
    p2.layer_id = args_.layer_id;
    p2.relu_enable = 0;
    ctx_.tiu_mul(&p2);

    cvk_tiu_min_param_t p3 = {0};
    p3.min = &tl_neg;
    p3.a = tl_output;
    p3.b_is_const = 1;
    p3.b_const.val = 0;
    p3.b_const.is_signed = 1;
    p3.layer_id = args_.layer_id;
    ctx_.tiu_min(&p3);

    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_neg;
    p4.a = &tl_neg;
    p4.b_const.val = args_.activation_le_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = args_.activation_le_rshift;
    p4.layer_id = args_.layer_id;
    p4.relu_enable = 0;
    ctx_.tiu_mul(&p4);

    cvk_tiu_or_int8_param_t p5 = {0};
    p5.res = tl_output;
    p5.a = &tl_relu;
    p5.b = &tl_neg;
    p5.layer_id = args_.layer_id;
    ctx_.tiu_or_int8(&p5);
  }
}

// gmOutputPoss shape (n_pos, ig_pos, oc_pos, oh_pos, ow_pos)
// lmIndex: (input, weight, output)
void Conv::compute(std::vector<uint32_t> gmOutputPoss,
    std::vector<uint32_t> lmIndexes, uint32_t cmdQueueIndex, uint32_t icPos) {
  uint32_t lm_input_index = lmIndexes[0];  // input
  uint32_t lm_weight_index = lmIndexes[1];  // weight
  uint32_t lm_output_index = lmIndexes[2];  // output

  // Input information also used in loadInput()
  // Output for TIU
  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);
  cvk_tl_t tl_output;
  cvk_tl_shape_t tl_output_shape = {
      gmOutputPossShapes[NGCHW::N], gmOutputPossShapes[NGCHW::C],
      gmOutputPossShapes[NGCHW::H], gmOutputPossShapes[NGCHW::W]};
  ctx_.lmem_init_tensor(&tl_output, tl_output_shape,
                        lmOutputDescs_[lm_output_index]->getDataFormat(),
                        lmOutputDescs_[lm_output_index]->getEuAlign());
  tl_output.start_address = lmOutputDescs_[lm_output_index]->getAddress();

  // Input for TIU
  std::vector<uint32_t> cur_gm_input_poss;
  std::vector<uint32_t> cur_gm_input_shapes;
  std::vector<uint32_t> cur_gm_input_paddings; // top, bottom, left, right
  getTiledGmPossAndShapesOfInputForTiu(
      gmOutputPoss, gmOutputPossShapes, cur_gm_input_poss,
      cur_gm_input_shapes, cur_gm_input_paddings, icPos);

  cvk_tl_t tl_input;
  cvk_tl_shape_t tl_input_shape = {
      cur_gm_input_shapes[NGCHW::N], cur_gm_input_shapes[NGCHW::C],
      cur_gm_input_shapes[NGCHW::H], cur_gm_input_shapes[NGCHW::W]};
  ctx_.lmem_init_tensor(&tl_input, tl_input_shape,
                        lmInputDescs_[lm_input_index]->getDataFormat(),
                        lmInputDescs_[lm_input_index]->getEuAlign());
  tl_input.start_address = lmInputDescs_[lm_input_index]->getAddress();

  // Bias for TIU
  std::vector<uint32_t> bias_shapes;
  cvk_tl_t tl_bias = {0};
  cvk_tl_shape_t tl_bias_shape = {0};
  if (getBiasAllowed(icPos)) {
    bias_shapes = getTiledLmShapesOfBiasForTiu(gmOutputPoss);
    tl_bias_shape = {
        bias_shapes[NGCHW::N], bias_shapes[NGCHW::C], bias_shapes[NGCHW::H],
        bias_shapes[NGCHW::W]};
    ctx_.lmem_init_tensor(&tl_bias, tl_bias_shape,
                          lmBiasDescs_[lm_weight_index]->getDataFormat(),
                          lmBiasDescs_[lm_weight_index]->getEuAlign());
    tl_bias.start_address = lmBiasDescs_[lm_weight_index]->getAddress();
  }

  // Weight for TIU, shapes (1, ic, tiledOc, kh, kw)
  //
  std::vector<uint32_t> weight_shapes =
      getTiledLmShapesOfWeightForTiu(gmOutputPoss, icPos);
  cvk_tl_t tl_weight;
  cvk_tl_shape_t tl_weight_shape = {
      weight_shapes[NGCHW::G], weight_shapes[NGCHW::C], weight_shapes[NGCHW::H],
      weight_shapes[NGCHW::W]};
  ctx_.lmem_init_tensor(&tl_weight, tl_weight_shape,
                        lmWeightDescs_[lm_weight_index]->getDataFormat(),
                        lmWeightDescs_[lm_weight_index]->getEuAlign());
  tl_weight.start_address = lmWeightDescs_[lm_weight_index]->getAddress();

  // Hardware constraint:
  //   if (des_ps32_md>=2)  {des_cmd_pre_exe <= 1};
  //
  //   The final stage of ps32 mode generates int8 result.
  //   Only enable early-store of cmd-pre-exe for no-ps32 mode or final stage of
  //   ps32 mode
  //
  uint8_t intraCmdParal = 0;
  if (ctx_.has_cmd_pre_exe() && cmdQueueIndex < cmdQueue_.size() &&
      (getPs32Mode(icPos) <= 1))
    intraCmdParal = cmdQueue_[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  LLVM_DEBUG(llvm::dbgs()
      << "    compute\n"
      << "      ifmap tl_addr " << llvm::format_hex(tl_input.start_address, 10)
      << ", shape (" << tl_input.shape.n
      << ", " << tl_input.shape.c
      << ", " << tl_input.shape.h
      << ", " << tl_input.shape.w << ")\n"
      << "      weight tl_addr "
      << llvm::format_hex(tl_weight.start_address, 10)
      <<", shape (" << tl_weight.shape.n
      << ", " << tl_weight.shape.c
      << ", " << tl_weight.shape.h
      << ", " << tl_weight.shape.w << ")\n"
      << "      bias " << llvm::format_hex(tl_bias.start_address, 10)
      <<", shape (" << tl_bias.shape.n
      << ", " << tl_bias.shape.c
      << ", " << tl_bias.shape.h
      << ", " << tl_bias.shape.w << ")\n"
      << "      ofmap " << llvm::format_hex(tl_output.start_address, 10)
      << ", shape (" << tl_output.shape.n
      << ", " << tl_output.shape.c
      << ", " << tl_output.shape.h
      << ", " << tl_output.shape.w << ")"
      << ", intraCmdParal " << (int)intraCmdParal
      << ", ps32mode " << (int)getPs32Mode(icPos)
      << "\n");

  // Use LayerId as trigger point
  uint32_t originalLayerId = args_.layer_id;
  cModelDebug_.updateLayerId(args_.layer_id, gmOutputPoss);

  if (args_.do_chl_quan && (getPs32Mode(icPos) <= 1))
    computeConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                cur_gm_input_paddings, intraCmdParal, icPos);
  else
    computePerTensorConv(&tl_output, &tl_input, &tl_weight, &tl_bias,
                         cur_gm_input_paddings, intraCmdParal, icPos);

  // Restore LayerId
  args_.layer_id = originalLayerId;

  if (args_.do_leaky_relu)
    computeLeakyRelu(&tl_output);
}

void Conv::storeOutput(std::vector<uint32_t> gmOutputPoss,
    uint32_t lmIndex, uint32_t cmdQueueIndex) {
  assert(gmOutputPoss.size() == 5 && "Expect 5D tensor");

  std::vector<uint32_t> gmOutputPossShapes =
      getTiledGmShapesOfOutputForTiu(gmOutputPoss);

  uint64_t ga_output_offset =
      gmOutputDesc_->getCurrentOffset(gmOutputPoss);
  uint64_t ga_output_store = gmOutputDesc_->getAddress() + ga_output_offset;

  std::vector<uint32_t> gm_output_strides = gmOutputDesc_->getStrides();
  cvk_tg_stride_t cvk_gm_output_stride = {
      gm_output_strides[NGCHW::N], gm_output_strides[NGCHW::C],
      gm_output_strides[NGCHW::H]};

  cvk_tl_t tl_output;
  cvk_tl_shape_t tl_output_shape = {
      gmOutputPossShapes[NGCHW::N], gmOutputPossShapes[NGCHW::C],
      gmOutputPossShapes[NGCHW::H], gmOutputPossShapes[NGCHW::W]};
  ctx_.lmem_init_tensor(&tl_output, tl_output_shape,
                        lmOutputDescs_[lmIndex]->getDataFormat(),
                        lmOutputDescs_[lmIndex]->getEuAlign());
  tl_output.start_address = lmOutputDescs_[lmIndex]->getAddress();

  uint8_t intraCmdParal = 0;
  if (ctx_.has_cmd_pre_exe() && cmdQueueIndex < cmdQueue_.size())
    intraCmdParal = cmdQueue_[cmdQueueIndex]->isIntraCmdParalEnabled() ? 1 : 0;

  LLVM_DEBUG(llvm::dbgs()
      << "  [ig=" << gmOutputPoss[NGCHW::G]
      << "][oc_pos=" << gmOutputPoss[NGCHW::C]
      << "][n_pos=" << gmOutputPoss[NGCHW::N]
      << "][oh_pos=" << gmOutputPoss[NGCHW::H]
      << "][ow_pos=" << gmOutputPoss[NGCHW::W]
      << "] new tdma_store_stride:\n"
      << "    tl_ofmap gaddr " << llvm::format_hex(ga_output_store, 10)
      << ", laddr " << llvm::format_hex(tl_output.start_address, 10)
      << ", shape (" << tl_output.shape.n
      << ", " << tl_output.shape.c
      << ", " << tl_output.shape.h
      << ", " << tl_output.shape.w
      << "), gm_output_stride (" << cvk_gm_output_stride.n
      << ", " << cvk_gm_output_stride.c
      << ", " << cvk_gm_output_stride.h << ")\n"
      << "    intraCmdParal " << (int)intraCmdParal << "\n");

  if (gmOutputDesc_->getCompressed()) {
    // Store compressed output
    cvk_cmpr_tg_t cmpr_dst = {0};
    cmpr_dst.bias0 = (gmOutputDesc_->getDataFormat() == CVK_FMT_BF16) ? 127 : 0;
    cmpr_dst.t.base_reg_index =
        ctx_.getTdmaBaseSelectIndexFromGaddr(gmOutputDesc_->getAddress());
    cmpr_dst.t.start_address = ga_output_store;
    cmpr_dst.t.fmt = gmOutputDesc_->getDataFormat();
    cmpr_dst.t.shape = {
        gmOutputPossShapes[NGCHW::N], gmOutputPossShapes[NGCHW::C],
        gmOutputPossShapes[NGCHW::H], gmOutputPossShapes[NGCHW::W]};
    cmpr_dst.t.stride = cvk_gm_output_stride;

    cvk_tdma_l2g_tensor_copy_compressed_param_t param = {0};
    param.src = &tl_output;
    param.dst = &cmpr_dst;
    ctx_.tdma_l2g_tensor_copy_compressed(&param);
  } else {

    // Store normal output
    cvk_tg_t ts_data = {0};
    ts_data.base_reg_index =
        ctx_.getTdmaBaseSelectIndexFromGaddr(gmOutputDesc_->getAddress());
    ts_data.fmt = tl_output.fmt;
    ts_data.start_address = ga_output_store;
    ts_data.shape = {
        tl_output.shape.n, tl_output.shape.c, tl_output.shape.h,
        tl_output.shape.w};
    ts_data.stride = cvk_gm_output_stride;

    cvk_tdma_l2g_tensor_copy_param_t param = {0};
    param.src = &tl_output;
    param.dst = &ts_data;
    param.intra_cmd_paral = intraCmdParal ? 1 : 0;

    ctx_.tdma_l2g_tensor_copy(&param);
  }

  cModelDebug_.recordOutput(args_.layer_id, gmOutputPoss, ga_output_store,
      ga_output_offset, gmOutputPossShapes);
}

bool Conv::isDwConv() {
  if ((groups() != 1) && (group_input_channels() == 1) &&
      (group_output_channels() == 1))
    return true;

  return false;
}

bool Conv::isConvPs32() {
  return (slices_.ic_step != group_input_channels()) ? true : false;
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Not split ic since it needs 32b ofmap for partial sum.
bool Conv::determineTileSize(bool useDoubleBuffer) {
  int32_t input_n = args_.input_n;
  int32_t input_c = args_.input_c;
  int32_t input_h = args_.input_h;
  int32_t input_w = args_.input_w;
  int32_t groups = args_.groups;
  int32_t output_c = args_.output_c;
  int32_t do_bias = args_.do_bias;
  bool do_chl_quan = args_.do_chl_quan;
  int32_t do_activation = args_.do_activation;
  float *activation_arg = args_.activation_arg;
  uint16_t kh = args_.kh;
  uint16_t kw = args_.kw;
  uint16_t dilation_h = args_.dilation_h;
  uint16_t dilation_w = args_.dilation_w;
  uint8_t pad_top = args_.pad_top;
  uint8_t pad_bottom = args_.pad_bottom;
  uint8_t pad_left = args_.pad_left;
  uint8_t pad_right = args_.pad_right;
  uint8_t stride_h = args_.stride_h;
  uint8_t stride_w = args_.stride_w;

  int32_t ic = input_c / groups;
  int32_t oc = output_c / groups;
  int32_t kh_extent = dilation_h * (kh - 1) + 1;
  int32_t kw_extent = dilation_w * (kw - 1) + 1;
  int32_t oh = (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int32_t ow = (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w + 1;
  int32_t ih = input_h;
  int32_t iw = input_w;
  int32_t n = input_n;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  determineTileSize =>\n"
      "    layer_id %d\n"
      "    groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
      "    kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
      "    stride (%d, %d), dilation (%d, %d)\n"
      "    useDoubleBuffer %d\n",
      args_.layer_id, groups, input_n, input_c, input_h, input_w, input_n, oc,
      oh, ow, kh, kw, pad_top, pad_bottom, pad_left, pad_right, stride_h,
      stride_w, dilation_h, dilation_w, useDoubleBuffer));

  int32_t npu_num = static_cast<int32_t>(getNpuNum());
  slices_.n = 1;
  slices_.oc = ceiling_func(oc, npu_num);  // lane parallelism
  slices_.ic = 1;
  slices_.h = (ih + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)
  slices_.w = (iw + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)

  int32_t num_oc_step = (oc + npu_num - 1) / npu_num;
  uint32_t ic_step =
      std::min(group_input_channels(), getMaxChannelOfHardware());

  // Not handle ps32 tiling here.
  if (ic_step < group_input_channels()) {
    LLVM_DEBUG(llvm::errs() << "  <= determineTileSize fail\n");
    return false;
  }

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;

  // Split ow
  for (int32_t ow_step = std::min(ow, MAX_WIDTH); ow_step > 0; --ow_step) {
    int32_t iw_step = ceiling_func((ow_step - 1) * stride_w + kw_extent,
                                   1 + insert_width());
    iw_step = std::min(iw_step, iw);

    if ((iw_step == iw) && (stride_w > 1)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, iw);
      slices_.iw_step = iw_step;
    }

    // Split oh
    for (int32_t oh_step = std::min(oh, MAX_HEIGHT); oh_step > 0; --oh_step) {
      // When the width tiling is used, there is no need to do height tiling.
      if (ow_step < std::min(ow, MAX_WIDTH))
        oh_step = 1;

      // int32_t oh_step = ceiling_func(oh, slices_.h);
      int32_t ih_step = ceiling_func((oh_step - 1) * stride_h + kh_extent,
                                      1 + insert_height());
      ih_step = std::min(ih_step, ih);

      // Split oc
      for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc) {
        // Downward, align lanes
        //   E.g. oc = 48, oc_step: 48, 32
        int32_t npu_num = static_cast<int32_t>(getNpuNum());
        int32_t oc_step =
            std::min((num_oc_step - slice_oc) * npu_num, oc);

        // We may need to put EU-alignment info in one place
        cvk_tl_shape_t coeff_shape_i16 = ctx_.tl_shape_t4(2, oc_step, 1, 1);

        uint32_t coeff_oc_step_size = 0;

        if (do_chl_quan) {
          if (do_bias) {
            cvk_tl_shape_t coeff_shape_9byte =
                ctx_.tl_shape_t4(1, oc_step, 1, 9);
            coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_9byte,
                                      args_.tiu_fmt, /*eu_align=*/0);
          } else {
            cvk_tl_shape_t coeff_shape_5byte = ctx_.tl_shape_t4(1, oc_step, 1, 5);
            coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_5byte,
                                      args_.tiu_fmt, /*eu_align=*/0);
          }
        } else if (do_bias) {
          // 16 bit
          coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_i16,
                                    args_.tiu_fmt, /*eu_align=*/0);
        }

        // Add weight size
        coeff_oc_step_size += ctx_.lmem_tensor_to_size(
                                  ctx_.tl_shape_t4(ic_step, oc_step, kh, kw),
                                  args_.tiu_fmt, /*eu_align=*/0);

        // split n
        for (slices_.n = 1; slices_.n <= n; ++slices_.n) {
          int32_t n_step = ceiling_func(n, slices_.n);

          uint32_t total_needed = 0;

          uint32_t ofmap_size =
              ctx_.lmem_tensor_to_size(
                  ctx_.tl_shape_t4(n_step, oc_step, oh_step, ow_step),
                  args_.tiu_fmt, /*eu_align=*/1);
          total_needed += ofmap_size;

          uint32_t ifmap_size = ctx_.lmem_tensor_to_size(
                              ctx_.tl_shape_t4(n_step, ic_step, ih_step, iw_step),
                              args_.tiu_fmt, /*eu_align=*/1);
          total_needed += ifmap_size;

          total_needed += coeff_oc_step_size;

          // Double buffers so that TDMA load and store can run during TIU
          // executes.
          total_needed *= bufferMultiplier;

          // Leaky relu need tl_neg, tl_relu.
          // tl_relu, tl_neg are not from tmda and not final output.
          // One copy is enough.
          if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
            total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
          }

  #if 0 // debug
          if (ic_step < group_input_channels()) {
            LLVM_DEBUG(llvm::dbgs()
                << "      [n_step=" << n_step
                << "][oc_step=" << oc_step
                << "][oh_step=" << oh_step
                << "][ow_step=" << ow_step
                << "][ih_step=" << ih_step
                << "][iw_step=" << iw_step
                << "][ic_step=" << ic_step
                << "] total_needed " << total_needed
                << ", LMEM_SIZE " << getLmSizePerLane()
                << "\n");
          }
  #endif

          if (total_needed <= getLmSizePerLane()) {
            // I try to maximize the local memory utilization,
            // but it causes large write latency, especially in cross-layer.
            // However TDMA engine can handle small data transfer efficiently.
            //
            // E.g. Resnet50 scale2b_branch2c in DDR3 platform.
            //   (1, 96, 56, 56) tiu 19471, store 31056, 77 fps
            //   (1, 32, 56, 56) tiu 6535, store 10376, 84 fps
            //
            // The load/store reorder may be useful in intra-layer and
            // inter-layer.
            //
            // The next-generation chip will do DMA store once intermediate
            // result is generated.
            //
            // The following is temporary solution.
            // I decrease the output channel size to trigger frequent DMA store.
            // So local memory is wasted.

            // DMA efficiency: OH * OW >= 256B
            const uint32_t dma_min_size = 256;
            uint32_t ofmap_plane_size = oh_step * ow_step;

            if ((oc_step > npu_num) &&
                (ofmap_plane_size > (1 * dma_min_size))) {
              continue;
            }
            if ((oc_step > (2 * npu_num)) &&
                (ofmap_plane_size < dma_min_size)) {
              // even oh*ow is smaller, use at most 2xlanes_num
              continue;
            }

            slices_.n_step = n_step;
            slices_.oc_step = oc_step;
            slices_.oh_step = oh_step;
            slices_.ow_step = ow_step;
            slices_.ih_step = ih_step;
            slices_.iw_step = iw_step;
            slices_.ic_step = ic_step;
            slices_.total_needed = total_needed;

            LLVM_DEBUG(llvm::errs() << llvm::format(
                "    Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d"
                ", ih_step=%d, iw_step=%d, ic_step=%d)\n"
                "      coeff_oc_step_size %d, total_needed %d\n"
                "      ifmap shape (%d, %d, %d, %d)\n"
                "      weight shape (%d, %d, %d, %d)\n"
                "      ofmap shape (%d, %d, %d, %d)\n",
                n_step, oc_step, oh_step, ow_step, ih_step, iw_step, ic_step,
                coeff_oc_step_size, total_needed, n_step, ic_step, ih_step,
                iw_step, oc_step, ic_step, kh, kw, n_step, oc_step, oh_step,
                ow_step));
            LLVM_DEBUG(llvm::errs() << "  <= determineTileSize succeed\n");
            return true;
          }

        }  // for (slices_.n = 1; slices_.n < n; ++slices_.n)

      }  // for (int32_t slice_oc = 0; slice_oc < num_oc_step; ++slice_oc)

    }  // for (slices_.h = 1; slices_.h <= oh; ++slices_.h)

  }  // for (slices_.w = 1; slices_.w <= ow; ++slices_.ow)
  LLVM_DEBUG(llvm::errs() << "  <= determineTileSize fail\n");

  return false;
}

bool Conv::determinePs32TileSize(bool useDoubleBuffer) {
  int32_t input_n = args_.input_n;
  int32_t input_c = args_.input_c;
  int32_t input_h = args_.input_h;
  int32_t input_w = args_.input_w;
  int32_t groups = args_.groups;
  int32_t output_c = args_.output_c;
  int32_t do_bias = args_.do_bias;
  bool do_chl_quan = args_.do_chl_quan;
  int32_t do_activation = args_.do_activation;
  float *activation_arg = args_.activation_arg;
  uint16_t kh = args_.kh;
  uint16_t kw = args_.kw;
  uint16_t dilation_h = args_.dilation_h;
  uint16_t dilation_w = args_.dilation_w;
  uint8_t pad_top = args_.pad_top;
  uint8_t pad_bottom = args_.pad_bottom;
  uint8_t pad_left = args_.pad_left;
  uint8_t pad_right = args_.pad_right;
  uint8_t stride_h = args_.stride_h;
  uint8_t stride_w = args_.stride_w;

  int32_t ic = input_c / groups;
  int32_t oc = output_c / groups;
  int32_t kh_extent = dilation_h * (kh - 1) + 1;
  int32_t kw_extent = dilation_w * (kw - 1) + 1;
  int32_t oh = (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int32_t ow = (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w + 1;
  int32_t ih = input_h;
  int32_t iw = input_w;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  LLVM_DEBUG(llvm::errs() << llvm::format(
      "  determinePs32TileSize =>\n"
      "    layer_id %d\n"
      "    groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
      "    kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
      "    stride (%d, %d), dilation (%d, %d)\n"
      "    useDoubleBuffer %d\n",
      args_.layer_id, groups, input_n, input_c, input_h, input_w, input_n, oc,
      oh, ow, kh, kw, pad_top, pad_bottom, pad_left, pad_right, stride_h,
      stride_w, dilation_h, dilation_w, useDoubleBuffer));

  int32_t npu_num = static_cast<int32_t>(getNpuNum());
  slices_.n = 1;
  slices_.oc = ceiling_func(oc, npu_num);  // lane parallelism
  slices_.ic = 1;
  slices_.h = (ih + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)
  slices_.w = (iw + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)

  uint32_t max_ic_step =
      std::min(group_input_channels(), getMaxChannelOfHardware());

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;

  int32_t n_step = 1;
  int32_t oc_step = std::min(static_cast<int32_t>(group_output_channels()),
                             static_cast<int32_t>(getNpuNum()));

  // Split ow
  for (int32_t ow_step = std::min(ow, MAX_WIDTH); ow_step > 0; --ow_step) {
    // int32_t ow_step = ceiling_func(ow, slices_.w);
    int32_t iw_step = std::min((ow_step - 1) * stride_w + kw_extent, iw);

    // if ((slices_.w == 1) && (stride_w > 1)) {
    if ((iw_step == iw) && (stride_w > 1)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, iw);
      slices_.iw_step = iw_step;
    }

    // Split oh
    // for (slices_.h = 1; slices_.h <= oh; ++slices_.h) {
    for (int32_t oh_step = std::min(oh, MAX_HEIGHT); oh_step > 0; --oh_step) {
      // When the width tiling is used, there is no need to do height tiling.
      if (ow_step < std::min(ow, MAX_WIDTH))
        oh_step = 1;

      // int32_t oh_step = ceiling_func(oh, slices_.h);
      int32_t ih_step =
          std::min((oh_step - 1) * stride_h + kh_extent, ih);

      // We may need to put EU-alignment info in one place
      cvk_tl_shape_t coeff_shape_i16 = ctx_.tl_shape_t4(2, oc_step, 1, 1);

      uint32_t coeff_oc_step_size = 0;

      if (do_chl_quan) {
        if (do_bias) {
          cvk_tl_shape_t coeff_shape_9byte =
              ctx_.tl_shape_t4(1, oc_step, 1, 9);
          coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_9byte,
                                    args_.tiu_fmt, /*eu_align=*/0);
        } else {
          cvk_tl_shape_t coeff_shape_5byte = ctx_.tl_shape_t4(1, oc_step, 1, 5);
          coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_5byte,
                                    args_.tiu_fmt, /*eu_align=*/0);
        }
      } else if (do_bias) {
        // 16 bit
        coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_i16,
                                  args_.tiu_fmt, /*eu_align=*/0);
      }

      // Split ic
      for (int32_t ic_step = max_ic_step; ic_step > 0;
           ic_step = align_up(ic_step / 2, npu_num)) {
        uint32_t ofmapSizeMultiplier =
            (ic_step < static_cast<int32_t>(group_input_channels())) ? 4 : 1;

        // Add weight size
        uint32_t weight_size =
            ctx_.lmem_tensor_to_size(
                            ctx_.tl_shape_t4(ic_step, oc_step, kh, kw),
                            args_.tiu_fmt, /*eu_align=*/0);

        uint32_t ofmap_size =
            ctx_.lmem_tensor_to_size(
                ctx_.tl_shape_t4(n_step, oc_step, oh_step, ow_step),
                args_.tiu_fmt, /*eu_align=*/1);

        uint32_t ifmap_size = ctx_.lmem_tensor_to_size(
                            ctx_.tl_shape_t4(n_step, ic_step, ih_step, iw_step),
                            args_.tiu_fmt, /*eu_align=*/1);

        uint32_t total_needed = coeff_oc_step_size + weight_size + ifmap_size +
                                ofmap_size * ofmapSizeMultiplier;

        // Double buffers so that TDMA load and store can run during TIU
        // executes.
        total_needed *= bufferMultiplier;

        // Leaky relu need tl_neg, tl_relu.
        // tl_relu, tl_neg are not from tmda and not final output.
        // One copy is enough.
        if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
          total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
        }

#if 0 // debug
        LLVM_DEBUG(llvm::dbgs()
            << "      [n_step=" << n_step
            << "][oc_step=" << oc_step
            << "][oh_step=" << oh_step
            << "][ow_step=" << ow_step
            << "][ih_step=" << ih_step
            << "][iw_step=" << iw_step
            << "][ic_step=" << ic_step
            << "] total_needed " << total_needed
            << ", LMEM_SIZE " << getLmSizePerLane()
            << ", bufferMultiplier " << bufferMultiplier
            << ", ofmapSizeMultiplier " << ofmapSizeMultiplier << "\n"
            << "        ifmap shape(" << n_step << ", " << ic_step
            << ", " << ih_step << ", " << iw_step
            << "), size " << ifmap_size << "\n"
            << "        weight shape(" << oc_step << ", " << ic_step
            << ", " << kh << ", " << kw << "), size " << weight_size << "\n"
            << "        ofmap shape (" << n_step << ", " << oc_step
            << ", " << oh_step << ", " << ow_step << "), size "
            << ofmap_size << "\n");
#endif

        if (total_needed <= getLmSizePerLane()) {
          LLVM_DEBUG(llvm::dbgs()
              << "      [n_step=" << n_step
              << "][oc_step=" << oc_step
              << "][oh_step=" << oh_step
              << "][ow_step=" << ow_step
              << "][ih_step=" << ih_step
              << "][iw_step=" << iw_step
              << "][ic_step=" << ic_step
              << "] total_needed " << total_needed
              << ", LMEM_SIZE " << getLmSizePerLane()
              << ", bufferMultiplier " << bufferMultiplier
              << ", ofmapSizeMultiplier " << ofmapSizeMultiplier << "\n"
              << "        ifmap shape(" << n_step << ", " << ic_step
              << ", " << ih_step << ", " << iw_step
              << "), size " << ifmap_size << "\n"
              << "        weight shape(" << oc_step << ", " << ic_step
              << ", " << kh << ", " << kw << "), size " << weight_size << "\n"
              << "        ofmap shape (" << n_step << ", " << oc_step
              << ", " << oh_step << ", " << ow_step << "), size "
              << ofmap_size << "\n");

          LLVM_DEBUG(llvm::errs() << llvm::format(
              "    Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d"
              ", ih_step=%d, iw_step=%d, ic_step=%d)\n"
              "      coeff_oc_step_size %d, total_needed %d\n"
              "      ifmap shape (%d, %d, %d, %d)\n"
              "      weight shape (%d, %d, %d, %d)\n"
              "      ofmap shape (%d, %d, %d, %d)\n",
              n_step, oc_step, oh_step, ow_step, ih_step, iw_step, ic_step,
              coeff_oc_step_size, total_needed, n_step, ic_step, ih_step,
              iw_step, oc_step, ic_step, kh, kw, n_step, oc_step, oh_step,
              ow_step));
          LLVM_DEBUG(llvm::errs() << "  <= determinePs32TileSize succeed\n");

          slices_.n_step = n_step;
          slices_.oc_step = oc_step;
          slices_.oh_step = oh_step;
          slices_.ow_step = ow_step;
          slices_.ih_step = ih_step;
          slices_.iw_step = iw_step;
          slices_.ic_step = ic_step;
          slices_.total_needed = total_needed;

          return true;
        }

      } // uint32_t ic_step = group_input_channels(); ic_step > 0; --ic_step

    }  // for (slices_.h = 1; slices_.h <= oh; ++slices_.h)

  }  // for (slices_.w = 1; slices_.w <= ow; ++slices_.ow)

  LLVM_DEBUG(llvm::errs() << "  <= determinePs32TileSize fail\n");

  return false;
}

//
// This function implemnets weight reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
// TIU/TDMA command execution flow:
//   DMA G2L,  cmd_id 1, wait_id_tpu 0
//   DMA G2L,  cmd_id 2, wait_id_tpu 0
//   DMA G2L,  cmd_id 3, wait_id_tpu 0, LD0
//   TIU conv, cmd_id 1, cmd_id_gdma 3, TIU0, wait LD0
//   DMA G2L,  cmd_id 4, wait_id_tpu 0, LD1, no wait
//   TIU conv, cmd_id 2, cmd_id_gdma 4, TIU1, wait LD1
//   DMA L2G,  cmd_id 5, wait_id_tpu 1, SD0, wait TIU1
//   DMA G2L,  cmd_id 6, wait_id_tpu 0, LD2, no wait
//   TIU conv, cmd_id 3, cmd_id_gdma 6, TIU2, wait LD2
//   DMA L2G,  cmd_id 7, wait_id_tpu 2, SD1, wait TIU2
//   DMA G2L,  cmd_id 8, wait_id_tpu 0, LD3, no wait
//   TIU conv, cmd_id 4, cmd_id_gdma 8, TIU3, wait LD3
//
//   TDMA      TIU
//   LD0
//   LD1       TIU03
//   SD0/LD2   TIU1
//   SD1/LD3   TIU2
//
void Conv::convReuseWeight() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "convReuseWeight =>\n"
      "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
      "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
      "  stride (%d, %d), dilation (%d, %d)\n"
      "  do_bias %d, do_chl_quan %d\n"
      "  Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, ih_step=%d"
      ", iw_step=%d, ic_step=%d)\n"
      "  store_compr_act %d, load_compr_act %d\n",
      args_.groups, args_.input_n, args_.input_c, args_.input_h, args_.input_w,
      args_.input_n, args_.output_c, output_height(), output_width(),
      args_.kh, args_.kw, args_.pad_top, args_.pad_bottom, args_.pad_left,
      args_.pad_right, args_.stride_h, args_.stride_w, args_.dilation_h,
      args_.dilation_w, args_.do_bias, args_.do_chl_quan,slices_.n_step,
      slices_.oc_step, slices_.oh_step, slices_.ow_step, slices_.ih_step,
      slices_.iw_step, slices_.ic_step, args_.store_compr_act,
      args_.load_compr_act));

  // Pre-alloc maximum one-step size
  // The local memory release must be in reverse order.
  allocateAllLocalMem();

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    int first = 1;
    uint32_t flip = 0;
    uint32_t coeff_flip = 0;
    std::vector<uint32_t> gmOutputPoss[2];

    ctx_.parallel_disable();

    // split oc
    for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
         oc_pos += slices_.oc_step) {
      loadBias({/*n_pos=*/0, ig, oc_pos, /*oh_pos=*/0, /*ow_pos=*/0},
                coeff_flip, 0);
      loadWeight({/*n_pos=*/0, ig, oc_pos, /*oh_pos=*/0, /*ow_pos=*/0},
                 coeff_flip, 0);

      // split n
      for (uint32_t n_pos = 0; n_pos < batch_size(); n_pos += slices_.n_step) {
        // split h
        for (uint32_t oh_pos = 0; oh_pos < output_height();
             oh_pos += slices_.oh_step) {

          // split w
          for (uint32_t ow_pos = 0; ow_pos < output_width();
               ow_pos += slices_.ow_step) {
            gmOutputPoss[flip] = {n_pos, ig, oc_pos, oh_pos, ow_pos};
            loadInput(gmOutputPoss[flip], flip, 0);

            ctx_.parallel_disable();
            ctx_.parallel_enable();

            compute(gmOutputPoss[flip], {flip, coeff_flip, flip}, 0);

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              uint32_t flip_back = 1 - flip;

              // Store back to global memory
              storeOutput(gmOutputPoss[flip_back], flip_back, 0);
            }

            flip = 1 - flip;

          }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)
        }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)
      }  // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx_.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    uint32_t flip_back = 1 - flip;

    // Store back to global memory
    storeOutput(gmOutputPoss[flip_back], flip_back, 0);
  }  // for (int group_i = 0; group_i < groups; ++groups)

  deallocateAllLocalMem();

  LLVM_DEBUG(llvm::errs() << "<=convReuseWeight" << "\n");
}

//
// This function implemnets activation(ifmap) reuse.
//   - 2x input and output buffer - load and store while tiu is busy
//   - 2x weight buffer - split oc
//
void Conv::convReuseActivation() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "convReuseActivation =>\n"
      "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
      "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
      "  stride (%d, %d), dilation (%d, %d)\n"
      "  do_bias %d, do_chl_quan %d\n"
      "  Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, ih_step=%d"
      ", iw_step=%d, ic_step=%d)\n"
      "  store_compr_act %d, load_compr_act %d\n",
      args_.groups, args_.input_n, args_.input_c, args_.input_h, args_.input_w,
      args_.input_n, args_.output_c, output_height(), output_width(),
      args_.kh, args_.kw, args_.pad_top, args_.pad_bottom, args_.pad_left,
      args_.pad_right, args_.stride_h, args_.stride_w, args_.dilation_h,
      args_.dilation_w, args_.do_bias, args_.do_chl_quan, slices_.n_step,
      slices_.oc_step, slices_.oh_step, slices_.ow_step, slices_.ih_step,
      slices_.iw_step, slices_.ic_step, args_.store_compr_act,
      args_.load_compr_act));

  // Pre-alloc maximum one-step size
  // The local memory release must be in reverse order.
  allocateAllLocalMem();

  auto loadInputCmd = [&](std::vector<uint32_t> poss, uint32_t index) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadInputCmdType, poss, index));
  };

  auto loadBiasCmd = [&](std::vector<uint32_t> poss, uint32_t index) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadBiasCmdType, poss, index));
  };

  auto loadWeightCmd = [&](std::vector<uint32_t> poss, uint32_t index) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadWeightCmdType, poss, index));
  };

  auto disParallelCmd = [&]{
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::ParallelCmdType, false));
  };

  auto enParallelCmd = [&]{
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::ParallelCmdType, true));
  };

  auto computeCmd = [&](std::vector<uint32_t> poss,
      std::vector<uint32_t> indexes) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::ComputCmdType, poss, indexes));
  };

  auto storeOutputCmd = [&](std::vector<uint32_t> poss, uint32_t index) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::StoreOutputCmdType, poss, index));
  };

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    int first = 1;
    uint32_t flip = 0;
    uint32_t coeff_flip = 0;
    std::vector<uint32_t> gmOutputPoss[2];

    disParallelCmd();

    // split n
    for (uint32_t n_pos = 0; n_pos < batch_size(); n_pos += slices_.n_step) {

      // split h
      for (uint32_t oh_pos = 0; oh_pos < output_height();
           oh_pos += slices_.oh_step) {

        // split w
        for (uint32_t ow_pos = 0; ow_pos < output_width();
             ow_pos += slices_.ow_step) {
          loadInputCmd({n_pos, /*g_pos=*/0, /*oc_pos=*/0, oh_pos, ow_pos},
              flip);

          // split oc
          for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
               oc_pos += slices_.oc_step) {
            gmOutputPoss[coeff_flip] =
                {n_pos, ig, oc_pos, oh_pos, ow_pos};
            std::vector<uint32_t> cur_weight_pos =
                {/*n_pos=*/0, ig, oc_pos, /*oh_pos=*/0, /*ow_pos=*/0};

          loadBiasCmd(cur_weight_pos, coeff_flip);
          loadWeightCmd(cur_weight_pos, coeff_flip);

          disParallelCmd();
          enParallelCmd();

          computeCmd(gmOutputPoss[coeff_flip],
              {flip, coeff_flip, coeff_flip});

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              int coeff_flip_back = 1 - coeff_flip;
              // Store back to global memory

              storeOutputCmd(gmOutputPoss[coeff_flip_back],
                  coeff_flip_back);
            }

            coeff_flip = 1 - coeff_flip;

          }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

          flip = 1 - flip;

        }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

      }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

    }  // for (int n_i = 0; n_i < n; ni += n_step)

    // ctx_.parallel_disable();

    disParallelCmd();

    // the last iteration stored the other side, leave the last side not stored
    int coeff_flip_back = 1 - coeff_flip;

    // Store back to global memory
    storeOutputCmd(gmOutputPoss[coeff_flip_back], coeff_flip_back);

  }  // for (int group_i = 0; group_i < groups; ++groups)

  auto intraCmdAnalysis =
      std::make_unique<IntraCmdParallelAnalysis>(cmdQueue_);
  intraCmdAnalysis->analyze();
  // intraCmdAnalysis->dumpStates();

  // Generate command
  generateCmd();

  deallocateAllLocalMem();

  LLVM_DEBUG(llvm::errs() << "<= convReuseActivation\n");
}

// Split n, oh, ow, oc.
// Split oc as the number of lanes.
// Borrowed from BM1880v2ConvFixedParallelv2::split
bool Conv::determineDwTileSize(bool useDoubleBuffer) {
  int input_n = args_.input_n;
  int input_c = args_.input_c;
  int input_h = args_.input_h;
  int input_w = args_.input_w;
  //int groups = args_.groups;
  //int output_c = args_.output_c;
  int do_bias = args_.do_bias;
  //bool do_chl_quan = args_.do_chl_quan;
  int do_activation = args_.do_activation;
  float *activation_arg = args_.activation_arg;
  uint16_t kh = args_.kh;
  uint16_t kw = args_.kw;
  uint16_t dilation_h = args_.dilation_h;
  uint16_t dilation_w = args_.dilation_w;
  uint8_t pad_top = args_.pad_top;
  uint8_t pad_bottom = args_.pad_bottom;
  uint8_t pad_left = args_.pad_left;
  uint8_t pad_right = args_.pad_right;
  uint8_t stride_h = args_.stride_h;
  uint8_t stride_w = args_.stride_w;

  int ic = input_c;
  int oc = input_c;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  LLVM_DEBUG(llvm::errs() << llvm::format(
             "determineDwTileSize =>\n"
             "  ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
             "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
             "  ins_h %d, ins_w %d "
             "  stride (%d, %d), dilation (%d, %d)\n",
             input_n, input_c, input_h, input_w, input_n, oc, oh, ow, kh, kw,
             pad_top, pad_bottom, pad_left, pad_right, insert_height(), insert_width(),
             stride_h, stride_w, dilation_h, dilation_w));

  int32_t npu_num = static_cast<int32_t>(getNpuNum());
  slices_.n = 1;
  slices_.oc = ceiling_func(oc, npu_num);  // lane parallelism
  slices_.ic = ic;
  slices_.h = (ih + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)
  slices_.w = (iw + (4095 - 32 - 1)) / (4095 - 32);  // 12bit, max 4095-32(lanes)

  int oc_step = (oc >= npu_num) ? npu_num : oc;  // use all lanes
  int ic_step = 1;

  // We may need to put EU-alignment info in one place
  cvk_tl_shape_t coeff_shape_i16 = ctx_.tl_shape_t4(2, oc_step, 1, 1);

  uint32_t coeff_oc_step_size = 0;
  if (do_bias) {
    // 16 bit
    coeff_oc_step_size += ctx_.lmem_tensor_to_size(coeff_shape_i16,
                              args_.tiu_fmt, /*eu_align=*/0);
  }

  // Add weight size
  coeff_oc_step_size += ctx_.lmem_tensor_to_size(
                            ctx_.tl_shape_t4(ic_step, oc_step, kh, kw),
                            args_.tiu_fmt, /*eu_align=*/0);

  uint32_t bufferMultiplier = useDoubleBuffer ? 2 : 1;

  //
  // Slices may not be a good way to find size
  // We may try to increase or decrease width in aligned with 4, 8, 16 ...
  // or specific height/width (8, 8), (16, 16) ...
  //
  // Split ow
  for (slices_.w = 1; slices_.w <= ow; ++slices_.w) {
    int ow_step = ceiling_func(ow, slices_.w);
    int iw_step = ceiling_func((ow_step - 1) * stride_w + kw_extent, 1 + insert_width());
    iw_step = std::min(iw_step, iw);

    // Split oh
    for (slices_.h = 1; slices_.h <= oh; ++slices_.h) {
      // split n
      for (slices_.n = 1; slices_.n <= n; ++slices_.n) {
        int n_step = ceiling_func(n, slices_.n);

        int oh_step = ceiling_func(oh, slices_.h);
        int ih_step = ceiling_func((oh_step - 1) * stride_h + kh_extent, 1 + insert_height());
        ih_step = std::min(ih_step, ih);

        uint32_t total_needed = 0;

        uint32_t ofmap_size = ctx_.lmem_tensor_to_size(
                            ctx_.tl_shape_t4(n_step, oc_step, oh_step, ow_step),
                            args_.tiu_fmt, /*eu_align=*/1);
        total_needed += ofmap_size;

        uint32_t ifmap_size = ctx_.lmem_tensor_to_size(
                            ctx_.tl_shape_t4(n_step, oc_step, ih_step, iw_step),
                            args_.tiu_fmt, /*eu_align=*/1);
        total_needed += ifmap_size;

        total_needed += coeff_oc_step_size;

        // Double buffers so that TDMA load and store can run during TIU executes.
        total_needed *= bufferMultiplier;

        // Leaky relu need tl_neg, tl_relu.
        // tl_relu, tl_neg are not from tmda and not final output.
        // One copy is enough.
        if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
          total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
        }

        if (total_needed <= getLmSizePerLane()) {
          slices_.n_step = n_step;

          LLVM_DEBUG(llvm::errs() << llvm::format(
                  "  Slices(n=%d, oc=%d, ic=%d, h=%d, w=%d), n_step %d, oh_step %d, ih_step %d"
                  ", coeff_oc_step_size %d, total_needed %d\n",
                  slices_.n, slices_.oc, slices_.ic, slices_.h, slices_.w, n_step, oh_step, ih_step,
                  coeff_oc_step_size, total_needed));
          LLVM_DEBUG(llvm::errs() << "<= determineDwTileSize succeed" << "\n");
          return true;
        }

      }  // for (slices_.n = 1; slices_.n < n; ++slices_.n)

    }  // for (slices_.h = 1; slices_.h <= oh; ++slices_.h)

  }  // for (slices_.w = 1; slices_.w <= ow; ++slices_.ow)

  LLVM_DEBUG(llvm::errs() << "<= determineDwTileSize fail" << "\n";);

  return false;
}

void Conv::dwConv() {
  int input_n = args_.input_n;
  int input_c = args_.input_c;
  int input_h = args_.input_h;
  int input_w = args_.input_w;
  //int groups = args_.groups;
  int output_c = args_.output_c;
  bool do_bias = args_.do_bias;
  bool do_chl_quan = args_.do_chl_quan;
  bool do_activation = args_.do_activation;
  float *activation_arg = args_.activation_arg;
  uint16_t kh = args_.kh;
  uint16_t kw = args_.kw;
  uint16_t dilation_h = args_.dilation_h;
  uint16_t dilation_w = args_.dilation_w;
  uint8_t pad_top = args_.pad_top;
  uint8_t pad_bottom = args_.pad_bottom;
  uint8_t pad_left = args_.pad_left;
  uint8_t pad_right = args_.pad_right;
  uint8_t stride_h = args_.stride_h;
  uint8_t stride_w = args_.stride_w;
  //bool load_compr_act = args_.load_compr_act;
  //bool store_compr_act = args_.store_compr_act;
  gaddr_t ga_ifmap = args_.ga_ifmap;
  gaddr_t ga_ofmap = args_.ga_ofmap;
  gaddr_t ga_weight = args_.ga_weight;
  gaddr_t ga_bias = args_.ga_bias;
  int activation_gt_rshift = args_.activation_gt_rshift;
  int activation_gt_scale = args_.activation_gt_scale;
  int activation_le_scale = args_.activation_le_scale;
  int activation_le_rshift = args_.activation_le_rshift;
  int right_shift_width = args_.right_shift_width;
  uint32_t layer_id = args_.layer_id;

  int oc = output_c;
  int ic = 1;
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int oh = (inserted_input_height() + pad_top + pad_bottom - kh_ext) / stride_h + 1;
  int ow = (inserted_input_width() + pad_left + pad_right - kw_ext) / stride_w + 1;

  int n_step = ceiling_func(input_n, slices_.n);
  assert(n_step == static_cast<int>(slices_.n_step));

  int oh_step = ceiling_func(oh, slices_.h);
  int ow_step = ceiling_func(ow, slices_.w);
  int ih_step = input_h;
  int iw_step = input_w;
  int oc_step = oc;

  // Always use all lanes.
  // Not divided by slices_.oc.
  // It is better to store step.
  if (slices_.oc > 1) {
    ASSERT(oc > static_cast<int32_t>(getNpuNum()));
    oc_step = static_cast<int32_t>(getNpuNum());
  }

  if (slices_.h > 1) {
    // max input height inside feature map
    ih_step = (oh_step - 1) * stride_h + kh_ext;
    ih_step = ceiling_func(ih_step, 1 + insert_height());
  }
  if (slices_.w > 1) {
    // max input width inside feature map
    iw_step = (ow_step - 1) * stride_w + kw_ext;
    iw_step = ceiling_func(iw_step, 1 + insert_width());
  }

  bool fused_conv_relu =
      ((do_activation && (!activation_arg || (activation_arg[0] == 0.0f))))
          ? true
          : false;

  cvk_tl_t *tl_weight[2] = {nullptr, nullptr}, *tl_bias[2] = {nullptr, nullptr};
  cvk_tl_t *tl_chl_quan[2] = {nullptr, nullptr};
  cvk_tl_t *tl_ifmap[2] = {nullptr};
  cvk_tl_t *tl_ofmap[2] = {nullptr};
  cvk_tl_t *tl_neg = nullptr, *tl_relu = nullptr;

  // Global memory stride from global memory shape
  // input_c, output_c, not ic, oc
  cvk_tg_stride_t ofmap_gstride =
      ctx_.tg_default_stride(output_c, oh, ow, args_.output_fmt);
  cvk_tg_stride_t ifmap_gstride = ctx_.tg_default_stride(
      input_c, input_h, input_w, args_.input_fmt);
  cvk_tg_stride_t bias_gstride =
      ctx_.tg_default_stride(output_c, 1, 1, args_.tiu_fmt);
  cvk_tg_stride_t weight_gstride =
      ctx_.tg_default_stride(oc, kh * kw, ic, args_.tiu_fmt);

  uint32_t ifmap_gstride_w = (args_.input_fmt == CVK_FMT_BF16) ? 2 : 1;
  uint32_t weight_gstride_w = (args_.tiu_fmt == CVK_FMT_BF16) ? 2 : 1;
  uint32_t ofmap_gstride_w = (args_.output_fmt == CVK_FMT_BF16) ? 2 : 1;

  int ofmap_multiplier = 1;
  if (args_.ps32_output && args_.tiu_fmt == CVK_FMT_BF16)
    ofmap_multiplier = 2;
  else if (args_.ps32_output && args_.tiu_fmt == CVK_FMT_I8)
    ofmap_multiplier = 4;

  //
  // Pre-alloc maximum one-step size
  //
  // Need vector to track the order of local memory.
  // The local memory release must be in reverse order.
  //
  tl_weight[0] =
      ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(1, oc_step, kh, kw), args_.tiu_fmt, /*eu_align=*/0);
  tl_weight[1] =
      ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(1, oc_step, kh, kw), args_.tiu_fmt, /*eu_align=*/0);
  tl_ifmap[0] = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(n_step, oc_step, ih_step, iw_step), args_.tiu_fmt,
                                      /*eu_align=*/1);
  tl_ifmap[1] = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(n_step, oc_step, ih_step, iw_step), args_.tiu_fmt,
                                      /*eu_align=*/1);
  tl_ofmap[0] = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(ofmap_multiplier * n_step, oc_step, oh_step, ow_step), args_.tiu_fmt,
                                      /*eu_align=*/1);
  tl_ofmap[1] = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(ofmap_multiplier * n_step, oc_step, oh_step, ow_step), args_.tiu_fmt,
                                      /*eu_align=*/1);
  ASSERT(tl_weight[0] && tl_weight[1] && tl_ifmap[0] && tl_ifmap[1] && tl_ofmap[0] && tl_ofmap[1]);

  cvk_tl_shape_t coeff_shape_i16 = ctx_.tl_shape_t4(2, oc_step, 1, 1);

  if (do_chl_quan) {
    // Per-channel quantization

    if (do_bias) {
      cvk_tl_shape_t coeff_shape_9byte = ctx_.tl_shape_t4(1, oc_step, 1, 9);
      tl_chl_quan[0] = ctx_.lmem_alloc_tensor(coeff_shape_9byte, CVK_FMT_U8, /*eu_align=*/0);
      tl_chl_quan[1] = ctx_.lmem_alloc_tensor(coeff_shape_9byte, CVK_FMT_U8, /*eu_align=*/0);
    } else {
      cvk_tl_shape_t coeff_shape_5byte = ctx_.tl_shape_t4(1, oc_step, 1, 5);
      tl_chl_quan[0] = ctx_.lmem_alloc_tensor(coeff_shape_5byte, CVK_FMT_U8, /*eu_align=*/0);
      tl_chl_quan[1] = ctx_.lmem_alloc_tensor(coeff_shape_5byte, CVK_FMT_U8, /*eu_align=*/0);
    }

    ASSERT(tl_chl_quan[0] && tl_chl_quan[1]);
  } else if (do_bias) {
    // Only allocate resource without per-channel quantization

    // 16 bit
    tl_bias[0] = ctx_.lmem_alloc_tensor(coeff_shape_i16, args_.tiu_fmt, /*eu_align=*/0);
    tl_bias[1] = ctx_.lmem_alloc_tensor(coeff_shape_i16, args_.tiu_fmt, /*eu_aling=*/0);
    ASSERT(tl_bias[0] && tl_bias[1]);
  }

  // Leaky relu needs tl_neg, tl_relu.
  if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
    tl_neg = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(n_step, oc_step, oh_step, ow_step), args_.tiu_fmt,
                                   /*eu_align=*/1);
    tl_relu = ctx_.lmem_alloc_tensor(ctx_.tl_shape_t4(n_step, oc_step, oh_step, ow_step), args_.tiu_fmt,
                                    /*eu_align=*/1);
    ASSERT(tl_neg && tl_relu);
  }

  // split groups
  for (int ig = 0; ig < 1; ++ig) {
    int first = 1;
    uint32_t flip = 0;
    uint32_t coeff_flip = 0;
    gaddr_t ga_ofmap_cur[2] = {0};

    ctx_.parallel_disable();

    // split oc
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = std::min(oc - oc_pos, oc_step);

      uint64_t coeff_offset = ig * oc + oc_pos;

      // Actual shape for tdma, tiu
      coeff_shape_i16 = ctx_.tl_shape_t4(2, cur_oc, 1, 1);

      if (do_chl_quan) {
        assert(args_.tiu_fmt == CVK_FMT_I8 && "Per-channel data only for i8");

        if (do_bias) {
          tl_chl_quan[coeff_flip]->shape = ctx_.tl_shape_t4(1, cur_oc, 1, 9);
          tl_chl_quan[coeff_flip]->stride =
              ctx_.tl_default_stride(tl_chl_quan[coeff_flip]->shape, args_.tiu_fmt, /*eu_aign=*/0);
          ctx_.tdma_load(tl_chl_quan[coeff_flip], ga_bias + coeff_offset * 9);
        } else {
          tl_chl_quan[coeff_flip]->shape = ctx_.tl_shape_t4(1, cur_oc, 1, 5);
          tl_chl_quan[coeff_flip]->stride =
              ctx_.tl_default_stride(tl_chl_quan[coeff_flip]->shape, args_.tiu_fmt, /*eu_aign=*/0);

          ctx_.tdma_load(tl_chl_quan[coeff_flip], ga_bias + coeff_offset * 5);
        }
      } else if (do_bias) {
        // 16 bit
        // bmk does not keep eu-align info, user need to update stride if shape changed
        tl_bias[coeff_flip]->shape = coeff_shape_i16;
        tl_bias[coeff_flip]->stride =
            ctx_.tl_default_stride(tl_bias[coeff_flip]->shape, args_.tiu_fmt, /*eu_aign=*/0);

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "  [ig=%d][oc_pos=%d] tdma_load_stride:\n"
                        "    tl_bias gaddr 0x%lx, laddr 0x%x, shape (%d, %d, "
                        "%d, %d), gstride (%d, %d, %d)\n",
                        ig, oc_pos, ga_bias + coeff_offset, tl_bias[coeff_flip]->start_address,
                        tl_bias[coeff_flip]->shape.n, tl_bias[coeff_flip]->shape.c,
                        tl_bias[coeff_flip]->shape.h, tl_bias[coeff_flip]->shape.w, bias_gstride.n,
                        bias_gstride.c, bias_gstride.h));

        if (args_.tiu_fmt == CVK_FMT_I8)
          ctx_.tdma_load_stride(
              tl_bias[coeff_flip], ga_bias + coeff_offset, bias_gstride);
        else if (args_.tiu_fmt == CVK_FMT_BF16)
          ctx_.tdma_load_stride(
              tl_bias[coeff_flip], ga_bias + coeff_offset * weight_gstride_w,
              bias_gstride);
        else {
          assert(0 && "dw-conv bias only support i8/bf16");
        }
      }

      // Weight shape for load != shape for tiu
      // bmk does not keep eu-align info, user need to update stride if shape changed
      tl_weight[coeff_flip]->shape = ctx_.tl_shape_t4(1, cur_oc, kh, kw);
      tl_weight[coeff_flip]->stride =
          ctx_.tl_default_stride(tl_weight[coeff_flip]->shape, args_.tiu_fmt,
                                /*eu_aign*/ 1);

      // uint64_t weight_offset = ga_weight + oc_pos * kh * kw;
      uint64_t weight_offset = ga_weight + oc_pos * weight_gstride.c;
      {
        // Same local address, different shape, stride
        cvk_tl_t tl_tmp;
        tl_tmp.start_address = tl_weight[coeff_flip]->start_address;
        tl_tmp.fmt = args_.tiu_fmt;
        // cant set ic_step, else shape will be wrong at multi-batch
        ASSERT(tl_weight[coeff_flip]->shape.n == 1);
        tl_tmp.shape = ctx_.tl_shape_t4(1, cur_oc, kh * kw, 1);
//        tl_tmp.shape = ctx_.tl_shape_t4(1, cur_oc, kh * kw, ic_step);
        tl_tmp.stride = ctx_.tl_default_stride(tl_tmp.shape, args_.tiu_fmt,
                                              /*eu_align=*/0);

        if (args_.tiu_fmt == CVK_FMT_I8)
          ctx_.tdma_load_stride(&tl_tmp, weight_offset, weight_gstride);
        else if (args_.tiu_fmt == CVK_FMT_BF16)
          ctx_.tdma_load_stride(&tl_tmp, weight_offset, weight_gstride);
        else {
          assert(0 && "dw-conv weight only supports i8/bf16");
        }
      }

      // split n
      for (int n_pos = 0; n_pos < input_n; n_pos += n_step) {
        int cur_n = std::min(input_n - n_pos, n_step);

        // split h
        for (int oh_pos = 0; oh_pos < oh; oh_pos += oh_step) {
          int cur_oh = std::min(oh - oh_pos, oh_step);

          int oh_top = oh_pos;
          int oh_bot = oh_top + cur_oh;
          int ih_top = std::max(oh_top * stride_h - pad_top, 0);
          int ih_bot = std::min((oh_bot - 1) * stride_h + kh_ext - pad_top, (int)inserted_input_height());
          ih_top = ceiling_func(ih_top, 1 + insert_height());
          ih_bot = ceiling_func(ih_bot, 1 + insert_height());
          int cur_ih = ih_bot - ih_top;

          int ph_top = 0;
          if (ih_top == 0) {
            ph_top = pad_top - oh_top * stride_h;
          } else {
            int gap = (oh_top * stride_h - pad_top) % (1 + insert_height());
            ph_top = (gap == 0) ? 0 : (1 + insert_height() - gap);
          }

          int ph_bot = 0;
          if (ih_bot == static_cast<int>(input_height())) {
            ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top -
                     inserted_input_height();
          } else {
            ph_bot = (oh_bot - 1) * stride_h + kh_ext - pad_top -
                     (ih_bot + (ih_bot - 1) * insert_height());
          }

          // split w
          for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step) {
            int cur_ow = std::min(ow - ow_pos, ow_step);

            int ow_left = ow_pos;
            int ow_right = ow_left + cur_ow;
            int iw_left = std::max(ow_left * stride_w - pad_left, 0);
            int iw_right = std::min((ow_right - 1) * stride_w + kw_ext - pad_left, (int)inserted_input_width());
            iw_left = ceiling_func(iw_left, 1 + insert_width());
            iw_right = ceiling_func(iw_right, 1 + insert_width());
            int cur_iw = iw_right - iw_left;

            int pw_left = 0;
            if (iw_left == 0) {
              pw_left = pad_left - ow_left * stride_w;
            } else {
              int gap = (ow_left * stride_w - pad_left) % (1 + insert_width());
              pw_left = (gap == 0) ? 0 : (1 + insert_width() - gap);
            }

            int pw_right = 0;
            if (iw_right == static_cast<int>(input_width())) {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left -
                         inserted_input_width();
            } else {
              pw_right = (ow_right - 1) * stride_w + kw_ext - pad_left -
                         (iw_right + (iw_right - 1) * insert_width());
            }

            LLVM_DEBUG(llvm::errs() << llvm::format(
                       "  [ig=%d][oc_pos=%d][n_pos=%d][oh_pos=%d][ow_pos=%d]"
                       " cur_oh %d, cur_ih %d, ih_top %d, ih_bot %d"
                       ", cur_ow %d, cur_iw %d, iw_left %d, iw_right %d ins_h:%d\n",
                       ig, oc_pos, n_pos, oh_pos, ow_pos, cur_oh, cur_ih, ih_top,
                       ih_bot, cur_ow, cur_iw, iw_left, iw_right, insert_height()));

            // Adjust current shape and stride
            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ofmap[flip]->shape = ctx_.tl_shape_t4(ofmap_multiplier * cur_n, cur_oc, cur_oh, cur_ow);
            tl_ofmap[flip]->stride =
                ctx_.tl_default_stride(tl_ofmap[flip]->shape, args_.tiu_fmt,
                                      /*eu_aign=*/1);

            // bmk does not keep eu-align info, user need to update stride if shape changed
            tl_ifmap[flip]->shape = ctx_.tl_shape_t4(cur_n, cur_oc, cur_ih, cur_iw);
            tl_ifmap[flip]->stride =
                ctx_.tl_default_stride(tl_ifmap[flip]->shape, args_.tiu_fmt,
                                      /*eu_align=*/1);

            uint64_t ifmap_offset =
                ga_ifmap + n_pos * ifmap_gstride.n + oc_pos * ifmap_gstride.c +
                ih_top * ifmap_gstride.h + iw_left * ifmap_gstride_w;

            if ((args_.input_fmt == CVK_FMT_I8) && (args_.tiu_fmt == CVK_FMT_I8))
              ctx_.tdma_load_stride(tl_ifmap[flip], ifmap_offset, ifmap_gstride);
            else if ((args_.input_fmt == CVK_FMT_BF16) && (args_.tiu_fmt == CVK_FMT_BF16))
              ctx_.tdma_load_stride(tl_ifmap[flip], ifmap_offset, ifmap_gstride);
            else {
              assert(0 && "dw-conv input only supports i8/bf16");
            }

            ctx_.parallel_disable();
            ctx_.parallel_enable();

            if (do_chl_quan) {
              // tiu logical shape != allocated local memory shape
              cvk_tl_t tl_chl_quan_tiu = {0};
              tl_chl_quan_tiu.start_address = tl_chl_quan[coeff_flip]->start_address;
              tl_chl_quan_tiu.fmt = tl_chl_quan[coeff_flip]->fmt;
              tl_chl_quan_tiu.shape = {1, tl_chl_quan[coeff_flip]->shape.c, 1, 1};
              tl_chl_quan_tiu.stride =
                  ctx_.tl_default_stride(tl_chl_quan_tiu.shape, args_.tiu_fmt,
                                        /*eu_align=*/0);

              cvk_tiu_depthwise_convolution_param_t param = {nullptr};
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.chl_quan_param = &tl_chl_quan_tiu;
              param.ins_h = (tl_ifmap[flip]->shape.h > 1) ? insert_height() : 0;
              param.ins_w = (tl_ifmap[flip]->shape.w > 1) ? insert_width() : 0;
              param.ins_last_h = 0;
              param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pw_left;
              param.pad_right = pw_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.has_bias = do_bias ? 1 : 0;
              param.relu_enable = fused_conv_relu;
              param.layer_id = layer_id;
              param.ins_val = 0;                             // symmetric quantization
              param.ins_fp = ctx_.convert_fp32_to_bf16(0.0); // symmetric quantization

              LLVM_DEBUG(llvm::errs() << llvm::format(
                         "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] dwconv:\n"
                         "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                         "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                         "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                         ig, n_pos, oh_pos, ow_pos, param.ifmap->start_address,
                         param.ifmap->shape.n, param.ifmap->shape.c,
                         param.ifmap->shape.h, param.ifmap->shape.w,
                         param.weight->start_address, param.weight->shape.n,
                         param.weight->shape.c, param.weight->shape.h,
                         param.weight->shape.w, param.ofmap->start_address,
                         param.ofmap->shape.n, param.ofmap->shape.c,
                         param.ofmap->shape.h, param.ofmap->shape.w));

              ctx_.tiu_depthwise_convolution(&param);

            } else {
              cvk_tiu_depthwise_pt_convolution_param_t param = {nullptr};
              param.ofmap = tl_ofmap[flip];
              param.ifmap = tl_ifmap[flip];
              param.weight = tl_weight[coeff_flip];
              param.bias = tl_bias[coeff_flip];
              param.ins_h = (tl_ifmap[flip]->shape.h > 1) ? insert_height() : 0;
              param.ins_w = (tl_ifmap[flip]->shape.w > 1) ? insert_width() : 0;
              param.ins_last_h = 0;
              param.ins_last_w = 0;
              param.pad_top = ph_top;
              param.pad_bottom = ph_bot;
              param.pad_left = pw_left;
              param.pad_right = pw_right;
              param.stride_h = stride_h;
              param.stride_w = stride_w;
              param.dilation_h = dilation_h;
              param.dilation_w = dilation_w;
              param.relu_enable = fused_conv_relu;
              param.rshift_bits = right_shift_width;
              param.ps32_mode = args_.ps32_output ? 2 : 0;
              param.layer_id = layer_id;
              param.ins_val = 0;                             // symmetric quantization
              param.ins_fp = ctx_.convert_fp32_to_bf16(0.0); // symmetric quantization

              LLVM_DEBUG(llvm::errs() << llvm::format(
                         "  [ig=%d][n_pos=%d][oh_pos=%d][ow_pos=%d] conv:\n"
                         "    ifmap la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                         "    weight la_addr 0x%x, shape (%d, %d, %d, %d)\n"
                         "    ofmap la_addr 0x%x, shape (%d, %d, %d, %d)\n",
                         ig, n_pos, oh_pos, ow_pos, param.ifmap->start_address,
                         param.ifmap->shape.n, param.ifmap->shape.c,
                         param.ifmap->shape.h, param.ifmap->shape.w,
                         param.weight->start_address, param.weight->shape.n,
                         param.weight->shape.c, param.weight->shape.h,
                         param.weight->shape.w, param.ofmap->start_address,
                         param.ofmap->shape.n, param.ofmap->shape.c,
                         param.ofmap->shape.h, param.ofmap->shape.w));

              ctx_.tiu_pt_depthwise_convolution(&param);
            }

            if (do_activation) {
              if (activation_arg && activation_arg[0] != 0.0f) {
                // leaky relu

                // bmk does not keep eu-align info, user need to update
                // stride if shape changed
                tl_relu->shape = tl_ofmap[flip]->shape;
                tl_relu->stride =
                    ctx_.tl_default_stride(tl_relu->shape, args_.tiu_fmt,
                                          /*eu_align=*/1);

                tl_neg->shape = tl_ofmap[flip]->shape;
                tl_neg->stride =
                    ctx_.tl_default_stride(tl_neg->shape, args_.tiu_fmt,
                                          /*eu_align=*/1);

                bool isIgnorePosPart = (activation_gt_scale == 0);
                bool isSlopeSmallerThanOne = ((activation_le_scale >> activation_le_rshift) == 0);

                if(isIgnorePosPart) {
                  cvk_tiu_mul_param_t p4 = {0};
                  p4.res_high = nullptr;
                  p4.res_low = tl_relu;
                  p4.a = tl_ofmap[flip];
                  p4.b_const.val = activation_le_scale;
                  p4.b_const.is_signed = true;
                  p4.b_is_const = 1;
                  p4.rshift_bits = activation_le_rshift;
                  p4.layer_id = layer_id;
                  p4.relu_enable = 0;
                  ctx_.tiu_mul(&p4);

                  if(isSlopeSmallerThanOne) {
                    cvk_tiu_max_param_t p1 = {0};
                    p1.max = tl_ofmap[flip];
                    p1.a = tl_ofmap[flip];
                    p1.b = tl_relu;
                    p1.b_is_const = 0;
                    p1.layer_id = layer_id;
                    ctx_.tiu_max(&p1);
                  } else {
                    cvk_tiu_min_param_t p1 = {0};
                    p1.min = tl_ofmap[flip];
                    p1.a = tl_ofmap[flip];
                    p1.b = tl_relu;
                    p1.b_is_const = 0;
                    p1.layer_id = layer_id;
                    ctx_.tiu_min(&p1);
                  }
                } else {
                  cvk_tiu_max_param_t p1 = {0};
                  p1.max = tl_relu;
                  p1.a = tl_ofmap[flip];
                  p1.b_is_const = 1;
                  p1.b_const.is_signed = 1;
                  p1.b_const.val = 0;
                  p1.layer_id = layer_id;
                  ctx_.tiu_max(&p1);

                  cvk_tiu_mul_param_t p2 = {0};
                  p2.res_high = nullptr;
                  p2.res_low = tl_relu;
                  p2.a = tl_relu;
                  p2.b_const.val = activation_gt_scale;
                  p2.b_const.is_signed = true;
                  p2.b_is_const = 1;
                  p2.rshift_bits = activation_gt_rshift;
                  p2.layer_id = layer_id;
                  p2.relu_enable = 0;
                  ctx_.tiu_mul(&p2);

                  cvk_tiu_min_param_t p3 = {0};
                  p3.min = tl_neg;
                  p3.a = tl_ofmap[flip];
                  p3.b_is_const = 1;
                  p3.b_const.val = 0;
                  p3.b_const.is_signed = 1;
                  p3.layer_id = layer_id;
                  ctx_.tiu_min(&p3);

                  cvk_tiu_mul_param_t p4 = {0};
                  p4.res_high = nullptr;
                  p4.res_low = tl_neg;
                  p4.a = tl_neg;
                  p4.b_const.val = activation_le_scale;
                  p4.b_const.is_signed = true;
                  p4.b_is_const = 1;
                  p4.rshift_bits = activation_le_rshift;
                  p4.layer_id = layer_id;
                  p4.relu_enable = 0;
                  ctx_.tiu_mul(&p4);

                  cvk_tiu_or_int8_param_t p5 = {0};
                  p5.res = tl_ofmap[flip];
                  p5.a = tl_relu;
                  p5.b = tl_neg;
                  p5.layer_id = layer_id;
                  ctx_.tiu_or_int8(&p5);
                }
              }
            }    // if (do_activation)

            ga_ofmap_cur[flip] =
                ga_ofmap + (n_pos * ofmap_gstride.n + oc_pos * ofmap_gstride.c +
                oh_top * ofmap_gstride.h + ow_left * ofmap_gstride_w) * (args_.ps32_output ? 2 : 1);

            if (first) {
              // postponse first result to next loop
              // loop0: LD0 TIU0
              // loop1: LD1 TIU1 SD0
              // loop2: LD2 TIU2 SD1
              first = 0;
            } else {
              uint32_t flip_back = 1 - flip;

              // Store back to global memory
              if ((args_.tiu_fmt == CVK_FMT_I8) && (args_.output_fmt == CVK_FMT_I8)) {
                ctx_.tdma_store_stride(tl_ofmap[flip_back],
                                      ga_ofmap_cur[flip_back], ofmap_gstride);
              } else if ((args_.tiu_fmt == CVK_FMT_BF16) && (args_.output_fmt == CVK_FMT_BF16)) {
                if (!args_.ps32_output) {
                  ctx_.tdma_store_stride(tl_ofmap[flip_back],
                                        ga_ofmap_cur[flip_back], ofmap_gstride);
                } else {
                  cvk_tl_t *tl_res = tl_ofmap[flip_back];
                  gaddr_t ga_dst = ga_ofmap_cur[flip_back];

                  cvi_backend_tl_bf16_ps32_to_fp32(ctx_, args_.layer_id,
                      tl_res->start_address,
                      tl_res->shape.n, tl_res->shape.c, tl_res->shape.h,
                      tl_res->shape.w);

                  cvi_backend_tl_store_fp32(ctx_, args_.layer_id,
                      ga_dst, tl_res->start_address,
                      tl_res->shape.n, tl_res->shape.c, tl_res->shape.h,
                      tl_res->shape.w);

                }
              } else {
                assert(0 && "dw-conv output only supports i8/bf16");
              }
            }

            flip = 1 - flip;

          }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

        }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      }  // for (int n_i = 0; n_i < n; ni += n_step)

      coeff_flip = 1 - coeff_flip;

    }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

    ctx_.parallel_disable();

    // the last iteration stored the other side, leave the last side not stored
    uint32_t flip_back = 1 - flip;

    // Store back to global memory
    if ((args_.tiu_fmt == CVK_FMT_I8) && (args_.output_fmt == CVK_FMT_I8)) {
      ctx_.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                            ofmap_gstride);
    } else if ((args_.tiu_fmt == CVK_FMT_BF16) && (args_.output_fmt == CVK_FMT_BF16)) {
      if (!args_.ps32_output) {
        ctx_.tdma_store_stride(tl_ofmap[flip_back], ga_ofmap_cur[flip_back],
                                    ofmap_gstride);
      } else {
        cvk_tl_t *tl_res = tl_ofmap[flip_back];
        gaddr_t ga_dst = ga_ofmap_cur[flip_back];

        cvi_backend_tl_bf16_ps32_to_fp32(ctx_, args_.layer_id,
            tl_res->start_address,
            tl_res->shape.n, tl_res->shape.c, tl_res->shape.h,
            tl_res->shape.w);

        cvi_backend_tl_store_fp32(ctx_, args_.layer_id,
            ga_dst, tl_res->start_address,
            tl_res->shape.n, tl_res->shape.c, tl_res->shape.h,
            tl_res->shape.w);
      }
    } else {
      assert(0 && "dw-conv output only supports i8/bf16");
    }

  }  // for (int group_i = 0; group_i < groups; ++groups)

  //
  // Release resource in reverse order
  //
  if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
    ctx_.lmem_free_tensor(tl_relu);
    ctx_.lmem_free_tensor(tl_neg);
  }
  if (do_chl_quan) {
    ctx_.lmem_free_tensor(tl_chl_quan[1]);
    ctx_.lmem_free_tensor(tl_chl_quan[0]);
  } else if (do_bias) {
    // only allocated without per-channel quantization
    ctx_.lmem_free_tensor(tl_bias[1]);
    ctx_.lmem_free_tensor(tl_bias[0]);
  }
  ctx_.lmem_free_tensor(tl_ofmap[1]);
  ctx_.lmem_free_tensor(tl_ofmap[0]);
  ctx_.lmem_free_tensor(tl_ifmap[1]);
  ctx_.lmem_free_tensor(tl_ifmap[0]);
  ctx_.lmem_free_tensor(tl_weight[1]);
  ctx_.lmem_free_tensor(tl_weight[0]);
}

// No tiling, no parallel.
bool Conv::canNoTile() {
  // H/W does not support group convolution.
  if (args_.groups > 1)
    return false;

  // Only support per-channel convolution.
  if (!args_.do_chl_quan)
    return false;

  // Hardware limit
  if ((group_input_channels() > getMaxChannelOfHardware()) ||
      (group_output_channels() > getMaxChannelOfHardware()) ||
      (input_height() > getMaxHeightOfHardware()) ||
      (input_width() > getMaxWidthOfHardware()))
    return false;

  int input_n = args_.input_n;
  int input_c = args_.input_c;
  int input_h = args_.input_h;
  int input_w = args_.input_w;
  int groups = args_.groups;
  int output_c = args_.output_c;
  int do_bias = args_.do_bias;
  // bool do_chl_quan = args_.do_chl_quan;
  int do_activation = args_.do_activation;
  float *activation_arg = args_.activation_arg;
  uint16_t kh = args_.kh;
  uint16_t kw = args_.kw;
  uint16_t dilation_h = args_.dilation_h;
  uint16_t dilation_w = args_.dilation_w;
  uint8_t pad_top = args_.pad_top;
  uint8_t pad_bottom = args_.pad_bottom;
  uint8_t pad_left = args_.pad_left;
  uint8_t pad_right = args_.pad_right;
  uint8_t stride_h = args_.stride_h;
  uint8_t stride_w = args_.stride_w;

  int ic = input_c / groups;
  int oc = output_c / groups;
  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;
  int oh = (inserted_input_height() + pad_top + pad_bottom - kh_extent) / stride_h + 1;
  int ow = (inserted_input_width() + pad_left + pad_right - kw_extent) / stride_w + 1;
  int ih = input_h;
  int iw = input_w;
  int n = input_n;

  assert(static_cast<uint32_t>(ic) == group_input_channels());
  assert(static_cast<uint32_t>(oc) == group_output_channels());
  assert(static_cast<uint32_t>(kh_extent) == dilated_kernel_height());
  assert(static_cast<uint32_t>(kw_extent) == dilated_kernel_width());
  assert(static_cast<uint32_t>(oh) == output_height());
  assert(static_cast<uint32_t>(ow) == output_width());

  uint32_t coeff_size = 0;

  if (do_bias) {
    cvk_tl_shape_t coeff_shape_9byte =
        ctx_.tl_shape_t4(1, oc, 1, 9);
    coeff_size += ctx_.lmem_tensor_to_size(coeff_shape_9byte,
                      args_.tiu_fmt, /*eu_align=*/0);
  } else {
    cvk_tl_shape_t coeff_shape_5byte =
        ctx_.tl_shape_t4(1, oc, 1, 5);
    coeff_size += ctx_.lmem_tensor_to_size(coeff_shape_5byte,
                      args_.tiu_fmt, /*eu_align=*/0);
  }

  // Add weight size
  coeff_size += ctx_.lmem_tensor_to_size(ctx_.tl_shape_t4(ic, oc, kh, kw),
                                         args_.tiu_fmt,/*eu_align=*/0);

  uint32_t total_needed = 0;

  uint32_t ofmap_size =
      ctx_.lmem_tensor_to_size(ctx_.tl_shape_t4(n, oc, oh, ow), args_.tiu_fmt,
                              /*eu_align=*/1);
  total_needed += ofmap_size;

  uint32_t ifmap_size =
      ctx_.lmem_tensor_to_size(ctx_.tl_shape_t4(n, ic, ih, iw), args_.tiu_fmt,
                              /*eu_align=*/1);
  total_needed += ifmap_size;

  total_needed += coeff_size;

  // Leaky relu need tl_neg, tl_relu.
  // tl_relu, tl_neg are not from tmda and not final output.
  if (do_activation && activation_arg && activation_arg[0] != 0.0f) {
    total_needed += 2 * ofmap_size;  // tl_relu + tl_neg
  }

  // LLVM_DEBUG(llvm::dbgs()
  //     << "canNoTile:\n"
  //     << "    layer_id " << args_.layer_id
  //     << ",   ifmap shape(" << input_n << ", " << input_c
  //     << ", " << input_h << ", " << input_w << ")\n"
  //     << "    kernel shape(oc=" << output_c
  //     << ", kh " << kh << ", kw " << kw << ", ic=" << input_c << ")\n"
  //     << "    total_needed " << static_cast<int>(total_needed)
  //     << ", lane_size " << getLmSizePerLane()
  //     << "\n");

  if (total_needed <= getLmSizePerLane()) {
    slices_.n_step = n;
    slices_.oc_step = oc;
    slices_.oh_step = oh;
    slices_.ow_step = ow;
    slices_.ih_step = ih;
    slices_.iw_step = iw;
    slices_.ic_step = ic;
    slices_.total_needed = total_needed;
    return true;
  }

  return false;
}

// No tiling, no parallel execution, maximized local memory utilization.
// 1. For activation compression.
// 2. For maximum TDMA load/store efficiency and local memory utilization
//    With tiu/tdma outstanding feature in 1822,
//    the backend does not need to split output channels to reduce TDMA
//    store latency.
//    Without it, the compiler has to implement inter-layer outstanding.
void Conv::convNoTile() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "cvi_backend_tg_fixed_conv_kernel w/ one tiu:\n"
      "    layer_id %d\n"
      "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
      "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
      "    kernel = (%d, %d), dilation = (%d, %d)\n"
      "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
      args_.layer_id, args_.ga_ifmap, args_.ga_ofmap, args_.ga_weight,
      args_.ga_bias, args_.input_n, args_.input_c, args_.input_h, args_.input_w,
      args_.groups, args_.output_c, args_.kh, args_.kw, args_.dilation_h,
      args_.dilation_w, args_.pad_top, args_.pad_bottom, args_.pad_left,
      args_.pad_right, args_.stride_h, args_.stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "    activation_gt_scale = %d, activation_gt_scale = %d\n"
      "    activation_le_rshift = %d, activation_le_scale = %d\n"
      "    do_activation = %d\n"
      "    do_ic_alignment = %d\n"
      "    store_compr_act %d, load_compr_act %d\n",
      args_.activation_gt_scale, args_.activation_gt_scale,
      args_.activation_le_rshift, args_.activation_le_scale,
      args_.do_activation, args_.do_ic_alignment,
      args_.store_compr_act, args_.load_compr_act));

  allocateAllLocalMem();

  std::vector<uint32_t> poss = {0, 0, 0, 0, 0};
  std::vector<uint32_t> indexes = {0, 0, 0};

  cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadBiasCmdType, poss, indexes[0]));

  cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadInputCmdType, poss, indexes[0]));

  cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::LoadWeightCmdType, poss, indexes[0]));

  cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::ComputCmdType, poss, indexes));

  cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
      CmdDescriptor::StoreOutputCmdType, poss, indexes[0]));

  auto intraCmdAnalysis =
      std::make_unique<IntraCmdParallelAnalysis>(cmdQueue_);
  intraCmdAnalysis->analyze();
  // intraCmdAnalysis->dumpStates();

  // Generate command
  generateCmd();

  deallocateAllLocalMem();
}

// Straightforward tiling, no double buffer
// Modified from convReuseActivation
void Conv::convNaive() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
      "convNaive =>\n"
      "  groups %d, ifmap (%d, %d, %d, %d), ofmap(%d, %d, %d, %d)\n"
      "  kernel (%d, %d), pad (top=%d, bot=%d, left=%d, right=%d)\n"
      "  stride (%d, %d), dilation (%d, %d)\n"
      "  do_bias %d, do_chl_quan %d\n"
      "  Slices (n_step=%d, oc_step=%d, oh_step=%d, ow_step=%d, ih_step=%d"
      ", iw_step=%d, ic_step=%d)\n"
      "  store_compr_act %d, load_compr_act %d\n",
      args_.groups, args_.input_n, args_.input_c, args_.input_h, args_.input_w,
      args_.input_n, args_.output_c, output_height(), output_width(),
      args_.kh, args_.kw, args_.pad_top, args_.pad_bottom, args_.pad_left,
      args_.pad_right, args_.stride_h, args_.stride_w, args_.dilation_h,
      args_.dilation_w, args_.do_bias, args_.do_chl_quan, slices_.n_step,
      slices_.oc_step, slices_.oh_step, slices_.ow_step, slices_.ih_step,
      slices_.ih_step, slices_.ic_step, args_.store_compr_act,
      args_.load_compr_act));

  // Pre-alloc maximum one-step size
  // The local memory release must be in reverse order.
  allocateAllLocalMem();

  auto loadInputCmd = [&](std::vector<uint32_t> poss, uint32_t cmdQueueIndex,
                          uint32_t icPos) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadInputCmdType, poss, cmdQueueIndex, icPos));
  };

  auto loadBiasCmd = [&](std::vector<uint32_t> poss, uint32_t cmdQueueIndex) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadBiasCmdType, poss, cmdQueueIndex));
  };

  auto loadWeightCmd = [&](std::vector<uint32_t> poss, uint32_t cmdQueueIndex,
                           uint32_t icPos) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::LoadWeightCmdType, poss, cmdQueueIndex, icPos));
  };

  auto computeCmd = [&](std::vector<uint32_t> poss,
      std::vector<uint32_t> indexes, uint32_t icPos) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::ComputCmdType, poss, indexes, icPos));
  };

  auto storeOutputCmd =
      [&](std::vector<uint32_t> poss, uint32_t cmdQueueIndex) {
    cmdQueue_.push_back(std::make_unique<CmdDescriptor>(
        CmdDescriptor::StoreOutputCmdType, poss, cmdQueueIndex));
  };

  // split groups
  for (uint32_t ig = 0; ig < groups(); ++ig) {
    std::vector<uint32_t> gmOutputPoss;
    // split oc
    for (uint32_t oc_pos = 0; oc_pos < group_output_channels();
         oc_pos += slices_.oc_step) {
      // split n
      for (uint32_t n_pos = 0; n_pos < batch_size(); n_pos += slices_.n_step) {
        // split h
        for (uint32_t oh_pos = 0; oh_pos < output_height();
            oh_pos += slices_.oh_step) {
          // split w
          for (uint32_t ow_pos = 0; ow_pos < output_width();
              ow_pos += slices_.ow_step) {
              gmOutputPoss = {n_pos, ig, oc_pos, oh_pos, ow_pos};
              std::vector<uint32_t> cur_weight_pos =
                  {/*n_pos=*/0, ig, oc_pos, /*oh_pos=*/0, /*ow_pos=*/0};

              loadBiasCmd(cur_weight_pos, /*flip=*/0);

              for (uint32_t ic_pos = 0; ic_pos < group_input_channels();
                   ic_pos += slices_.ic_step) {

                loadInputCmd({n_pos, /*g_pos=*/0, /*oc_pos=*/0, oh_pos, ow_pos},
                             /*flip*/0, ic_pos);

                loadWeightCmd(cur_weight_pos, /*flip=*/0, ic_pos);

                computeCmd(gmOutputPoss, {/*flip=*/0, /*flip=*/0, /*flip=*/0},
                           ic_pos);
              }

              storeOutputCmd(gmOutputPoss, /*flip=*/0);

            }  // for (int oc_i = 0; oc_i < oc; oc_i += oc_step

        }  // for (int oh_i = 0; oh_i < oh; oh_i += oh_step)

      }  // for (int n_i = 0; n_i < n; ni += n_step)

    }  // for (int ow_pos = 0; ow_pos < ow; ow_pos += ow_step)

  }  // for (int group_i = 0; group_i < groups; ++groups)


  auto intraCmdAnalysis =
      std::make_unique<IntraCmdParallelAnalysis>(cmdQueue_);
  intraCmdAnalysis->analyze();
  // intraCmdAnalysis->dumpStates();

  // Generate command
  generateCmd();

  deallocateAllLocalMem();

  LLVM_DEBUG(llvm::errs() << "<= convNaive\n");
}

void Conv::generateCmd() {
  auto genParallCmd = [&](uint32_t index) {
    if (cmdQueue_[index]->isParallelEnabled())
      ctx_.parallel_enable();
    else
      ctx_.parallel_disable();
  };

  for (uint32_t i = 0; i < cmdQueue_.size(); ++i) {
    CmdDescriptor::CmdTypeEnum cmdType = cmdQueue_[i]->getCmdType();
    std::vector<uint32_t> gmOutputPoss = cmdQueue_[i]->getGmOutputPoss();
    std::vector<uint32_t> lmIndexes = cmdQueue_[i]->getLmIndexes();
    uint32_t icPos = cmdQueue_[i]->getIcPos();

    if (cmdType == CmdDescriptor::LoadBiasCmdType) {
      loadBias(gmOutputPoss, lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::LoadInputCmdType) {
      loadInput(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::LoadWeightCmdType) {
      loadWeight(gmOutputPoss, lmIndexes[0], i, icPos);
    } else if (cmdType == CmdDescriptor::ComputCmdType) {
      compute(gmOutputPoss, lmIndexes, i, icPos);
    } else if (cmdType == CmdDescriptor::StoreOutputCmdType) {
      storeOutput(gmOutputPoss, lmIndexes[0], i);
    } else if (cmdType == CmdDescriptor::ParallelCmdType) {
      genParallCmd(i);
    } else {
      assert(0 && "Expect valid command");
    }
  }
}

// Priority:
//   1. No tiling
//   2. Reuse weight w/ double buffer
//   3. Reuse activation w/ double buffer
//   4. Tile w/ single buffer
//
void Conv::determineTilePolicy() {
  if (canNoTile()) {
    // No tiling should be the best condition
    tilePolicy_ = NoTilePolicyType;
  } else if (determineTileSize(/*useDoubleBuffer*/true)) {
    // Now we have cmdQueue_, we can use it to develop cost model later.
    if (args_.kh == 1 && args_.kw == 1)
      tilePolicy_ = ReuseActivationPolicyType;
    else
      tilePolicy_ = ReuseWeightPolicyType;
  } else if (determineTileSize(/*useDoubleBuffer*/false)) {
    tilePolicy_ = SingleBufferPolicyType;
  } else if (determinePs32TileSize(/*useDoubleBuffer=*/false)) {
    tilePolicy_ = SingleBufferPs32PolicyType;
  } else {
    assert(0 && "Expect valid tile policy");
    tilePolicy_ = MaxTilePolicyType;
  }
}

void Conv::doConvByTilePolicy() {

  configCModelDebug();

  switch (tilePolicy_) {
    case NoTilePolicyType:
      // Update tiling again for ic aligment.
      initializeTile();

      use_double_buffer_ = false;
      convNoTile();
    break;

    case SingleBufferPolicyType:
      use_double_buffer_ = false;
      convNaive();
    break;

    case SingleBufferPs32PolicyType:
      use_double_buffer_ = false;
      convNaive();
    break;

    case ReuseWeightPolicyType:
      use_double_buffer_ = true;
      convReuseWeight();
    break;

    case ReuseActivationPolicyType:
      use_double_buffer_ = true;
      convReuseActivation();
    break;

    default:
      return;
  }

  cModelDebug_.dump();
}

void cvi_backend_tg_fixed_conv_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias, int input_n,
    int input_c, int input_h, int input_w, int groups, int output_c,
    uint16_t kh, uint16_t kw, uint16_t dilation_h, uint16_t dilation_w,
    uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left, uint8_t pad_right,
    uint8_t insert_h, uint8_t insert_w, uint8_t stride_h, uint8_t stride_w,
    int do_bias, int do_activation, float activation_arg[],
    int activation_gt_scale, int activation_gt_rshift, int activation_le_scale,
    int activation_le_rshift, int right_shift_width, bool do_chl_quan,
    bool do_ic_alignment, bool store_compr_act, bool load_compr_act, int pad_value) {
  // this message is too long for llvm::format, so seperate it
  LLVM_DEBUG(llvm::errs() << llvm::format(
             "cvi_backend_tg_fixed_conv_kernel:\n"
             "    layer_id %d\n"
             "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
             "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
             "    kernel = (%d, %d), dilation = (%d, %d)\n"
             "    pad = (%d, %d, %d, %d), stride = (%d, %d)\n",
             layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n,
             input_c, input_h, input_w, groups, output_c, kh, kw,
             dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
             pad_right, stride_h, stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
             "    activation_gt_rshift = %d, activation_gt_scale = %d\n"
             "    activation_le_rshift = %d, activation_le_scale = %d\n"
             "    do_activation = %d\n"
             "    do_ic_alignment = %d\n"
             "    store_compr_act %d, load_compr_act %d\n",
             activation_gt_rshift, activation_gt_scale, activation_le_rshift,
             activation_le_scale, do_activation,
             do_ic_alignment, store_compr_act, load_compr_act));

  //
  // Convolution initialization
  //   Too many arguments come from pure-C api.
  //
  auto conv(std::make_unique<Conv>(ctx));
  conv->args_.ga_ifmap = ga_ifmap;
  conv->args_.ga_ofmap = ga_ofmap;
  conv->args_.ga_weight = ga_weight;
  conv->args_.ga_bias = ga_bias;
  conv->args_.input_n = input_n;
  conv->args_.input_c = input_c;
  conv->args_.input_h = input_h;
  conv->args_.input_w = input_w;
  conv->args_.groups = groups;
  conv->args_.output_c = output_c;
  conv->args_.kh = kh;
  conv->args_.kw = kw;
  conv->args_.dilation_h = dilation_h;
  conv->args_.dilation_w = dilation_w;
  conv->args_.pad_top = pad_top;
  conv->args_.pad_bottom = pad_bottom;
  conv->args_.pad_left = pad_left;
  conv->args_.pad_right = pad_right;
  conv->args_.insert_h = insert_h;
  conv->args_.insert_w = insert_w;
  conv->args_.stride_h = stride_h;
  conv->args_.stride_w = stride_w;
  conv->args_.do_bias = static_cast<bool>(do_bias);
  conv->args_.do_activation = static_cast<bool>(do_activation);
  conv->args_.activation_arg = activation_arg;
  conv->args_.activation_gt_scale = activation_gt_scale;
  conv->args_.activation_gt_rshift = activation_gt_rshift;
  conv->args_.activation_le_scale = activation_le_scale;
  conv->args_.activation_le_rshift = activation_le_rshift;
  conv->args_.right_shift_width = right_shift_width;
  conv->args_.do_chl_quan = do_chl_quan;
  conv->args_.layer_id = layer_id;
  conv->args_.do_ic_alignment = do_ic_alignment;
  conv->args_.store_compr_act = store_compr_act;
  conv->args_.load_compr_act = load_compr_act;
  conv->args_.pad_value = pad_value;
  // Mix-precision tdma load/store from dialect
  // E.g. input int8 -> tiu bf16 -> output fp32
  conv->args_.input_fmt = CVK_FMT_I8;
  conv->args_.output_fmt = CVK_FMT_I8;
  conv->args_.tiu_fmt = CVK_FMT_I8;

  // Global memory region from dialect
  conv->initializeGlobalMem();

  conv->initializeFusedActivation();
  conv->initializeTile();

  // For tdma
  ctx.set_layer_id(layer_id);

  // Try depthwise convolution.
  if (conv->isDwConv()) {
    if (conv->determineDwTileSize(/*useDoubleBuffer*/true))
      return conv->dwConv();
    else
      assert(0 && "DwConv does not support single buffer yet");
  }

  // For double convolution, weight and output already altered.
  // But the input is still unchanged and needs to restore original channels.
  // In dialect, ifmap tensor<1x3x85x85xi8>, weight tensor<64x4x5x5xi8>
  if (do_ic_alignment && (input_c % 2 != 0)) {
    assert(input_c >= 1);
    conv->args_.input_c = input_c + 1;
  }

  conv->determineTilePolicy();
  conv->doConvByTilePolicy();
}

void Conv::configCModelDebug() {
  // WZC-0, batch 8
  // [ig=0][oc_pos=736][n_pos=6][oh_pos=31][ow_pos=0][ic_pos=0]
  //cModelDebug_.assignOutput(21, {6, 0, 736, 31, 0});

  // onnx
  // cModelDebug_.assignOutput(1, {6, 0, 736, 31, 0});

  // WZC-6, batch 4 in onnx
  // [ig=0][oc_pos=1152][n_pos=2][oh_pos=15][ow_pos=34][ic_pos=0]
  // cModelDebug_.assignOutput(1, {2, 0, 1152, 15, 34});
}

void cvi_backend_tg_bf16_conv_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_ifmap,
    gaddr_t ga_ofmap, gaddr_t ga_weight, gaddr_t ga_bias,
    int input_n, int input_c, int input_h, int input_w, int groups,
    int output_c, uint16_t kh, uint16_t kw, uint16_t dilation_h,
    uint16_t dilation_w, uint8_t pad_top, uint8_t pad_bottom, uint8_t pad_left,
    uint8_t pad_right, uint8_t ins_h, uint8_t ins_w,
    uint8_t stride_h, uint8_t stride_w, int do_bias,
    int do_activation, bool fp32_output) {

  // this message is too long for llvm::format, so seperate it
  LLVM_DEBUG(llvm::errs() << llvm::format(
             "cvi_backend_tg_bf16_conv_kernel:\n"
             "    layer_id %d\n"
             "    bottom = %lx, top = %lx, weight = %lx, bias = %lx\n"
             "    nchw = (%d, %d, %d, %d), group = %d, oc = (%d)\n"
             "    kernel = (%d, %d), dilation = (%d, %d)\n"
             "    pad = (%d, %d, %d, %d), ins=(%d, %d) stride = (%d, %d)\n",
             layer_id, ga_ifmap, ga_ofmap, ga_weight, ga_bias, input_n,
             input_c, input_h, input_w, groups, output_c, kh, kw,
             dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
             pad_right, ins_h, ins_w, stride_h, stride_w));
  LLVM_DEBUG(llvm::errs() << llvm::format(
             "    do_activation = %d\n",
             do_activation));

  //
  // Convolution initialization
  //   Too many arguments come from pure-C api.
  //
  auto conv(std::make_unique<Conv>(ctx));
  conv->args_.ga_ifmap = ga_ifmap;
  conv->args_.ga_ofmap = ga_ofmap;
  conv->args_.ga_weight = ga_weight;
  conv->args_.ga_bias = ga_bias;
  conv->args_.input_n = input_n;
  conv->args_.input_c = input_c;
  conv->args_.input_h = input_h;
  conv->args_.input_w = input_w;
  conv->args_.groups = groups;
  conv->args_.output_c = output_c;
  conv->args_.kh = kh;
  conv->args_.kw = kw;
  conv->args_.dilation_h = dilation_h;
  conv->args_.dilation_w = dilation_w;
  conv->args_.pad_top = pad_top;
  conv->args_.pad_bottom = pad_bottom;
  conv->args_.pad_left = pad_left;
  conv->args_.pad_right = pad_right;
  conv->args_.insert_h = ins_h;
  conv->args_.insert_w = ins_w;
  conv->args_.stride_h = stride_h;
  conv->args_.stride_w = stride_w;
  conv->args_.do_bias = static_cast<bool>(do_bias);
  conv->args_.do_activation = static_cast<bool>(do_activation);
  conv->args_.layer_id = layer_id;

  // Mix-precision tdma load/store from dialect
  // E.g. input int8 -> tiu bf16 -> output fp32
  conv->args_.input_fmt = CVK_FMT_BF16;
  conv->args_.output_fmt = CVK_FMT_BF16;
  conv->args_.tiu_fmt = CVK_FMT_BF16;
  conv->args_.ps32_output = fp32_output;

  // Global memory region from dialect
  conv->initializeGlobalMem();

  conv->initializeFusedActivation();
  conv->initializeTile();

  // For tdma
  ctx.set_layer_id(layer_id);

  // Try depthwise convolution.
  if (conv->isDwConv()) {
    if (conv->determineDwTileSize(/*useDoubleBuffer*/true))
      return conv->dwConv();
    else
      assert(0 && "DwConv does not support single buffer yet");
  }

  conv->determineTilePolicy();
  conv->doConvByTilePolicy();
}
