#include "tpuc/TPUCompressUtil.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

#define DEBUG_TYPE "compress-util-test"

namespace mlir {

void testCompress(void) {
  // 221
  uint8_t plainData[] = {
      0x00, 0x00, 0xfb, 0xf0, 0xf9, 0x00, 0xf7, 0x00,
      0x00, 0xfb, 0x00, 0x00, 0x08, 0x00, 0x0d, 0x00,
      0xfa, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0e,
      0x07, 0xfa, 0xf5, 0x00, 0xfa, 0xfb, 0xf2, 0x06,
      0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0xf3, 0x00,
      0x0f, 0x07, 0x00, 0x00, 0x08, 0x00, 0x00, 0xf8,
      0xee, 0x00, 0x00, 0x00, 0xf9, 0xbe, 0x00, 0xf5,
      0x0e, 0x00, 0xf7, 0x00, 0x00, 0x00, 0x0b, 0x00,
      0xf2, 0xeb, 0x16, 0x00, 0x00, 0x05, 0x00, 0x00,
      0xfb, 0x00, 0x00, 0xf0, 0x0a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x05, 0x05, 0x00, 0x0e, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00,
      0xf8, 0xf6, 0x00, 0xf3, 0x00, 0xf5, 0xf6, 0x00,
      0x00, 0x0a, 0x00, 0x00, 0x0e, 0x0a, 0x00, 0xfb,
      0x00, 0x00, 0x00, 0x00, 0xf9, 0xf5, 0x00, 0xf7,
      0x00, 0x17, 0x00, 0xf8, 0x00, 0x00, 0x06, 0x0b,
      0x00, 0xf7, 0x00, 0xfa, 0x00, 0xfb, 0x0c, 0x05,
      0xfb, 0x00, 0x00, 0x09, 0x0b, 0x05, 0xf8, 0x00,
      0x05, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00,
      0x09, 0x05, 0x0a, 0x00, 0x07, 0x00, 0x06, 0x06,
      0x00, 0x05, 0x00, 0xfb, 0x00, 0xf7, 0x00, 0x0b,
      0xf2, 0x00, 0x00, 0xf7, 0xe3, 0xf5, 0xf9, 0x00,
      0x00, 0xf5, 0x05, 0x00, 0x00, 0x09, 0x00, 0x09,
      0x00, 0x00, 0xf9, 0x00, 0x1d, 0xfb, 0x00, 0x07,
      0x0a, 0xf4, 0x00, 0xf0, 0xee, 0x08, 0x00, 0x00,
      0x00, 0x00, 0xf3, 0x0b, 0x00, 0x06, 0x00, 0x00,
      0xf2, 0xf4, 0x00, 0xfb, 0x00, 0xf4, 0x00, 0xf5,
      0x0e, 0x00, 0x00, 0xf0, 0x00
  };

  // 160
  uint8_t refCompressedData[] = {
      0x80, 0x00, 0x00, 0x10, 0x04, 0x04, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x3e, 0x52, 0x4e, 0x77, 0x5d, 0x35, 0x57, 0x54,
      0x4c, 0x3a, 0x5b, 0x55, 0x5b, 0x57, 0x00, 0x00,
      0x5c, 0x02, 0x07, 0x40, 0x86, 0x1f, 0x06, 0x70,
      0x80, 0x5d, 0xc0, 0xd4, 0xe1, 0x41, 0xe3, 0xa1,
      0x48, 0x80, 0x08, 0x83, 0x77, 0x18, 0x74, 0x6e,
      0x2c, 0x41, 0x08, 0x10, 0x64, 0x10, 0x3e, 0x00,
      0x60, 0xd9, 0x7b, 0x20, 0x21, 0x04, 0x01, 0x02,
      0x04, 0xf0, 0x1f, 0xc4, 0x03, 0x00, 0xac, 0x01,
      0xf8, 0x0f, 0xe0, 0x35, 0xc0, 0x21, 0x00, 0x19,
      0x46, 0x8e, 0x83, 0x38, 0x2c, 0x02, 0x80, 0xe2,
      0x8b, 0x19, 0xc0, 0x2e, 0xaa, 0x04, 0x21, 0xe2,
      0xe5, 0xc3, 0x27, 0xb6, 0x00, 0x80, 0xf8, 0x18,
      0x14, 0x18, 0x93, 0x28, 0x79, 0x82, 0x01, 0x9f,
      0x11, 0x4e, 0x00, 0x88, 0x16, 0x20, 0x21, 0x85,
      0x8c, 0x67, 0x6e, 0x00, 0xb8, 0x1a, 0x04, 0x1a,
      0x08, 0x88, 0x41, 0x20, 0x1f, 0x62, 0x5f, 0x45,
      0x18, 0x41, 0x80, 0x78, 0x8c, 0xe0, 0xe0, 0x03,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  const int refBlockSize = 256;
  int blockSize =
      getCompressedDataSize(sizeof(plainData)/sizeof(char), /*dataType*/0);

  if (blockSize != refBlockSize)
    llvm::errs() << "Error ! blockSize(" << blockSize
                 << ") != refBlockSize(" << refBlockSize << ")\n";
  assert(blockSize == refBlockSize && "Expect same block size");

  CompressCommandInfo refCmdInfo;
  refCmdInfo.signedness = 1;
  refCmdInfo.is_bfloat16 = 0;
  refCmdInfo.bias0 = 0x04;
  refCmdInfo.bias1 = 0x04;
  refCmdInfo.zero_guard_en = 0;

  CompressCommandInfo cmdInfo;
  std::memset(&cmdInfo, 0, sizeof(cmdInfo));
  cmdInfo.signedness = 1; // int8
  cmdInfo.is_bfloat16 = 0;
  getCompressParameter(plainData, sizeof(plainData), cmdInfo.signedness,
                       cmdInfo.is_bfloat16, &cmdInfo);
  if (cmdInfo.bias0 != refCmdInfo.bias0)
    llvm::errs() << "Error ! bias0(" << cmdInfo.bias0
                 << ") != refBias0(" << refCmdInfo.bias0;
  assert(cmdInfo.bias0 == refCmdInfo.bias0 && "Expect same bias0");

  if (cmdInfo.bias1 != refCmdInfo.bias1)
    llvm::errs() << "Error ! bias1(" << cmdInfo.bias1
                 << ") != refBia1(" << refCmdInfo.bias1;
  assert(cmdInfo.bias1 == refCmdInfo.bias1 && "Expect same bias1");

  auto blockData = std::make_unique<std::vector<uint8_t> >(blockSize);
  auto compressedData = blockData->data();
  int compressDataSize = 0;
  compressInt8Data(plainData, sizeof(plainData), compressedData,
                   &compressDataSize, &cmdInfo);
  if (compressDataSize != sizeof(refCompressedData))
    llvm::errs() << "Error ! compressDataSize(" << compressDataSize
                 << " != refCompressDataSize(" << sizeof(refCompressedData)
                 << "\n";
  assert(compressDataSize == sizeof(refCompressedData) &&
         "Expect same compress data size");

  for (int i = 0; i < (int)sizeof(refCompressedData); ++i) {
    if (compressedData[i] != refCompressedData[i]) {
      llvm::errs() << "Error ! compressedData[" << i << "]("
                   << compressedData[i]
                   << ") != refCompressedData[" << i << "]("
                   << refCompressedData[i] << ")\n";
      assert(compressedData[i] == refCompressedData[i] &&
             "Expect same compressed data");
    }
  }

  llvm::errs() << "Compress test pass !\n";
}

} // namespace mlir
