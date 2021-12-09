//===- FileUtilities.cpp - utilities for working with files ---------------===//
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
// Definitions of common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Support/PixelHelper.h"

using namespace mlir;

namespace mlir {

void setPixelAlign(std::string &chip_name, std::string &pixel_format,
                   uint32_t &y_align, uint32_t &w_align,
                   uint32_t &channel_align) {
  if ("cv183x" == chip_name) {
    y_align = 32;
    w_align = 32;
    channel_align = 0x1000;
  } else {
    y_align = 64;
    w_align = 64;
    channel_align = 64;
  }
  if ("YUV420_PLANAR" == pixel_format) {
    y_align = w_align * 2;
  }
}

int aligned_image_size(int n, int c, int h, int w, std::string &pixel_format,
               int y_align, int w_align, int channel_align) {
  if ("YUV420_PLANAR" == pixel_format) {
    assert(c == 3);
    int y_w_aligned = align_up(w, y_align);
    int uv_w_aligned = align_up(w / 2, w_align);
    int u = align_up(h * y_w_aligned, channel_align);
    int v = align_up(u + h / 2 * uv_w_aligned, channel_align);
    int n_stride = align_up(v + h / 2 * uv_w_aligned, channel_align);
    return n * n_stride;
  } else if ("YUV_NV21" == pixel_format || "YUV_NV12" == pixel_format) {
    int y_w_aligned = align_up(w, y_align);
    int uv_w_aligned = align_up(w, w_align);
    int uv = align_up(h * y_w_aligned, channel_align);
    int n_stride = align_up(uv + h / 2 * uv_w_aligned, channel_align);
    return n * n_stride;
  } else if ("RGB_PLANAR" == pixel_format || "BGR_PLANAR" == pixel_format ||
             "ARGB_PLANAR" == pixel_format) {
    int aligned_w = align_up(w, w_align);
    int n_stride = align_up(aligned_w * h, channel_align) * c;
    return n * n_stride;
  } else if ("RGB_PACKED" == pixel_format || "BGR_PACKED" == pixel_format ||
             "GRAYSCALE" == pixel_format) {
    int aligned_w = align_up(w * c, w_align);
    int n_stride = aligned_w * h;
    return n * n_stride;
  } else {
    assert(0 && "unsupported pixel format");
    return 0;
  }
}

} // namespace mlir
