
//===- FileUtilities.h - utilities for working with tensor files -------*- C++ -*-===//
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
// Common utilities for working with tensor files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <assert.h>
#include <string>
#include <atomic>
#include <fstream>

namespace mlir {

inline int align_up(int x, int n) {
  if (n <= 0) {
    return x;
  }
  return (x + n - 1) & ~(n - 1);
}

void setPixelAlign(std::string &chip_name, std::string &pixel_format,
                   uint32_t &y_align, uint32_t &w_align,
                   uint32_t &channel_align);

int aligned_image_size(int n, int c, int h, int w, std::string &pixel_format,
               int y_align, int w_align, int channel_align);

} // namespace mlir