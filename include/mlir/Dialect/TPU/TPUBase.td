//===-- TPUOps.td - TPU dialect operation definitions ------*- tablegen -*-===//
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
// Defines some operations of the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_BASE
#define TPU_BASE

include "mlir/IR/OpBase.td"

def TPU_Dialect : Dialect {
  let name = "tpu";

  let description = [{
    The TPU dialect.

    This dialect maps to TPU operations.
  }];

  let cppNamespace = "tpu";
}

//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

class TPU_TG_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

class TPU_TL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

class CPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TPU type class.
//===----------------------------------------------------------------------===//

// Tensor or None type.
class TPU_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
    AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

// MemRef or None type.
class TPU_MemRefOfOrNone<list<Type> allowedTypes, string description = ""> :
    AnyTypeOf<[MemRefOf<allowedTypes>, NoneType], description>;

#endif // TPU_BASE
