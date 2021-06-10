#ifndef TPUC_ALL_PASSES
#define TPUC_ALL_PASSES

include "mlir/Pass/PassBase.td"

def DoPrintTpuOp : FunctionPass<"print-tpu-op-info"> {
  let summary = "Print TPU operation information.";
  let constructor = "mlir::createPrintTpuOpPass()";
}

def DoTpuQuant : FunctionPass<"tpu-quant"> {
  let summary = "Do quantization on TPU Ops";
  let constructor = "mlir::createTpuQuantPass()";
}

def DoImportCalibrationTable : FunctionPass<"import-calibration-table"> {
  let summary = "Import calibration table from external tools";
  let constructor = "mlir::createImportCalibrationTablePass()";
}

def DoTgOpTile : FunctionPass<"tg-op-tile"> {
  let summary = "TG Op tiling";
  let constructor = "mlir::createTgOpTilePass()";
}

def DoTgLeakyReluFuse : FunctionPass<"tg-fuse-leakyrelu"> {
  let summary = "TG Op tiling";
  let constructor = "mlir::createTgFuseLeakyReluPass()";
}

def DoEliminateDeadcode : FunctionPass<"dce"> {
  let summary = "Dead Code Eliminate such as tdma";
  let constructor = "mlir::createEliminateDeadcodePass()";
}

def DoGroupOps : FunctionPass<"group-ops"> {
  let summary = "Group ops together to speedup";
  let constructor = "mlir::createGroupOpsPass()";
}

def DoDeepFusionSimple : FunctionPass<"deep-fusion-simple"> {
  let summary = "Apply simple deep fusion.";
  let constructor = "mlir::createDeepFusionSimple()";
}

def DoDeepFusionOpt : FunctionPass<"deep-fusion-opt"> {
  let summary = "deep fusion opt";
  let constructor = "mlir::createDeepFusionOpt()";
}

def DoDeepFusionGroupSlice : FunctionPass<"deep-fusion-group-slice"> {
  let summary = "Apply deep fusion slice.";
  let constructor = "mlir::createDeepFusionGroupSlice()";
}

def DoDivideOpsToFunc : FunctionPass<"divide-ops-to-func"> {
  let summary = "divide ops into functions";
  let constructor = "mlir::createDivideOpsToFuncPass()";
}

def DoRefactorOddIcConv : FunctionPass<"conv-ic-alignment"> {
  let summary = "Enable padding odd ic to even to enable double conv for conv layer";
  let constructor = "mlir::createRefactorOddIcConvPass()";
}

def DoFuseReshape : FunctionPass<"fuse-reshape"> {
  let summary = "Fuse the input and the output of reshape op";
  let constructor = "mlir::createFuseReshapePass()";
}

def DoConvertSwishToRelu : FunctionPass<"convert-swish-to-relu"> {
  let summary = "Convert Swish Activation to Relu";
  let constructor = "mlir::createConvertSwishToReluPass()";
}

def DoConvertClipToRelu6 : FunctionPass<"convert-clip-to-relu6"> {
  let summary = "Convert clip to relu6";
  let constructor = "mlir::createConvertClipToRelu6Pass()";
}

def DoTpuLower : FunctionPass<"tpu-lower"> {
  let summary = "Lower TPU Dialect from TPU Ops to TPU_TG Ops";
  let constructor = "mlir::createTpuLowerPass()";
}

def DoConvertPriorBox : FunctionPass<"convert-priorbox-to-loadweight"> {
  let summary = "convert priorbox to leadweight to save each priorbox result";
  let constructor = "mlir::createConvertPriorBoxPass()";
}

def DoMoveConvStrideToEltwiseOp : FunctionPass<"eltwise-early-stride"> {
  let summary = "Refactor hStride of elt and conv op";
  let constructor = "mlir::createMoveConvStrideToEltwiseOpPass()";
}

def DoMergeConvConvPoolOp : FunctionPass<"merge-conv-conv-pool"> {
  let summary = "Merge conv-conv-pool pattern";
  let constructor = "mlir::createMergeConvConvPoolPass()";
}

def DoMergePermuteOp : FunctionPass<"merge-permute"> {
  let summary = "Merge permute pattern";
  let constructor = "mlir::createMergePermutePass()";
}

def DoAssignChipName : FunctionPass<"assign-chip-name"> {
  let summary = "Assign Chip Name to each tpu op";
  let constructor = "mlir::createAssignChipNamePass()";
}

def DoConvertUpsampleToDeconv : FunctionPass<"convert-upsample-to-deconv"> {
  let summary = "Convert a upsample operation to deconv";
  let constructor = "mlir::createConvertUpsampleToDeconvPass()";
}

def DoFusePad : FunctionPass<"fuse-pad"> {
  let summary = "Fuse pad op into next op (pooling/crop etc)";
  let constructor = "mlir::createFusePadPass()";
}

def DoConvertBnToScale : FunctionPass<"convert-bn-to-scale"> {
  let summary = "Convert a BN operation to Scale operation";
  let constructor = "mlir::createConvertBnToScalePass()";
}

def DoAssignNeuronAddress : FunctionPass<"assign-neuron-address"> {
  let summary = "Assign address to each neuron";
  let constructor = "mlir::createAssignNeuronAddressPass()";
}

def DoFuseAsymmetricZeroPoint : FunctionPass<"fuse-asymmetric-zero-point"> {
  let summary = "Fuse asymmetric zero point";
  let constructor = "mlir::createFuseAsymmetricZeroPointPass()";
}

def DoDecomposeNormalize : FunctionPass<"normalize-decompose"> {
  let summary = "Decompose Normalize to ltwise(prod)+conv2D+sqrt+";
  let constructor = "mlir::createDecomposeNormalizePass()";
}

def DoFoldScale : FunctionPass<"fold-scale"> {
  let summary = "Fold two consecutive scale operations into one";
  let constructor = "mlir::createFoldScalePass()";
}

def DoMergeScaleIntoConv : FunctionPass<"merge-scale-into-conv"> {
  let summary = "Merge scale op into conv op";
  let constructor = "mlir::createMergeScaleIntoConvPass()";
}

def DoConvertScaleToDWConv : FunctionPass<"convert-scale-to-dwconv"> {
  let summary = "Convert a scale operation to a dwconv operation";
  let constructor = "mlir::createConvertScaleToDWConvPass()";
}

def DoReorderOp : FunctionPass<"reorder-op"> {
  let summary = "Reorder OPs to make defs closing to its uses";
  let constructor = "mlir::createReorderOpPass()";
}

def DoCompressActivation : FunctionPass<"compress-activation"> {
  let summary = "Compress activation";
  let constructor = "mlir::createCompressActivationPass()";
}

def DoFuseRelu : FunctionPass<"fuse-relu"> {
  let summary = "Fuse relu op into previous op (conv/eltwise etc)";
  let constructor = "mlir::createFuseReluPass()";
}

def DoAssignWeightAddress : FunctionPass<"assign-weight-address"> {
  let summary = "Convert .npz weight file into a .bin file, ";
  let constructor = "mlir::createAssignWeightAddressPass()";
}

def DoAssignLayerId : FunctionPass<"assign-layer-id"> {
  let summary = "Assign layer id to each tpu op";
  let constructor = "mlir::createAssignLayerIdPass()";
}

def DoCompressWeight : FunctionPass<"compress-weight"> {
  let summary = "Compress weight";
  let constructor = "mlir::createCompressWeightPass()";
}

def DoGenPseudoWeightNpz : FunctionPass<"gen-pseudo-weight-npz"> {
  let summary = "Generic fake weight npz file if only have mlir file";
  let constructor = "mlir::createGenPseudoWeightNpzPass()";
}

def DoAddTpuPreprocess: FunctionPass<"add-tpu-preprocess"> {
  let summary = "Add tpu preprocess (crop, mean, scale, swap channel) to input";
  let constructor = "mlir::createAddTpuPreprocessPass()";
}

def TpucCanonicalizer : Pass<"canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createTpucCanonicalizerPass()";
}

#endif // MLIR_TRANSFORMS_PASSES
