#ifndef TG_OP_TILE_PASS_H
#define TG_OP_TILE_PASS_H

namespace mlir {
namespace tpu {

void PopulateSplitPoolPatterns(MLIRContext *context,
                               OwningRewritePatternList *patterns,
                               MInfo &mInfo);

} // namespace tpu
} // namespace mlir

#endif // TG_OP_TILE_PASS_H
