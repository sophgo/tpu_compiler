add_executable(tpuc-opt
  tpuc-opt.cpp
)
target_link_libraries(
  tpuc-opt
  MLIRAnalysis
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRTPU
  MLIROptLib
  TPUCSupport
)
install(TARGETS tpuc-opt DESTINATION bin)