add_executable(tpuc-interpreter
  tpuc-interpreter.cpp
)
target_link_libraries(
  tpuc-interpreter
  MLIRAnalysis
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRTPU
  MLIROptLib
  TPUCSupport
)
install(TARGETS tpuc-interpreter DESTINATION bin)