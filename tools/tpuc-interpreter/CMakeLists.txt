add_executable(tpuc-interpreter
  tpuc-interpreter.cpp
)
target_link_libraries(
  tpuc-interpreter
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRTPU
  MLIRSupport
  TPUCSupport
)
install(TARGETS tpuc-interpreter DESTINATION bin)