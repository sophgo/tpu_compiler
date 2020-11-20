add_executable(tpuc-translate
  tpuc-translate.cpp
)
target_link_libraries(
  tpuc-translate
  MLIRParser
  MLIRPass
  MLIRTPU
  MLIRTPUTranslate
  MLIRTranslation
  MLIRSupport
  TPUCSupport
)
install(TARGETS tpuc-translate DESTINATION bin)