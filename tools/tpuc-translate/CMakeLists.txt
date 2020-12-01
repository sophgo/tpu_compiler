add_executable(tpuc-translate
  tpuc-translate.cpp
)
target_link_libraries(tpuc-translate
  MLIRTPUTranslate
  MLIRTPU
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransformUtils
  MLIRTranslation
  MLIRSupport
  TPUCSupport
)
install(TARGETS tpuc-translate DESTINATION bin)