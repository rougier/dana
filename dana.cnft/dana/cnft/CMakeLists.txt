# C Library
SET (SRC
asynchronous_unit.cc  environment.h  spec.h      synchronous_unit.h
asynchronous_unit.h   environment.cc  spec.cc    synchronous_unit.cc)
ADD_LIBRARY (dana_cnft SHARED ${SRC})
TARGET_LINK_LIBRARIES(dana_cnft dana_core boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_cnft MODULE cnft_export.cc)
SET_TARGET_PROPERTIES (_cnft PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_cnft dana_cnft)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/cnft)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/cnft)
INSTALL(TARGETS _cnft LIBRARY DESTINATION ${SITE_PACKAGE}/cnft)
INSTALL(TARGETS dana_cnft LIBRARY DESTINATION lib)
