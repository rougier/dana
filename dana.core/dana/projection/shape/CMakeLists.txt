# C Library
ADD_LIBRARY (dana_projection_shape SHARED
             shape.cc shape.h)
TARGET_LINK_LIBRARIES (dana_projection_shape
                       dana_core
                       boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_shape MODULE shape_export.cc)
SET_TARGET_PROPERTIES (_shape PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_shape dana_projection_shape)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/projection/shape)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/projection/shape)
INSTALL(TARGETS _shape LIBRARY DESTINATION ${SITE_PACKAGE}/projection/shape)
INSTALL(TARGETS dana_projection_shape LIBRARY DESTINATION lib)

