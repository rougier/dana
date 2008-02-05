
# C Library
ADD_LIBRARY (dana_projection_density SHARED
             density.cc density.h)
TARGET_LINK_LIBRARIES (dana_projection_density
                       dana_core
                       boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_density MODULE density_export.cc)
SET_TARGET_PROPERTIES (_density PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_density dana_projection_density)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/projection/density)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/projection/density)
INSTALL(TARGETS _density LIBRARY DESTINATION ${SITE_PACKAGE}/projection/density)
INSTALL(TARGETS dana_projection_density LIBRARY DESTINATION lib)

