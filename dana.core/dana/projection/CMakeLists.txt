# C Library
ADD_LIBRARY (dana_projection SHARED projection.cc projection.h)
TARGET_LINK_LIBRARIES (dana_projection
                       dana_core
                       dana_projection_distance
                       dana_projection_profile
                       dana_projection_shape
                       dana_projection_density
                       boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_projection MODULE projection_export.cc)
SET_TARGET_PROPERTIES (_projection PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_projection dana_projection)

# Install targets
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/projection)
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/projection)
INSTALL(TARGETS _projection LIBRARY DESTINATION ${SITE_PACKAGE}/projection)
INSTALL(TARGETS dana_projection LIBRARY DESTINATION lib)

ADD_SUBDIRECTORY (density)
ADD_SUBDIRECTORY (distance)
ADD_SUBDIRECTORY (profile)
ADD_SUBDIRECTORY (shape)
