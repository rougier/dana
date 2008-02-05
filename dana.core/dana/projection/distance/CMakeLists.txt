# C Library
ADD_LIBRARY (dana_projection_distance SHARED
             distance.cc distance.h)
TARGET_LINK_LIBRARIES (dana_projection_distance
                       dana_core
                       boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_distance MODULE distance_export.cc)
SET_TARGET_PROPERTIES (_distance PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_distance dana_projection_distance)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/projection/distance)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/projection/distance)
INSTALL(TARGETS _distance LIBRARY DESTINATION ${SITE_PACKAGE}/projection/distance)
INSTALL(TARGETS dana_projection_distance LIBRARY DESTINATION lib)

