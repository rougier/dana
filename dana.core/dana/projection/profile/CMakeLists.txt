# C Library
ADD_LIBRARY (dana_projection_profile SHARED
             profile.cc profile.h)
TARGET_LINK_LIBRARIES (dana_projection_profile
                       dana_core
                       boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_profile MODULE profile_export.cc)
SET_TARGET_PROPERTIES (_profile PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_profile dana_projection_profile)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/projection/profile)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/projection/profile)
INSTALL(TARGETS _profile LIBRARY DESTINATION  ${SITE_PACKAGE}/projection/profile)
INSTALL(TARGETS dana_projection_profile LIBRARY DESTINATION lib)

