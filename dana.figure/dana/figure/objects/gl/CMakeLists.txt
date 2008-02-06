# C Library
SET (SRC array.cc array.h)
ADD_LIBRARY (dana_figure_objects_gl SHARED ${SRC})
TARGET_LINK_LIBRARIES(dana_figure_objects_gl
  dana_figure boost_python ${PYTHON_LIBRARIES} ${OPENGL_LIBRARIES})

# Python module
ADD_LIBRARY (_gl MODULE figure_objects_gl_export.cc)
SET_TARGET_PROPERTIES (_gl PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_gl dana_figure_objects_gl)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/figure/objects/gl)
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
INSTALL(FILES ${files} DESTINATION ${SITE_PACKAGE}/figure/objects/gl)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/figure/objects/gl)
INSTALL(TARGETS _gl LIBRARY DESTINATION ${SITE_PACKAGE}/figure/objects/gl)
INSTALL(TARGETS dana_figure_objects_gl LIBRARY DESTINATION lib)
