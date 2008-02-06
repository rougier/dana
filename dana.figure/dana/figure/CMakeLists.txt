# C Library
SET (SRC
color.cc  colormap.cc  colormaps.cc  object.h
color.h   colormap.h   colormaps.h   object.cc)
ADD_LIBRARY (dana_figure SHARED ${SRC})
TARGET_LINK_LIBRARIES(dana_figure boost_python ${PYTHON_LIBRARIES} ${OPENGL_LIBRARIES})

# Python module
ADD_LIBRARY (_figure MODULE figure_export.cc)
SET_TARGET_PROPERTIES (_figure PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_figure dana_figure)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/figure)
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
INSTALL(FILES ${files} DESTINATION ${SITE_PACKAGE}/figure/)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/figure)
INSTALL(TARGETS _figure LIBRARY DESTINATION ${SITE_PACKAGE}/figure)
INSTALL(TARGETS dana_figure LIBRARY DESTINATION lib)

ADD_SUBDIRECTORY (objects)
