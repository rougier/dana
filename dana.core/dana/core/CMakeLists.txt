# C Library
SET (SRC
environment.cc  layer.h  model.cc    object.h       random.cc  unit.h
environment.h   link.cc  model.h     observable.cc  random.h
event.cc        link.h   network.cc  observable.h   spec.cc
event.h         map.cc   network.h   observer.cc    spec.h
layer.cc        map.h    object.cc   observer.h     unit.cc)

ADD_LIBRARY (dana_core SHARED ${SRC})
TARGET_LINK_LIBRARIES(dana_core boost_python ${PYTHON_LIBRARIES} ${LIBXML2_LIBRARIES})

# Python module
ADD_LIBRARY (_core MODULE core_export.cc)
SET_TARGET_PROPERTIES (_core PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(_core dana_core)

# Install targets
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
INSTALL(FILES ${files} DESTINATION include/dana/core)
INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" DESTINATION ${SITE_PACKAGE}/core)
INSTALL(TARGETS _core LIBRARY DESTINATION ${SITE_PACKAGE}/core)
INSTALL(TARGETS dana_core LIBRARY DESTINATION lib)
