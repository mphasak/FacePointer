CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT( NoseTracking )

IF( WIN32 )
  IF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        SET( CMAKE_PLATFORM_NAME "x64" )
  ELSE( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        SET( CMAKE_PLATFORM_NAME "Win32" )
  ENDIF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
ELSEIF (UNIX )
  IF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        SET( CMAKE_PLATFORM_NAME "unix64" )
  ELSE( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        SET( CMAKE_PLATFORM_NAME "unix32" )
  ENDIF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
ELSEIF ()
  SET( CMAKE_PLATFORM_NAME "Unknown" )
ENDIF ()

MESSAGE( STATUS "PLATFORM:" ${CMAKE_PLATFORM_NAME} )

SET( EXE_SOURCE_FILES
        FacePointer.cpp
)

# Set the executables's name
SET( TARGET_EXE trackNose )

# Find OpenCV
FIND_PACKAGE( OpenCV REQUIRED )
#FIND_PACKAGE( LibSVM REQUIRED )

#INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/libs/ImageProcessingLib )

#LINK_DIRECTORIES( ${CMAKE_BINARY_DIR}/libs/ImageProcessingLib/${CMAKE_CFG_INTDIR} )

ADD_EXECUTABLE( ${TARGET_EXE} ${EXE_SOURCE_FILES})

#ADD_DEPENDENCIES( ${TARGET_EXE} ImageProcessing)

TARGET_LINK_LIBRARIES( ${TARGET_EXE} ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES( ${TARGET_EXE} ${LIBSVM_LIBRARY})

#ADD_SUBDIRECTORY( libs )
#ADD_SUBDIRECTORY( apps )
