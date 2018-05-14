cmake_minimum_required( VERSION 2.8 )
project (practice)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3")

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include_directories( ${PROJECT_SOURCE_DIR}/include )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

# sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# g2o Eigen Cholmod
find_package(G2O REQUIRED)
include_directories(
  ${G2O_INCLUDE_DIRS}
  "/usr/include/eigen3"
)

find_package( Cholmod REQUIRED )
include_directories( ${CHOLMOD_INCLUDE_DIR} )

# OpenCV
find_package( OpenCV 3.2.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
# difference between CERES_INCLUDE_DIRS and CERES_INCLUDE_DIR?

# gtsam
find_package( GTSAMCMakeTools )
find_package( GTSAM REQUIRED )
include_directories( ${GTSAM_INCLUDE_DIR} )
# gtsam


## executable
add_executable(find_min src/find_min.cpp)
target_link_libraries(find_min ${CERES_LIBRARIES} )

add_executable(curve_fitting src/curve_fitting.cpp)
target_link_libraries(curve_fitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
