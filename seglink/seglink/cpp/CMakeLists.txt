CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(seglink)

#EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3';import tensorflow as tf; print(tf.sysconfig.get_lib())"
#                OUTPUT_VARIABLE TF_LIB)

#MESSAGE(STATUS "Found TF_LIB: " ${TF_LIB})
#-L$TF_LIB -ltensorflow_framework
# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3';import tensorflow as tf; print(tf.sysconfig.get_include())"
                OUTPUT_VARIABLE TF_INC)



MESSAGE(STATUS "Found TF_INC: " ${TF_INC})



# boost
# find_package (Boost REQUIRED COMPONENTS graph)

#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


# target
#link_directories(${TF_LIB})
INCLUDE_DIRECTORIES(${TF_INC})
#LINK_LIBRARIES(${TF_LIB})
ADD_LIBRARY(seglink SHARED 
  utilities.h
  sample_crop_bbox_op.cc
  encode_groundtruth_op.cc
  decode_segments_links_op.cc
  combine_segments_op.cc
  detection_mask_op.cc
  clip_rboxes_op.cc
  polygons_to_rboxes_op.cc
  project_polygons_op.cc)
