# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build

# Include any dependencies generated for this target.
include CMakeFiles/dift.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dift.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dift.dir/flags.make

CMakeFiles/dift.dir/main.cpp.o: CMakeFiles/dift.dir/flags.make
CMakeFiles/dift.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dift.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dift.dir/main.cpp.o -c /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/main.cpp

CMakeFiles/dift.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dift.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/main.cpp > CMakeFiles/dift.dir/main.cpp.i

CMakeFiles/dift.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dift.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/main.cpp -o CMakeFiles/dift.dir/main.cpp.s

CMakeFiles/dift.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/dift.dir/main.cpp.o.requires

CMakeFiles/dift.dir/main.cpp.o.provides: CMakeFiles/dift.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/dift.dir/build.make CMakeFiles/dift.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/dift.dir/main.cpp.o.provides

CMakeFiles/dift.dir/main.cpp.o.provides.build: CMakeFiles/dift.dir/main.cpp.o

# Object files for target dift
dift_OBJECTS = \
"CMakeFiles/dift.dir/main.cpp.o"

# External object files for target dift
dift_EXTERNAL_OBJECTS =

dift: CMakeFiles/dift.dir/main.cpp.o
dift: CMakeFiles/dift.dir/build.make
dift: /usr/lib/x86_64-linux-gnu/libboost_system.so
dift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
dift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
dift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
dift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
dift: /usr/lib/x86_64-linux-gnu/libpthread.so
dift: /usr/lib/libpcl_common.so
dift: /usr/lib/libOpenNI.so
dift: /usr/lib/libvtkCommon.so.5.8.0
dift: /usr/lib/libvtkRendering.so.5.8.0
dift: /usr/lib/libvtkHybrid.so.5.8.0
dift: /usr/lib/libvtkCharts.so.5.8.0
dift: /usr/lib/libpcl_io.so
dift: /usr/lib/x86_64-linux-gnu/libboost_system.so
dift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
dift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
dift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
dift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
dift: /usr/lib/x86_64-linux-gnu/libpthread.so
dift: /usr/lib/libpcl_common.so
dift: /usr/lib/libpcl_octree.so
dift: /usr/lib/x86_64-linux-gnu/libboost_system.so
dift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
dift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
dift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
dift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
dift: /usr/lib/x86_64-linux-gnu/libpthread.so
dift: /usr/lib/libpcl_common.so
dift: /usr/lib/x86_64-linux-gnu/libboost_system.so
dift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
dift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
dift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
dift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
dift: /usr/lib/x86_64-linux-gnu/libpthread.so
dift: /usr/lib/libpcl_common.so
dift: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
dift: /usr/lib/libpcl_kdtree.so
dift: /usr/lib/libpcl_octree.so
dift: /usr/lib/libpcl_search.so
dift: /usr/lib/libpcl_sample_consensus.so
dift: /usr/lib/libpcl_filters.so
dift: /usr/lib/libpcl_tracking.so
dift: /usr/lib/libOpenNI.so
dift: /usr/lib/libvtkCommon.so.5.8.0
dift: /usr/lib/libvtkRendering.so.5.8.0
dift: /usr/lib/libvtkHybrid.so.5.8.0
dift: /usr/lib/libvtkCharts.so.5.8.0
dift: /usr/lib/libpcl_io.so
dift: /usr/lib/libpcl_features.so
dift: /usr/lib/libpcl_segmentation.so
dift: /usr/lib/libqhull.so
dift: /usr/lib/libpcl_surface.so
dift: /usr/lib/libpcl_registration.so
dift: /usr/lib/libpcl_recognition.so
dift: /usr/lib/libpcl_keypoints.so
dift: /usr/lib/libpcl_visualization.so
dift: /usr/lib/libpcl_outofcore.so
dift: /usr/lib/libpcl_people.so
dift: /usr/lib/libpcl_apps.so
dift: /usr/lib/x86_64-linux-gnu/libboost_system.so
dift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
dift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
dift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
dift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
dift: /usr/lib/x86_64-linux-gnu/libpthread.so
dift: /usr/lib/libqhull.so
dift: /usr/lib/libOpenNI.so
dift: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
dift: /usr/lib/libvtkCommon.so.5.8.0
dift: /usr/lib/libvtkRendering.so.5.8.0
dift: /usr/lib/libvtkHybrid.so.5.8.0
dift: /usr/lib/libvtkCharts.so.5.8.0
dift: /usr/local/lib/libopencv_videostab.so.3.0.0
dift: /usr/local/lib/libopencv_video.so.3.0.0
dift: /usr/local/lib/libopencv_ts.a
dift: /usr/local/lib/libopencv_superres.so.3.0.0
dift: /usr/local/lib/libopencv_stitching.so.3.0.0
dift: /usr/local/lib/libopencv_softcascade.so.3.0.0
dift: /usr/local/lib/libopencv_shape.so.3.0.0
dift: /usr/local/lib/libopencv_photo.so.3.0.0
dift: /usr/local/lib/libopencv_optim.so.3.0.0
dift: /usr/local/lib/libopencv_ocl.so.3.0.0
dift: /usr/local/lib/libopencv_objdetect.so.3.0.0
dift: /usr/local/lib/libopencv_nonfree.so.3.0.0
dift: /usr/local/lib/libopencv_ml.so.3.0.0
dift: /usr/local/lib/libopencv_legacy.so.3.0.0
dift: /usr/local/lib/libopencv_imgproc.so.3.0.0
dift: /usr/local/lib/libopencv_highgui.so.3.0.0
dift: /usr/local/lib/libopencv_flann.so.3.0.0
dift: /usr/local/lib/libopencv_features2d.so.3.0.0
dift: /usr/local/lib/libopencv_cudawarping.so.3.0.0
dift: /usr/local/lib/libopencv_cudastereo.so.3.0.0
dift: /usr/local/lib/libopencv_cudaoptflow.so.3.0.0
dift: /usr/local/lib/libopencv_cudaimgproc.so.3.0.0
dift: /usr/local/lib/libopencv_cudafilters.so.3.0.0
dift: /usr/local/lib/libopencv_cudafeatures2d.so.3.0.0
dift: /usr/local/lib/libopencv_cudacodec.so.3.0.0
dift: /usr/local/lib/libopencv_cudabgsegm.so.3.0.0
dift: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
dift: /usr/local/lib/libopencv_cuda.so.3.0.0
dift: /usr/local/lib/libopencv_core.so.3.0.0
dift: /usr/local/lib/libopencv_contrib.so.3.0.0
dift: /usr/local/lib/libopencv_calib3d.so.3.0.0
dift: /usr/local/lib/libopencv_bioinspired.so.3.0.0
dift: /usr/lib/libpcl_common.so
dift: /usr/lib/libpcl_io.so
dift: /usr/lib/libpcl_octree.so
dift: /usr/lib/libpcl_kdtree.so
dift: /usr/lib/libpcl_search.so
dift: /usr/lib/libpcl_sample_consensus.so
dift: /usr/lib/libpcl_filters.so
dift: /usr/lib/libpcl_tracking.so
dift: /usr/lib/libpcl_features.so
dift: /usr/lib/libpcl_segmentation.so
dift: /usr/lib/libpcl_surface.so
dift: /usr/lib/libpcl_registration.so
dift: /usr/lib/libpcl_recognition.so
dift: /usr/lib/libpcl_keypoints.so
dift: /usr/lib/libpcl_visualization.so
dift: /usr/lib/libpcl_outofcore.so
dift: /usr/lib/libpcl_people.so
dift: /usr/lib/libpcl_apps.so
dift: /usr/lib/libvtkViews.so.5.8.0
dift: /usr/lib/libvtkInfovis.so.5.8.0
dift: /usr/lib/libvtkWidgets.so.5.8.0
dift: /usr/lib/libvtkHybrid.so.5.8.0
dift: /usr/lib/libvtkParallel.so.5.8.0
dift: /usr/lib/libvtkVolumeRendering.so.5.8.0
dift: /usr/lib/libvtkRendering.so.5.8.0
dift: /usr/lib/libvtkGraphics.so.5.8.0
dift: /usr/lib/libvtkImaging.so.5.8.0
dift: /usr/lib/libvtkIO.so.5.8.0
dift: /usr/lib/libvtkFiltering.so.5.8.0
dift: /usr/lib/libvtkCommon.so.5.8.0
dift: /usr/lib/libvtksys.so.5.8.0
dift: /usr/local/lib/libopencv_cudawarping.so.3.0.0
dift: /usr/local/lib/libopencv_legacy.so.3.0.0
dift: /usr/local/lib/libopencv_cudaimgproc.so.3.0.0
dift: /usr/local/lib/libopencv_cudafilters.so.3.0.0
dift: /usr/local/lib/libopencv_nonfree.so.3.0.0
dift: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
dift: /usr/local/lib/libopencv_ocl.so.3.0.0
dift: /usr/local/lib/libopencv_video.so.3.0.0
dift: /usr/local/lib/libopencv_objdetect.so.3.0.0
dift: /usr/local/lib/libopencv_ml.so.3.0.0
dift: /usr/local/lib/libopencv_calib3d.so.3.0.0
dift: /usr/local/lib/libopencv_features2d.so.3.0.0
dift: /usr/local/lib/libopencv_highgui.so.3.0.0
dift: /usr/local/lib/libopencv_imgproc.so.3.0.0
dift: /usr/local/lib/libopencv_flann.so.3.0.0
dift: /usr/local/lib/libopencv_core.so.3.0.0
dift: CMakeFiles/dift.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable dift"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dift.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dift.dir/build: dift
.PHONY : CMakeFiles/dift.dir/build

CMakeFiles/dift.dir/requires: CMakeFiles/dift.dir/main.cpp.o.requires
.PHONY : CMakeFiles/dift.dir/requires

CMakeFiles/dift.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dift.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dift.dir/clean

CMakeFiles/dift.dir/depend:
	cd /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build /home/jichy/Documents/MKR/Project/mkr_6d_registration/Program/build/CMakeFiles/dift.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dift.dir/depend

