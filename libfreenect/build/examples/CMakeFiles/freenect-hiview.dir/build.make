# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/freenect-hiview.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/freenect-hiview.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/freenect-hiview.dir/flags.make

examples/CMakeFiles/freenect-hiview.dir/hiview.c.o: examples/CMakeFiles/freenect-hiview.dir/flags.make
examples/CMakeFiles/freenect-hiview.dir/hiview.c.o: ../examples/hiview.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/freenect-hiview.dir/hiview.c.o"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/freenect-hiview.dir/hiview.c.o   -c /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/hiview.c

examples/CMakeFiles/freenect-hiview.dir/hiview.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect-hiview.dir/hiview.c.i"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/hiview.c > CMakeFiles/freenect-hiview.dir/hiview.c.i

examples/CMakeFiles/freenect-hiview.dir/hiview.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect-hiview.dir/hiview.c.s"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/hiview.c -o CMakeFiles/freenect-hiview.dir/hiview.c.s

examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.requires:

.PHONY : examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.requires

examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.provides: examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.requires
	$(MAKE) -f examples/CMakeFiles/freenect-hiview.dir/build.make examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.provides.build
.PHONY : examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.provides

examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.provides.build: examples/CMakeFiles/freenect-hiview.dir/hiview.c.o


# Object files for target freenect-hiview
freenect__hiview_OBJECTS = \
"CMakeFiles/freenect-hiview.dir/hiview.c.o"

# External object files for target freenect-hiview
freenect__hiview_EXTERNAL_OBJECTS =

bin/freenect-hiview: examples/CMakeFiles/freenect-hiview.dir/hiview.c.o
bin/freenect-hiview: examples/CMakeFiles/freenect-hiview.dir/build.make
bin/freenect-hiview: lib/libfreenect.so.0.6.0
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libGL.so
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libGLU.so
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libglut.so
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libXmu.so
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libXi.so
bin/freenect-hiview: /usr/lib/arm-linux-gnueabihf/libusb-1.0.so
bin/freenect-hiview: examples/CMakeFiles/freenect-hiview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/freenect-hiview"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-hiview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/freenect-hiview.dir/build: bin/freenect-hiview

.PHONY : examples/CMakeFiles/freenect-hiview.dir/build

examples/CMakeFiles/freenect-hiview.dir/requires: examples/CMakeFiles/freenect-hiview.dir/hiview.c.o.requires

.PHONY : examples/CMakeFiles/freenect-hiview.dir/requires

examples/CMakeFiles/freenect-hiview.dir/clean:
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/freenect-hiview.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/freenect-hiview.dir/clean

examples/CMakeFiles/freenect-hiview.dir/depend:
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples/CMakeFiles/freenect-hiview.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/freenect-hiview.dir/depend

