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
include examples/CMakeFiles/freenect-regview.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/freenect-regview.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/freenect-regview.dir/flags.make

examples/CMakeFiles/freenect-regview.dir/regview.c.o: examples/CMakeFiles/freenect-regview.dir/flags.make
examples/CMakeFiles/freenect-regview.dir/regview.c.o: ../examples/regview.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/freenect-regview.dir/regview.c.o"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/freenect-regview.dir/regview.c.o   -c /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/regview.c

examples/CMakeFiles/freenect-regview.dir/regview.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect-regview.dir/regview.c.i"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/regview.c > CMakeFiles/freenect-regview.dir/regview.c.i

examples/CMakeFiles/freenect-regview.dir/regview.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect-regview.dir/regview.c.s"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples/regview.c -o CMakeFiles/freenect-regview.dir/regview.c.s

examples/CMakeFiles/freenect-regview.dir/regview.c.o.requires:

.PHONY : examples/CMakeFiles/freenect-regview.dir/regview.c.o.requires

examples/CMakeFiles/freenect-regview.dir/regview.c.o.provides: examples/CMakeFiles/freenect-regview.dir/regview.c.o.requires
	$(MAKE) -f examples/CMakeFiles/freenect-regview.dir/build.make examples/CMakeFiles/freenect-regview.dir/regview.c.o.provides.build
.PHONY : examples/CMakeFiles/freenect-regview.dir/regview.c.o.provides

examples/CMakeFiles/freenect-regview.dir/regview.c.o.provides.build: examples/CMakeFiles/freenect-regview.dir/regview.c.o


# Object files for target freenect-regview
freenect__regview_OBJECTS = \
"CMakeFiles/freenect-regview.dir/regview.c.o"

# External object files for target freenect-regview
freenect__regview_EXTERNAL_OBJECTS =

bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/regview.c.o
bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/build.make
bin/freenect-regview: lib/libfreenect.so.0.6.0
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libGL.so
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libGLU.so
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libglut.so
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libXmu.so
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libXi.so
bin/freenect-regview: /usr/lib/arm-linux-gnueabihf/libusb-1.0.so
bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/freenect-regview"
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-regview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/freenect-regview.dir/build: bin/freenect-regview

.PHONY : examples/CMakeFiles/freenect-regview.dir/build

examples/CMakeFiles/freenect-regview.dir/requires: examples/CMakeFiles/freenect-regview.dir/regview.c.o.requires

.PHONY : examples/CMakeFiles/freenect-regview.dir/requires

examples/CMakeFiles/freenect-regview.dir/clean:
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/freenect-regview.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/freenect-regview.dir/clean

examples/CMakeFiles/freenect-regview.dir/depend:
	cd /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/examples /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples /home/kevinleeor/Desktop/workspace/DepthMapping/libfreenect/build/examples/CMakeFiles/freenect-regview.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/freenect-regview.dir/depend

