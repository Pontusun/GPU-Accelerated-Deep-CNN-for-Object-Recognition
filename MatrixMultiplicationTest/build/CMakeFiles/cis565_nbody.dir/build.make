# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.3.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.3.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build

# Include any dependencies generated for this target.
include CMakeFiles/cis565_nbody.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cis565_nbody.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cis565_nbody.dir/flags.make

CMakeFiles/cis565_nbody.dir/src/main.cpp.o: CMakeFiles/cis565_nbody.dir/flags.make
CMakeFiles/cis565_nbody.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cis565_nbody.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cis565_nbody.dir/src/main.cpp.o -c /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/src/main.cpp

CMakeFiles/cis565_nbody.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cis565_nbody.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/src/main.cpp > CMakeFiles/cis565_nbody.dir/src/main.cpp.i

CMakeFiles/cis565_nbody.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cis565_nbody.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/src/main.cpp -o CMakeFiles/cis565_nbody.dir/src/main.cpp.s

CMakeFiles/cis565_nbody.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/cis565_nbody.dir/src/main.cpp.o.requires

CMakeFiles/cis565_nbody.dir/src/main.cpp.o.provides: CMakeFiles/cis565_nbody.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cis565_nbody.dir/build.make CMakeFiles/cis565_nbody.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/cis565_nbody.dir/src/main.cpp.o.provides

CMakeFiles/cis565_nbody.dir/src/main.cpp.o.provides.build: CMakeFiles/cis565_nbody.dir/src/main.cpp.o


# Object files for target cis565_nbody
cis565_nbody_OBJECTS = \
"CMakeFiles/cis565_nbody.dir/src/main.cpp.o"

# External object files for target cis565_nbody
cis565_nbody_EXTERNAL_OBJECTS =

cis565_nbody: CMakeFiles/cis565_nbody.dir/src/main.cpp.o
cis565_nbody: CMakeFiles/cis565_nbody.dir/build.make
cis565_nbody: /usr/local/cuda/lib/libcudart_static.a
cis565_nbody: src/libsrc.a
cis565_nbody: ../external/lib/osx/libglfw3.a
cis565_nbody: ../external/lib/osx/libGLEW.a
cis565_nbody: /usr/local/cuda/lib/libcudart_static.a
cis565_nbody: CMakeFiles/cis565_nbody.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cis565_nbody"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cis565_nbody.dir/link.txt --verbose=$(VERBOSE)
	/usr/local/Cellar/cmake/3.3.2/bin/cmake -E copy_directory /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/shaders /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build/shaders

# Rule to build all files generated by this target.
CMakeFiles/cis565_nbody.dir/build: cis565_nbody

.PHONY : CMakeFiles/cis565_nbody.dir/build

CMakeFiles/cis565_nbody.dir/requires: CMakeFiles/cis565_nbody.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/cis565_nbody.dir/requires

CMakeFiles/cis565_nbody.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cis565_nbody.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cis565_nbody.dir/clean

CMakeFiles/cis565_nbody.dir/depend:
	cd /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2 /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2 /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build /Users/Sun/Dropbox/2015Fall/CIS565/Project1-CUDA-Introduction/Project1-Part2/build/CMakeFiles/cis565_nbody.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cis565_nbody.dir/depend

