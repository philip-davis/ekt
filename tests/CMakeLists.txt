ExternalProject_Add(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG release-1.10.0
    TIMEOUT 10
    # Force separate output paths for debug and release builds to allow easy
    # identification of correct lib in subsequent TARGET_LINK_LIBRARIES commands
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -Dgtest_force_shared_crt=ON
    # Disable install step
    INSTALL_COMMAND ""
    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
include_directories(${source_dir}/googletest/include)

enable_language(CXX)

add_executable(simple_test simple_test.cpp)
set_target_properties(simple_test PROPERTIES LINKER_LANGUAGE CXX)
ExternalProject_Get_Property(googletest binary_dir)
find_library(GTEST gtest ${binary_dir}/lib)
target_link_libraries(simple_test ${GTEST} pthread)

include(GoogleTest)

add_executable(ekt_test_local ekt_test_local.c)
target_link_libraries(ekt_test_local ekt)

gtest_add_tests(TARGET simple_test)
