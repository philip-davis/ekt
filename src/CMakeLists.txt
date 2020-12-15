# load package helper for generating cmake CONFIG packages
include (CMakePackageConfigHelpers)

# where to install files for "find_package"
set (ekt-pkg "share/cmake/ekt")

set (ekt-vers "${ekt_VERSION_MAJOR}.${ekt_VERSION_MINOR}")

add_library(ekt ekt.c)
target_link_libraries (ekt margo)
target_include_directories (ekt PUBLIC $<INSTALL_INTERFACE:include>)

# local include's BEFORE, in case old incompatable .h files in prefix/include
target_include_directories (ekt BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

# for shared libs, establish the lib version
set_target_properties (ekt
    PROPERTIES VERSION ${ekt_VERSION}
    SOVERSION ${ekt_VERSION_MAJOR})

#
# installation stuff (packaging and install commands)
#
write_basic_package_version_file(
    "ekt-config-version.cmake"
    VERSION ${ekt_VERSION}
    COMPATIBILITY AnyNewerVersion)

# generate our config file for find_package()
configure_file (ekt-config.cmake.in ekt-config.cmake @ONLY)

# some bits for the pkg-config file
set (DEST_DIR "${CMAKE_INSTALL_PREFIX}")
set (PRIVATE_LIBS "-lekt -lmargo -labt -lmercury")
configure_file ("ekt.pc.in" "ekt.pc" @ONLY)

#
# "make install" rules
#
install (TARGETS ekt EXPORT ekt-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib)
install (EXPORT ekt-targets
         DESTINATION ${ekt-pkg}
         FILE "ekt-targets.cmake")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/ekt-config.cmake"
               "${CMAKE_CURRENT_BINARY_DIR}/ekt-config-version.cmake"
               "../cmake/xpkg-import.cmake"
         DESTINATION ${ekt-pkg} )
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/ekt.pc"
		DESTINATION "lib/pkgconfig/")
