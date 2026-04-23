
# Set IDE folders for targets
set(EXTERNAL_IDE_FOLDER ExternalLibraries)
set(CMAKE_FOLDER ${EXTERNAL_IDE_FOLDER})


add_subdirectory(fmt-10.1.1)


add_subdirectory( glad )


set( GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
set( GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
set( GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" )
set( GLFW_INSTALL OFF CACHE BOOL "Generate installation target")

# Filter out warnings that cause problems with GLFW.
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  string( REPLACE "-Wdeclaration-after-statement" "" filtered_c_flags ${CMAKE_C_FLAGS} )
  string( REPLACE "-Wsign-compare" "" filtered_c_flags ${filtered_c_flags} )
  push_variable( CMAKE_C_FLAGS "${filtered_c_flags} -Wno-format-truncation -Wno-deprecated" )
endif()
add_subdirectory( glfw )
set_property( TARGET glfw PROPERTY C_STANDARD 99 )
if( WARNINGS_AS_ERRORS AND (USING_GNU_CXX OR USING_CLANG_CXX))
  pop_variable( CMAKE_C_FLAGS )
endif()

add_subdirectory( imgui )

function( add_glm )
    # temporarily disable building shared and static libraries such that glm does not add them
    # need to wrap this in a function
    set(BUILD_STATIC_LIBS OFF)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory( glm )
endfunction()
add_glm()


#set(PUGIXML_BUILD_SHARED_AND_STATIC_LIBS ON)
add_subdirectory(pugixml)

add_subdirectory(libnpy)

add_subdirectory(stb)

add_subdirectory(tinyexr)

add_subdirectory(tinyobj)

add_subdirectory(whereami)

function( add_nfd )
  set(BUILD_SHARED_LIBS OFF)
  set(NFD_PORTAL ON)
  set(NFD_BUILD_TESTS OFF)
  set(NFD_INSTALL OFF)
  add_subdirectory(nativefiledialog-extended)
endfunction()
add_nfd()
