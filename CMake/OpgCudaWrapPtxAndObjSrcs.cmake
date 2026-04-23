
# Given a list of source files, parse the "*.cu" filese and check for `#pragma cuda_source_property_format=*` definitions.
# If the pragma is found, the source property is `CUDA_SOURCE_PROPERTY_FORMAT` is set to the specified value.
# Otherwise the `CUDA_SOURCE_PROPERTY_FORMAT` is set to the default value `PTX`.
macro(OPG_CUDA_SOURCE_PROPERTY_FORMAT_FROM_PRAGMA)
foreach(source_file ${ARGN})
  # Define the pragma we use to determine the value of CUDA_SOURCE_PROPERTY_FORMAT
  set(pragma_expression "^[ \t]*#[ \t]*pragma[ \t]+cuda_source_property_format[ \t]*=[ \t]*([a-zA-Z]+)[ \t]*$")
  # Process any *.cu source files
  if(${source_file} MATCHES "\\.cu$")
    # Read all lines from the source file defining the pragma we are looking for.
    file(STRINGS ${source_file} matched_content REGEX ${pragma_expression})
    if (matched_content)
      # For all pragma definitions that we found, should only be one usually...
      foreach(line_string ${matched_content})
        # Filter out the defined value
        string(REGEX REPLACE ${pragma_expression} "\\1" source_property_format ${line_string})
        # Apply the property to the source file
        set_source_files_properties(${source_file} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT ${source_property_format})
      endforeach()
    else()
      # Apply the default property format.
      set_source_files_properties(${source_file} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT "PTX")
    endif()
  endif()
endforeach()
endmacro()

# Given a list of source files that should be part of a library, check the `CUDA_SOURCE_PROPERTY_FORMAT` property
# of each source file to decide if the file should be part of the regular shared/static library, or if the source
# file should be compiled into a ptx file intead.
macro(OPG_CUDA_SPLIT_PTX_SOURCE_FILES regular_source_files source_files_to_ptx)
set(${regular_source_files})
set(${source_files_to_ptx})
foreach(source_file ${ARGN})
  # Retrieve property from file
  get_source_file_property(source_property_format ${source_file} CUDA_SOURCE_PROPERTY_FORMAT)
  if (${source_property_format} STREQUAL "PTX")
    # .cu -> .ptx files are added to a special object library.
    list(APPEND ${source_files_to_ptx} ${source_file})
  else()
    # Regular {.cu,.cpp,.h} -> .obj files are added to the normal library.
    list(APPEND ${regular_source_files} ${source_file})
  endif()
endforeach()
endmacro()

macro(OPG_ADD_PTX_LIBRARY target_name target_name_var)
  # Set the name of the PTX library.
  set(${target_name_var} ${target_name}-ptx)
  # Create the library.
  add_library(${target_name}-ptx OBJECT ${ARGN})
  # Enable ptx compilation.
  set_target_properties("${target_name}-ptx" PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options("${target_name}-ptx" PRIVATE -ptx)
  # Copy the ptx files from the CMake internal build directory into the output library folder.
  set(OBJECT_OUTPUT_DIR "${OPG_CUDA_GENERATED_OUTPUT_DIR_PTX}/${target_name}")
  add_custom_target(${target_name}-ptx-copy
    DEPENDS ${target_name}-ptx
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJECT_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    $<TARGET_OBJECTS:${target_name}-ptx> ${OBJECT_OUTPUT_DIR}
    COMMAND_EXPAND_LISTS
  )
  # Make the base library depend on the copied ptx library.
  add_dependencies(${target_name} ${target_name}-ptx-copy)
endmacro()
