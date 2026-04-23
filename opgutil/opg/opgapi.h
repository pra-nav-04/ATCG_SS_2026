#pragma once

// We are currently building opgutil as static library
#define OPG_API

#ifndef OPG_API
#  if opgutil_EXPORTS // Set by CMake
#    if defined( _WIN32 ) || defined( _WIN64 )
#      define OPG_API __declspec(dllexport)
#      define OPG_CLASSAPI
#    elif defined( linux ) || defined( __linux__ ) || defined ( __CYGWIN__ )
#      define OPG_API __attribute__ ((visibility ("default")))
#      define OPG_CLASSAPI OPG_API
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define OPG_API __attribute__ ((visibility ("default")))
#      define OPG_CLASSAPI OPG_API
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  else /* opgutil_EXPORTS */

#    if defined( _WIN32 ) || defined( _WIN64 )
#      define OPG_API __declspec(dllimport)
#      define OPG_CLASSAPI
#    elif defined( linux ) || defined( __linux__ ) || defined ( __CYGWIN__ )
#      define OPG_API __attribute__ ((visibility ("default")))
#      define OPG_CLASSAPI OPG_API
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define OPG_API __attribute__ ((visibility ("default")))
#      define OPG_CLASSAPI OPG_API
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  endif /* opgutil_EXPORTS */
#endif
