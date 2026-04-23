#pragma once

#ifdef __optix_optix_stubs_h__
#error "optix_stubs.h already included before opg_optix_stubs.h!"
#endif

#include "opg/opgapi.h"

#define OPTIX_STUBS_API OPG_API
#include <optix_stubs.h>
