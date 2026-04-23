# OptiX include files

The current optix include files can be found [here](https://developer.nvidia.com/designworks/optix/download).
The framework currently requires version 7.4 of the OptiX headers.

Note that we had to modify the headers to make them work with our framework!
In case you want to update the headers, you need to add `OPTIX_STUBS_API` to the declaration of `g_optixFunctionTable` in `optix_stubs.h` like so:
```
OPTIX_STUBS_API extern OptixFunctionTable g_optixFunctionTable;
```

Also don't include optix_stubs.h directly, but use our wrapper `opg/opg_optix_stubs.h`.
This is required to use a shared instance of the `g_optixFunctionTable` across DLL boundaries!
