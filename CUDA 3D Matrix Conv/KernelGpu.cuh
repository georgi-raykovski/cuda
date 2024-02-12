#pragma once
#include "CpuGpuMat.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus


void gpuMatrixConvulation3D(struct CpuGpuMat* image, struct CpuGpuMat* mask, struct CpuGpuMat* result);
void gpuMatrixConvulation3DShared(struct CpuGpuMat* image, struct CpuGpuMat* mask, struct CpuGpuMat* result);
