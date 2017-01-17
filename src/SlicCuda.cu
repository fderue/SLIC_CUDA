
#include "SlicCudaHost.h"
#include "SlicCudaDevice.h"

// Texture/surface ref can only be declared in one unique file
// This is why we need this files organisation
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texFrameBGRA;
surface<void, cudaSurfaceType2D> surfFrameLab;
surface<void, cudaSurfaceType2D> surfLabels;

#include "SlicCudaHost.hcu"
#include "SlicCudaDevice.dcu"


