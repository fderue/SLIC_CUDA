
#include "SlicCudaHost.h"
#include "SlicCudaDevice.h"

// Texture/surface ref can only be declared in one unique file
// This is why we need this files organisation
extern texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texFrameBGRA;
extern surface<void, cudaSurfaceType2D> surfFrameLab;
extern surface<void, cudaSurfaceType2D> surfLabels;

#include "SlicCudaHost.hcu"
#include "SlicCudaDevice.dcu"


