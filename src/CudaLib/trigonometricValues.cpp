/**
 * @file   CudaLib/trigonometricValues.cpp
 * @date   Nov 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include <cmath>

//! Prepare trigonometric values
void trigonometricValues(float **d_cosAlpha, float **d_sinAlpha, int num_rot)
{
	if (!num_rot) return;

	float angleStepRadians = 2.0 * M_PI / num_rot;

	float angle;
	*d_cosAlpha = cuda_alloc_float(num_rot);
	*d_sinAlpha = cuda_alloc_float(num_rot);
	float *cosAlpha = (float *)malloc(num_rot * sizeof(float));
	float *sinAlpha = (float *)malloc(num_rot * sizeof(float));

	for (int i = 0; i < num_rot; ++i) {
		angle = (i+1) * angleStepRadians;
	    cosAlpha[i] = cos(angle);
        sinAlpha[i] = sin(angle);
	}

	cuda_copyHostToDevice_float(*d_cosAlpha, cosAlpha, num_rot);
	cuda_copyHostToDevice_float(*d_sinAlpha, sinAlpha, num_rot);

	// Free memory
	free(cosAlpha);
	free(sinAlpha);
}
