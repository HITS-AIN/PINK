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

	float angleStepRadians = 0.5 * M_PI / num_rot;
	int num_rot_m1 = num_rot - 1;

	float angle;
	*d_cosAlpha = cuda_alloc_float(num_rot_m1);
	*d_sinAlpha = cuda_alloc_float(num_rot_m1);
	float *cosAlpha = (float *)malloc(num_rot_m1 * sizeof(float));
	float *sinAlpha = (float *)malloc(num_rot_m1 * sizeof(float));

	for (int i = 0; i < num_rot_m1; ++i) {
		angle = (i+1) * angleStepRadians;
	    cosAlpha[i] = cos(angle);
        sinAlpha[i] = sin(angle);
	}

	cuda_copyHostToDevice_float(*d_cosAlpha, cosAlpha, num_rot_m1);
	cuda_copyHostToDevice_float(*d_sinAlpha, sinAlpha, num_rot_m1);

	// Free memory
	free(cosAlpha);
	free(sinAlpha);
}
