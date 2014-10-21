/**
 * @file   cuda_rotate.h
 * @brief  Image rotataion using CUDA.
 * @date   Oct 17, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef CUDA_ROTATE_H_
#define CUDA_ROTATE_H_

void cuda_rotate(int height, int width, float *source, float *dest, float angle);

#endif /* CUDA_ROTATE_H_ */
