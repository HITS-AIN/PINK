/**
 * @file   cuda_print_properties.h
 * @brief  Print device properties of GPU cards.
 * @date   Oct 21, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef CUDALIB_H_
#define CUDALIB_H_

void cuda_print_properties();

void cuda_rotate(int height, int width, float *source, float *dest, float angle);

#endif /* CUDALIB_H_ */
