/**
 * @file   UtilitiesLib/DistributionFunctions.h
 * @date   Nov 14, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef DISTRIBUTIONFUNCTIONS_H_
#define DISTRIBUTIONFUNCTIONS_H_

#define UPDATE_NEURONS_SIGMA     1.1
#define UPDATE_NEURONS_DAMPING   0.2

//! 2.0 / ( math.sqrt(3.0 * sigma) * math.pow(math.pi, 0.25)) * (1- x**2.0 / sigma**2.0) * math.exp(-x**2.0/(2.0 * sigma**2))
float mexicanHat(float x, float sigma);

//! 1.0 / (sigma * math.sqrt(2.0 * math.pi)) * math.exp(-1.0/2.0 * (x / sigma)**2 );
float gaussian(float x, float sigma);

#endif /* DISTRIBUTIONFUNCTIONS_H_ */
