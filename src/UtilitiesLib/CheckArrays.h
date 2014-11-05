/**
 * @file   CheckArrays.h
 * @date   Nov 5, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#ifndef CHECKARRAYS_H_
#define CHECKARRAYS_H_

#include <string>

void checkArrayForNaN(float* a, int length, std::string const& msg);

void checkArrayForNanAndNegative(float* a, int length, std::string const& msg);

#endif /* CHECKARRAYS_H_ */
