/**
 * @file   EqualFloatArrays.h
 * @date   Nov 11, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "gtest/gtest.h"
#include <cmath>

//! Threshold for equality check of two floating point numbers.
const float defaultTolerance = float(1.0 / (1 << 22));

//! Equality check of two floating point numbers.
template <class T>
::testing::AssertionResult EqualFloatArrays(const T* const expected,
    const T* const actual, unsigned long length, float tolerance = defaultTolerance)
{
    ::testing::AssertionResult result = ::testing::AssertionFailure();
    int errorsFound = 0;
    const char* separator = " ";
    for (unsigned long index = 0; index < length; index++)
    {
        if (fabs(expected[index] - actual[index]) > tolerance)
        {
            if (errorsFound == 0)
            {
                result << "Differences found:";
            }
            if (errorsFound < 3)
            {
                result << separator
                        << expected[index] << " != " << actual[index]
                        << " @ " << index;
                separator = ", ";
            }
            errorsFound++;
        }
    }
    if (errorsFound > 0)
    {
        result << separator << errorsFound << " differences in total";
        result << separator << "tolerance = " << tolerance;
        return result;
    }
    return ::testing::AssertionSuccess();
}
