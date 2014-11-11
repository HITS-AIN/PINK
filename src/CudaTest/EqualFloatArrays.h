/**
 * @file   EqualFloatArrays.h
 * @date   Nov 11, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "gtest/gtest.h"
#include <cmath>

const float FLOAT_INEQUALITY_TOLERANCE = float(1.0 / (1 << 22));

template <class T>
::testing::AssertionResult EqualFloatArrays(
                                const T* const expected,
                                const T* const actual,
                                unsigned long length)
{
    ::testing::AssertionResult result = ::testing::AssertionFailure();
    int errorsFound = 0;
    const char* separator = " ";
    for (unsigned long index = 0; index < length; index++)
    {
        if (fabs(expected[index] - actual[index]) > FLOAT_INEQUALITY_TOLERANCE)
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
        return result;
    }
    return ::testing::AssertionSuccess();
}
