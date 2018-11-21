/**
 * @file   PythonBinding/pink.cpp
 * @date   Nov 21, 2018
 * @author Bernd Doser, HITS gGmbH
 */

/// Outsource code into impl file to avoid code duplicity compiling CPU and GPU version.
/// Splitting of PYBIND11_MODULE is not possible.
#include "pink_impl.cpp"
