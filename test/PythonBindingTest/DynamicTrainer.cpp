/**
 * @file   PythonBindingTest/DynamicTrainer.cpp
 * @date   Aug 9, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>
#include <vector>

#include "PythonBinding/DynamicData.h"
#include "PythonBinding/DynamicSOM.h"
#include "PythonBinding/DynamicTrainer.h"
#include "UtilitiesLib/DistributionFunctor.h"

using namespace pink;

TEST(GenericPythonBindingTest, DynamicTrainer)
{
    std::vector<ssize_t> shape{3, 3};
    std::vector<float> ptr{2.0f, 3.0f, 0.0f, 1.0f};

    DynamicData data("float32", "cartesian-2d", shape, static_cast<void*>(&ptr[0]));

    std::vector<uint32_t> som_shape{2, 2, 3, 3};
    std::vector<float> som_ptr(16, 0.0f);

    DynamicSOM som("float32", "cartesian-2d", "cartesian-2d", som_shape, static_cast<void*>(&som_ptr[0]));

    auto&& f = GaussianFunctor(1.1f, 0.2f);

    DynamicTrainer trainer(som, f, 0, 16, true, -1.0, Interpolation::BILINEAR, true, 3, DataType::UINT8);

    trainer(data);
}
