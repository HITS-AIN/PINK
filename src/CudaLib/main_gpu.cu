/**
 * @file   Pink/main_gpu.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "main_gpu.h"
#include "Pink/main_generic.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

void main_gpu(InputData const & input_data)
{
    cuda_print_properties();

    if (input_data.m_layout == Layout::CARTESIAN)
        if (input_data.m_dimensionality == 1)
            main_generic<CartesianLayout<1>, float, true>(input_data);
        else if (input_data.m_dimensionality == 2)
            main_generic<CartesianLayout<2>, float, true>(input_data);
        else if (input_data.m_dimensionality == 3)
            main_generic<CartesianLayout<3>, float, true>(input_data);
        else
            throw pink::exception("Unsupported dimensionality of " + input_data.m_dimensionality);
    else if (input_data.m_layout == Layout::HEXAGONAL)
        main_generic<HexagonalLayout, float, true>(input_data);
    else
        throw pink::exception("Unknown layout");
}

} // namespace pink
