/**
 * @file   Pink/main_gpu.cpp
 * @brief  Main routine of PINK.
 * @date   Oct 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "main_gpu.h"
#include "SelfOrganizingMapLib/main_generic.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

void main_gpu(InputData const & input_data)
{
	cuda_print_properties();

    if (input_data.layout == Layout::CARTESIAN)
        if (input_data.dimensionality == 1)
            main_generic<CartesianLayout<1>, CartesianLayout<2>, float, true>(input_data);
        else if (input_data.dimensionality == 2)
            main_generic<CartesianLayout<2>, CartesianLayout<2>, float, true>(input_data);
        else if (input_data.dimensionality == 3)
            main_generic<CartesianLayout<3>, CartesianLayout<2>, float, true>(input_data);
        else
            pink::exception("Unsupported dimensionality of " + input_data.dimensionality);
    else if (input_data.layout == Layout::HEXAGONAL)
        main_generic<HexagonalLayout, CartesianLayout<2>, float, true>(input_data);
    else
        pink::exception("Unknown layout");
}

} // namespace pink
