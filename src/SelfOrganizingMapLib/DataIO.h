/**
 * @file   SelfOrganizingMapLib/DataIO.h
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iomanip>
#include <iostream>

#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/Data.h"

namespace pink {

template <typename T>
std::ostream& operator << (std::ostream& os, Data<CartesianLayout<1>, T> const& data)
{
	for (uint32_t i = 0; i != data.get_dimension()[0]; ++i) {
        os << std::setw(6) << data[i] << " ";
    }
    os << std::endl;

    return os;
}

template <typename T>
std::ostream& operator << (std::ostream& os, Data<CartesianLayout<2>, T> const& data)
{
	for (uint32_t i = 0, p = 0; i != data.get_dimension()[0]; ++i) {
		for (uint32_t j = 0; j != data.get_dimension()[1]; ++j, ++p) {
            os << std::setw(6) << data[p] << " ";
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

template <typename T>
std::ostream& operator << (std::ostream& os, Data<CartesianLayout<3>, T> const& data)
{
	for (uint32_t i = 0, p = 0; i != data.get_dimension()[0]; ++i) {
		for (uint32_t j = 0; j != data.get_dimension()[1]; ++j, ++p) {
			for (uint32_t k = 0; k != data.get_dimension()[2]; ++k, ++p) {
                os << std::setw(6) << data[p] << " ";
	        }
	        os << std::endl;
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

template <typename T>
std::ostream& operator << (std::ostream& os, Data<HexagonalLayout, T> const& data)
{
	for (uint32_t i = 0, p = 0; i != data.get_dimension()[0]; ++i) {
		for (uint32_t j = 0; j != data.get_dimension()[1]; ++j, ++p) {
            os << std::setw(6) << data[p] << " ";
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

} // namespace pink
