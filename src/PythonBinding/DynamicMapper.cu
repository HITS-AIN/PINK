/**
 * @file   PythonBinding/DynamicMapper.cpp
 * @date   Sep 3, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicMapper.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

DynamicMapper::DynamicMapper(DynamicSOM const& som, int verbosity, uint32_t number_of_rotations, bool use_flip,
    Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
    [[maybe_unused]] DataType euclidean_distance_type)
 : m_use_gpu(use_gpu)
{
    if (som.m_data_type != "float32") throw std::runtime_error("data-type not supported");
    if (som.m_som_layout != "cartesian-2d") throw std::runtime_error("som_layout not supported");
    if (som.m_neuron_layout != "cartesian-2d") throw std::runtime_error("neuron_layout not supported");

    if (euclidean_distance_dim == 0) {
        euclidean_distance_dim = static_cast<uint32_t>(som.m_shape[2]);
        if (number_of_rotations != 1)
            euclidean_distance_dim = static_cast<uint32_t>(euclidean_distance_dim * std::sqrt(2.0) / 2);
    }
    assert(euclidean_distance_dim != 0);

    if (m_use_gpu)
    {
        m_mapper = std::make_shared<Mapper<CartesianLayout<2>, CartesianLayout<2>, float, true>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.m_data)),
            verbosity, number_of_rotations, use_flip, interpolation, euclidean_distance_dim, 256, euclidean_distance_type);
    }
    else
    {
        m_mapper = std::make_shared<Mapper<CartesianLayout<2>, CartesianLayout<2>, float, false>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.m_data)),
            verbosity, number_of_rotations, use_flip, interpolation, euclidean_distance_dim);
    }
}

void DynamicMapper::operator () (DynamicData const& data)
{
    auto s_data = *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data));
    
    if (m_use_gpu)
    {
        auto s_mapper = std::dynamic_pointer_cast<
    	    Mapper<CartesianLayout<2>, CartesianLayout<2>, float, true>>(m_mapper);
        s_mapper->operator()(s_data);
    }
    else
    {
        auto s_mapper = std::dynamic_pointer_cast<
            Mapper<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m_mapper);
        s_mapper->operator()(s_data); 
    }
}

} // namespace pink
