/**
 * @file   PythonBinding/DynamicData.cpp
 * @date   Aug 8, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicTrainer.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

DynamicTrainer::DynamicTrainer(DynamicSOM& som, std::function<float(float)> const& distribution_function,
    int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
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
        m_trainer = std::make_shared<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.m_som)),
            distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim, 256, euclidean_distance_type);
    }
    else
    {
        m_trainer = std::make_shared<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.m_som)),
            distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim);
    }
}

void DynamicTrainer::operator () (DynamicData const& data)
{
    auto s_data = *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data));
    
    if (m_use_gpu)
    {
        auto s_trainer = std::dynamic_pointer_cast<
    	    Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true>>(m_trainer);
        s_trainer->operator()(s_data);
    }
    else
    {
        auto s_trainer = std::dynamic_pointer_cast<
            Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m_trainer);
        s_trainer->operator()(s_data); 
    }
}

void DynamicTrainer::update_som()
{
    if (m_use_gpu)
    {
		auto s_trainer = std::dynamic_pointer_cast<
			Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true>>(m_trainer);
        s_trainer->update_som();
    }
    else
    {
		auto s_trainer = std::dynamic_pointer_cast<
			Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m_trainer);
        s_trainer->update_som();
    }
}

} // namespace pink
