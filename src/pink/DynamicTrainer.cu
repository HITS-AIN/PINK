/**
 * @file   pink/DynamicData.cu
 * @date   Aug 8, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicTrainer.h"

namespace pink {

DynamicTrainer::DynamicTrainer(DynamicSOM& dynamic_som, std::function<float(float)> const& distribution_function,
    int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
    Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
    EuclideanDistanceShape euclidean_distance_shape, DataType euclidean_distance_type)
 : m_data_type(dynamic_som.m_data_type),
   m_som_layout(dynamic_som.m_som_layout),
   m_neuron_layout(dynamic_som.m_neuron_layout),
   m_use_gpu(use_gpu)
{
    if (m_data_type != "float32") throw pink::exception("data-type not supported");
    if (euclidean_distance_dim == 0) throw pink::exception("euclidean_distance_dim not defined");

    if (m_som_layout == "cartesian-2d") {
        m_trainer = get_trainer<CartesianLayout<2>>(dynamic_som, distribution_function,
            verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim, euclidean_distance_shape, euclidean_distance_type);
    } else if (m_som_layout == "hexagonal-2d") {
        m_trainer = get_trainer<HexagonalLayout>(dynamic_som, distribution_function,
            verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim, euclidean_distance_shape, euclidean_distance_type);
    } else {
        throw pink::exception("som layout " + m_som_layout + " is not supported");
    }
}

void DynamicTrainer::operator () (DynamicData const& data)
{
    if (m_som_layout == "cartesian-2d") {
        train<CartesianLayout<2>>(data);
    } else if (m_som_layout == "hexagonal-2d") {
        train<HexagonalLayout>(data);
    } else {
        throw pink::exception("som layout " + m_som_layout + " is not supported");
    }
}

void DynamicTrainer::update_som()
{
    m_trainer->update_som();
}

} // namespace pink
