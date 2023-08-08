/**
 * @file   pink/DynamicMapper.cu
 * @date   Sep 3, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <cassert>

#include "DynamicMapper.h"

namespace pink {

DynamicMapper::DynamicMapper(DynamicSOM const& dynamic_som, int verbosity, uint32_t number_of_rotations,
    bool use_flip, Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
    EuclideanDistanceShape euclidean_distance_shape, [[maybe_unused]] DataType euclidean_distance_type)
 : m_data_type(dynamic_som.m_data_type),
   m_som_layout(dynamic_som.m_som_layout),
   m_neuron_layout(dynamic_som.m_neuron_layout),
   m_use_gpu(use_gpu)
{
    if (m_data_type != "float32") throw pink::exception("data-type not supported");
    if (euclidean_distance_dim == 0) throw pink::exception("euclidean_distance_dim not defined");

    if (m_som_layout == "cartesian-2d") {
        m_mapper = get_mapper<CartesianLayout<2>>(dynamic_som, verbosity, number_of_rotations, use_flip,
            interpolation, euclidean_distance_dim, euclidean_distance_shape, euclidean_distance_type);
    } else if (m_som_layout == "hexagonal-2d") {
        m_mapper = get_mapper<HexagonalLayout>(dynamic_som, verbosity, number_of_rotations, use_flip,
            interpolation, euclidean_distance_dim, euclidean_distance_shape, euclidean_distance_type);
    } else {
        throw pink::exception("som layout " + m_som_layout + " is not supported");
    }
}

auto DynamicMapper::operator () (DynamicData const& data) const
    -> std::tuple<std::vector<float>, std::vector<uint32_t>>
{
    if (m_som_layout == "cartesian-2d") {
        return map<CartesianLayout<2>>(data);
    } else if (m_som_layout == "hexagonal-2d") {
        return map<HexagonalLayout>(data);
    } else {
        throw pink::exception("som layout " + m_som_layout + " is not supported");
    }
}

} // namespace pink
