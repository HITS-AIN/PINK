/**
 * @file   pink/DynamicMapper.h
 * @date   Sep 3, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>

#include "DynamicData.h"
#include "DynamicSOM.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/Mapper.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

struct DynamicMapper
{
    DynamicMapper(DynamicSOM const& som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
        EuclideanDistanceShape euclidean_distance_shape, DataType euclidean_distance_type);

    DynamicMapper(DynamicMapper const&) = delete;

    auto operator () (DynamicData const& data) const
        -> std::tuple<std::vector<float>, std::vector<uint32_t>>;

private:

    template <typename SOM_Layout>
    auto get_mapper(DynamicSOM const& dynamic_som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, uint32_t euclidean_distance_dim, EuclideanDistanceShape euclidean_distance_shape,
        DataType euclidean_distance_type) -> std::shared_ptr<MapperBase>
    {
        if (m_neuron_layout == "cartesian-2d") {
            return get_mapper<SOM_Layout, CartesianLayout<2>>(dynamic_som, verbosity,
                number_of_rotations, use_flip, interpolation, euclidean_distance_dim,
                euclidean_distance_shape, euclidean_distance_type);
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    template <typename SOM_Layout, typename Neuron_Layout>
    auto get_mapper(DynamicSOM const& dynamic_som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, uint32_t euclidean_distance_dim, EuclideanDistanceShape euclidean_distance_shape,
        [[maybe_unused]] DataType euclidean_distance_type) -> std::shared_ptr<MapperBase>
    {
        if (m_use_gpu == false) {
            return std::make_shared<Mapper<SOM_Layout, Neuron_Layout, float, false>>(
                *(std::dynamic_pointer_cast<SOM<SOM_Layout, Neuron_Layout, float>>(dynamic_som.m_som)),
                verbosity, number_of_rotations, use_flip, interpolation,
                euclidean_distance_dim, euclidean_distance_shape);
        } else {
#ifdef __CUDACC__
            return std::make_shared<Mapper<SOM_Layout, Neuron_Layout, float, true>>(
                *(std::dynamic_pointer_cast<SOM<SOM_Layout, Neuron_Layout, float>>(dynamic_som.m_som)),
                verbosity, number_of_rotations, use_flip, interpolation,
                euclidean_distance_dim, euclidean_distance_shape, 256, euclidean_distance_type);
#else
            throw pink::exception("GPU support is not supported");
#endif
        }
    }

    template <typename SOM_Layout>
    auto map(DynamicData const& data) const
        -> std::tuple<std::vector<float>, std::vector<uint32_t>>
    {
        if (m_neuron_layout == "cartesian-2d") {
            return map<SOM_Layout, CartesianLayout<2>>(data);
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    template <typename SOM_Layout, typename Neuron_Layout>
    auto map(DynamicData const& data) const
        -> std::tuple<std::vector<float>, std::vector<uint32_t>>
    {
        if (m_use_gpu == false) {
            return std::dynamic_pointer_cast<Mapper<SOM_Layout, Neuron_Layout, float, false>>(m_mapper)->operator()(
                *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data)));
        } else {
#ifdef __CUDACC__
            return std::dynamic_pointer_cast<Mapper<SOM_Layout, Neuron_Layout, float, true>>(m_mapper)->operator()(
                *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data)));
#else
            throw pink::exception("GPU support is not supported");
#endif
        }
    }

    std::shared_ptr<MapperBase> m_mapper;

    std::string m_data_type;

    std::string m_som_layout;

    std::string m_neuron_layout;

    bool m_use_gpu;
};

} // namespace pink
