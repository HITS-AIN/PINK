/**
 * @file   PythonBinding/DynamicTrainer.h
 * @date   Aug 8, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>

#include "DynamicData.h"
#include "DynamicSOM.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

struct DynamicTrainer
{
    DynamicTrainer(DynamicSOM& som, std::function<float(float)> const& distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
        DataType euclidean_distance_type);

    DynamicTrainer(DynamicTrainer const&) = delete;

    void operator () (DynamicData const& data);

    void update_som();

private:

    template <typename SOM_Layout>
    auto get_trainer(DynamicSOM& dynamic_som, std::function<float(float)> const& distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        DataType euclidean_distance_type) -> std::shared_ptr<TrainerBase>
    {
        if (m_neuron_layout == "cartesian-2d") {
            return get_trainer<SOM_Layout, CartesianLayout<2>>(dynamic_som, distribution_function,
                verbosity, number_of_rotations, use_flip, max_update_distance,
                interpolation, euclidean_distance_dim, euclidean_distance_type);
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    template <typename SOM_Layout, typename Neuron_Layout>
    auto get_trainer(DynamicSOM& dynamic_som, std::function<float(float)> const& distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim,
        [[maybe_unused]] DataType euclidean_distance_type) -> std::shared_ptr<TrainerBase>
    {
        if (m_use_gpu == true) {
            return std::make_shared<Trainer<SOM_Layout, Neuron_Layout, float, true>>(
                *(std::dynamic_pointer_cast<SOM<SOM_Layout, Neuron_Layout, float>>(dynamic_som.m_som)),
                distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
                interpolation, euclidean_distance_dim, 256, euclidean_distance_type);
        } else {
            return std::make_shared<Trainer<SOM_Layout, Neuron_Layout, float, false>>(
                *(std::dynamic_pointer_cast<SOM<SOM_Layout, Neuron_Layout, float>>(dynamic_som.m_som)),
                distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
                interpolation, euclidean_distance_dim);
        }
    }

    template <typename SOM_Layout>
    void train(DynamicData const& data)
    {
        if (m_neuron_layout == "cartesian-2d") {
            train<SOM_Layout, CartesianLayout<2>>(data);
        } else {
            throw pink::exception("neuron layout " + m_neuron_layout + " is not supported");
        }
    }

    template <typename SOM_Layout, typename Neuron_Layout>
    void train(DynamicData const& data)
    {
        if (m_use_gpu == true) {
            std::dynamic_pointer_cast<Trainer<SOM_Layout, Neuron_Layout, float, true>>(m_trainer)->operator()(
                *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data)));
        } else {
            std::dynamic_pointer_cast<Trainer<SOM_Layout, Neuron_Layout, float, false>>(m_trainer)->operator()(
                *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data)));
        }
    }

    std::shared_ptr<TrainerBase> m_trainer;

    std::string m_data_type;

    std::string m_som_layout;

    std::string m_neuron_layout;

    bool m_use_gpu;
};

} // namespace pink
