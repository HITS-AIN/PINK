/**
 * @file   PythonBinding/DynamicData.cpp
 * @date   Aug 8, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "DynamicTrainer.h"
#include "SelfOrganizingMapLib/CartesianLayout.h"
#include "SelfOrganizingMapLib/HexagonalLayout.h"
#include "UtilitiesLib/pink_exception.h"

namespace pink {

DynamicTrainer::DynamicTrainer(DynamicSOM& som, std::function<float(float)> const& distribution_function,
    int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
    Interpolation interpolation, uint32_t euclidean_distance_dim, bool use_gpu,
    [[maybe_unused]] DataType euclidean_distance_type)
{
    if (som.data_type != "float32") throw std::runtime_error("data-type not supported");
    if (som.som_layout != "cartesian-2d") throw std::runtime_error("som_layout not supported");
    if (som.neuron_layout != "cartesian-2d") throw std::runtime_error("neuron_layout not supported");

    if (use_gpu)
    {
#ifdef __CUDACC__
        m_trainer = std::make_shared<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.data)),
            distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim, 256, euclidean_distance_type);
#else
        pink::exception("PINK not compiled with CUDA support");
#endif
    }
    else
    {
        m_trainer = std::make_shared<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(
            *(std::dynamic_pointer_cast<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(som.data)),
            distribution_function, verbosity, number_of_rotations, use_flip, max_update_distance,
            interpolation, euclidean_distance_dim);
    }
}

void DynamicTrainer::operator () (DynamicData const& data)
{
    auto s_trainer = *(std::dynamic_pointer_cast<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m_trainer));
    auto s_data = *(std::dynamic_pointer_cast<Data<CartesianLayout<2>, float>>(data.m_data));

    s_trainer(s_data);
}

} // namespace pink
