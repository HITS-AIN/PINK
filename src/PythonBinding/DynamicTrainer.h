/**
 * @file   PythonBinding/DynamicTrainer.h
 * @date   Aug 8, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>

#include "DynamicData.h"
#include "DynamicSOM.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"

namespace pink {

struct DynamicTrainer
{
    DynamicTrainer(DynamicSOM& som, std::function<float(float)> const& distribution_function,
        int verbosity, uint32_t number_of_rotations, bool use_flip, float max_update_distance,
        Interpolation interpolation, uint32_t euclidean_distance_dim, bool use_gpu,
        DataType euclidean_distance_type);

    void operator () (DynamicData const& data);

    std::shared_ptr<TrainerBase> m_trainer;
};

} // namespace pink
