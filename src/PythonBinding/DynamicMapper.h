/**
 * @file   PythonBinding/DynamicMapper.h
 * @date   Sep 3, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <memory>

#include "DynamicData.h"
#include "DynamicSOM.h"
#include "SelfOrganizingMapLib/Mapper.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Layout.h"

namespace pink {

struct DynamicMapper
{
    DynamicMapper(DynamicSOM const& som, int verbosity, uint32_t number_of_rotations, bool use_flip,
        Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
        DataType euclidean_distance_type);

    DynamicMapper(DynamicMapper const&) = delete;

    auto operator () (DynamicData const& data)
        -> std::tuple<std::vector<float>, std::vector<uint32_t>>;

    void update_som();

    std::shared_ptr<MapperBase> m_mapper;

    bool m_use_gpu;
};

} // namespace pink
