/**
 * @file   CudaLib/circular_mapping.cu
 * @date   Mar 25, 2020
 * @author Bernd Doser, HITS gGmbH
 */

#include <vector>

namespace pink {

void circular_mapping(uint32_t const& dim)
{
    std::vector<uint32_t> delta(dim);
    std::vector<uint32_t> offset(dim + 1);

    delta[0] = std::sqrt(dim * 0.5 - std::pow(0.5, 2));
    offset[0] = 0;
    for (uint32_t i = 1; i < dim; ++i) {
        delta[i] = std::sqrt(dim * (i + 0.5) - std::pow((i + 0.5), 2));
        offset[i] = offset[i - 1] + delta[i - 1];
    }
    offset[dim] = offset[dim - 1] + delta[dim - 1];
}

} // namespace pink
