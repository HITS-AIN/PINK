/**
 * @file   SelfOrganizingMapLib/Trainer.h
 * @date   Oct 11, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

namespace pink {

/// Primary template of Trainer should never be instantiated
template <typename SOMLayout, typename DataLayout, typename T, bool use_gpu>
class Trainer;

} // namespace pink
