/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "SOM.h"

namespace pink {

template <>
SOM<CartesianLayout<1>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.get_size() * neuron_layout.get_size())
{}

template <>
SOM<CartesianLayout<2>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width, input_data.som_height}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.get_size() * neuron_layout.get_size())
{}

template <>
SOM<CartesianLayout<3>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width, input_data.som_height, input_data.som_depth}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.get_size() * neuron_layout.get_size())
{}

template <>
SOM<HexagonalLayout, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.get_size() * neuron_layout.get_size())
{}

} // namespace pink
