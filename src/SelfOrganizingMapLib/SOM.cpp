/**
 * @file   SelfOrganizingMapLib/SOM.h
 * @date   Sep 25, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include "SOM.h"
#include "UtilitiesLib/Filler.h"

namespace pink {

template <>
SOM<CartesianLayout<1>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.size() * neuron_layout.size())
{}

template <>
SOM<CartesianLayout<2>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width, input_data.som_height}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.size() * neuron_layout.size())
{
    // Initialize SOM
    if (input_data.init == SOMInitialization::ZERO)
        fill_value(&data[0], data.size());
    else if (input_data.init == SOMInitialization::RANDOM)
        fill_random_uniform(&data[0], data.size(), input_data.seed);
    else if (input_data.init == SOMInitialization::RANDOM_WITH_PREFERRED_DIRECTION) {
        fill_random_uniform(&data[0], data.size(), input_data.seed);
        for (int n = 0; n < input_data.som_size; ++n)
            for (uint32_t i = 0; i < input_data.neuron_dim; ++i)
                data[n * input_data.neuron_size + i * input_data.neuron_dim + i] = 1.0;
    }
    else if (input_data.init == SOMInitialization::FILEINIT) {
        std::ifstream is(input_data.som_filename);
        if (!is) throw pink::exception("Error opening " + input_data.som_filename);

        // Skip header
        std::string line;
        int binary_start_position = 0;
        while (std::getline(is, line)) {
            if (line == "# END OF HEADER") {
                binary_start_position = is.tellg();
                break;
            }
        }

        // Keep header
        is.clear();
        is.seekg(0, is.beg);
        if (binary_start_position != 0) {
            while (std::getline(is, line)) {
                header += line + '\n';
                if (line == "# END OF HEADER") break;
            }
        }

        is.clear();
        is.seekg(binary_start_position, is.beg);

        // <file format version> 1 <data-type> <som layout> <neuron layout> <data>
        int tmp;
        is.read((char*)&tmp, sizeof(int));
        if (tmp != 2) throw pink::exception("read SOM: wrong binary file version");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != 1) throw pink::exception("read SOM: wrong file type");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != 0) throw pink::exception("read SOM: wrong data type");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != 0) throw pink::exception("read SOM: wrong SOM layout");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != som_layout.dimensionality) throw pink::exception("read SOM: wrong SOM dimensionality");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != static_cast<int>(som_layout.dimension[0])) throw pink::exception("read SOM: wrong SOM dimension[0]");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != static_cast<int>(som_layout.dimension[1])) throw pink::exception("read SOM: wrong SOM dimension[1]");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != 0) throw pink::exception("read SOM: wrong neuron layout");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != neuron_layout.dimensionality) throw pink::exception("read SOM: wrong neuron dimensionality");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != static_cast<int>(neuron_layout.dimension[0])) throw pink::exception("read SOM: wrong neuron dimension[0]");
        is.read((char*)&tmp, sizeof(int));
        if (tmp != static_cast<int>(neuron_layout.dimension[1])) throw pink::exception("read SOM: wrong neuron dimension[1]");
        is.read((char*)&data[0], data.size() * sizeof(float));
    } else
        throw pink::exception("Unknown SOMInitialization");
}

template <>
SOM<CartesianLayout<3>, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width, input_data.som_height, input_data.som_depth}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.size() * neuron_layout.size())
{}

template <>
SOM<HexagonalLayout, CartesianLayout<2>, float>::SOM(InputData const& input_data)
 : som_layout{{input_data.som_width, input_data.som_height}},
   neuron_layout{{input_data.neuron_dim, input_data.neuron_dim}},
   data(som_layout.size() * neuron_layout.size())
{}

} // namespace pink
