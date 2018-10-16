/**
 * @file   SelfOrganizingMapLib/FileIO.h
 * @date   Oct 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/SOM.h"

namespace pink {

//! Write SOM in binary mode
template <typename SOMLayout, typename NeuronLayout, typename T>
void write(SOM<SOMLayout, NeuronLayout, T> const& som, std::string const& filename)
{
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("Error opening " + filename);

    auto&& som_layout = som.get_som_layout();
    auto&& neuron_layout = som.get_neuron_layout();

    os << "# " << som.header << std::endl;
    os << "# " << som_layout.type << " ";
    for (int dim = 0; dim != som_layout.dimensionality; ++dim) os << som_layout.dimension[dim] << " ";
    for (int dim = 0; dim != neuron_layout.dimensionality; ++dim) os << neuron_layout.dimension[dim] << " ";
    os << std::endl;

    // binary part
    for (int dim = 0; dim != som_layout.dimensionality; ++dim) {
    	int tmp = som_layout.dimension[dim];
    	os.write((char*)&tmp, sizeof(int));
    }
    for (int dim = som_layout.dimensionality; dim != 3; ++dim) {
    	int tmp = 1;
    	os.write((char*)&tmp, sizeof(int));
    }
    for (int dim = 0; dim != neuron_layout.dimensionality; ++dim) {
    	int tmp = neuron_layout.dimension[dim];
	    os.write((char*)&tmp, sizeof(int));
    }
    for (int dim = neuron_layout.dimensionality; dim != 3; ++dim) {
    	int tmp = 1;
    	os.write((char*)&tmp, sizeof(int));
    }
    os.write((char*)som.get_data_pointer(), som_layout.get_size() * neuron_layout.get_size() * sizeof(T));
}

} // namespace pink
