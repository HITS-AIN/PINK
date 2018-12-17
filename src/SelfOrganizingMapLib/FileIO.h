/**
 * @file   SelfOrganizingMapLib/FileIO.h
 * @date   Oct 15, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "Data.h"
#include "SOM.h"

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

    // <file format version> 1 <data-type> <som layout> <neuron layout> <data>
    int version = 2;
    int file_type = 1;
    int data_type_idx = 0;
    int som_layout_idx = 0;
    int neuron_layout_idx = 0;
    int som_dimensionality = som_layout.dimensionality;
    int neuron_dimensionality = neuron_layout.dimensionality;

    os.write((char*)&version, sizeof(int));
    os.write((char*)&file_type, sizeof(int));
    os.write((char*)&data_type_idx, sizeof(int));
    os.write((char*)&som_layout_idx, sizeof(int));
    os.write((char*)&som_dimensionality, sizeof(int));
    for (int dim = 0; dim != som_layout.dimensionality; ++dim) {
        int tmp = som_layout.dimension[dim];
        os.write((char*)&tmp, sizeof(int));
    }
    os.write((char*)&neuron_layout_idx, sizeof(int));
    os.write((char*)&neuron_dimensionality, sizeof(int));
    for (int dim = 0; dim != neuron_layout.dimensionality; ++dim) {
        int tmp = neuron_layout.dimension[dim];
        os.write((char*)&tmp, sizeof(int));
    }
    os.write((char*)som.get_data_pointer(), som.size() * sizeof(T));
}

} // namespace pink
