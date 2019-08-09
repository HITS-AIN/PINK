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

    os << som.m_header;

    // <file format version> 1 <data-type> <som layout> <neuron layout> <data>
    int version = 2;
    int file_type = 1;
    int data_type_idx = 0;
    int som_layout_idx = 0;
    int neuron_layout_idx = 0;
    int som_dimensionality = som_layout.dimensionality;
    int neuron_dimensionality = neuron_layout.dimensionality;

    os.write(reinterpret_cast<char*>(&version), sizeof(int));
    os.write(reinterpret_cast<char*>(&file_type), sizeof(int));
    os.write(reinterpret_cast<char*>(&data_type_idx), sizeof(int));
    os.write(reinterpret_cast<char*>(&som_layout_idx), sizeof(int));
    os.write(reinterpret_cast<char*>(&som_dimensionality), sizeof(int));
    for (auto d : som_layout.m_dimension) os.write(reinterpret_cast<char*>(&d), sizeof(int));
    os.write(reinterpret_cast<char*>(&neuron_layout_idx), sizeof(int));
    os.write(reinterpret_cast<char*>(&neuron_dimensionality), sizeof(int));
    for (auto d : neuron_layout.m_dimension) os.write(reinterpret_cast<char*>(&d), sizeof(int));
    os.write(reinterpret_cast<const char*>(som.get_data_pointer()), static_cast<std::streamsize>(som.size() * sizeof(T)));
}

} // namespace pink
