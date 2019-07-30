/**
 * @file   SelfOrganizingMapTest/add_binary_section.cpp
 * @date   Jul 30, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include "add_binary_section.h"

void add_binary_section(std::stringstream& ss, std::vector<std::vector<float>> const& images)
{
    int version = 2;
    int binary_file_type = 0;
    int data_type = 0;
    int number_of_data_entries = 2;
    int layout = 0;
    int dimensionality = 2;
    int width = 2;
    int height = 2;

    ss.write(reinterpret_cast<const char*>(&version), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&binary_file_type), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&data_type), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&number_of_data_entries), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&layout), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&dimensionality), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&width), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&height), sizeof(int));
    ss.write(reinterpret_cast<const char*>(&images[0][0]), width * height * sizeof(float));
    ss.write(reinterpret_cast<const char*>(&images[1][0]), width * height * sizeof(float));
}
