/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <iostream>

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIO.h"
#include "SelfOrganizingMapLib/DataIterator.h"
#include "SelfOrganizingMapLib/DataIteratorShuffled.h"
#include "SelfOrganizingMapLib/FileIO.h"
#include "SelfOrganizingMapLib/Mapper.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/Trainer.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/DistributionFunctor.h"
#include "UtilitiesLib/InputData.h"
#include "UtilitiesLib/pink_exception.h"
#include "UtilitiesLib/ProgressBar.h"

namespace pink {

template <typename SOMLayout, typename T, bool UseGPU>
void main_generic(InputData const& input_data)
{
    if (input_data.m_data_layout == Layout::CARTESIAN)
        if (input_data.m_data_dimension.size() == 1)
            main_generic<SOMLayout, CartesianLayout<1U>, T, UseGPU>(input_data);
        else if (input_data.m_data_dimension.size() == 2)
            main_generic<SOMLayout, CartesianLayout<2U>, T, UseGPU>(input_data);
        else if (input_data.m_data_dimension.size() == 3)
            main_generic<SOMLayout, CartesianLayout<3U>, T, UseGPU>(input_data);
        else
            throw pink::exception("Unsupported data dimensionality: " + std::to_string(input_data.m_data_dimension.size()));
    else
        throw pink::exception("Unsupported data layout: " + std::to_string(input_data.m_data_layout));
}

template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
void main_generic(InputData const& input_data)
{
    if (input_data.m_verbose)
        std::cout << "SOM layout:  " << SOMLayout::type
                  << "<" << static_cast<int>(SOMLayout::dimensionality)  << ">" << "\n"
                  << "Data layout: " << DataLayout::type
                  << "<" << static_cast<int>(DataLayout::dimensionality) << ">" << "\n"
                  << std::endl;

    SOM<SOMLayout, DataLayout, T> som(input_data);

    std::ifstream ifs(input_data.m_data_filename);
    if (!ifs) throw std::runtime_error("Error opening " + input_data.m_data_filename);

    if (input_data.m_executionPath == ExecutionPath::TRAIN)
    {
        Trainer<SOMLayout, DataLayout, T, UseGPU> trainer(
            som
            ,input_data.get_distribution_function()
            ,input_data.m_verbose
            ,input_data.m_number_of_rotations
            ,input_data.m_use_flip
            ,input_data.m_max_update_distance
            ,input_data.m_interpolation
            ,input_data.m_euclidean_distance_dim
            ,input_data.m_euclidean_distance_shape
#ifdef __CUDACC__
            ,input_data.m_block_size_1
            ,input_data.m_euclidean_distance_type
#endif
        );


        ProgressBar progress_bar(static_cast<int>(
            input_data.m_number_of_data_entries * input_data.m_number_of_iterations),
            70, input_data.m_max_number_of_progress_prints);
        uint32_t count = 0;
        for (uint32_t i = 0; i < input_data.m_number_of_iterations; ++i)
        {
            // Change the seed for DataIteratorShuffled for every iteration by adding
            // the loop index number, so that the image order is different in every iteration.
            auto&& iter_data_cur = DataIteratorShuffled<DataLayout, T>(ifs,
                static_cast<uint64_t>(input_data.m_seed) + i, input_data.m_shuffle_data_input);
            auto&& iter_data_end = DataIteratorShuffled<DataLayout, T>(ifs, true);
            for (; iter_data_cur != iter_data_end; ++iter_data_cur, ++progress_bar)
            {
                trainer(*iter_data_cur);

                if (progress_bar.valid() and input_data.m_intermediate_storage != IntermediateStorageType::OFF) {
                    std::string interStore_filename = input_data.m_result_filename;
                    if (input_data.m_intermediate_storage == IntermediateStorageType::KEEP) {
                        interStore_filename.insert(interStore_filename.find_last_of("."),
                            "_" + std::to_string(count++));
                    }
                    if (input_data.m_verbose) {
                        std::cout << "  Write intermediate SOM to " << interStore_filename << " ... " << std::flush;
                    }
                    #ifdef __CUDACC__
                        trainer.update_som();
                    #endif
                    write(som, interStore_filename);
                    if (input_data.m_verbose) std::cout << "done." << std::endl;
                }
            }
        }

        std::cout << "  Write final SOM to " << input_data.m_result_filename << " ... " << std::flush;
#ifdef __CUDACC__
        trainer.update_som();
#endif
        write(som, input_data.m_result_filename);
        std::cout << "done." << std::endl;

        if (input_data.m_verbose) {
            std::cout << "\n  Number of updates of each neuron:\n\n"
                      << trainer.get_update_info()
                      << std::endl;
        }
    }
    else if (input_data.m_executionPath == ExecutionPath::MAP)
    {
        // File for euclidean distances
        std::ofstream result_file(input_data.m_result_filename);
        if (!result_file) throw pink::exception("Error opening " + input_data.m_result_filename);

        auto&& iter_data_cur = DataIterator<DataLayout, T>(ifs);
        auto&& iter_data_end = DataIterator<DataLayout, T>(ifs, true);

        // <file format version> 2 <data-type> <number of entries> <som layout> <data>
        int version = 2;
        int file_type = 2;
        int data_type_idx = 0;
        int som_layout_idx = 0;
        int som_dimensionality = static_cast<int>(som.get_som_layout().dimensionality);
        int number_of_data_entries = static_cast<int>(input_data.m_number_of_data_entries);

        result_file.write(reinterpret_cast<char*>(&version), sizeof(int));
        result_file.write(reinterpret_cast<char*>(&file_type), sizeof(int));
        result_file.write(reinterpret_cast<char*>(&data_type_idx), sizeof(int));
        result_file.write(reinterpret_cast<char*>(&number_of_data_entries), sizeof(int));
        result_file.write(reinterpret_cast<char*>(&som_layout_idx), sizeof(int));
        result_file.write(reinterpret_cast<char*>(&som_dimensionality), sizeof(int));
        for (auto d : som.get_som_layout().m_dimension) result_file.write(reinterpret_cast<char*>(&d), sizeof(int));

        // File for spatial_transformations (optional)
        std::ofstream spatial_transformation_file;
        if (input_data.m_write_rot_flip) {
            spatial_transformation_file.open(input_data.m_rot_flip_filename);
            if (!spatial_transformation_file) throw pink::exception("Error opening " + input_data.m_rot_flip_filename);

            // <file format version> 3 <number of entries> <som layout> <data>
            file_type = 3;

            spatial_transformation_file.write(reinterpret_cast<char*>(&version), sizeof(int));
            spatial_transformation_file.write(reinterpret_cast<char*>(&file_type), sizeof(int));
            spatial_transformation_file.write(reinterpret_cast<char*>(&number_of_data_entries), sizeof(int));
            spatial_transformation_file.write(reinterpret_cast<char*>(&som_layout_idx), sizeof(int));
            spatial_transformation_file.write(reinterpret_cast<char*>(&som_dimensionality), sizeof(int));
            for (auto d : som.get_som_layout().m_dimension) spatial_transformation_file.write(reinterpret_cast<char*>(&d), sizeof(int));
        }

        Mapper<SOMLayout, DataLayout, T, UseGPU> mapper(
            som
            ,input_data.m_verbose
            ,input_data.m_number_of_rotations
            ,input_data.m_use_flip
            ,input_data.m_interpolation
            ,input_data.m_euclidean_distance_dim
            ,input_data.m_euclidean_distance_shape
#ifdef __CUDACC__
            ,input_data.m_block_size_1
            ,input_data.m_euclidean_distance_type
#endif
        );

        ProgressBar progress_bar(number_of_data_entries, 70, input_data.m_max_number_of_progress_prints);
        for (; iter_data_cur != iter_data_end; ++iter_data_cur, ++progress_bar)
        {
            auto result = mapper(*iter_data_cur);
            // corresponds to structured binding with C++17:
            //auto& [euclidean_distance_matrix, best_rotation_matrix] = mapper(data);

            result_file.write(reinterpret_cast<char*>(&std::get<0>(result)[0]),
                static_cast<std::streamsize>(som.get_number_of_neurons() * sizeof(float)));

            if (input_data.m_write_rot_flip) {
                float angle_step_radians = static_cast<float>(2.0 * M_PI) / input_data.m_number_of_rotations;
                for (uint32_t i = 0; i != som.get_number_of_neurons(); ++i) {
                    char flip = static_cast<char>(std::get<1>(result)[i] / input_data.m_number_of_rotations);
                    float angle = (std::get<1>(result)[i] % input_data.m_number_of_rotations) * angle_step_radians;
                    spatial_transformation_file.write(&flip, sizeof(char));
                    spatial_transformation_file.write(reinterpret_cast<char*>(&angle), sizeof(float));
                }
            }
        }
    }
    else
    {
        throw pink::exception("Unknown execution path");
    }
}

} // namespace pink
