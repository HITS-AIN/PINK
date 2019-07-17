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

template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
void main_generic(InputData const& input_data)
{
    if (input_data.verbose)
        std::cout << "SOM layout:  " << SOMLayout::type  << "<" << static_cast<int>(SOMLayout::dimensionality)  << ">" << "\n"
                  << "Data layout: " << DataLayout::type << "<" << static_cast<int>(DataLayout::dimensionality) << ">" << "\n"
                  << std::endl;

    SOM<SOMLayout, DataLayout, T> som(input_data);

    std::ifstream ifs(input_data.data_filename);
    if (!ifs) throw std::runtime_error("Error opening " + input_data.data_filename);

    if (input_data.executionPath == ExecutionPath::TRAIN)
    {
        Trainer<SOMLayout, DataLayout, T, UseGPU> trainer(
            som
            ,input_data.get_distribution_function()
            ,input_data.verbose
            ,input_data.number_of_rotations
            ,input_data.use_flip
            ,input_data.max_update_distance
            ,input_data.interpolation
            ,input_data.euclidean_distance_dim
#ifdef __CUDACC__
            ,input_data.block_size_1
            ,input_data.euclidean_distance_type
#endif
        );

        auto&& iter_data_cur = DataIteratorShuffled<DataLayout, T>(ifs, static_cast<uint64_t>(input_data.seed), input_data.shuffle_data_input);
        auto&& iter_data_end = DataIteratorShuffled<DataLayout, T>(ifs, true);

        ProgressBar progress_bar(iter_data_cur.get_number_of_entries() * input_data.numIter, 70, input_data.max_number_of_progress_prints);
        uint32_t count = 0;
        for (int i = 0; i < input_data.numIter; ++i)
        {
            iter_data_cur.set_to_begin();
            for (; iter_data_cur != iter_data_end; ++iter_data_cur, ++progress_bar)
            {
                trainer(*iter_data_cur);

                if (progress_bar.valid() and input_data.intermediate_storage != IntermediateStorageType::OFF) {
                    std::string interStore_filename = input_data.result_filename;
                    if (input_data.intermediate_storage == IntermediateStorageType::KEEP) {
                        interStore_filename.insert(interStore_filename.find_last_of("."), "_" + std::to_string(count++));
                    }
                    if (input_data.verbose) std::cout << "  Write intermediate SOM to " << interStore_filename << " ... " << std::flush;
                    #ifdef __CUDACC__
                        trainer.update_som();
                    #endif
                    write(som, interStore_filename);
                    if (input_data.verbose) std::cout << "done." << std::endl;
                }
            }
        }

        std::cout << "  Write final SOM to " << input_data.result_filename << " ... " << std::flush;
#ifdef __CUDACC__
        trainer.update_som();
#endif
        write(som, input_data.result_filename);
        std::cout << "done." << std::endl;

        if (input_data.verbose) {
            std::cout << "\n  Number of updates of each neuron:\n\n"
                      << trainer.get_update_info()
                      << std::endl;
        }
    }
    else if (input_data.executionPath == ExecutionPath::MAP)
    {
        // File for euclidean distances
        std::ofstream result_file(input_data.result_filename);
        if (!result_file) throw pink::exception("Error opening " + input_data.result_filename);

        auto&& iter_data_cur = DataIterator<DataLayout, T>(ifs);
        auto&& iter_data_end = DataIterator<DataLayout, T>(ifs, true);

        // <file format version> 2 <data-type> <number of entries> <som layout> <data>
        int version = 2;
        int file_type = 2;
        int data_type_idx = 0;
        int som_layout_idx = 0;
        int som_dimensionality = som.get_som_layout().dimensionality;
        int number_of_data_entries = iter_data_cur.get_number_of_entries();

        result_file.write((char*)&version, sizeof(int));
        result_file.write((char*)&file_type, sizeof(int));
        result_file.write((char*)&data_type_idx, sizeof(int));
        result_file.write((char*)&number_of_data_entries, sizeof(int));
        result_file.write((char*)&som_layout_idx, sizeof(int));
        result_file.write((char*)&som_dimensionality, sizeof(int));
        for (int dim = 0; dim != som_dimensionality; ++dim) {
            int tmp = som.get_som_layout().dimension[dim];
            result_file.write((char*)&tmp, sizeof(int));
        }

        // File for spatial_transformations (optional)
        std::ofstream spatial_transformation_file;
        if (input_data.write_rot_flip) {
            spatial_transformation_file.open(input_data.rot_flip_filename);
            if (!spatial_transformation_file) throw pink::exception("Error opening " + input_data.rot_flip_filename);

            // <file format version> 3 <number of entries> <som layout> <data>
            int file_type = 3;

            spatial_transformation_file.write((char*)&version, sizeof(int));
            spatial_transformation_file.write((char*)&file_type, sizeof(int));
            spatial_transformation_file.write((char*)&number_of_data_entries, sizeof(int));
            spatial_transformation_file.write((char*)&som_layout_idx, sizeof(int));
            spatial_transformation_file.write((char*)&som_dimensionality, sizeof(int));
            for (int dim = 0; dim != som_dimensionality; ++dim) {
                int tmp = som.get_som_layout().dimension[dim];
                spatial_transformation_file.write((char*)&tmp, sizeof(int));
            }
        }

        Mapper<SOMLayout, DataLayout, T, UseGPU> mapper(
            som
            ,input_data.verbose
            ,input_data.number_of_rotations
            ,input_data.use_flip
            ,input_data.interpolation
            ,input_data.euclidean_distance_dim
#ifdef __CUDACC__
            ,input_data.block_size_1
            ,input_data.euclidean_distance_type
#endif
        );

        ProgressBar progress_bar(iter_data_cur.get_number_of_entries(), 70, input_data.max_number_of_progress_prints);
        for (; iter_data_cur != iter_data_end; ++iter_data_cur, ++progress_bar)
        {
            auto result = mapper(*iter_data_cur);
            // corresponds to structured binding with C++17:
            //auto& [euclidean_distance_matrix, best_rotation_matrix] = mapper(data);

            result_file.write((char*)&std::get<0>(result)[0], som.get_number_of_neurons() * sizeof(float));

            if (input_data.write_rot_flip) {
                float angle_step_radians = 0.5 * M_PI / input_data.number_of_rotations / 4;
                for (uint32_t i = 0; i != som.get_number_of_neurons(); ++i) {
                    char flip = std::get<1>(result)[i] / input_data.number_of_rotations;
                    float angle = (std::get<1>(result)[i] % input_data.number_of_rotations) * angle_step_radians;
                    spatial_transformation_file.write(&flip, sizeof(char));
                    spatial_transformation_file.write((char*)&angle, sizeof(float));
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
