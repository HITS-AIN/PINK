/**
 * @file   Pink/main_generic.h
 * @brief  Generic main routine of PINK
 * @date   Oct 8, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <iostream>

#include "ImageProcessingLib/ImageIterator.h"
#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/DataIO.h"
#include "SelfOrganizingMapLib/FileIO.h"
#include "SOM.h"
#include "Mapper_generic.h"
#include "Trainer_generic.h"
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
                  << "GPU usage:   " << UseGPU << "\n"
                  << std::endl;

    SOM<SOMLayout, DataLayout, T> som(input_data);

    auto&& distribution_function = GaussianFunctor(input_data.sigma, input_data.damping);

    if (input_data.executionPath == ExecutionPath::TRAIN)
    {
        Trainer_generic<SOMLayout, DataLayout, T, UseGPU> trainer(
            som
            ,distribution_function
            ,input_data.verbose
            ,input_data.numberOfRotations
            ,input_data.use_flip
            ,input_data.max_update_distance
            ,input_data.interpolation
#ifdef __CUDACC__
            ,input_data.block_size_1
            ,input_data.useMultipleGPUs
            ,input_data.euclidean_distance_type
#endif
        );

        auto&& iter_image_cur = ImageIterator<T>(input_data.imagesFilename);
        ProgressBar progress_bar(iter_image_cur.getNumberOfImages(), 70, input_data.number_of_progress_prints);
        uint32_t count = 0;
        for (auto&& iter_image_end = ImageIterator<T>(); iter_image_cur != iter_image_end; ++iter_image_cur, ++progress_bar)
        {
            auto&& beg = iter_image_cur->getPointerOfFirstPixel();
            auto&& end = beg + iter_image_cur->getSize();
            Data<DataLayout, T> data({input_data.image_dim, input_data.image_dim}, std::vector<T>(beg, end));
            trainer(data);

            if (progress_bar.valid() and input_data.intermediate_storage != IntermediateStorageType::OFF) {
                std::string interStoreFilename = input_data.resultFilename;
                if (input_data.intermediate_storage == IntermediateStorageType::KEEP) {
                    interStoreFilename.insert(interStoreFilename.find_last_of("."), "_" + std::to_string(count++));
                }
                if (input_data.verbose) std::cout << "  Write intermediate SOM to " << interStoreFilename << " ... " << std::flush;
                #ifdef __CUDACC__
                    trainer.update_som();
                #endif
                write(som, interStoreFilename);
                if (input_data.verbose) std::cout << "done." << std::endl;
            }
        }

        std::cout << "  Write final SOM to " << input_data.resultFilename << " ... " << std::flush;
#ifdef __CUDACC__
        trainer.update_som();
#endif
        write(som, input_data.resultFilename);
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
        std::ofstream result_file(input_data.resultFilename);
        if (!result_file) throw pink::exception("Error opening " + input_data.resultFilename);
        result_file.write((char*)&input_data.number_of_images, sizeof(int));
        som.write_file_header(result_file);

        // File for spatial_transformations (optional)
        std::ofstream spatial_transformation_file;
        if (input_data.write_rot_flip) {
            spatial_transformation_file.open(input_data.rot_flip_filename);
            spatial_transformation_file.write((char*)&input_data.number_of_images, sizeof(int));
            som.write_file_header(spatial_transformation_file);
        }

        Mapper_generic<SOMLayout, DataLayout, T, UseGPU> mapper(
            som
            ,input_data.verbose
            ,input_data.numberOfRotations
            ,input_data.use_flip
            ,input_data.interpolation
#ifdef __CUDACC__
            ,input_data.block_size_1
            ,input_data.useMultipleGPUs
            ,input_data.euclidean_distance_type
#endif
        );

        auto&& iter_image_cur = ImageIterator<T>(input_data.imagesFilename);
        ProgressBar progress_bar(iter_image_cur.getNumberOfImages(), 70, input_data.number_of_progress_prints);
        for (auto&& iter_image_end = ImageIterator<T>(); iter_image_cur != iter_image_end; ++iter_image_cur, ++progress_bar)
        {
            auto&& beg = iter_image_cur->getPointerOfFirstPixel();
            auto&& end = beg + iter_image_cur->getSize();
            Data<DataLayout, T> data({input_data.image_dim, input_data.image_dim}, std::vector<T>(beg, end));

            auto result = mapper(data);
            // corresponds to structured binding with C++17:
            //auto& [euclidean_distance_matrix, best_rotation_matrix] = mapper(data);

            result_file.write((char*)&std::get<0>(result)[0], som.get_number_of_neurons() * sizeof(float));

            if (input_data.write_rot_flip) {
                float angle_step_radians = 0.5 * M_PI / input_data.numberOfRotations / 4;
                for (uint32_t i = 0; i != som.get_number_of_neurons(); ++i) {
                    char flip = std::get<1>(result)[i] / input_data.numberOfRotations;
                    float angle = (std::get<1>(result)[i] % input_data.numberOfRotations) * angle_step_radians;
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
