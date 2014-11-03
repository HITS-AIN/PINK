/**
 * @file   cuda_trainSelfOrganizingMap.cu
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#include "CudaLib.h"
#include "cublas_v2.h"
#include <iostream>
#include <omp.h>

void cuda_trainSelfOrganizingMap(InputData const& inputData)
{
    if (inputData.verbose) {
    	cuda_print_properties();
		std::cout << "  === WARNING === Number of CPU threads must be one using CUDA." << std::endl;
	}
    omp_set_num_threads(1);
}
