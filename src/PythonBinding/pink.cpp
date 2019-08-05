/**
 * @file   PythonBinding/pink.cpp
 * @date   Nov 21, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "DynamicData.h"
//#include "DynamicMapper.h"
//#include "DynamicSOM.h"
//#include "DynamicTrainer.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/Interpolation.h"
#include "UtilitiesLib/Layout.h"
#include "UtilitiesLib/Version.h"

namespace py = pybind11;

PYBIND11_MODULE(pink, m)
{
    using namespace pink;

    m.doc() = "PINK python interface";
    m.attr("__version__") = std::string(PROJECT_VERSION) + " revision " + std::string(GIT_REVISION);

    py::enum_<Interpolation>(m, "interpolation")
       .value("NEAREST_NEIGHBOR", Interpolation::NEAREST_NEIGHBOR)
       .value("BILINEAR", Interpolation::BILINEAR)
       .export_values();

    py::enum_<DataType>(m, "data_type")
       .value("FLOAT", DataType::FLOAT)
       .value("UINT16", DataType::UINT16)
       .value("UINT8", DataType::UINT8)
       .export_values();

    py::enum_<Layout>(m, "layout")
       .value("CARTESIAN", Layout::CARTESIAN)
       .value("HEXAGONAL", Layout::HEXAGONAL)
       .export_values();

    py::class_<DynamicData>(m, "data", py::buffer_protocol())
        .def(py::init([](DataType data_type, Layout layout, py::buffer b)
        {
            py::buffer_info info = b.request();
            return new DynamicData(data_type, layout, info.shape, info.ptr);
        }))
        .def_buffer([](DynamicData &m) -> py::buffer_info
        {
            auto&& info = m.get_buffer_info();
            return py::buffer_info(
                info.ptr,
                info.itemsize,
                info.format,
                info.ndim,
                info.shape,
                info.strides
            );
        });

//    py::class_<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m, "som", py::buffer_protocol())
//        .def(py::init())
//        .def(py::init([](py::buffer b)
//        {
//            py::buffer_info info = b.request();
//
//            if (info.ndim != 4) throw std::runtime_error("Incompatible buffer dimension!");
//
//            auto&& p = static_cast<float*>(info.ptr);
//            auto&& dim0 = static_cast<uint32_t>(info.shape[0]);
//            auto&& dim1 = static_cast<uint32_t>(info.shape[1]);
//            auto&& dim2 = static_cast<uint32_t>(info.shape[2]);
//            auto&& dim3 = static_cast<uint32_t>(info.shape[3]);
//            return new SOM<CartesianLayout<2>, CartesianLayout<2>, float>({dim0, dim1}, {dim2, dim3},
//                std::vector<float>(p, p + dim0 * dim1 * dim2 * dim3));
//        }))
//        .def_buffer([](SOM<CartesianLayout<2>, CartesianLayout<2>, float> &m) -> py::buffer_info {
//
//             auto&& som_dimension = m.get_som_dimension();
//             auto&& neuron_dimension = m.get_neuron_dimension();
//
//             return py::buffer_info(
//                 m.get_data_pointer(),                   /* Pointer to buffer */
//                 sizeof(float),                          /* Size of one scalar */
//                 py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
//                 4,                                      /* Number of dimensions */
//                 { som_dimension[0],
//                   som_dimension[1],
//                   neuron_dimension[0],
//                   neuron_dimension[1]},                 /* Buffer dimensions */
//                 { sizeof(float) * neuron_dimension[1] * neuron_dimension[0] * som_dimension[1],
//                   sizeof(float) * neuron_dimension[1] * neuron_dimension[0],
//                   sizeof(float) * neuron_dimension[1],
//                   sizeof(float) }                       /* Strides (in bytes) for each index */
//             );
//         });
//
//    py::class_<DynamicTrainer>(m, "trainer")
//        .def(py::init<DynamicSOM>&, std::function<float(float)>, int, uint32_t, bool, float, Interpolation, int>(),
//            py::arg("som"),
//            py::arg("distribution_function"),
//            py::arg("verbosity") = 0,
//            py::arg("number_of_rotations") = 360,
//            py::arg("use_flip") = true,
//            py::arg("max_update_distance") = -1.0,
//            py::arg("interpolation") = Interpolation::BILINEAR,
//            py::arg("euclidean_distance_dim") = -1
//        )
//        .def("__call__", [](Trainer<CartesianLayout<2>, CartesianLayout<2>,
//            float, false>& trainer, Data<CartesianLayout<2>, float> const& data)
//        {
//            return trainer(data);
//        });
//
//    py::class_<Mapper<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m, "mapper")
//        .def(py::init<SOM<CartesianLayout<2>, CartesianLayout<2>,
//            float>&, int, uint32_t, bool, Interpolation, int>(),
//            py::arg("som"),
//            py::arg("verbosity") = 0,
//            py::arg("number_of_rotations") = 360,
//            py::arg("use_flip") = true,
//            py::arg("interpolation") = Interpolation::BILINEAR,
//            py::arg("euclidean_distance_dim") = -1
//        )
//        .def("__call__", [](Mapper<CartesianLayout<2>, CartesianLayout<2>,
//            float, false>& mapper, Data<CartesianLayout<2>, float> const& data)
//        {
//            return mapper(data);
//        });

}
