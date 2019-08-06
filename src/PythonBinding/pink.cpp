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
#include "DynamicSOM.h"
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
        .def(py::init([](py::buffer b, std::string data_type, std::string layout)
        {
            py::buffer_info info = b.request();
            return new DynamicData(data_type, layout, info.shape, info.ptr);
        }),
            py::arg().noconvert(),
            py::arg("data_type") = "float32",
            py::arg("layout") = "cartesian-2d"
        )
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

    py::class_<DynamicSOM>(m, "som", py::buffer_protocol())
        .def(py::init([](py::buffer b, std::string const& data_type,
            std::string const& som_layout, std::string const& neuron_layout)
        {
            py::buffer_info info = b.request();
            return new DynamicSOM(data_type, som_layout, neuron_layout, info.shape, info.ptr);
        }),
            py::arg().noconvert(),
            py::arg("data_type") = "float32",
            py::arg("som_layout") = "cartesian-2d",
            py::arg("neuron_layout") = "cartesian-2d"
        )
        .def_buffer([](DynamicSOM &m) -> py::buffer_info
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
