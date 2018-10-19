/**
 * @file   SelfOrganizingMapLib/pink.cpp
 * @date   Oct 12, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Data.h"
#include "SelfOrganizingMapLib/SOM.h"
#include "SelfOrganizingMapLib/TrainerCPU.h"
#include "UtilitiesLib/Version.h"

namespace py = pybind11;
using namespace pink;

PYBIND11_MODULE(pink, m)
{
    m.doc() = "PINK python interface";
    m.attr("__version__") = std::string(PROJECT_VERSION) + " revision " + std::string(GIT_REVISION);

    py::class_<Data<CartesianLayout<2>, float>>(m, "data", py::buffer_protocol())
        .def(py::init())
        .def("__init__", [](Data<CartesianLayout<2>, float> &m, py::buffer b)
        {
            py::buffer_info info = b.request();

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            auto&& p = static_cast<float*>(info.ptr);
            auto&& dim0 = static_cast<uint32_t>(info.shape[0]);
            auto&& dim1 = static_cast<uint32_t>(info.shape[1]);
            new (&m) Data<CartesianLayout<2>, float>({dim0, dim1}, std::vector<float>(p, p + dim0 * dim1));
        })
        .def_buffer([](Data<CartesianLayout<2>, float> &m) -> py::buffer_info {

             auto&& dimension = m.get_dimension();

             return py::buffer_info(
                 m.get_data_pointer(),                   /* Pointer to buffer */
                 sizeof(float),                          /* Size of one scalar */
                 py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                 2,                                      /* Number of dimensions */
                 { dimension[0],
                   dimension[1]},                        /* Buffer dimensions */
                 { sizeof(float) * dimension[1],
                   sizeof(float) }                       /* Strides (in bytes) for each index */
             );
         });

    py::class_<SOM<CartesianLayout<2>, CartesianLayout<2>, float>>(m, "som", py::buffer_protocol())
        .def(py::init())
        .def("__init__", [](SOM<CartesianLayout<2>, CartesianLayout<2>, float> &m, py::buffer b)
        {
            py::buffer_info info = b.request();

            if (info.ndim != 4) throw std::runtime_error("Incompatible buffer dimension!");

            auto&& p = static_cast<float*>(info.ptr);
            auto&& dim0 = static_cast<uint32_t>(info.shape[0]);
            auto&& dim1 = static_cast<uint32_t>(info.shape[1]);
            auto&& dim2 = static_cast<uint32_t>(info.shape[2]);
            auto&& dim3 = static_cast<uint32_t>(info.shape[3]);
            new (&m) SOM<CartesianLayout<2>, CartesianLayout<2>, float>({dim0, dim1}, {dim2, dim3}, p);
        })
        .def_buffer([](SOM<CartesianLayout<2>, CartesianLayout<2>, float> &m) -> py::buffer_info {

             auto&& som_dimension = m.get_som_dimension();
             auto&& neuron_dimension = m.get_neuron_dimension();

             return py::buffer_info(
                 m.get_data_pointer(),                   /* Pointer to buffer */
                 sizeof(float),                          /* Size of one scalar */
                 py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                 4,                                      /* Number of dimensions */
                 { som_dimension[0],
                   som_dimension[1],
                   neuron_dimension[0],
                   neuron_dimension[1]},                 /* Buffer dimensions */
                 { sizeof(float) * neuron_dimension[1] * neuron_dimension[0] * som_dimension[1],
                   sizeof(float) * neuron_dimension[1] * neuron_dimension[0],
                   sizeof(float) * neuron_dimension[1],
                   sizeof(float) }                       /* Strides (in bytes) for each index */
             );
         });

    py::class_<Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false>>(m, "trainer")
        .def(py::init<SOM<CartesianLayout<2>, CartesianLayout<2>, float>&, std::function<float(float)>, int, int, bool, float>(),
            py::arg("som"),
            py::arg("distribution_function"),
            py::arg("verbosity") = 0,
            py::arg("number_of_rotations") = 360,
            py::arg("use_flip") = true,
            py::arg("max_update_distance") = 0.0
        )
        .def("__call__", [](Trainer<CartesianLayout<2>, CartesianLayout<2>, float, false> const& trainer, Data<CartesianLayout<2>, float> const& image)
        {
            return trainer(image);
        });
}
