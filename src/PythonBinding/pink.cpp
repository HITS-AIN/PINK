#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"
#include "SelfOrganizingMapLib/SOM_generic.h"
#include "SelfOrganizingMapLib/Trainer.h"

namespace py = pybind11;
using namespace pink;

PYBIND11_MODULE(pink, m)
{
    m.doc() = "PINK python interface";

    py::class_<Cartesian<2, float>>(m, "cartesian_2d_float", py::buffer_protocol())
        .def(py::init())
		.def("__init__", [](Cartesian<2, float> &m, py::buffer b)
		{
			py::buffer_info info = b.request();

	        if (info.ndim != 2)
	            throw std::runtime_error("Incompatible buffer dimension!");

			auto&& p = static_cast<float*>(info.ptr);
			auto&& dim0 = static_cast<uint32_t>(info.shape[0]);
			auto&& dim1 = static_cast<uint32_t>(info.shape[1]);
			new (&m) Cartesian<2, float>({dim0, dim1}, std::vector<float>(p, p + dim0 * dim1));
		})
        .def_buffer([](Cartesian<2, float> &m) -> py::buffer_info {
             return py::buffer_info(
                 m.get_data_pointer(),                   /* Pointer to buffer */
                 sizeof(float),                          /* Size of one scalar */
                 py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                 2,                                      /* Number of dimensions */
                 { m.get_dimension()[0],
				   m.get_dimension()[1]},                /* Buffer dimensions */
                 { sizeof(float) * m.get_dimension()[1],
                   sizeof(float) }                       /* Strides (in bytes) for each index */
             );
         })
        .def("info", &Cartesian<2, float>::info);

    py::class_<SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float>>(m, "som_cartesian_2d_cartesian_2d_float", py::buffer_protocol())
        .def(py::init())
		.def("__init__", [](SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float> &m, py::buffer b)
		{
			py::buffer_info info = b.request();

	        if (info.ndim != 4) throw std::runtime_error("Incompatible buffer dimension!");

			auto&& p = static_cast<float*>(info.ptr);
			auto&& dim0 = static_cast<uint32_t>(info.shape[0]);
			auto&& dim1 = static_cast<uint32_t>(info.shape[1]);
			auto&& dim2 = static_cast<uint32_t>(info.shape[2]);
			auto&& dim3 = static_cast<uint32_t>(info.shape[3]);
			new (&m) SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float>({dim0, dim1}, {dim2, dim3}, p);
		})
        .def_buffer([](SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float> &m) -> py::buffer_info {
             return py::buffer_info(
                 m.get_data_pointer(),                   /* Pointer to buffer */
                 sizeof(float),                          /* Size of one scalar */
                 py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                 4,                                      /* Number of dimensions */
                 { m.get_som_dimension()[0],
                   m.get_som_dimension()[1],
				   m.get_neuron_dimension()[0],
				   m.get_neuron_dimension()[1]},         /* Buffer dimensions */
                 { sizeof(float) * m.get_neuron_dimension()[1] * m.get_neuron_dimension()[0] * m.get_som_dimension()[1],
                   sizeof(float) * m.get_neuron_dimension()[1] * m.get_neuron_dimension()[0],
                   sizeof(float) * m.get_neuron_dimension()[1],
                   sizeof(float) }                       /* Strides (in bytes) for each index */
             );
         });

    py::class_<Trainer>(m, "trainer")
        .def(py::init<std::function<float(float)>, int, int, bool, float, bool, int>(),
           	py::arg("distribution_function"),
        	py::arg("verbosity") = 0,
			py::arg("number_of_rotations") = 360,
			py::arg("use_flip") = true,
			py::arg("progress_factor") = 0.1,
			py::arg("use_gpu") = true,
			py::arg("max_update_distance") = 0
	    )
        .def("__call__", [](Trainer const& trainer, SOM_generic<CartesianLayout<2>, CartesianLayout<2>, float>& som, Cartesian<2, float> const& image)
        {
    	    return trainer(som, image);
        });
}
