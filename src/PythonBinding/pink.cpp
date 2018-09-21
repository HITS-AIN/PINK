#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"
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
			new (&m) Cartesian<2, float>({info.shape[0], info.shape[1]}, std::vector<float>(p, p + info.shape[0] * info.shape[1]));
		})
        .def("info", &Cartesian<2, float>::info);

    py::class_<Cartesian<2, Cartesian<2, float>>>(m, "cartesian_2d_cartesian_2d_float", py::buffer_protocol())
        .def(py::init())
		.def("__init__", [](Cartesian<2, Cartesian<2, float>> &m, py::buffer b)
		{
			py::buffer_info info = b.request();

	        if (info.ndim != 4)
	            throw std::runtime_error("Incompatible buffer dimension!");

			auto&& p = static_cast<float*>(info.ptr);
			std::vector<Cartesian<2, float>> som;
			for (int i = 0; i != info.shape[0] * info.shape[1]; ++i) {
				som.push_back(Cartesian<2, float>({info.shape[2], info.shape[3]}, std::vector<float>(p, p + info.shape[2] * info.shape[3])));
				p += info.shape[2] * info.shape[3];
			}

			new (&m) Cartesian<2, Cartesian<2, float>>({info.shape[0], info.shape[1]}, std::move(som));
		})
        .def("info", &Cartesian<2, Cartesian<2, float>>::info);

    py::class_<Trainer>(m, "trainer")
        .def(py::init())
        .def("__call__", [](Trainer const& trainer, Cartesian<2, Cartesian<2, float>>& som, Cartesian<2, float> const& image)
        {
    	    return trainer(som, image);
        });
}
