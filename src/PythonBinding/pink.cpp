#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"
#include "SelfOrganizingMapLib/Trainer.h"

namespace py = pybind11;
using namespace pink;

PYBIND11_MODULE(pink, m)
{
    m.doc() = "pybind11 PINK plugin";

    py::class_<Cartesian<2, float>>(m, "cartesian_2d_float")
        .def(py::init())
		.def("info", &Cartesian<2, float>::info);

    py::class_<Cartesian<2, Cartesian<2, float>>>(m, "cartesian_2d_cartesian_2d_float")
        .def(py::init())
		.def("info", &Cartesian<2, Cartesian<2, float>>::info);

    py::class_<Trainer>(m, "trainer")
        .def(py::init())
        .def("__call__",[](const Trainer& trainer, Cartesian<2, Cartesian<2, float>>& som, Cartesian<2, float> const& image) { return trainer(som, image); });
}
