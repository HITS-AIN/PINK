#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"

namespace py = pybind11;
using namespace pink;

PYBIND11_MODULE(pink, m)
{
    m.doc() = "pybind11 PINK plugin";

    py::class_<Cartesian<2, Cartesian<2, float>>>(m, "cartesian_2d_cartesian_2d_float")
        .def(py::init())
		.def("info", &Cartesian<2, Cartesian<2, float>>::info);
}
