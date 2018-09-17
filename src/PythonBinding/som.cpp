#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"

namespace py = pybind11;
using namespace pink;

PYBIND11_MODULE(som, m)
{
    m.doc() = "pybind11 PINK plugin";

    py::class_<Cartesian<2, Cartesian<2, float>>>(m, "som");
}
