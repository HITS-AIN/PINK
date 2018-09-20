#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SelfOrganizingMapLib/Cartesian.h"
#include "SelfOrganizingMapLib/Hexagonal.h"
#include "SelfOrganizingMapLib/Trainer.h"

namespace py = pybind11;
using namespace pink;

template<typename T>
void declare_layout(py::module &m, std::string const& typestr)
{
    py::class_<T>(m, typestr.c_str(), py::buffer_protocol())
        .def(py::init())
        .def("info", &T::info);
}

class Matrix {
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new float[rows*cols];
    }
    float *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    float *m_data;
};

PYBIND11_MODULE(pink, m)
{
    m.doc() = "pybind11 PINK plugin";

    declare_layout<Cartesian<2, float>>(m, "cartesian_2d_float");
    declare_layout<Cartesian<2, Cartesian<2, float>>>(m, "cartesian_2d_cartesian_2d_float");

    py::class_<Trainer>(m, "trainer")
        .def(py::init())
        .def("__call__", [](Trainer const& trainer, Cartesian<2, Cartesian<2, float>>& som, Cartesian<2, float> const& image)
        {
    	    return trainer(som, image);
        })
        .def("__call__", [](Trainer const& trainer, Cartesian<2, Cartesian<2, float>>& som, py::array_t<float> const& image)
        {
    	    //return trainer(som, image);
        });

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * m.rows(),             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        })
		.def("__init__", [](Matrix &m, py::buffer b)
		{
			py::buffer_info info = b.request();
			new (&m) Matrix(info.shape[0], info.shape[1]);
		});
}
