#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<pybind11/stl.h>
#include<manifold.hpp>

namespace py = pybind11;

struct PyManifold : Manifold {
  PyManifold(int n, int k, double vf_bound, Eigen::VectorXd upper, Eigen::VectorXd lower)
    : Manifold(n, k, vf_bound, std::move(upper), std::move(lower)) {}

  Eigen::VectorXd coord(const Eigen::VectorXd& u) override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, Manifold, coord, u);
  }

  Eigen::MatrixXd pushforward(const Eigen::VectorXd& u) override {
    PYBIND11_OVERRIDE_PURE(Eigen::MatrixXd, Manifold, pushforward, u);
  }
};

PYBIND11_MODULE(saman, m) {
  py::class_<Manifold, PyManifold>(m, "Manifold")
    .def(py::init<int, int, double, Eigen::VectorXd, Eigen::VectorXd>(),
         py::arg("n"),
         py::arg("k"),
         py::arg("vf_bound"),
         py::arg("lower"),
         py::arg("upper"))
    .def_readonly("n", &Manifold::n)
    .def_readonly("k", &Manifold::k)
    .def_readonly("vf_bound", &Manifold::vf_bound)
    .def_readonly("lower", &Manifold::lower)
    .def_readonly("upper", &Manifold::upper)
    .def("metric", &Manifold::metric)
    .def("volume_form", &Manifold::volume_form)
    .def("sample", &Manifold::sample, py::arg("n_samples"));
}
