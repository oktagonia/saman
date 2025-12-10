#include<iostream>
#include<numbers>
#include"manifold.hpp"

using namespace Eigen;
using namespace std;

class Torus : public Manifold {
public:
  double R, r;

  Torus(double R, double r)
    : Manifold(3, 2, 1.5,
               VectorXd::Constant(2, 0.0),
               VectorXd::Constant(2, 2.0 * numbers::pi)),
      R(R), r(r) {}

  VectorXd coord(const VectorXd& u) override {
    double phi = u[0];
    double theta = u[1];
    VectorXd x(3);
    x << (R + r * cos(theta)) * cos(phi), (R + r * cos(theta)) * sin(phi), r * sin(theta);
    return x;
  }

  MatrixXd pushforward(const VectorXd& u) override {
    double phi = u[0];
    double theta = u[1];
    MatrixXd J(3, 2);
    J << -(R + r * cos(theta)) * sin(phi), -r * sin(theta) * cos(phi),
         (R + r * cos(theta)) * cos(phi), -r * sin(theta) * sin(phi),
         0, r * cos(theta);
    return J;
  }
};

int main() {
  Torus torus(1.0, 0.3);
  auto samples = torus.sample(10);
  for (const auto &x : samples)
    cout << x.transpose() << endl;
  return 0;
}
