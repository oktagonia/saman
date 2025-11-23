#include<iostream>
#include <numbers>

#include "manifold.hpp"

using namespace Eigen;
using namespace std;

class Ellipse : public Manifold {
  public:
    double a, b;

    Ellipse(double a, double b)
      : Manifold(2, 1, VectorXd::Constant(1, 0.0), VectorXd::Constant(1, 2.0 * numbers::pi)),
        a(a), b(b) {}

    VectorXd coord(const VectorXd& u) override {
      double t = u[0];
      VectorXd x(2);
      x << a * cos(t), b * sin(t);
      return x;
    }

    MatrixXd pushforward(const VectorXd& u) override {
      double t = u[0];
      MatrixXd J(2, 1);
      J << -a * sin(t), b * cos(t);
      return J;
    }
};

int main() {
  Ellipse e(2.0, 1.0);
  VectorXd u(1);
  u << 1.0;

  for (int i = 0; i < 10; i++)
    cout << "random coord:\n" << e.sample(2.5).transpose() << endl;
}
