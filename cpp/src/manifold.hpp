#ifndef MANIFOLD
#define MANIFOLD

#include<iostream>
#include<utility>

#include<Eigen/Dense>

class Manifold {
  public:
    virtual ~Manifold() = default;

    const int n, k;
    const Eigen::VectorXd upper, lower;

    virtual Eigen::VectorXd coord(const Eigen::VectorXd& u) = 0;
    virtual Eigen::MatrixXd pushforward(const Eigen::VectorXd& u) = 0;

    Eigen::MatrixXd metric(const Eigen::VectorXd u) {
      Eigen::MatrixXd J = this->pushforward(u);
      return J.transpose() * J;
    }

    double volume_form(const Eigen::VectorXd u) {
      return std::sqrt(this->metric(u).determinant());
    }

    Eigen::VectorXd sample(double M) {
      Eigen::VectorXd x;
      double u;
      do {
        x = sample_param();
        u = 0.5 * (Eigen::VectorXd::Random(1)(0) + 1.0);
      } while (u * M > this->volume_form(x));
      return this->coord(x);
    }

  private:
    Eigen::VectorXd sample_param() const {
      Eigen::VectorXd u = Eigen::VectorXd::Random(k);
      u = 0.5 * (u.array() + 1.0);
      return lower.array() + u.array() * (upper - lower).array();
    }

  protected:
    Manifold(int n, int k, Eigen::VectorXd upper, Eigen::VectorXd lower)
      : n(n), k(k), upper(std::move(upper)), lower(std::move(lower)) {}
};

#endif
