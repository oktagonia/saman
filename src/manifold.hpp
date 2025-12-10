#ifndef MANIFOLD
#define MANIFOLD

#include<iostream>
#include<utility>
#include<vector>

#include<Eigen/Dense>

class Manifold {
  public:
    virtual ~Manifold() = default;

    const int n, k;
    const double vf_bound;
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

    std::vector<Eigen::VectorXd> sample(int n_sample) {
      std::vector<Eigen::VectorXd> samples;

      for (int i = 0; i < n_sample; i++) {
        double u;
        Eigen::VectorXd x;
        do {
          x = sample_param();
          u = 0.5 * (Eigen::VectorXd::Random(1)(0) + 1.0);
        } while (u * vf_bound > this->volume_form(x));
        samples.push_back(this->coord(x));
      }

      return samples;
    }

  private:
    Eigen::VectorXd sample_param() const {
      Eigen::VectorXd u = Eigen::VectorXd::Random(k);
      u = 0.5 * (u.array() + 1.0);
      return lower.array() + u.array() * (upper - lower).array();
    }

  protected:
    Manifold(int n, int k, double vf_bound, Eigen::VectorXd lower, Eigen::VectorXd upper)
      : n(n), k(k), vf_bound(vf_bound), upper(std::move(upper)), lower(std::move(lower)) {}
};

#endif
