// [[Rcpp::depends(RcppEigen)]]
#include<RcppEigen.h>
#include "manifold.hpp"

class RManifold : public Manifold {
    public:
      Rcpp::Function coord_fun, pushforward_fun;

      RManifold(int n, int k, double vf_bound,
                const Eigen::VectorXd &lower,
                const Eigen::VectorXd &upper,
                Rcpp::Function coord_fun,
                Rcpp::Function pushforward_fun)
        : Manifold(n, k, vf_bound, lower, upper),
          coord_fun(coord_fun),
          pushforward_fun(pushforward_fun) {}

      Eigen::VectorXd coord(const Eigen::VectorXd &u) override {
        Rcpp::NumericVector ru(u.data(), u.data() + u.size());
        Rcpp::NumericVector rx = coord_fun(ru);
        Eigen::VectorXd x(rx.size());
        for (int i = 0; i < rx.size(); i++)
            x(i) = rx[i];
        return x;
      }

      Eigen::MatrixXd pushforward(const Eigen::VectorXd &u) override {
        Rcpp::NumericVector ru(u.data(), u.data() + u.size());
        Rcpp::NumericMatrix rJ = pushforward_fun(ru);
        Eigen::MatrixXd J(n, k);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                J(i,j) = rJ(i, j);
        return J;
      }
};

// [[Rcpp::export]]
SEXP manifold_cpp(
    int n, int k, double vf_bound,
    const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper,
    Rcpp::Function coord_fun,
    Rcpp::Function pushforward_fun
) {
    if (lower.size() != k || upper.size() != k)
        Rcpp::stop("lower and upper bound must have dimension k");

    RManifold* M = new RManifold(n, k, vf_bound, lower, upper, coord_fun, pushforward_fun);
    Rcpp::XPtr<RManifold> ptr(M, true);
    return ptr;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix manifold_sample_cpp(SEXP M_ptr, int n_samples) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    std::vector<Eigen::VectorXd> samples = M->sample(n_samples);
    int n = M->n;

    Rcpp::NumericMatrix out(n_samples, n);
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < n; j++)
            out(i, j) = samples[i](j);

    return out;
}

// [[Rcpp::export]]
Eigen::MatrixXd manifold_metric_cpp(SEXP M_ptr, const Eigen::VectorXd &u) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    if (u.size() != M->k)
        Rcpp::stop("u must have length k");
    return M->metric(u);
}

// [[Rcpp::export]]
double manifold_volume_form_cpp(SEXP M_ptr, const Eigen::VectorXd &u) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    if (u.size() != M->k)
        Rcpp::stop("u must have length k");
    return M->volume_form(u);
}

// [[Rcpp::export]]
Eigen::VectorXd manifold_coord_cpp(SEXP M_ptr, const Eigen::VectorXd &u) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    if (u.size() != M->k)
        Rcpp::stop("u must have length k");
    return M->coord(u);
}

// [[Rcpp::export]]
Eigen::MatrixXd manifold_pushforward_cpp(SEXP M_ptr, const Eigen::VectorXd &u) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    if (u.size() != M->k)
        Rcpp::stop("u must have length k");
    return M->pushforward(u);
}

// [[Rcpp::export]]
int manifold_n_cpp(SEXP M_ptr) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    return M->n;
}

// [[Rcpp::export]]
int manifold_k_cpp(SEXP M_ptr) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    return M->k;
}

// [[Rcpp::export]]
double manifold_vf_bound_cpp(SEXP M_ptr) {
    Rcpp::XPtr<RManifold> M(M_ptr);
    return M->vf_bound;
}