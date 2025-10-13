import numpy as np
import pymc as pm
from scipy.integrate import quad, solve_ivp


def sample_arbitrary_pdf(log_pdf_func, n_sample=1000):
    with pm.Model():
        x = pm.DensityDist('x', logp=log_pdf_func)
        trace = pm.sample(n_sample, return_inferencedata=False, progressbar=False)
    return trace['x'][:n_sample]


class Curve:
    def __init__(self, x, v, bounds=(0,1)):
        self.x = x
        self.v = v
        self.bounds = bounds
        self.length, _ = quad(lambda t: np.linalg.norm(self.v(t)), bounds[0], bounds[1])

        sol = solve_ivp(
            lambda s, y: 1.0 / np.linalg.norm(self.v(y[0])),
            (0, self.length),
            [self.bounds[0]],
            dense_output=True,
            rtol=1e-10,
            atol=1e-12
        ).sol

        self.reparam = lambda s: sol(s)[0]

    def uniform_sample(self, n_sample=1):
        out = []
        for _ in range(n_sample):
            s = np.random.random() * self.length
            out.append(self.x(self.reparam(s)))
        return out

    def uniform_sample2(self, n_sample=1):
        t_samples = sample_arbitrary_pdf(lambda t: np.linalg.norm(self.v(t)), n_sample=n_sample)
        return [self.x(t) for t in t_samples]

