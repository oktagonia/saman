# Manifold Sampling

Sampling algorithms for probability distributions on manifolds using Riemannian geometry. Generates samples from arbitrary curves and surfaces by performing pullback on the input density.

$$
 \int_{\phi(U)} f\,dV = \int_U  f(\phi(x)) \det(\phi'(x)^T \phi'(x))\,dx
$$

![Torus Sampling](output.png)

## Usage

```python
from curve import Curve
from surface import Surface

# Sample from ellipse
ellipse = Curve(
    lambda t: np.array([2*np.cos(t), np.sin(t)]), 
    lambda t: np.array([-2*np.sin(t), np.cos(t)]), 
    bounds=(0, 2*np.pi)
)
samples = ellipse.uniform_sample2(100)

# Sample from torus
torus = Surface(torus_chart, pushforward=torus_pushforward, 
                t_bounds=(0, 2*np.pi), s_bounds=(0, 2*np.pi))
samples = torus.uniform_sample(100)
```

See `manifold_sampling.ipynb` for full examples.

