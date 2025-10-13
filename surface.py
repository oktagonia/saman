import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Op, Apply
from pytensor.tensor import TensorType

class PDFLogp(Op):
    def __init__(self, pdf_func, bounds):
        self.pdf_func = pdf_func
        self.x_min, self.x_max = bounds[0]
        self.y_min, self.y_max = bounds[1]
    
    def make_node(self, x, y):
        x = pt.as_tensor_variable(x)
        y = pt.as_tensor_variable(y)
        out_type = TensorType(dtype='float64', shape=x.type.shape)
        return Apply(self, [x, y], [out_type()])
    
    def perform(self, node, inputs, outputs):
        x, y = inputs
        x_np = np.asarray(x, dtype='float64')
        y_np = np.asarray(y, dtype='float64')
        
        prob = self.pdf_func(x_np, y_np)
        log_prob = np.log(np.asarray(prob, dtype='float64') + 1e-10)
        
        in_bounds = ((x_np >= self.x_min) & (x_np <= self.x_max) & 
                     (y_np >= self.y_min) & (y_np <= self.y_max))
                     
        result = np.where(in_bounds, log_prob, -np.inf)
        
        outputs[0][0] = np.asarray(result, dtype='float64')


def sample_2d_pdf(pdf_func, bounds, n_samples=1000, tune=1000, random_seed=None):
    (x_min, x_max), (y_min, y_max) = bounds
    
    pdf_op = PDFLogp(pdf_func, bounds)
    
    def logp(value):
        x = value[..., 0]
        y = value[..., 1]
        return pdf_op(x, y)
    
    def random_fn(rng, size):
        if size is None:
            size = (2,)
        else:
            size = (*size, 2)
        samples = np.zeros(size)
        samples[..., 0] = rng.uniform(x_min, x_max, size=size[:-1])
        samples[..., 1] = rng.uniform(y_min, y_max, size=size[:-1])
        return samples
    
    with pm.Model() as model:
        xy = pm.CustomDist('xy', logp=logp, random=random_fn, shape=2)
        trace = pm.sample(draws=n_samples, tune=tune, random_seed=random_seed,
                         return_inferencedata=True, progressbar=True)
    
    samples = trace.posterior['xy'].values.reshape(-1, 2)
    return samples[:n_samples]


class Surface:
    def __init__(self, chart, pushforward=None, t_bounds=(0,1), s_bounds=(0,1)):
        self.chart = chart
        self.pushforward = pushforward
        self.metric = lambda x: pushforward(x).T @ pushforward(x)
        self.area_element = lambda x: np.sqrt(np.linalg.det(self.metric(x)))
        self.t_bounds = t_bounds
        self.s_bounds = s_bounds

    def uniform_sample(self, n_sample=1):
        parameter_samples = sample_2d_pdf(
            pdf_func=lambda x, y: self.area_element(np.array([x,y])),
            bounds=[self.t_bounds, self.s_bounds],
            n_samples = n_sample
        )

        return [self.chart(p) for p in parameter_samples]
