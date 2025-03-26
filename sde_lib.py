
import torch
from models import utils as mutils
from scipy import integrate
        


class RectifiedFlow():
    def __init__(self, init_type='gaussian', noise_scale=1.0, reflow_flag=False, reflow_t_schedule='u_shape', reflow_loss='l2', use_ode_sampler='rk45', sigma_var=0.0, ode_tol=1e-5, sample_N=None, u_shape_td_a=2, num_numericals=0, scale_ohe=False):
        if sample_N is not None:
            self.sample_N = sample_N
            print('Number of sampling steps:', self.sample_N)
        self.init_type = init_type
        
        self.noise_scale = noise_scale
        self.use_ode_sampler = use_ode_sampler
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1. - t) * sigma_var
        print('Init. Distribution Variance:', self.noise_scale)
        print('SDE Sampler Variance:', sigma_var)
        print('ODE Tolerence:', self.ode_tol)
                
        self.reflow_flag = reflow_flag
        self.u_shape_td_a = u_shape_td_a # a: exp(au) + exp(-au) in RFPP
        self.num_numericals = num_numericals
        self.switch_loss_to_hurber = False
        self.scale_ohe = scale_ohe
        if self.reflow_flag:
            self.reflow_t_schedule = reflow_t_schedule
            self.reflow_loss = reflow_loss

    @property
    def T(self):
      return 1.

    @torch.no_grad()
    def ode(self, init_input: torch.Tensor, model, reverse=False):
        ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
        rtol = 1e-5
        atol = 1e-5
        method = 'RK45'
        eps = 1e-3

        # Initial sample
        x = init_input.detach().clone()

        model_fn = mutils.get_model_fn(model, train=False)
        shape = init_input.shape
        device = init_input.device

        def ode_func(t, x: torch.Tensor):
            x = mutils.from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model_fn(x, vec_t*999)
            return mutils.to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse:
            solution = integrate.solve_ivp(ode_func, (self.T, eps), mutils.to_flattened_numpy(x), rtol=rtol, atol=atol, method=method)
        else:
            solution = integrate.solve_ivp(ode_func, (eps, self.T), mutils.to_flattened_numpy(x), rtol=rtol, atol=atol, method=method)
        
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        # nfe = solution.nfev

        return x

    @torch.no_grad()
    def euler_ode(self, init_input: torch.Tensor, model, reverse=False, N=100):
        ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
        eps = 1e-3
        dt = 1./N

        # Initial sample
        x = init_input.detach().clone()

        model_fn = mutils.get_model_fn(model, train=False)
        shape = init_input.shape
        device = init_input.device
        
        for i in range(N):  
            num_t = i / N * (self.T - eps) + eps      
            t = torch.ones(shape[0], device=device) * num_t
            pred = model_fn(x, t*999)
            
            x = x.detach().clone() + pred * dt         

        return x

    def get_z0(self, batch: torch.Tensor, train=True):

        if self.init_type == 'gaussian':
            ### standard gaussian #+ 0.5
            return torch.randn_like(batch) * self.noise_scale
        elif self.init_type == "concat":
            num_initial_dist = torch.randn((batch.shape[0], self.num_numericals))
            if self.scale_ohe:
                # {-1, 1}
                cat_initial_dist = torch.empty(batch.shape[0], batch.shape[1] - self.num_numericals).uniform_(-1, 1)
            else:
                # {0, 1}
                cat_initial_dist = torch.empty(batch.shape[0], batch.shape[1] - self.num_numericals).uniform_()
            return torch.cat([num_initial_dist, cat_initial_dist], dim=-1) * self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
      