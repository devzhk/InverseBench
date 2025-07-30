import torch
from tqdm import tqdm
from .base import Algo
import numpy as np

import wandb

from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler



def get_cov_diag(z):
    particles = z.reshape(len(z), -1)
    diff = particles - particles.mean(dim=0, keepdim=True)
    diff_sq = diff ** 2
    diag = diff_sq.sum(dim=0) / len(particles)
    return diag

def get_nonzero_eigenvalues(z, threshold=1e-5):
    """
    Compute non-zero eigenvalues of the covariance matrix of ensemble particles.
    
    Args:
        z: ensemble particles with shape (num_particles, ...)
        threshold: eigenvalues below this threshold are considered zero
    
    Returns:
        nonzero_eigenvals: tensor of non-zero eigenvalues in descending order
    """
    particles = z.reshape(z.shape[0], -1)  # (num_particles, flattened_dim)
    diff = particles - particles.mean(dim=0, keepdim=True)
    cov = diff.T @ diff / diff.shape[0]  # (flattened_dim, flattened_dim)
    
    eigenvals = torch.linalg.eigvals(cov).real
    nonzero_eigenvals = eigenvals[eigenvals > threshold]
    return torch.sort(nonzero_eigenvals, descending=True)[0]


def create_step_scheduler(x0, scale, step_size, N):
    scheduler = np.full(N + 1, x0)
    
    for i in range(step_size, N, step_size):
        scheduler[i:] *= scale
        
    scheduler[N] = scheduler[N - 1]
    
    return scheduler


def exp_decay_fn(rho_max, rho_min, N):
    decay = (rho_min / rho_max) ** (1 / N)
    rho_schedule = np.power(decay, np.arange(N)) * rho_max
    return rho_schedule

def linear_decay_fn(rho_max, rho_min, N):
    rho_schedule = np.linspace(rho_max, rho_min, N, endpoint=True)
    return rho_schedule

def cosine_decay_fn(rho_max, rho_min, N):
    rho_schedule = 0.5 * (1 + np.cos(np.pi * np.arange(1, N+1) / N)) * (rho_max - rho_min) + rho_min
    return rho_schedule

def edm_decay_fn(rho_max, rho_min, N, order=4):
    step_indices = np.arange(N)
    rho_schedule = (rho_max ** (1 / order) 
                    + step_indices / (N - 1) * (rho_min ** (1 / order) - rho_max ** (1 / order))
                    ) ** order
    return rho_schedule

def quadratic_concave_fn(rho_max, rho_min, N):
    """
    Generate a concave quadratic decay scheduler that starts at rho_max and ends at rho_min.
    The curve accelerates as it approaches rho_min.
    """
    t = np.linspace(0, 1, N, endpoint=True)
    # Apply quadratic transformation to create concavity
    # y = 1 - x^2 is a concave function when normalized to [0,1] range
    normalized_values = 1 - t**2
    
    # Scale to desired range
    rho_schedule = rho_min + (rho_max - rho_min) * normalized_values
    return rho_schedule


_rho_schedule_fn_dict = {
    'exp': exp_decay_fn,
    'linear': linear_decay_fn,
    'cosine': cosine_decay_fn,
    'edm': edm_decay_fn,
    'concave': quadratic_concave_fn
}


class Blade(Algo):
    def __init__(self, 
             net,
             forward_op,
             guidance_scale,
             num_steps,
             rho_min,
             rho_max,
             likelihood_steps=None,
             total_steps=None,
             rho_schedule='edm',
             tau=None,
             batch_size=64,
             resample=True,
             init_ensemble=1024,
             clean_init=False,
             last_step_prior=True, 
             mode='correction',         # correction, diag, original
             diffusion_scheduler_config={},
             **kwargs):
        super(Blade, self).__init__(net, forward_op, **kwargs)
        self.scale = guidance_scale
        self.N = num_steps
        if likelihood_steps is None:
            self.num_l_steps = total_steps // num_steps
        else:
            self.num_l_steps = likelihood_steps

        if total_steps != likelihood_steps * num_steps:
            print(f"Warning: total_steps should be equal to num_steps * likelihood_steps, \n but got {total_steps} != {num_steps} * {likelihood_steps}"
                  f"Setting total_steps to {num_steps * likelihood_steps}")

        self.rho_min = rho_min
        self.rho_max = rho_max
        self.batch_size = batch_size
        # assert batch_size == init_ensemble, "batch_size must equal init_ensemble"
        self.resample = resample
        self.mode = mode
        self.init_ensemble = init_ensemble
        self.tau = tau if tau is not None else forward_op.sigma_noise
        # self.growth_rate = growth_rate
        # self.ensemble_schedule = create_step_scheduler(init_ensemble, growth_rate, scheduler_steps, num_steps)
        self.rho_schedule = _rho_schedule_fn_dict[rho_schedule](rho_max, rho_min, num_steps)
        self.clean_init = clean_init
        self.last_step_prior = last_step_prior

        self.diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        
    @torch.no_grad()
    def inference(self,  observation, num_samples=1, **kwargs):
        print(observation.dtype)
        observation = torch.view_as_real(observation) if observation.is_complex() else observation
        # observation = observation.to(self.dtype)
        device = self.forward_op.device
        
        if self.clean_init:
            x_initial = torch.randn(self.init_ensemble, *self.net.shape, device=device, dtype=self.dtype) * self.diffusion_scheduler.sigma_max
            sampler = DiffusionSampler(self.diffusion_scheduler)
            num_batches = len(x_initial) // self.batch_size
            for b in range(num_batches):
                start = b * self.batch_size
                end = (b + 1) * self.batch_size
                x_initial[start : end] = sampler.sample(self.net, x_initial[start : end])
        else:
            x_initial = torch.randn(self.init_ensemble, *self.net.shape, device=device, dtype=self.dtype) * self.rho_max
        print('Starting inference...')
        
        x = x_initial
        threshold = self.N if self.last_step_prior else self.N - 1
        for i in range(self.N):
            rho_cur = self.rho_schedule[i]
            
            print(f'Iteration {i}, rho = {rho_cur}, ensemble size = {len(x)}\n')
            # Likelihood Step
            z = self.ll_step(observation, x, rho_cur)

            # Prior Step
            if i < threshold:
                diff_scheduler = Scheduler.get_partial_scheduler(self.diffusion_scheduler, rho_cur)
                sampler = DiffusionSampler(diff_scheduler)

                x = torch.empty_like(z)
                num_batches = len(x) // self.batch_size
                pbar = range(num_batches)
                for b in pbar:
                    start = b * self.batch_size
                    end = (b + 1) * self.batch_size
                    x[start : end] = sampler.sample(self.net, z[start : end])
            else:
                x = z
        return x
    
    @torch.no_grad()
    def ll_step(self, y, particles, rho, resample=True):
        x = particles
        z_next = particles.clone()
        
        J, *spatial = particles.shape
        
        total_time = 0.
                        
        pbar = tqdm(range(self.num_l_steps))
        for _ in pbar:
            
            z_diff = (z_next - z_next.mean(dim=0, keepdim=True)).reshape(J, -1)
            
            if self.mode == 'diag':
                cov_diag = get_cov_diag(z_next)
                dz_reg = ((x - z_next).reshape(J, -1) * cov_diag).reshape(J, *spatial) / (rho ** 2) 
            else:
                cov = z_diff.T @ z_diff / len(z_diff)
                dz_reg = ((x - z_next).reshape(J, -1) @ cov).reshape(J, *spatial) / (rho ** 2)
            
            if self.mode == 'correction':
                dz_reg = dz_reg + z_diff.reshape(J, *spatial) * (z_diff.shape[-1] + 1) / J
            
            std_y = self.tau if self.tau > 0 else self.forward_op.sigma_noise
            std_y = std_y if std_y > 0 else 1.0
            dz_ll, loss = self.ek_update(self.forward_op, y, std_y, 
                                         z_next, z_next, return_loss=True)
            
            lr = self.scale / torch.linalg.matrix_norm((dz_ll + dz_reg).reshape(J, -1))
            total_time += lr
            
            z_next -= dz_ll * lr
            z_next += dz_reg * lr

            if self.mode == 'correction':
                eps = torch.randn(J, J, device=z_next.device, dtype=z_next.dtype)
                noise = eps @ z_diff / np.sqrt(J) * torch.sqrt(2 * lr)
            elif self.mode == 'diag':
                eps = torch.randn_like(z_next).reshape(J, -1)
                cov_sqrt = torch.sqrt(cov_diag)
                noise = (eps * cov_sqrt) * torch.sqrt(2 * lr)
            elif self.mode == 'original':
                eps = torch.randn_like(z_next).reshape(J, -1)
                cov_sqrt = torch.linalg.cholesky(cov + 1e-3 * torch.eye(len(cov), device=z_next.device))
                noise = (eps @ cov_sqrt) * torch.sqrt(2 * lr)
            else:
                raise ValueError(f"only 'correction', 'diag' and 'original' modes are expected, but got {self.mode}")

            z_next += noise.reshape(J, *spatial)
            if wandb.run is not None:
                wandb.log({'data_misfit': loss.item(), 'll_step_lr': lr})
        
        # pred = self.forward_op.forward(z_next)
        # diff = pred - y
        # l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1).mean()

        print(f'time horizon: {total_time}')
        if resample:
            z_cov_diag = get_cov_diag(z_next)
            noise_diff = max(rho - torch.sqrt(z_cov_diag.mean()), 0)
            return z_next + torch.randn_like(z_next) * noise_diff
        else:
            return z_next
            
    @torch.no_grad()
    def ek_update(self, forward_operator, y, std_y, x, x_clean, return_loss=False):
    
        N, *spatial = x.shape
        
        preds = forward_operator.forward(x_clean)        
        
        xs_diff = x - x.mean(dim=0, keepdim=True)
        pred_err = (preds - y)  # (N, *spatial)
        pred_diff = preds - preds.mean(dim=0, keepdim=True) # (N, *spatial)
            
        coef = (
            torch.matmul(
                pred_err.reshape(pred_err.shape[0], -1) / (std_y ** 2),
                pred_diff.reshape(pred_diff.shape[0], -1).T,
            )
            / N
        )   # (N, N)
                
        dx = (coef @ xs_diff.reshape(N, -1)).reshape(N, *spatial)
        if return_loss:
            loss = torch.linalg.norm(pred_err.reshape(pred_err.shape[0], -1), dim=1).mean()
            return dx, loss
        else:
            return dx