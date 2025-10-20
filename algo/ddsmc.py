# -------------------------------------------------------------------------------------------
# Paper: Solving Linear-Gaussian Bayesian Inverse Problems with Decoupled Diffusion Sequential Monte Carlo
# Implementation adapted for the InverseBench repository
# Official implementation: https://github.com/filipekstrm/ddsmc
# -------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
from utils.helper import has_svd

import tqdm

# SMC helper class from https://github.com/filipekstrm/ddsmc/blob/main/ddsmc/utils/smc_utils.py
class BatchedSMCHelper:
    def __init__(self, resampling_method, num_particles, device):
        self.device = device
        self.num_particles = num_particles
        resampling_fns = {"multinomial": self.multinomial_resampling, 
                          "systematic": self.systematic_resampling,
                          "stratified": self.stratified_resampling}

        self._resample = resampling_fns[resampling_method]
        self.device = device

    def resample(self, nu, log=False):
        assert len(nu.shape) == 1
        if log:
            nu = self.normalize(nu, log=True)
        nu = nu.reshape(-1, self.num_particles)
        assert not torch.any(torch.abs(torch.sum(nu, dim=-1) - 1.0) > 1e-5)
        return self._resample(nu)

    def multinomial_resampling(self, nu):
        a = nu.multinomial(self.num_particles, replacement=True).flatten()
        particle_offset = (self.num_particles * torch.arange(nu.shape[0],
                                                             device=nu.device)).repeat_interleave(self.num_particles)
        a = a + particle_offset
        return a.flatten()

    def _sampling_help(self, p, nu):
        nu_cumsum = nu.cumsum(dim=-1)
        p = p.flatten().unsqueeze(-1)
        nu_cumsum = nu_cumsum.repeat_interleave(self.num_particles, dim=0)
        indices = self.num_particles - torch.sum(p < nu_cumsum, dim=-1)
        indices = indices + (self.num_particles * torch.arange(nu.shape[0],
                                                               device=self.device)).repeat_interleave(self.num_particles)
        return indices.flatten()

    def systematic_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], 1), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    def stratified_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], self.num_particles), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    def compute_ess(self, w, log=False):
        if log:
            w = self.normalize(w, log=True)
        w = w.reshape(-1, self.num_particles)
        assert not torch.any(torch.abs(torch.sum(w, dim=-1) - 1.0) > 1e-5)
        n_eff = 1 / torch.sum(w ** 2, dim=-1)
        return n_eff.flatten()

    def normalize(self, vec, log=False):
        assert len(vec.shape) == 1
        vec = vec.reshape((-1, self.num_particles))
        if log:
            vec = torch.exp(vec - torch.max(vec, dim=-1, keepdim=True)[0])
        return F.normalize(vec, p=1, dim=-1).flatten()

    def normalize_log(self, vec):
        assert len(vec.shape) == 1
        vec = vec.reshape((-1, self.num_particles))
        return (vec - vec.logsumexp(-1, keepdim=True)).flatten()
    
    def importance_sampling(self, x, w, log=False):
        assert len(w.shape) == 1
        w = w.reshape(-1, self.num_particles)
        if log:
            distr = torch.distributions.Categorical(logits=w)
        else:
            distr = torch.distributions.Categorical(probs=w)
        s = distr.sample((1,)).flatten()
        s = s + self.num_particles * torch.arange(s.shape[0], device=x.device)
        return x[s]

def diag_gauss_logpdf(x, mean, diag):
    """
    x: torch.Tensor (batch, D)
    mean: mean in MV Gaussian, torch.Tensor (batch, D)
    diag: diagonal in the MV Gaussian (batch, D)
    """
    return torch.sum(-1/2 * (x-mean)**2/diag, dim=-1)


class OperatorWrapper:
    """
    Wrapper of forward operator to enable defining log_likelihood
    """
    def __init__(self, forward_op):
        self.forward_op = forward_op
    
    def log_likelihood(self, x0hat_prime, y_prime, rho_t=0.):
        """
        Compute log_likelihood in the prime basis, i.e., when likelihood is diagonal
        x0hat_prime: [batch, Dx]
        y_prime: [batch, Dy]
        rho_t: float
        """
        mean = self.non_zero_Sigma * x0hat_prime[:, self.M.flatten().bool()]
        sigma2 = self.sigma_noise**2 + rho_t**2 * (self.non_zero_Sigma * self.non_zero_Sigma.conj())
        return torch.sum(-1/2 * (mean - y_prime[:, self.M.flatten().bool()])**2/sigma2, dim=-1)
    
    @property
    def non_zero_Sigma(self):
        return self.S.flatten()[self.M.flatten().bool()]
    
    def __getattr__(self, name):
        return getattr(self.forward_op, name)

class DDSMC(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 smc_config,
                 eta,
                 annealing_scheduler_config, 
                 rho_scheduler_config={},
                 diffusion_scheduler_config={},
                 recon_fn="tweedie",
                 verbose=False):
        super(DDSMC, self).__init__(net, OperatorWrapper(forward_op))
        assert has_svd(forward_op), "DDSMC requires access to SVD of observation model"
        self.eta = eta
        self.num_particles = smc_config.num_particles
        self.device = forward_op.device
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.rho_t = Scheduler(**rho_scheduler_config).sigma_steps[::-1]
        self.verbose = verbose

        self.smc_helper = BatchedSMCHelper(**smc_config, device=self.device)

        recon_fns = {"ode": self.ode_reconstruct, "tweedie": self.tweedie_reconstruct}
        assert recon_fn.lower() in recon_fns, f"Reconstruction function {recon_fn} is not available"
        self._reconstruct = recon_fns[recon_fn.lower()]
    
    @property
    def num_steps(self):
        return self.annealing_scheduler.num_steps

    def t_from_step(self, step):
        return self.num_steps - step
    
    def step_from_t(self, t):
        return self.num_steps - t

    def proposal(self, x0hat_prev, xtprev, y, tprev):
        step_prev = self.step_from_t(tprev)
        sigma2_t = self.annealing_scheduler.sigma_steps[step_prev + 1]**2
        delta_sigma2 = self.annealing_scheduler.sigma_steps[step_prev]**2 - sigma2_t
        assert delta_sigma2 > 0

        mean_factor = delta_sigma2 / (self.eta * sigma2_t + delta_sigma2)
        prev_sample_factor = self.eta * sigma2_t / (self.eta * sigma2_t + delta_sigma2)
        rho_t = self.rho_t[tprev]
        M_inv, mean = self.get_Minv_Minvb(rho_t**2, x0hat_prev, y)
        if tprev > 1:
            diag = (sigma2_t * mean_factor - (mean_factor * rho_t)**2) + mean_factor**2*M_inv
        else:
            diag = torch.zeros_like(M_inv)

        mean = mean_factor * mean + prev_sample_factor * xtprev.flatten(1, -1)
        z = torch.randn_like(mean)
        x_new = mean + torch.sqrt(diag) * z
        if tprev <= 1:
            log_r = torch.zeros(xtprev.shape[0], device=xtprev.device)
        else:
            log_r = diag_gauss_logpdf(x_new, mean, diag)

        return x_new, log_r

    def reconstruct(self, x_t, t):
        step = self.step_from_t(t)
        return self._reconstruct(x_t, step)
        
    def ode_reconstruct(self, x_t, start_step):
        sigma = self.annealing_scheduler.sigma_steps[start_step]
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
        sampler = DiffusionSampler(diffusion_scheduler)
        x0hat = self.forward_op.Vt(sampler.sample(self.net, self.forward_op.V(x_t), SDE=False, verbose=False))
        return x0hat
    
    def tweedie_reconstruct(self, x_t, current_step):
        sigma = self.annealing_scheduler.sigma_steps[current_step]
        return self.forward_op.Vt(self.net(self.forward_op.V(x_t), torch.as_tensor(sigma).to(x_t.device)))
    
    def evaluate_target_transition(self, xt, xtprev, x0hat_prev, t):
        if t > 0:
            sigma2_t = self.annealing_scheduler.sigma_steps[self.step_from_t(t)]**2
            delta_sigma2 = self.annealing_scheduler.sigma_steps[self.step_from_t(t) - 1]**2 - sigma2_t
            assert delta_sigma2 > 0.
            diag = delta_sigma2 * sigma2_t/(self.eta * sigma2_t + delta_sigma2) * torch.ones(xt.shape[-1], device=xt.device)
            mean = self.eta / (self.eta + delta_sigma2 / sigma2_t) * xtprev +  1 /(self.eta * sigma2_t / delta_sigma2 + 1) * x0hat_prev
            return diag_gauss_logpdf(xt, mean, diag)
        else:
            return torch.zeros(xt.shape[0], device=xt.device)
        
    def get_Minv_Minvb(self, rho2_t, x0hat, y):
        '''
        Helper function to obtain M^{-1} and \tilde \mu (i.e., M^{-1}b) from equation 20 and 21 in paper.
        Here, though, it is not assumed that the observed variables are the "first" coordinates in x_prime, but these are found using the mask matrix M in the operator
        '''
        num_unobserved = x0hat.shape[-1] - self.forward_op.non_zero_Sigma.shape[-1]
        num_observed = self.forward_op.non_zero_Sigma.shape[-1]
        denominator = self.forward_op.non_zero_Sigma * self.forward_op.non_zero_Sigma.conj() * rho2_t + self.forward_op.sigma_noise**2
        Minv_obs = rho2_t * self.forward_op.sigma_noise**2 / denominator
        Minv_unobs = rho2_t * torch.ones(num_unobserved, device=Minv_obs.device, dtype=Minv_obs.dtype)
        Minv = torch.empty(num_unobserved + num_observed, device=Minv_obs.device, dtype=Minv_obs.dtype)
        Minv[self.forward_op.M.flatten().bool()] = Minv_obs
        Minv[~(self.forward_op.M.flatten().bool())] = Minv_unobs     

        Minvb_obs = rho2_t * self.forward_op.non_zero_Sigma / denominator * y[:, self.forward_op.M.flatten().bool()] + self.forward_op.sigma_noise**2 / denominator * x0hat[:, self.forward_op.M.flatten().bool()]
        Minvb = torch.empty(x0hat.shape, device=x0hat.device, dtype=Minvb_obs.dtype)
        Minvb[:, self.forward_op.M.flatten().bool()] = Minvb_obs
        Minvb[:, ~(self.forward_op.M.flatten().bool())] = x0hat[:, ~(self.forward_op.M.flatten().bool())]
        assert Minv.shape[-1] == Minvb.shape[-1] == x0hat.shape[-1]
        return Minv, Minvb

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):

        # initialize and obtain x'
        xt = torch.randn(num_samples*self.num_particles, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=self.device) * self.annealing_scheduler.sigma_max
        xt = self.forward_op.Vt(xt)

        # Following MCGDiff, perform normalization of observation
        observation = observation / self.forward_op.unnorm_scale - self.forward_op.forward(self.forward_op.unnorm_shift * torch.ones(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=self.device), unnormalize=False)
        observation = observation.repeat_interleave(num_samples * self.num_particles, dim=0)
        # obtain y'
        observation = self.forward_op.Ut(observation)
        assert observation.shape[0] == xt.shape[0]

        if self.verbose:
            pbar = tqdm.trange(self.num_steps)
        else:
            pbar = range(self.num_steps)

        ancestors = torch.arange(self.num_particles, device=self.device).unsqueeze(0).repeat(num_samples, 1)
        assert ancestors.shape[0] == num_samples

        for step in pbar:
            t = self.t_from_step(step)
            # reconstruct
            x0hat = self.reconstruct(xt, t)

            # weight
            logp_y_x0_t = self.forward_op.log_likelihood(x0hat, observation, self.rho_t[t])

            if step == 0:
                log_weights = self.smc_helper.normalize_log(logp_y_x0_t)
            else:
                log_q0 = self.evaluate_target_transition(xt, xtprev, x0hat_prev, t)
                log_weights = self.smc_helper.normalize_log(logp_y_x0_t + log_q0 - logp_y_x0_tplusone - log_qt_xt)

            # Resample
            ancestors = self.smc_helper.resample(log_weights, log=True)
            x0hat_prev = x0hat[ancestors]
            logp_y_x0_tplusone = logp_y_x0_t[ancestors]
            xtprev = xt[ancestors]

            # proposal
            xt, log_qt_xt = self.proposal(x0hat_prev, xtprev, observation, t)

        logp_y_x0_t = self.forward_op.log_likelihood(xt, observation)
        log_q0 = self.evaluate_target_transition(xt, xtprev, x0hat_prev, 0)
        log_weights = self.smc_helper.normalize_log(logp_y_x0_t + log_q0 - logp_y_x0_tplusone - log_qt_xt)
        x0 = self.smc_helper.importance_sampling(xt, log_weights, log=True)
        return self.forward_op.V(x0)