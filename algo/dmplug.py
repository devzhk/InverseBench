import copy
import json
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class DMPlug(Algo):
    
    '''
    DMPlug algorithm implemented in EDM framework.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale=1.0,
                 sde=False,
                 iteration=5000,
                 lr=0.1,
                 weight_decay=0.0,
                 loss_scaling='residual',
                 solver='euler',
                 ):
        super(DMPlug, self).__init__(net, forward_op)
        self.net.eval().requires_grad_(False)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.guidance_scale = guidance_scale
        self.sde = sde
        self.iteration = iteration
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_scaling = loss_scaling
        self.solver = solver
        if self.loss_scaling not in {'residual', 'mse', 'none'}:
            raise ValueError("loss_scaling must be one of {'residual', 'mse', 'none'}")


    def _measurement_numel(self, observation):
        def numel(data):
            if torch.is_tensor(data):
                return data.numel()
            size = getattr(data, "size", None)
            return size() if callable(size) else size

        if torch.is_tensor(observation):
            return observation.numel()
        if isinstance(observation, (list, tuple)):
            total = 0
            for item in observation:
                data = getattr(item, "data", item)
                total += numel(data)
            return total
        data = getattr(observation, "data", None)
        if data is not None:
            return numel(data)
        raise TypeError("Cannot infer measurement size for loss_scaling='mse'.")
        
    def inference(self, observation, num_samples=1, **kwargs):
        target = kwargs.get('target')
        evaluator = kwargs.get('evaluator')
        trace_dir = kwargs.get('trace_dir')
        device = self.forward_op.device
        if num_samples > 1:
            if not torch.is_tensor(observation):
                raise ValueError("DMPlug num_samples > 1 requires tensor observations.")
            observation = observation.repeat(num_samples, *([1] * (observation.ndim - 1)))
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_initial.requires_grad = True
        
        sampler = DiffusionSampler(self.scheduler, solver=self.solver)
        pbar = tqdm(range(self.iteration))
        
        optimizer = torch.optim.AdamW([x_initial], lr=self.lr, weight_decay=self.weight_decay)
        
        for iteration in pbar:
            optimizer.zero_grad()
            denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)

            gradient, loss_scale = self.forward_op.gradient(denoised, observation, return_loss=True)

            x_initial_grad = torch.autograd.grad(
                outputs=denoised,
                inputs=x_initial,
                grad_outputs=gradient,
            )[0]
            if self.loss_scaling == 'residual':
                x_initial_grad = x_initial_grad * 0.5 / torch.sqrt(loss_scale).clamp_min(1e-8)
            elif self.loss_scaling == 'mse':
                x_initial_grad = x_initial_grad / self._measurement_numel(observation)
            x_initial.grad = x_initial_grad * self.guidance_scale
            desc = f'Iteration {iteration + 1}/{self.iteration}. Data fitting loss: {display_loss}, x_initial.grad norm: {grad_norm}'
            
            optimizer.step()
            pbar.set_description(desc)

        with torch.no_grad():
            denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)
        return denoised
