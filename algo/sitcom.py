import torch
import tqdm
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


# -----------------------------------------------------------------------------------------------
# Paper: SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems
# Official implementation: https://github.com/sjames40/SITCOM
# Adapted for InverseBench EDM-style network (direct x0 prediction)
# -----------------------------------------------------------------------------------------------


class SITCOM(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 annealing_scheduler_config,
                 diffusion_scheduler_config,
                 num_inner_steps=20,
                 learning_rate=0.01,
                 noise_level=0.05,
                 loss_weight=1.0,
                 measurement_start_sigma=float('inf')):

        super(SITCOM, self).__init__(net, forward_op)
        self.net.eval()
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.num_inner_steps = num_inner_steps
        self.lr = learning_rate
        self.noise_level = noise_level
        self.loss_weight = loss_weight
        self.measurement_start_sigma = measurement_start_sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        xt = torch.randn(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution,
            self.net.img_resolution,
            device=device
        ) * self.annealing_scheduler.sigma_max

        pbar = tqdm.trange(self.annealing_scheduler.num_steps)

        for step in pbar:
            sigma      = self.annealing_scheduler.sigma_steps[step]
            sigma_next = self.annealing_scheduler.sigma_steps[step + 1]

            dsc = self.diffusion_scheduler_config
            diffusion_scheduler = Scheduler(
                num_steps=int(dsc.num_steps),
                sigma_max=float(sigma),
                sigma_min=float(dsc.sigma_min),
                sigma_final=float(dsc.sigma_final),
                schedule=str(dsc.schedule),
                timestep=str(dsc.timestep),
            )
            sampler = DiffusionSampler(diffusion_scheduler)
            with torch.no_grad():
                x0hat = sampler.sample(self.net, xt, SDE=False, verbose=False)

            if sigma < self.measurement_start_sigma:
                xt_opt = xt.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([xt_opt], lr=self.lr)

                for _ in range(self.num_inner_steps):
                    optimizer.zero_grad()
                    with torch.enable_grad():
                        pred_x0 = self.net(
                            xt_opt,
                            torch.as_tensor(sigma).to(device)
                        ).clamp(-1, 1)
                        meas = self.forward_op.forward(pred_x0)
                        loss = self.loss_weight * (meas - observation).square().flatten(start_dim=1).sum()

                    loss.backward(retain_graph=True)
                    optimizer.step()

                with torch.no_grad():
                    pred_x0 = self.net(
                        xt_opt.detach(),
                        torch.as_tensor(sigma).to(device)
                    ).clamp(-1, 1)
            else:

                pred_x0 = x0hat

            xt = pred_x0 + torch.randn_like(pred_x0) * (sigma_next + self.noise_level)

            with torch.no_grad():
                display_loss = self.forward_op.loss(pred_x0, observation).sum().sqrt()
            pbar.set_description(
                f'Iter {step + 1}/{self.annealing_scheduler.num_steps} '
                f'loss={display_loss.item():.4f}'
            )

        return pred_x0