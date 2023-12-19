import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class VASSO(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho, theta) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert theta <= 1 and theta >= 0, f"theta must lives in [0, 1]."
        self.rho = rho
        self.theta = theta
        super(VASSO, self).__init__(params, dict(rho=rho, theta=theta))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group['theta'] = theta

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho,
            'theta': args.theta,
        }

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            theta = group['theta']
            for p in group["params"]:
                if p.grad is None: continue
                if 'ema' not in self.state[p]:
                    self.state[p]['ema'] = p.grad.clone().detach()
                else:
                    self.state[p]['ema'].mul_(1 - theta)
                    self.state[p]['ema'].add_(p.grad, alpha=theta)


        avg_grad_norm = self._avg_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (avg_grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = self.state[p]['ema'] * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _avg_grad_norm(self):
        norm = torch.norm(
            torch.stack([
                self.state[p]['ema'].norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
