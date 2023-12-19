import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class INFOSAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho, alpha) -> None:
        # variance inversed - sam
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        assert 0 <= alpha, f"rho should be non-negative:{alpha}"
        self.theta = alpha
        super(INFOSAM, self).__init__(params, dict(rho=rho, theta=alpha))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group['alpha'] = alpha

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho,
            'alpha': args.alpha,
        }

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            alpha = self.alpha
            for p in group["params"]:
                if p.grad is None: continue
                if 'wfw' not in self.state[p]:
                    self.state[p]['wfw'] = p.grad.clone().detach()
                    self.state[p]['belief'] = torch.ones_like(p.grad)
                else:
                    self.state[p]['wfw'].mul_(alpha).add_(p.grad, alpha=1 - alpha)
                    self.state[p]['belief'] = torch.square(self.state[p]['wfw'] - p.grad)


        weighted_grad_norm = self._weighted_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (weighted_grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = torch.div(p.grad, self.state[p]['belief']) * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _weighted_grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                torch.div(p.grad, self.state[p]['belief']).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm