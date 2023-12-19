import math
import torch
import torch.optim

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class SAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
    
    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
        }
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
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

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

