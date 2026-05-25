import torch 
import torch.nn as nn
from einops import rearrange, einsum
import math
from collections.abc import Callable, Iterable
from typing import Optional


def cross_entropy(logits: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
    # 先减去最大值防止数值溢出
    max_num =  torch.max(logits,dim=-1, keepdim=True).values
    shifted  = logits - max_num

    logsumexp =  torch.log(torch.sum(torch.exp(shifted),dim=-1))
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = -(target_logits - logsumexp)
    return loss.mean()



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取 learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取和 p 关联的状态.
                t = state.get("t", 0)  # 从状态中获取迭代次数，若不存在则为 0.
                grad = p.grad.data  # 获取 loss 对 p 的梯度.
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重 tensor.
                state["t"] = t + 1  # 迭代次数加 1.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,betas=(0.9, 0.95),eps=1e-8,weight_decay=0.0,):
        if lr < 0:
            raise ValueError(f"Invalid : {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay")
        defaults = {"lr": lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                state["step"] +=1
                t = state["step"]
                step_lr = lr*(math.sqrt(1-beta2**t)/(1-beta1**t))
                with torch.no_grad():
                    p.mul_(1-lr*weight_decay)
                    # m ← β1m + (1 − β1)g
                    # v ← β2v + (1 − β2)g^2
                    m = state["exp_avg"]
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    v = state["exp_avg_sq"]
                    v.mul_(beta2).add_(grad**2, alpha=1 - beta2)
                    p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-step_lr)

        return loss


def cosine_learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    alpha_t = None
    if(t <T_w):
        alpha_t = t/T_w*alpha_max
    if(t >= T_w and t <= T_c):
        alpha_t = alpha_min + 0.5*(1+math.cos(math.pi*(t-T_w)/(T_c - T_w)))*(alpha_max - alpha_min)
    if(t > T_c):
        alpha_t = alpha_min
    return alpha_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients in-place so that their global L2 norm is at most max_l2_norm.

    Args:
        parameters: Iterable of torch.nn.Parameter.
        max_l2_norm: Maximum allowed L2 norm.
    """
    eps = 1e-6

    # 过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # 计算 global L2 norm: sqrt(sum_i ||grad_i||_2^2)
    total_norm_sq = torch.zeros((), device=params_with_grad[0].grad.device)

    for p in params_with_grad:
        total_norm_sq += torch.sum(p.grad.detach() ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    # 如果超过 max_l2_norm，就原地缩放所有梯度
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)

        for p in params_with_grad:
            p.grad.mul_(clip_coef)

if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e-3)

    for t in range(10):
        opt.zero_grad()  # 重置所有可学习参数的梯度.
        loss = (weights**2).mean()  # 计算一个标量 loss.
        print(f"{t}",loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients. 
        opt.step() # Run optimizer step.