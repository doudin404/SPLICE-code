import torch
import numpy as np


def edm_sampler(
    net,
    latents,
    class_labels=None,
    num_steps=18*5,  # 18
    sigma_min=0.002,
    sigma_max=80,
    rho=15,  # 7, 0.002
    S_churn=40/18*5, S_min=0.05, S_max=50, S_noise=1.003,
    #S_churn=0.1, S_min=0.0, S_max=float('inf'), S_noise=1,#2
    #S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    cond=None,  # 新增cond参数
    mask=None,  # 新增mask参数
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    if cond is not None and mask is not None:
            x_next = x_next + cond * ~mask
        

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        # gamma = (
        #     min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # )
        gamma = (
            min(S_churn, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

        # 将 denoised 中 mask 为 0 的部分替换为 cond 对应位置的值
        denoised=adj_denoised(denoised,t_cur,cond,mask)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_hat, class_labels).to(torch.float64)

            # 再次进行替换
            denoised=adj_denoised(denoised,t_cur,cond,mask)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def adj_denoised(denoised,t,cond= None , mask = None):
    # 将 denoised 中 mask 为 0 的部分替换为 cond 对应位置的值
    if cond is not None and mask is not None:
        denoised = denoised * mask + cond * ~mask
    # if t<2:
    #     print(f"{t:.3f}:{denoised[denoised.norm(2,-1)<5].norm(2,-1).mean():.3f},{denoised[denoised.norm(2,-1)>5].norm(2,-1).mean():.3f}")
    #     denoised[denoised.norm(2,-1)<5]=0
        
    return denoised

def mse_to_closest_binary(value):
    """
    计算每个生成数据项到0或1中较近者的均方误差
    """
    closest_binary = torch.where(value < 0.5, torch.zeros_like(value), torch.ones_like(value))

    return ((value-closest_binary)**2).mean(),closest_binary