"""
Minimal Path B math test.

Constructs a tiny critic with the same interface as the real one,
runs a synthetic batch through the Path B target computation
(extracted by hand from TD3_BC_ct.py), and verifies:
- shapes are correct
- HVP equals the true delta_s^T H delta_s on a hand-checked example
- backward into the online critic loss works without error

Does NOT touch gym, simglucose, the replay buffer, or training loops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Mini critic with the same Q1/Q2 + forward shape as the real one.
class MiniCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.l2 = nn.Linear(32, 1)
        self.l3 = nn.Linear(state_dim + action_dim, 32)
        self.l4 = nn.Linear(32, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.l2(F.relu(self.l1(sa))), self.l4(F.relu(self.l3(sa)))

    def Q1(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.l2(F.relu(self.l1(sa)))


class MiniActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, action_dim)

    def forward(self, s):
        return torch.tanh(self.l2(F.relu(self.l1(s))))


def run_path_b_step():
    torch.manual_seed(0)
    device = "cpu"
    state_size, action_size = 11, 1
    phys_idx = [8, 9, 10]
    B = 4

    critic = MiniCritic(state_size, action_size).to(device)
    critic_target = MiniCritic(state_size, action_size).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_target = MiniActor(state_size, action_size).to(device)

    # Synthetic batch
    state = torch.randn(B, state_size)
    action = torch.randn(B, action_size).clamp(-1, 1)
    next_state = state + 0.01 * torch.randn(B, state_size)
    reward = torch.randn(B, 1) * 0.1
    done = torch.ones(B, 1)

    dt = 3.0
    beta = -math.log(0.99) / 3.0
    discount = math.exp(-beta * dt)
    policy_noise = 0.2
    noise_clip = 0.5
    max_action = 1.0

    # === The Path B block, extracted from TD3_BC_ct.py ===
    s_g = state.detach().clone().requires_grad_(True)
    a_g = action.detach().clone().requires_grad_(True)

    q_sa = critic_target.Q1(s_g, a_g)
    grad_s = torch.autograd.grad(
        q_sa.sum(), s_g, create_graph=True, retain_graph=True
    )[0]
    grad_a = torch.autograd.grad(
        q_sa.sum(), a_g, create_graph=True, retain_graph=True
    )[0]
    assert grad_s.shape == (B, state_size), f"grad_s shape {grad_s.shape}"
    assert grad_a.shape == (B, action_size), f"grad_a shape {grad_a.shape}"

    phys_mask = torch.zeros(state_size, device=device)
    phys_mask[phys_idx] = 1.0
    delta_s = (next_state - state) * phys_mask
    # Check the mask zeroed out non-physical dims
    for d in range(state_size):
        if d in phys_idx:
            continue
        assert (delta_s[:, d].abs().sum() == 0), f"delta_s[:, {d}] not zero"

    with torch.no_grad():
        noise = (torch.randn_like(action) * policy_noise).clamp(
            -noise_clip, noise_clip
        )
        next_action = (actor_target(next_state) + noise).clamp(
            -max_action, max_action
        )
    delta_a = next_action - action

    first_order_s = (grad_s * delta_s).sum(dim=1, keepdim=True)
    first_order_a = (grad_a * delta_a).sum(dim=1, keepdim=True)
    assert first_order_s.shape == (B, 1)
    assert first_order_a.shape == (B, 1)

    # HVP
    v_scalar = (grad_s * delta_s.detach()).sum()
    H_delta_s = torch.autograd.grad(
        v_scalar, s_g, create_graph=False, retain_graph=True
    )[0]
    assert H_delta_s.shape == (B, state_size)
    second_order = 0.5 * (H_delta_s * delta_s).sum(dim=1, keepdim=True)
    assert second_order.shape == (B, 1)

    # Verify HVP matches true Δs^T H Δs on a 3-d sub-Hessian for one sample.
    # We compute the full 11x11 Hessian via autograd for sample 0 and compare.
    s_one = state[0:1].detach().clone().requires_grad_(True)
    a_one = action[0:1].detach().clone().requires_grad_(True)
    q_one = critic_target.Q1(s_one, a_one)
    grad_one = torch.autograd.grad(q_one.sum(), s_one, create_graph=True)[0]
    # Build full 11x11 Hessian by differentiating each component of grad_one
    H_full = torch.zeros(state_size, state_size)
    for k in range(state_size):
        gk = torch.autograd.grad(grad_one[0, k], s_one, retain_graph=True)[0][0]
        H_full[k] = gk
    delta_s_0 = delta_s[0].detach()
    true_quadratic = 0.5 * delta_s_0 @ H_full @ delta_s_0
    hvp_quadratic = second_order[0, 0]
    err = float(abs(true_quadratic - hvp_quadratic))
    print(f"True 1/2 dS^T H dS  = {true_quadratic.item():.6e}")
    print(f"HVP  1/2 dS^T H dS  = {hvp_quadratic.item():.6e}")
    print(f"abs error           = {err:.2e}")
    assert err < 1e-5, f"HVP does not match true Hessian! err={err}"

    target_inner = (
        q_sa.detach() + first_order_s + first_order_a + second_order
    )
    target_Q = (reward + done * discount * target_inner).detach()
    assert target_Q.shape == (B, 1)

    # Backward through online critic loss
    current_Q1, current_Q2 = critic(state, action)
    loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    loss.backward()

    # Check critic params got grads, critic_target did not.
    online_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in critic.parameters()
    )
    target_has_grad = any(
        p.grad is not None for p in critic_target.parameters()
    )
    assert online_has_grad, "Online critic got no gradient!"
    assert not target_has_grad, "Target critic should not receive gradient!"

    print(f"Loss: {loss.item():.6f}")
    print("All shape, HVP-correctness, and backward checks PASSED.")


if __name__ == "__main__":
    run_path_b_step()
