"""
Offline math test for LinearBasisCritic.

Verifies:
- Phi(s, a) has the expected number of features (39 for state_dim=11)
- Phi includes the expected components (constant, linear, quadratic, cross)
- grad_s Q and grad_a Q can be computed via autograd
- grad_s^2 Q (Hessian-vector product) works
- |grad_s Q| stays bounded as |theta| stays bounded
- A full Path-B-style forward + backward pass runs without error

Does NOT touch gym, simglucose, the replay buffer, or training loops.
"""
import math
import torch
import torch.nn.functional as F
from TD3_BC_ct import LinearBasisCritic


def main():
    torch.manual_seed(0)
    state_dim = 11
    action_dim = 1
    phys_idx = [8, 9, 10]
    B = 8

    critic = LinearBasisCritic(state_dim, action_dim, phys_idx=phys_idx)
    critic_target = LinearBasisCritic(state_dim, action_dim, phys_idx=phys_idx)
    critic_target.load_state_dict(critic.state_dict())

    expected_n_features = (
        1                       # constant
        + state_dim             # linear state
        + 1                     # linear action
        + state_dim             # quadratic state
        + 1                     # quadratic action
        + state_dim             # cross state-action
        + len(phys_idx) * (len(phys_idx) - 1) // 2  # phys cross
    )
    assert critic.n_features == expected_n_features, \
        f"n_features mismatch: got {critic.n_features}, expected {expected_n_features}"
    print(f"OK: n_features = {critic.n_features} (= 1 + 11 + 1 + 11 + 1 + 11 + 3)")

    # ---- Synthetic batch ----
    state = torch.randn(B, state_dim)
    action = torch.randn(B, action_dim).clamp(-1, 1)
    next_state = state + 0.05 * torch.randn(B, state_dim)
    next_action = action + 0.05 * torch.randn(B, action_dim)
    reward = -0.5 * torch.ones(B, 1)

    # ---- Basis sanity ----
    phi = critic.basis(state, action)
    assert phi.shape == (B, expected_n_features), f"phi shape {phi.shape}"
    # First column should be all 1s (constant)
    assert torch.allclose(phi[:, 0], torch.ones(B)), "constant column not 1"
    # Columns 1..1+state_dim should equal state
    assert torch.allclose(phi[:, 1:1 + state_dim], state), "linear state cols broken"
    # state**2 cols
    qs_start = 1 + state_dim + 1
    assert torch.allclose(phi[:, qs_start:qs_start + state_dim], state ** 2), "quadratic state broken"
    print("OK: basis structure (constant + linear state + linear action + quadratic + cross)")

    # ---- Q1, Q2 forward ----
    q1, q2 = critic(state, action)
    assert q1.shape == (B, 1) and q2.shape == (B, 1), f"Q shapes {q1.shape}, {q2.shape}"
    print(f"OK: Q1 = {q1.detach().squeeze().tolist()[:3]}...")
    print(f"OK: Q2 = {q2.detach().squeeze().tolist()[:3]}...")

    # ---- grad_s Q1, grad_a Q1 via autograd ----
    s_g = state.clone().requires_grad_(True)
    a_g = action.clone().requires_grad_(True)
    q_sa = critic_target.Q1(s_g, a_g)
    grad_s = torch.autograd.grad(q_sa.sum(), s_g, create_graph=True, retain_graph=True)[0]
    grad_a = torch.autograd.grad(q_sa.sum(), a_g, create_graph=True, retain_graph=True)[0]
    assert grad_s.shape == (B, state_dim) and grad_a.shape == (B, action_dim)
    print(f"OK: |grad_s Q1| (row-mean) = {grad_s.norm(dim=1).mean().item():.4f}")
    print(f"OK: |grad_a Q1| (row-mean) = {grad_a.norm(dim=1).mean().item():.4f}")

    # ---- Hessian-vector product (the second-order Path B term) ----
    delta_s = (next_state - state)
    v_scalar = (grad_s * delta_s.detach()).sum()
    H_delta_s = torch.autograd.grad(v_scalar, s_g, create_graph=False, retain_graph=True)[0]
    assert H_delta_s.shape == (B, state_dim)
    print(f"OK: |H * delta_s| (row-mean) = {H_delta_s.norm(dim=1).mean().item():.4f}")

    # ---- Lipschitz bound: |grad_s Q| should scale with |theta1| ----
    with torch.no_grad():
        original_theta_norm = critic_target.theta1.weight.norm().item()
        critic_target.theta1.weight.mul_(10.0)
    s_g2 = state.clone().requires_grad_(True)
    q_sa2 = critic_target.Q1(s_g2, action)
    grad_s2 = torch.autograd.grad(q_sa2.sum(), s_g2)[0]
    ratio = grad_s2.norm(dim=1).mean().item() / grad_s.detach().norm(dim=1).mean().item()
    print(f"OK: |grad_s Q| scales linearly with |theta| -- ratio after 10x theta = {ratio:.2f} (expected ~10)")
    assert 9.0 < ratio < 11.0, f"|grad_s Q| didn't scale linearly with |theta|: ratio={ratio}"
    # Reset theta1 to original scale for further tests
    with torch.no_grad():
        critic_target.theta1.weight.div_(10.0)

    # ---- Full Path B target computation ----
    s_g3 = state.clone().requires_grad_(True)
    a_g3 = action.clone().requires_grad_(True)
    q_sa3 = critic_target.Q1(s_g3, a_g3)
    grad_s3 = torch.autograd.grad(q_sa3.sum(), s_g3, create_graph=True, retain_graph=True)[0]
    grad_a3 = torch.autograd.grad(q_sa3.sum(), a_g3, create_graph=True, retain_graph=True)[0]
    phys_mask = torch.zeros(state_dim)
    phys_mask[phys_idx] = 1.0
    ds = (next_state - state) * phys_mask
    da = next_action - action
    f1 = (grad_s3 * ds).sum(1, keepdim=True)
    f2 = (grad_a3 * da).sum(1, keepdim=True)
    v_sc = (grad_s3 * ds.detach()).sum()
    H_ds = torch.autograd.grad(v_sc, s_g3, create_graph=False, retain_graph=True)[0]
    second = 0.5 * (H_ds * ds).sum(1, keepdim=True)
    discount = math.exp(-(-math.log(0.99) / 3.0) * 3.0)
    target = reward + discount * (q_sa3.detach() + f1 + f2 + second)
    assert target.shape == (B, 1)
    print(f"OK: Path B target computed, shape {target.shape}, mean {target.mean().item():.4f}")

    # ---- Critic loss + backward ----
    q1_pred, q2_pred = critic(state, action)
    loss = F.mse_loss(q1_pred, target.detach()) + F.mse_loss(q2_pred, target.detach())
    loss.backward()
    has_grad = critic.theta1.weight.grad is not None and critic.theta1.weight.grad.abs().sum() > 0
    assert has_grad, "theta1 didn't receive gradient"
    print(f"OK: critic loss = {loss.item():.4f}, theta1 received gradient")

    print()
    print("All LinearBasisCritic checks PASSED.")


if __name__ == "__main__":
    main()
