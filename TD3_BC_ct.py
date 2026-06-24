#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous-time variant of TD3-BC (Approach A, normalised form).

The professor's continuous-time value function
    v(s) = r * dt + exp(-beta * dt) * E[v(s') | s]
becomes, after dividing both sides by dt and identifying r as the
reward-rate, the equivalent reward-rate Bellman recursion
    v(s) = r + exp(-beta * dt) * E[v(s') | s]
which we use as the critic target:
    target_Q = reward + done * exp(-beta * dt) * target_Q

This is the only structural change vs the paper baseline. At the
default beta = -ln(0.99) / 3 per min and dt = 3 min:
    exp(-beta * dt) = 0.99 = paper's gamma
so target_Q stays on the same numerical scale as the baseline; the
BC adaptive lambda = alpha / |Q|.mean is preserved and the critic
loss does not blow up. Tuning beta or dt now changes the effective
discount in a sampling-rate-independent way -- which is the whole
point of the continuous-time formulation, and the foundation for
Approach C (HJB regularisation).

Approach A keeps the replay buffer untouched: the existing buffer
file can be reused as-is.
"""

import math
import numpy as np
import copy, random, torch, gym, pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from utils import unpackage_replay, get_batch, test_algorithm, create_graph


"""
Simple feedforward neural network for the Actor.
"""
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


"""
Simple feedforward neural network for the Critic.

Optionally wraps every Linear layer with spectral normalization
(Miyato et al., 2018). When enabled, each layer's spectral norm
(largest singular value of its weight matrix) is constrained to <= 1,
so the overall network is Lipschitz with bounded constant. This
addresses the open problem from Zhu's PhiBE-Q paper §3.3.2 / §5:
the Taylor-expanded target depends on ∇_s Q, which is unbounded for
an unconstrained DNN critic; the linear-basis Lemma 3.1 (Lipschitz of
the iteration operator) does not transfer. Forcing a Lipschitz network
restores the assumption the paper's stability theory needs.

See RESEARCH_POSITION.md for full motivation.
"""
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_spectral_norm=False,
                 sn_scale=1.0):
        super(Critic, self).__init__()

        # If use_spectral_norm=True we wrap each Linear with
        # nn.utils.spectral_norm so its spectral norm is forced <= 1.
        # The overall network is then Lipschitz with constant
        # <= prod(per-layer spectral norms) <= 1.
        #
        # sn_scale relaxes the constraint: in forward we multiply each
        # spec-norm-wrapped layer's output by sn_scale, so the layer's
        # effective Lipschitz constant becomes sn_scale (not 1). This
        # gives the critic more expressive power than Lip=1 while still
        # keeping a finite, controllable bound.
        # sn_scale=1 reproduces the strict Lip=1 setting.
        # sn_scale > 1 gives critic more capacity at the cost of looser
        # stability guarantee.
        self.use_spectral_norm = use_spectral_norm
        self.sn_scale = float(sn_scale)

        def maybe_sn(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Q1 architecture
        self.l1 = maybe_sn(nn.Linear(state_dim + action_dim, 256))
        self.l2 = maybe_sn(nn.Linear(256, 256))
        self.l3 = maybe_sn(nn.Linear(256, 1))

        # Q2 architecture
        self.l4 = maybe_sn(nn.Linear(state_dim + action_dim, 256))
        self.l5 = maybe_sn(nn.Linear(256, 256))
        self.l6 = maybe_sn(nn.Linear(256, 1))

    def _apply_layer(self, layer, x):
        # Helper: apply a layer; if spectral_norm is on AND sn_scale != 1,
        # multiply the output by sn_scale (this lifts the per-layer
        # Lipschitz bound from 1 to sn_scale).
        out = layer(x)
        if self.use_spectral_norm and self.sn_scale != 1.0:
            out = out * self.sn_scale
        return out

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self._apply_layer(self.l1, sa))
        q1 = F.relu(self._apply_layer(self.l2, q1))
        q1 = self._apply_layer(self.l3, q1)

        q2 = F.relu(self._apply_layer(self.l4, sa))
        q2 = F.relu(self._apply_layer(self.l5, q2))
        q2 = self._apply_layer(self.l6, q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self._apply_layer(self.l1, sa))
        q1 = F.relu(self._apply_layer(self.l2, q1))
        q1 = self._apply_layer(self.l3, q1)
        return q1


"""
Linear-basis Critic — Q(s, a) = Phi(s, a)^T theta.

Directly mirrors the function class used in Zhu's PhiBE-Q paper (§3.2,
Theorems 3.2/3.3) and the LQR experiment in §4.1. Under this class
Yuhua's Lemma 3.1 holds by construction: ∇_s Phi is a fixed, bounded
quantity, so ∇_s Q = (∇_s Phi)^T theta is automatically bounded
whenever |theta| is bounded. This eliminates the unbounded-gradient
issue that the neural Critic exhibits (the open problem from §3.3.2
of the paper).

Basis design for our 11-dim state + 1-dim action setting:
    Phi = [1, s_0, ..., s_10, a,            (1 + 11 + 1 = 13 linear)
           s_0^2, ..., s_10^2, a^2,         (11 + 1 = 12 quadratic state/action)
           s_0 a, ..., s_10 a,              (11 cross state-action)
           s_8 s_9, s_8 s_10, s_9 s_10]     (3 cross within physical dims)
    => 13 + 12 + 11 + 3 = 39 features.

This is ~13x the LQR basis (which used 3 features for 1-dim state) but
still negligible compared to 256x256 neural Critic (~200k params).
The Q1 and Q2 heads share the same Phi and have separate theta vectors.
"""
class LinearBasisCritic(nn.Module):
    def __init__(self, state_dim, action_dim, phys_idx=(8, 9, 10)):
        super(LinearBasisCritic, self).__init__()
        assert action_dim == 1, "LinearBasisCritic currently assumes scalar action"
        self.state_dim = state_dim
        self.phys_idx = list(phys_idx)
        # Number of basis features (see docstring above).
        n_features = 1 + state_dim + 1 + state_dim + 1 + state_dim + len(self.phys_idx) * (len(self.phys_idx) - 1) // 2
        self.n_features = n_features
        # Linear heads -- the ONLY trainable params here.
        # Both heads share the basis Phi but have separate theta.
        self.theta1 = nn.Linear(n_features, 1, bias=False)
        self.theta2 = nn.Linear(n_features, 1, bias=False)
        # Initialise small so the initial Q has reasonable magnitude.
        nn.init.normal_(self.theta1.weight, std=0.1)
        nn.init.normal_(self.theta2.weight, std=0.1)

    def basis(self, state, action):
        """
        Compute Phi(s, a) for a batch.
        state: (B, state_dim)
        action: (B, action_dim) = (B, 1)
        Returns: (B, n_features)
        """
        feats = [torch.ones_like(action[:, :1])]    # constant
        feats.append(state)                          # linear state (B, 11)
        feats.append(action)                         # linear action (B, 1)
        feats.append(state ** 2)                     # quadratic state (B, 11)
        feats.append(action ** 2)                    # quadratic action (B, 1)
        feats.append(state * action)                 # state-action cross (B, 11)
        # Physical-dim cross terms
        for ii, i in enumerate(self.phys_idx):
            for j in self.phys_idx[ii + 1:]:
                feats.append((state[:, i:i + 1] * state[:, j:j + 1]))
        return torch.cat(feats, dim=1)

    def forward(self, state, action):
        phi = self.basis(state, action)
        return self.theta1(phi), self.theta2(phi)

    def Q1(self, state, action):
        phi = self.basis(state, action)
        return self.theta1(phi)


"""
Drift estimator for HJB regularisation (Approach C variant B).

Predicts mu(s, a) -- the expected instantaneous rate of change of the
physical-dynamics subset of the state [BG_now, MOB, IOB] -- given the
current state and action. Trained by regression against the empirical
finite-difference (s'_phys - s_phys) / dt across the offline buffer.
The network's limited capacity acts as a smoother that averages out the
diffusion noise present in raw finite differences, which is the main
reason raw (s'-s)/dt makes HJB regularisation unstable.
"""
class DriftNet(nn.Module):
    def __init__(self, state_dim, action_dim, out_dim=3, hidden=256):
        super(DriftNet, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        h = F.relu(self.l1(sa))
        h = F.relu(self.l2(h))
        return self.l3(h)


class td3_bc_ct:

    def __init__(self, init_seed, patient_params, params):

        # ENVIRONMENT
        self.params = params
        self.env_name = patient_params["env_name"]
        self.folder_name = patient_params["folder_name"]
        self.replay_name = patient_params["replay_name"]
        self.bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
        self.env = gym.make(self.env_name)
        self.action_size, self.state_size = 1, 11
        self.params["state_size"] = self.state_size
        self.sequence_length = 80
        self.data_processing = "condensed"

        # HYPERPARAMETERS
        self.device = params["device"]
        self.batch_size = 256
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.alpha = 2.5

        # CONTINUOUS-TIME PARAMETERS (Approach A)
        # dt: simulator sampling interval in minutes (paper uses 3 min)
        # beta: continuous-time discount rate (per minute)
        # At default values: exp(-beta * dt) == 0.99 == paper's gamma,
        # so this file matches the paper baseline numerically.
        self.dt = params.get("dt", 3.0)
        self.beta = params.get("beta", -math.log(0.99) / 3.0)
        self.discount = math.exp(-self.beta * self.dt)
        # kept as alias so any downstream code that reads self.gamma still works
        self.gamma = self.discount

        # HJB REGULARISATION (Approach C, optional)
        # If use_hjb=True, the critic loss adds a soft constraint enforcing
        # the first-order Hamilton-Jacobi-Bellman equation:
        #     beta * Q(s, a) ~= r(s, a) + grad_s Q(s, a) . mu(s, a)
        # Approach A == use_hjb=False (default, identical to before).
        # Approach C == use_hjb=True.
        self.use_hjb = params.get("use_hjb", False)
        self.lambda_hjb = params.get("lambda_hjb", 1.0)

        # PATH B (continuous-time Q via Taylor expansion as target body)
        # When use_path_b=True the critic target is REPLACED (not augmented)
        # by the second-order Taylor expansion of the professor's
        # continuous-time Q function around (s_t, a_t):
        #     target = r + exp(-beta * dt) * [
        #         Q_target(s, a)
        #         + (s' - s)^T grad_s Q_target
        #         + (pi_target(s') - a)^T grad_a Q_target
        #         + 1/2 (s' - s)^T grad_s^2 Q_target (s' - s)
        #     ]
        # Differences from Approach C:
        #   - No HJB residual added to the loss; target itself is replaced.
        #   - Includes second-order Hessian term via Hessian-vector product.
        #   - Uses the same phys_idx=[8,9,10] mask on Delta_s as in C v2,
        #     since the lagged BG snapshots do not obey an SDE.
        # The Hessian-vector product trick (Pearlmutter 1994) computes
        #     Delta_s^T (grad_s^2 Q) Delta_s
        # by differentiating the scalar (grad_s Q . Delta_s) once more
        # w.r.t. s, avoiding the O(d^2) cost of the full Hessian.
        self.use_path_b = params.get("use_path_b", False)
        # Cannot turn on Path B and Approach C at the same time -- they
        # define mutually incompatible critic targets.
        if self.use_path_b and self.use_hjb:
            raise ValueError(
                "use_path_b=True and use_hjb=True are mutually exclusive: "
                "Path B replaces the target, Approach C augments it. "
                "Pick one."
            )

        # REWARD NORMALISATION (思路 3).
        # The raw reward = -Magni_risk(BG) ranges from ~-0.07 (BG=144 target)
        # to ~-100 (BG=600 extreme), plus a -1e5 termination cliff. This
        # makes |Q|.mean ~= 1000 (HANDOFF §6.7), which shrinks the BC
        # adaptive lambda = alpha / |Q|.mean to ~0.0025 -- compressing the
        # Q-side of the actor loss by ~400x relative to the BC side.
        # Setting normalize_reward=True triggers utils.data_processing's
        # built-in reward normalisation: reward becomes
        #     (raw_reward - reward_mean) / reward_std
        # so its std is 1. Q drops to a much smaller magnitude (~1-10),
        # restoring lambda ~ 0.1-1 and making BC/Q rebalance to the
        # D4RL-style regime that paper alpha=2.5 was tuned for.
        # CRITICAL: the SHAPE of the reward is preserved exactly (no clip,
        # no warp). Low-BG penalty stays 7x higher than high-BG penalty,
        # and the terminal cliff stays. Only the linear scale changes.
        self.normalize_reward = params.get("normalize_reward", False)

        # SPECTRAL NORMALIZATION on critic (RESEARCH_POSITION.md §5.1).
        # When use_spectral_norm=True, every Linear layer in Critic gets
        # wrapped with torch.nn.utils.spectral_norm so the network's
        # Lipschitz constant is bounded. This is our principled attack on
        # the Zhu (2025) open problem: PhiBE-Q's Lemma 3.1 needs a
        # Lipschitz bound on the iteration operator H, which is automatic
        # under linear basis (bounded c_1, c_2, c_3) but absent for an
        # unconstrained DNN. Spectral norm restores the Lipschitz property
        # as an architecture constraint.
        #
        # IMPORTANT: this also tightens the input gradient |∇_s Q|, so the
        # ad-hoc path_b_grad_clip is no longer the principal stabiliser.
        # We keep clip available but suggest setting it large (e.g. 5+) so
        # it only catches outliers, not the typical case.
        # lambda_mode controls how TD3-BC's actor BC balance is computed:
        #   "adaptive" (default, paper): lmbda = alpha / |Q|.mean (every step)
        #   "fixed_by_reward_std":       lmbda = alpha / reward_std (set once)
        # The adaptive form assumes |Q| evolves slowly — broken when the
        # critic is a linear basis since |Q| = |theta^T Phi| can swing
        # faster than the actor can adapt. Using reward_std as the divisor
        # (a stable buffer statistic) gives lmbda ~ D4RL-regime values
        # (~2.5 under normalized reward) without depending on Q-magnitude.
        self.lambda_mode = params.get("lambda_mode", "adaptive")

        # critic_type: 'neural' (default, 256x256 ReLU MLP) or 'linear_basis'
        # (Yuhua paper §3.2 / §4.1 alignment). When 'linear_basis', the
        # Critic is Q = Phi^T theta with fixed handcrafted basis Phi —
        # Lemma 3.1 then holds by construction and our work directly
        # extends the LQR experiment in §4.1 to glucose control.
        self.critic_type = params.get("critic_type", "neural")

        self.use_spectral_norm = params.get("use_spectral_norm", False)
        # sn_scale: relax the Lip=1 default of spectral_norm by lifting
        # each layer's effective Lipschitz to sn_scale. With L layers, the
        # network's Lipschitz constant becomes <= sn_scale^L (loose bound).
        # sn_scale=1 (default) -> strict Lip=1, may be too tight on
        # complex value landscapes like Magni risk with terminal cliff.
        # sn_scale=3-10 commonly used in spectral-norm regularised SAC.
        self.sn_scale = params.get("sn_scale", 1.0)

        # Path B input-gradient clipping. Caps the per-row L2 norm of
        # grad_s Q, grad_a Q, and H*Delta_s when they are computed
        # for the Taylor-expanded target. This prevents the runaway
        # observed at FULL 100k where |grad_s Q| grew 12x (75 -> 925)
        # and pushed the critic target to absurd magnitudes.
        # Set to 0 or None to disable (recover the original behaviour).
        # NOTE: this clips the INPUT gradient that enters the target
        # formula itself, NOT the optimizer's parameter gradient (which
        # was the wrong target of the previous clip attempt).
        self.path_b_grad_clip = params.get("path_b_grad_clip", 10.0)

        # diag_phibe: when True, record PRE-CLIP magnitudes of every PhiBE
        # target component each step (grad_s/grad_a norms, Hessian term, and
        # the four target pieces Q / ds.gradsQ / da.gradaQ / 2nd order) into
        # self._diag so a diagnostic script can log how the (non-standard,
        # derivative-bearing) PhiBE loss misbehaves. Off by default = no cost.
        self.diag_phibe = params.get("diag_phibe", False)
        self._diag = {}
        self.diag_log_freq = int(params.get("diag_log_freq", 1000))
        self._diag_history = []

        # drift_mode: how mu(s, a) is estimated.
        #   "fd"  : raw finite difference (s'_phys - s_phys) / dt -- noisy,
        #           but zero extra parameters. (Approach C v1/v2.)
        #   "net" : a small MLP DriftNet(s, a) -> R^3 trained by MSE against
        #           the finite difference. The network smooths the diffusion
        #           noise that contaminates the raw FD signal.
        # drift_warmup_steps: number of critic steps where the HJB term is
        #   suppressed (set to zero) so DriftNet has time to fit the data
        #   before being asked to drive a regularisation signal.
        self.drift_mode = params.get("drift_mode", "fd")
        self.drift_lr = params.get("drift_lr", 3e-4)
        self.drift_warmup_steps = params.get("drift_warmup_steps", 5000)
        # The 3 physical-dynamics state indices [BG_now, MOB, IOB]. Only
        # these dimensions obey real physiological dynamics; the other 8
        # are lagged BG snapshots and have no SDE.
        self.phys_idx = [8, 9, 10]
        
        # DISPLAY
        self.pid_bg, self.pid_insulin, self.pid_action, self.pid_reward  = [], [], [], 0
        self.training_timesteps = params["training_timesteps"]
        self.training_progress_freq = int(self.training_timesteps // 10)
        
        # SEEDING
        self.train_seed = init_seed
        self.env.seed(self.train_seed) 
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)  
        random.seed(self.train_seed)      
        
        # MEMORY
        self.memory_size = self.training_timesteps 
        self.memory = deque(maxlen=self.memory_size)
        
    """
    Initialise the Actor and the Critic.
    """        
    def init_model(self):
        
        # actor
        self.actor = Actor(self.state_size, self.action_size, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        # critic. Spectral normalization is opt-in via self.use_spectral_norm
        # (see __init__ comment). When True, each Linear layer has its
        # spectral norm enforced <= 1, bounding the network's Lipschitz
        # constant -- this is the key knob for the Zhu 2025 DNN-stability
        # open problem.
        # Critic construction depends on critic_type:
        #   'neural'       -> 256x256 ReLU MLP (paper baseline architecture).
        #   'linear_basis' -> Q = Phi^T theta with fixed handcrafted basis.
        # The linear_basis path aligns with Yuhua's PhiBE-Q paper §3.2/§4.1
        # and gives Lemma 3.1's Lipschitz bound by construction.
        if self.critic_type == "linear_basis":
            self.critic = LinearBasisCritic(
                self.state_size, self.action_size, phys_idx=self.phys_idx,
            ).to(self.device)
        else:
            self.critic = Critic(
                self.state_size, self.action_size,
                use_spectral_norm=self.use_spectral_norm,
                sn_scale=self.sn_scale,
            ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # drift estimator (only used when use_hjb=True and drift_mode="net")
        if self.use_hjb and self.drift_mode == "net":
            self.drift_net = DriftNet(
                self.state_size, self.action_size, out_dim=len(self.phys_idx)
            ).to(self.device)
            self.drift_optimizer = torch.optim.Adam(
                self.drift_net.parameters(), lr=self.drift_lr
            )
        else:
            self.drift_net = None
            self.drift_optimizer = None
        
    
    """
    Save the learned models. Tag differentiates the modes:
        ct          -- Approach A (use_hjb=False, use_path_b=False)
        ct_hjb      -- Approach C v1/v2 (use_hjb=True, drift_mode="fd")
        ct_hjb_net  -- Approach C v3   (use_hjb=True, drift_mode="net")
        ct_path_b   -- Path B (use_path_b=True): Taylor-expanded target
    This prevents the modes from overwriting each other's weights.
    """
    def save_model(self):
        if self.use_path_b:
            tag = "ct_path_b"
        elif not self.use_hjb:
            tag = "ct"
        elif self.drift_mode == "net":
            tag = "ct_hjb_net"
        else:
            tag = "ct_hjb"
        prefix = './Models/' + str(self.env_name) + str(self.train_seed) + 'TD3_offline_BC_' + tag + '_weights'
        suffix = self.replay_name.split("-")[-1]
        torch.save(self.actor.state_dict(),  prefix + '_actor'  + suffix)
        torch.save(self.critic.state_dict(), prefix + '_critic' + suffix)
            
    """
    Load pre-trained weights for testing.
    """
    def load_model(self, name):
        
        # load actor
        self.actor.load_state_dict(torch.load(name + '_actor'))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.eval()   
        
        # load critic
        self.critic.load_state_dict(torch.load(name + '_critic'))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic.eval()   
        
    """
    Determine the action based on the state.
    """        
    def select_action(self, state, action, timestep, prev_reward):
        
        # Feed state into model
        with torch.no_grad():
            tensor_state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            tensor_action = self.actor(tensor_state)
            
        return tensor_action.cpu().data.numpy().flatten()
    
    """
    Train the model on a pre-collected sample of training data.
    """
    def train_model(self):
        
        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file) 
            
        # Process the replay --------------------------------------------------

        # unpackage the replay. We capture reward_mean / reward_std here
        # since 思路 3 (reward normalisation) needs them to be plumbed
        # into self.params so that get_batch returns standardised rewards.
        (
            self.memory, self.state_mean, self.state_std,
            self.action_mean, self.action_std,
            reward_mean, reward_std,
        ) = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )

        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        self.max_action = float(((self.bas * 3.0) - self.action_mean) / self.action_std)

        # Reward normalisation (思路 3). Only activate when explicitly
        # enabled; legacy modes (baseline / A / C) without this flag get
        # raw rewards exactly as before, preserving reproducibility of
        # the 67.48% baseline number recorded in HANDOFF §6.2.
        if self.normalize_reward:
            self.params["reward_mean"] = float(reward_mean)
            self.params["reward_std"]  = float(reward_std)
            print(f"[normalize_reward] mean={reward_mean:.4f}  std={reward_std:.4f}")
            print(f"[normalize_reward] reward will be passed as "
                  f"(raw - {reward_mean:.4f}) / {reward_std:.4f}")
        
        # initialise the networks
        self.init_model()
        
        print('Processing Complete.')             
        
        for t in range(1, self.training_timesteps + 1):
            
            # Get the batch ------------------------------------------------
            
            # unpackage the samples and split
            state, action, reward, next_state, done, _, _, _, _, _ = get_batch(
                replay=self.memory, batch_size=self.batch_size, 
                data_processing=self.data_processing, 
                sequence_length=self.sequence_length, device=self.device, 
                params=self.params
            )

            # Training -----------------------------------------------

            # Track gradient-norm diagnostics for Path B sanity check.
            self._last_grad_s_norm = 0.0
            self._last_hess_s_norm = 0.0

            # Per-row L2 norm clipping used by Path B for input gradients.
            # Caps each sample's gradient norm at max_norm, so the per-sample
            # target value cannot grow unboundedly even if the critic learns
            # an arbitrarily curvy Q. Returns the (possibly rescaled) tensor.
            def _clip_input_grad(g, max_norm):
                if not max_norm or max_norm <= 0:
                    return g
                # row-wise norm, shape (B, 1)
                row_norm = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
                scale = (max_norm / row_norm).clamp(max=1.0)
                return g * scale

            if self.use_path_b:
                # ============== PATH B: Taylor-expanded target ==============
                # target_i = r_i + exp(-beta*dt) * [
                #     Q_target(s_i, a_i)
                #     + Delta_s_i^T grad_s Q_target
                #     + Delta_a_i^T grad_a Q_target
                #     + 1/2 Delta_s_i^T grad_s^2 Q_target Delta_s_i
                # ]
                # All gradients are taken w.r.t. the target critic at (s, a).
                # phys_idx mask is applied to Delta_s so non-physical (lagged
                # BG snapshot) dimensions do not contribute spurious drift,
                # following the lesson from Approach C v2.

                # ---- 1. Build inputs that require gradients (target net) ----
                # We need grad_s Q_target, grad_a Q_target, grad_s^2 Q_target.
                # PyTorch can compute grads of the target critic's output
                # w.r.t. its inputs even though the target critic's
                # parameters are not being optimised -- requires_grad on the
                # INPUTS is what matters.
                s_g = state.detach().clone().requires_grad_(True)
                a_g = action.detach().clone().requires_grad_(True)

                # Use Q1 head of the target critic for the gradient terms;
                # Q2 head is used analogously below for double-Q min.
                def _path_b_target(target_net):
                    q_sa = target_net.Q1(s_g, a_g)  # shape (B, 1)

                    # First-order gradients
                    grad_s = torch.autograd.grad(
                        outputs=q_sa.sum(), inputs=s_g,
                        create_graph=True, retain_graph=True,
                    )[0]  # (B, state_size)
                    grad_a = torch.autograd.grad(
                        outputs=q_sa.sum(), inputs=a_g,
                        create_graph=True, retain_graph=True,
                    )[0]  # (B, action_size)

                    # --- PhiBE loss diagnostics: capture PRE-CLIP magnitudes.
                    # These are the derivative terms that make the PhiBE target
                    # non-standard vs ordinary RL. Recorded raw (before any
                    # clipping) so we see the TRUE runaway, not the capped value.
                    if self.diag_phibe:
                        self._diag["grad_s_norm"] = float(grad_s.detach().norm(dim=1).mean())
                        self._diag["grad_a_norm"] = float(grad_a.detach().norm(dim=1).mean())
                        self._diag["grad_s_max"] = float(grad_s.detach().norm(dim=1).max())

                    # Clip per-sample gradient norms used inside the target.
                    # This is the FIX for the runaway observed at FULL 100k.
                    grad_s = _clip_input_grad(grad_s, self.path_b_grad_clip)
                    grad_a = _clip_input_grad(grad_a, self.path_b_grad_clip)

                    # phys_mask: 0 on non-physical dims, 1 on [BG_now, MOB, IOB]
                    phys_mask = torch.zeros(
                        self.state_size, device=self.device
                    )
                    phys_mask[self.phys_idx] = 1.0

                    delta_s = (next_state - state) * phys_mask  # (B, 11)
                    delta_a = next_action - action              # (B, action_dim)

                    first_order_s = (grad_s * delta_s).sum(
                        dim=1, keepdim=True
                    )
                    first_order_a = (grad_a * delta_a).sum(
                        dim=1, keepdim=True
                    )

                    # Hessian-vector product:
                    #     H delta_s = grad_s ( grad_s Q . delta_s )
                    # Then 1/2 delta_s^T H delta_s = 1/2 (H delta_s . delta_s).
                    # delta_s is detached so PyTorch only differentiates the
                    # Q-side of the inner product.
                    v_scalar = (grad_s * delta_s.detach()).sum()
                    H_delta_s = torch.autograd.grad(
                        outputs=v_scalar, inputs=s_g,
                        create_graph=False, retain_graph=True,
                    )[0]  # (B, state_size)
                    # Also clip H * delta_s -- second-order contribution
                    # is usually 0 under ReLU but we clip for safety.
                    H_delta_s = _clip_input_grad(H_delta_s, self.path_b_grad_clip)
                    second_order = 0.5 * (H_delta_s * delta_s).sum(
                        dim=1, keepdim=True
                    )

                    # Diagnostics (only need to record once; first call wins).
                    if self._last_grad_s_norm == 0.0:
                        self._last_grad_s_norm = float(
                            grad_s.detach().norm(dim=1).mean()
                        )
                        self._last_hess_s_norm = float(
                            H_delta_s.detach().norm(dim=1).mean()
                        )

                    # --- PhiBE per-component target diagnostics (Q1 head).
                    # Records the magnitude of each piece of the PhiBE target so
                    # we can see which term dominates / blows up. second_order is
                    # ~0 under ReLU (no curvature) -- itself key evidence.
                    if self.diag_phibe:
                        self._diag["comp_Q"] = float(q_sa.detach().abs().mean())
                        self._diag["comp_ds_gradsQ"] = float(first_order_s.detach().abs().mean())
                        self._diag["comp_da_gradaQ"] = float(first_order_a.detach().abs().mean())
                        self._diag["comp_2nd"] = float(second_order.detach().abs().mean())

                    target = (
                        q_sa.detach()
                        + first_order_s
                        + first_order_a
                        + second_order
                    )
                    return target

                # Build the target actor's next_action (used for delta_a).
                with torch.no_grad():
                    noise = (
                        torch.randn_like(action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)

                target_inner_1 = _path_b_target(self.critic_target)
                # For double-Q min, run the same computation but read Q2 of
                # the target critic. We re-create the inputs to avoid graph
                # collisions from the previous Q1 backward pass.
                s_g2 = state.detach().clone().requires_grad_(True)
                a_g2 = action.detach().clone().requires_grad_(True)

                q2_sa = self.critic_target(s_g2, a_g2)[1]  # Q2 head
                grad_s_q2 = torch.autograd.grad(
                    q2_sa.sum(), s_g2, create_graph=True, retain_graph=True
                )[0]
                grad_a_q2 = torch.autograd.grad(
                    q2_sa.sum(), a_g2, create_graph=True, retain_graph=True
                )[0]
                # Same input-gradient clipping as Q1 head above.
                grad_s_q2 = _clip_input_grad(grad_s_q2, self.path_b_grad_clip)
                grad_a_q2 = _clip_input_grad(grad_a_q2, self.path_b_grad_clip)
                phys_mask = torch.zeros(self.state_size, device=self.device)
                phys_mask[self.phys_idx] = 1.0
                delta_s = (next_state - state) * phys_mask
                delta_a = next_action - action
                first_order_s2 = (grad_s_q2 * delta_s).sum(dim=1, keepdim=True)
                first_order_a2 = (grad_a_q2 * delta_a).sum(dim=1, keepdim=True)
                v_scalar2 = (grad_s_q2 * delta_s.detach()).sum()
                H_delta_s2 = torch.autograd.grad(
                    v_scalar2, s_g2, create_graph=False, retain_graph=False
                )[0]
                H_delta_s2 = _clip_input_grad(H_delta_s2, self.path_b_grad_clip)
                second_order2 = 0.5 * (H_delta_s2 * delta_s).sum(dim=1, keepdim=True)
                target_inner_2 = (
                    q2_sa.detach()
                    + first_order_s2
                    + first_order_a2
                    + second_order2
                )

                # Double-Q min on the bracketed inner expression, then apply
                # r + exp(-beta*dt) * [...] outside the min.
                target_inner = torch.min(target_inner_1, target_inner_2)
                # Double-Q min then outer r + e^(-beta dt)*[...]
                target_Q = (reward + done * self.discount * target_inner).detach()

            else:
                # ============== Baseline / Approach A target ==============
                with torch.no_grad():
                    noise = (
                        torch.randn_like(action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + done * self.discount * target_Q

            # Update the critic -------------------------------------------

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute Bellman critic loss (same as Approach A / paper)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # --- PhiBE diagnostics: critic loss + target range this step.
            if self.diag_phibe:
                self._diag["critic_loss"] = float(critic_loss.detach())
                self._diag["target_Q_mean"] = float(target_Q.detach().mean())
                self._diag["target_Q_absmax"] = float(target_Q.detach().abs().max())
                self._diag["current_Q_absmean"] = float(current_Q1.detach().abs().mean())

            # ----- HJB regularisation (Approach C, first-order) -----
            self._last_hjb_loss = 0.0
            self._last_drift_loss = 0.0

            if self.use_hjb and not self.use_path_b:
                mu_fd_phys = ((next_state - state) / self.dt)[:, self.phys_idx].detach()

                if self.drift_mode == "net":
                    pred_mu = self.drift_net(state.detach(), action.detach())
                    drift_loss = F.mse_loss(pred_mu, mu_fd_phys)
                    self.drift_optimizer.zero_grad()
                    drift_loss.backward()
                    self.drift_optimizer.step()
                    self._last_drift_loss = float(drift_loss.detach())

                if t > self.drift_warmup_steps:
                    state_g = state.detach().clone().requires_grad_(True)
                    q1_g = self.critic.Q1(state_g, action)
                    grad_q = torch.autograd.grad(
                        outputs=q1_g.sum(), inputs=state_g,
                        create_graph=True, retain_graph=True,
                    )[0]
                    grad_q_phys = grad_q[:, self.phys_idx]
                    if self.drift_mode == "net":
                        mu_hat_phys = self.drift_net(state, action).detach()
                    else:
                        mu_hat_phys = mu_fd_phys
                    drift_term = (grad_q_phys * mu_hat_phys).sum(dim=1, keepdim=True)
                    hjb_residual = self.beta * q1_g - reward - drift_term
                    hjb_loss = (hjb_residual ** 2).mean()
                    critic_loss = critic_loss + self.lambda_hjb * hjb_loss
                    self._last_hjb_loss = float(hjb_loss.detach())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Perform the actor update ----------------------------------

            if t % self.policy_freq == 0:

                # lmbda balances Q-following vs BC. Paper default uses
                # |Q|.mean; linear-basis Critic can swing |Q| fast and
                # break that. lambda_mode='fixed_by_reward_std' uses the
                # buffer reward std (stable) -> recovers D4RL-regime
                # lmbda~alpha when reward is normalized.
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                if self.lambda_mode == "fixed_by_reward_std":
                    if self.normalize_reward:
                        effective_std = 1.0
                    else:
                        r_std = self.params.get("reward_std", None)
                        effective_std = float(r_std) if (r_std is not None and r_std > 0) else 1.0
                    lmbda = self.alpha / effective_std
                else:
                    lmbda = self.alpha / Q.abs().mean().detach()
                actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # --- PhiBE diagnostics: append a snapshot to history every
            # diag_log_freq steps (finer than the 10-point progress print) so
            # we get a smooth trend of how each PhiBE term evolves.
            if self.diag_phibe and (t % self.diag_log_freq == 0):
                snap = {"t": t}
                snap.update(self._diag)
                self._diag_history.append(snap)

            # Show progress
            if t % self.training_progress_freq == 0:
                msg = 'Timesteps {} - Actor Loss {} - Critic Loss {}'.format(
                    t, actor_loss, critic_loss)
                if self.use_hjb:
                    msg += ' - HJB Loss {}'.format(self._last_hjb_loss)
                    if self.drift_mode == "net":
                        msg += ' - Drift Loss {}'.format(self._last_drift_loss)
                if self.use_path_b:
                    msg += ' - |grad_s Q| {:.4f} - |H_s Q . dS| {:.4f}'.format(
                        self._last_grad_s_norm, self._last_hess_s_norm
                    )
                print(msg)
                self.save_model()

    def test_model(self, input_seed=0, input_max_timesteps=4800):
        env = gym.make(self.env_name)
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file)

        (self.memory, self.state_mean, self.state_std,
         self.action_mean, self.action_std, _, _) = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory,
            data_processing=self.data_processing, sequence_length=self.sequence_length
        )
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"] = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        self.max_action = float(((self.bas * 3) - self.action_mean) / self.action_std)
        self.init_model()

        if self.use_path_b:
            tag = "ct_path_b"
        elif not self.use_hjb:
            tag = "ct"
        elif self.drift_mode == "net":
            tag = "ct_hjb_net"
        else:
            tag = "ct_hjb"
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'TD3_offline_BC_' + tag + '_weights')
        test_seed, max_timesteps = input_seed, input_max_timesteps

        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing,
            pid_run=False, params=self.params
        )

        create_graph(
            rl_reward=rl_reward, rl_blood_glucose=rl_bg, rl_action=rl_action, rl_insulin=rl_insulin,
            rl_meals=rl_meals, pid_reward=pid_reward, pid_blood_glucose=pid_bg,
            pid_action=pid_action, params=self.params
        )
