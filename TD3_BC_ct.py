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
"""
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


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
        
        # critic
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
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
    Save the learned models. Tag differentiates the four modes:
        ct          -- Approach A (use_hjb=False)
        ct_hjb      -- Approach C v1/v2 (use_hjb=True, drift_mode="fd")
        ct_hjb_net  -- Approach C v3   (use_hjb=True, drift_mode="net")
    This prevents the modes from overwriting each other's weights.
    """
    def save_model(self):
        if not self.use_hjb:
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
        
        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )
        
        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std    
        self.max_action = float(((self.bas * 3.0) - self.action_mean) / self.action_std)
        
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
            
            with torch.no_grad():
                
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value (continuous-time form, normalised).
                # Discrete paper version was:
                #     target_Q = reward + done * gamma * target_Q
                # The continuous-time value function
                #     v(s) = r*dt + exp(-beta*dt) * E[v(s') | s]
                # is equivalent (after dividing both sides by dt) to a
                # reward-rate Bellman recursion
                #     v(s) = r + done * exp(-beta*dt) * E[v(s') | s]
                # so the only structural change vs the paper baseline is that
                # gamma is replaced by the analytic discount exp(-beta*dt).
                # At the default beta = -ln(0.99)/3 and dt = 3 this gives
                # exp(-beta*dt) = 0.99 = paper's gamma, and target_Q lives on
                # the same numerical scale as the baseline -- so the BC
                # adaptive lambda = alpha / |Q|.mean is preserved and the
                # critic loss does not blow up. Tuning beta or dt now changes
                # the discount in a principled, sampling-rate-independent way.
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + done * self.discount * target_Q
                
            # Update the critic -------------------------------------------

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute Bellman critic loss (same as Approach A / paper)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # ----- HJB regularisation (Approach C, first-order) -----
            # Soft-enforce the first-order Hamilton-Jacobi-Bellman equation
            #     beta * Q(s, a) ~= r(s, a) + grad_s Q . mu_phys(s, a)
            # State layout reminder: the 11-D condensed state is
            #     [BG_-4h, BG_-3.5h, ..., BG_-0.5h, BG_0, MOB, IOB]
            # Indices 0..7 are lagged BG snapshots (window shifts), not
            # physical state -- their FD obeys no SDE -- so we restrict the
            # HJB drift term to indices 8 (BG_now), 9 (MOB), 10 (IOB).
            #
            # Two ways to estimate mu on these three dims (drift_mode):
            #   "fd"  -- raw finite difference (s'_phys - s_phys) / dt
            #   "net" -- a learned DriftNet trained alongside the critic
            #            against the FD signal; smooths out diffusion noise.
            self._last_hjb_loss = 0.0
            self._last_drift_loss = 0.0

            if self.use_hjb:
                # FD target on the physical subset, used as both the "fd"
                # drift estimate and the DriftNet training target.
                mu_fd_phys = ((next_state - state) / self.dt)[:, self.phys_idx].detach()

                # ----- 1. (Optional) update DriftNet by MSE vs FD -----
                if self.drift_mode == "net":
                    pred_mu = self.drift_net(state.detach(), action.detach())
                    drift_loss = F.mse_loss(pred_mu, mu_fd_phys)
                    self.drift_optimizer.zero_grad()
                    drift_loss.backward()
                    self.drift_optimizer.step()
                    self._last_drift_loss = float(drift_loss.detach())

                # ----- 2. Add HJB term to critic loss (after warmup) -----
                # Skip the HJB term during warmup so the drift estimator
                # (especially DriftNet) has time to be roughly correct
                # before we use it to constrain the critic.
                if t > self.drift_warmup_steps:
                    state_g = state.detach().clone().requires_grad_(True)
                    q1_g = self.critic.Q1(state_g, action)
                    # grad_s Q1, shape [batch, 11]
                    grad_q = torch.autograd.grad(
                        outputs=q1_g.sum(), inputs=state_g,
                        create_graph=True, retain_graph=True,
                    )[0]
                    grad_q_phys = grad_q[:, self.phys_idx]
                    # Pick the drift estimate. DriftNet is detached so HJB
                    # only updates the critic, not the drift net.
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
            
            # Perform the actor update ---------------------------------------------------

            # Delayed policy updates
            if t % self.policy_freq == 0:

                # Compute the modfied actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / Q.abs().mean().detach()
                actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Show progress
            if t % self.training_progress_freq == 0:

                # show the updated loss; HJB residual is shown separately
                # so we can tell whether the regularisation is doing work.
                msg = 'Timesteps {} - Actor Loss {} - Critic Loss {}'.format(
                    t, actor_loss, critic_loss)
                if self.use_hjb:
                    msg += ' - HJB Loss {}'.format(self._last_hjb_loss)
                    if self.drift_mode == "net":
                        msg += ' - Drift Loss {}'.format(self._last_drift_loss)
                print(msg)
                self.save_model()
                
    """
    Test the learned weights against the PID controller.
    """
    def test_model(self, input_seed=0, input_max_timesteps=4800): 
        
        # initialise the environment
        env = gym.make(self.env_name)           
            
        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file)  
        
        # Process the replay --------------------------------------------------
        
        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )

        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas) 
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        self.max_action = float(((self.bas * 3) - self.action_mean) / self.action_std)
        self.init_model()          
              
        # load the learned model (continuous-time variant); pick the tag.
        if not self.use_hjb:
            tag = "ct"
        elif self.drift_mode == "net":
            tag = "ct_hjb_net"
        else:
            tag = "ct_hjb"
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'TD3_offline_BC_' + tag + '_weights')
        test_seed, max_timesteps = input_seed, input_max_timesteps
            
        # TESTING -------------------------------------------------------------------------------------------
        
        # test the algorithm's performance vs pid algorithm
        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing, 
            pid_run=False, params=self.params
        )
         
        # display the results
        create_graph(
            rl_reward=rl_reward, rl_blood_glucose=rl_bg, rl_action=rl_action, rl_insulin=rl_insulin,
            rl_meals=rl_meals, pid_reward=pid_reward, pid_blood_glucose=pid_bg, 
            pid_action=pid_action, params=self.params
        )