#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:35:30 2022

"""

import numpy as np 
import copy, random, torch, gym, pickle, os
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
    def __init__(self, state_dim, action_dim, activation="softplus"):
        super(Critic, self).__init__()

        self.activation = activation

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def _act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        return F.softplus(x)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self._act(self.l1(sa))
        q1 = self._act(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self._act(self.l4(sa))
        q2 = self._act(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self._act(self.l1(sa))
        q1 = self._act(self.l2(q1))
        q1 = self.l3(q1)
        return q1


    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = self._act(self.l4(sa))
        q2 = self._act(self.l5(q2))
        q2 = self.l6(q2)
        return q2        
        
        
class td3_bc_phibe: 
    
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
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.alpha = params.get("alpha", 2.5)

        # PhiBE critic options. The residual is computed in the normalised
        # state coordinates returned by get_batch.
        self.phibe_mode = params.get("phibe_mode", "second_order")
        self.lambda_phibe = params.get("lambda_phibe", 0.1)
        self.dt = float(params.get("dt", 3.0))
        self.beta = float(params.get("beta", -np.log(self.gamma) / self.dt))
        self.critic_activation = params.get("critic_activation", "softplus")
        self.use_safe_phibe = params.get("use_safe_phibe", False)
        self.lambda_hypo = params.get("lambda_hypo", 4.0)
        self.lambda_near = params.get("lambda_near", 1.0)
        self.current_glucose_index = params.get("current_glucose_index", 8)
        self.save_tag = params.get("save_tag", "phibe")

        # Optional reward normalisation (思路 3). When enabled, get_batch
        # rescales rewards to zero-mean / unit-std using buffer statistics,
        # which keeps |Q| in a D4RL-like range and restores the BC vs Q
        # balance that paper alpha=2.5 was tuned for. Off by default so
        # the published baseline numbers stay reproducible.
        self.normalize_reward = params.get("normalize_reward", False)

        # DISPLAY
        self.pid_bg, self.pid_insulin, self.pid_action, self.pid_reward  = [], [], [], 0
        self.training_timesteps = params["training_timesteps"]
        self.num_train_steps = int(params.get("num_train_steps", self.training_timesteps))
        self.training_progress_freq = max(1, int(self.num_train_steps // 10))
        
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
        self.critic = Critic(self.state_size, self.action_size, activation=self.critic_activation).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    
    """
    Save the learned models.
    """
    def save_model(self):
        
        os.makedirs('./Models', exist_ok=True)
        prefix = './Models/' + str(self.env_name) + str(self.train_seed) + 'TD3_offline_BC_' + self.save_tag + '_weights'
        suffix = self.replay_name.split("-")[-1]
        torch.save(self.actor.state_dict(), prefix + '_actor' + suffix)
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
    

    def _safe_weight(self, state):
        if not self.use_safe_phibe:
            return torch.ones((state.shape[0], 1), device=state.device)
        idx = self.current_glucose_index
        bg = state[:, idx:idx + 1] * self.params["state_std"][idx] + self.params["state_mean"][idx]
        return (
            torch.ones_like(bg)
            + self.lambda_hypo * (bg < 70.0).float()
            + self.lambda_near * ((bg >= 70.0) & (bg < 90.0)).float()
        )

    def _phibe_residual_loss(self, state, action, reward, next_state, q_head):
        # Work in normalised state coordinates because get_batch returns
        # normalised state and next_state. This keeps scales consistent with
        # the critic input.
        s_req = state.detach().clone().requires_grad_(True)
        a_det = action.detach()
        q = q_head(s_req, a_det)

        grad_s = torch.autograd.grad(
            outputs=q.sum(), inputs=s_req, create_graph=True, retain_graph=True
        )[0]

        delta_s = (next_state.detach() - state.detach())
        drift = delta_s / self.dt
        drift_term = (grad_s * drift).sum(dim=1, keepdim=True)

        diffusion_term = torch.zeros_like(q)
        hess_diag_norm = torch.zeros((), device=state.device)
        if self.phibe_mode == "second_order":
            sigma_diag = (delta_s ** 2) / self.dt
            diag_terms = []
            for j in range(self.state_size):
                grad_j = grad_s[:, j].sum()
                hess_j = torch.autograd.grad(
                    outputs=grad_j, inputs=s_req, create_graph=True, retain_graph=True
                )[0][:, j:j + 1]
                diag_terms.append(hess_j)
            hess_diag = torch.cat(diag_terms, dim=1)
            diffusion_term = 0.5 * (sigma_diag * hess_diag).sum(dim=1, keepdim=True)
            hess_diag_norm = hess_diag.detach().norm(dim=1).mean()
        elif self.phibe_mode == "full_second_order":
            # Full Hessian contraction using Hessian-vector product:
            #   0.5 / dt * delta_s^T Hessian(Q) delta_s
            # This includes cross-dimension curvature terms without explicitly
            # materialising the full batch of state_dim x state_dim Hessians.
            directional_grad = (grad_s * delta_s).sum()
            hess_vec = torch.autograd.grad(
                outputs=directional_grad,
                inputs=s_req,
                create_graph=True,
                retain_graph=True,
            )[0]
            hess_quad = (hess_vec * delta_s).sum(dim=1, keepdim=True)
            diffusion_term = 0.5 * hess_quad / self.dt
            hess_diag_norm = hess_vec.detach().norm(dim=1).mean()

        residual = self.beta * q - reward - drift_term - diffusion_term
        weight = self._safe_weight(state)
        loss = (weight * residual.pow(2)).mean()
        diagnostics = {
            "phibe_loss": float(loss.detach()),
            "grad_s_norm": float(grad_s.detach().norm(dim=1).mean()),
            "hess_diag_norm": float(hess_diag_norm.detach()),
            "drift_term_abs": float(drift_term.detach().abs().mean()),
            "diffusion_term_abs": float(diffusion_term.detach().abs().mean()),
            "residual_abs": float(residual.detach().abs().mean()),
        }
        return loss, diagnostics

    """
    Train the model on a pre-collected sample of training data.
    """
    def train_model(self):
        
        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file) 
            
        # Process the replay --------------------------------------------------
        
        # unpackage the replay
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

        # 思路 3 reward normalisation (opt-in).
        if self.normalize_reward:
            self.params["reward_mean"] = float(reward_mean)
            self.params["reward_std"]  = float(reward_std)
            print(f"[normalize_reward] mean={reward_mean:.4f}  std={reward_std:.4f}")
        
        # initialise the networks
        self.init_model()
        
        print('Processing Complete.')             
        
        for t in range(1, self.num_train_steps + 1):
            
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

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + done * self.gamma * target_Q
                
            # Update the critic -------------------------------------------

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            td_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            phibe_loss = torch.zeros((), device=self.device)
            phibe_diag = None
            if self.phibe_mode in ("first_order", "second_order", "full_second_order") and self.lambda_phibe > 0:
                phibe_loss_q1, phibe_diag = self._phibe_residual_loss(
                    state, action, reward, next_state, self.critic.Q1
                )
                phibe_loss_q2, _ = self._phibe_residual_loss(
                    state, action, reward, next_state, self.critic.Q2
                )
                phibe_loss = phibe_loss_q1 + phibe_loss_q2
            critic_loss = td_loss + self.lambda_phibe * phibe_loss

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
                
                # show the updated loss
                
                if phibe_diag is not None:
                    print('Timesteps {} - Actor Loss {} - Critic Loss {} - TD Loss {} - PhiBE Loss {} - |grad_s| {:.4f} - |hess_diag| {:.4f} - |drift| {:.4f} - |diff| {:.4f}'.format(
                        t, actor_loss, critic_loss, td_loss, phibe_loss,
                        phibe_diag['grad_s_norm'], phibe_diag['hess_diag_norm'],
                        phibe_diag['drift_term_abs'], phibe_diag['diffusion_term_abs']), flush=True)
                else:
                    print('Timesteps {} - Actor Loss {} - Critic Loss {}'.format(t, actor_loss, critic_loss), flush=True)
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
              
        # load the learned model
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'TD3_offline_BC_weights')
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
