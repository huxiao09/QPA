import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils
import hydra
import wandb

from agent import Agent
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor

def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(
                obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
            )
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, wandb,
                 normalize_state_entropy=True):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic_cfg = critic_cfg
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        self.actor_cfg = actor_cfg
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr
        self.wandb = wandb
        
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
            betas=alpha_betas)
        
        # change mode
        self.train()
        self.critic_target.train()

        self.critic_loss_last = 100
    
    def reset_critic(self):
        self.critic = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(self.critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr,
            betas=self.alpha_betas)
        
        # reset actor
        self.actor = hydra.utils.instantiate(self.actor_cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas)
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, step, weight=1, weighted_loss=False):
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        if step >= 360000:
            weighted_loss = False

        if weighted_loss:
            critic_loss = (weight * F.mse_loss(current_Q1, target_Q, reduction='none').flatten() + weight * F.mse_loss(
            current_Q2, target_Q, reduction='none').flatten()).mean()
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if self.wandb and step % 5000 == 0:
            wandb.log({'Q_loss':critic_loss}, step=step)
            if weighted_loss:
                wandb.log({'w_mean':weight.mean().item(), 'w_max':weight.max().item(), 
                           'w_min':weight.min().item()}, step=step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
        
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done,
        step, K=5):
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        
        # compute state entropy
        state_entropy = compute_state_entropy(obs, full_obs, k=K)

        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std

        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy
        
        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
    
    def update_actor_and_alpha(self, obs, step, weight=1, weighted_loss=False):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        if step >= 360000:
            weighted_loss = False

        if weighted_loss:
            actor_loss = ((self.alpha.detach() * log_prob - actor_Q) * weight).mean()
        else:
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        if self.wandb and step % 5000 == 0:
            wandb.log({'actor_loss':actor_loss}, step=step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
    def update(self, replay_buffer, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            critic_loss = self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
    
    def update_onpolicy_sample(self, replay_buffer, step, size, gradient_update=1, her_ratio=0.5):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                int(self.batch_size*(1-her_ratio)))
            obs_on, action_on, reward_on, next_obs_on, not_done_on, not_done_no_max_on = replay_buffer.sample_onpolicy(
                int(self.batch_size*(her_ratio)), size=size)
            
            obs = torch.cat([obs, obs_on], axis=0)
            action = torch.cat([action, action_on], axis=0)
            reward = torch.cat([reward, reward_on], axis=0)
            next_obs = torch.cat([next_obs, next_obs_on], axis=0)
            not_done_no_max = torch.cat([not_done_no_max, not_done_no_max_on], axis=0)

            critic_loss = self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
    
    def update_critic_only(self, replay_buffer, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
    
    def update_with_uncertainty(self, beta, disc, reward_model, replay_buffer, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)
            
            sa_t = torch.cat([obs, action], axis=-1)
            _, disagree = reward_model.r_hat_disagreement(sa_t)
            dl_dd = disc.predict(sa_t).flatten()

            weight = beta / disagree
            # weight[weight > 100] = 100
            weight = weight ** 0.2
            weight = weight / weight.mean()

            # dl_dd = dl_dd ** 0.2
            weight = dl_dd / dl_dd.mean()
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               step, weight, weighted_loss=True)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, step, weight, weighted_loss=True)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
    def update_after_reset(self, replay_buffer, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, step)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)
            
    def update_state_ent(self, replay_buffer, step, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
                self.batch_size)
                
            self.update_critic_state_ent(
                obs, full_obs, action, next_obs, not_done_no_max, step, K=K)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)