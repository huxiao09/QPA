#!/usr/bin/env python3
import numpy as np
import torch
import os
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra
import wandb

class Workspace(object):
    def __init__(self, cfg):
        if cfg.wandb:
            wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=cfg)
            wandb.run.name = f"{cfg.env}_{cfg.experiment}_{cfg.max_feedback}_{cfg.num_interact}_{cfg.seed}"
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
            self.env.seed(cfg.seed)
            # need to set seed here
            self.env.action_space.seed(cfg.seed)
            self.env.observation_space.seed(cfg.seed)
            self.env.goal_space.seed(cfg.seed)
        else:
            self.env = utils.make_env(cfg)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
            max_episode_len=self.env._max_episode_steps)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            device=cfg.device,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            max_size=cfg.max_reward_buffer_size,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            data_aug_ratio=cfg.data_aug_ratio)
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                episode_reward += self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                obs, reward, done, extra = self.env.step(action)
                
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        return average_true_episode_reward, average_episode_reward, success_rate

    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        # self.reward_model.size_segment += 5
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            labeled_queries = self.reward_model.uniform_sampling(self.cfg.explore)
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            if 'metaworld' in self.cfg.env:
                for _ in range(self.cfg.reward_update):
                    train_acc, reward_loss = self.reward_model.train_reward()
                    total_acc = np.mean(train_acc)
                    if total_acc > 0.97:
                        break;
            else:
                num_iters = int(np.ceil(self.cfg.reward_update*self.labeled_feedback/self.reward_model.train_batch_size))
                train_acc, reward_loss = self.reward_model.train_reward_iter(num_iters)
                total_acc = np.mean(train_acc)
            
            if self.cfg.wandb:
                wandb.log({'train_acc':total_acc, 'reward_loss':reward_loss}, step=self.step)


    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 

        interact_count = 0
        with tqdm(total=self.cfg.num_train_steps) as pbar:
            while self.step < self.cfg.num_train_steps:
                if done:
                    # evaluate agent periodically
                    if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                        ep_reward, ep_reward_hat, success_rate  = self.evaluate()
                        if self.cfg.wandb:
                            if self.log_success:
                                wandb.log({'episode_reward':ep_reward,
                                        'episode_reward_hat':ep_reward_hat,
                                        'success_rate': success_rate}, step=self.step)
                            else:
                                wandb.log({'episode_reward':ep_reward,
                                    'episode_reward_hat':ep_reward_hat}, step=self.step)
                    
                    obs = self.env.reset()
                    self.agent.reset()
                    done = False
                    episode_reward = 0
                    avg_train_true_return.append(true_episode_reward)
                    true_episode_reward = 0
                    if self.log_success:
                        episode_success = 0
                    episode_step = 0
                    episode += 1
                            
                # sample action for data collection
                if self.step < self.cfg.num_seed_steps:
                    action = self.env.action_space.sample()
                else:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=True)

                # run training update                
                if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                    # update schedule
                    if self.cfg.reward_schedule == 1:
                        frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                        if frac == 0:
                            frac = 0.01
                    elif self.cfg.reward_schedule == 2:
                        frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                    else:
                        frac = 1
                    self.reward_model.change_batch(frac)
                    
                    # update margin --> not necessary / will be updated soon
                    new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                    self.reward_model.set_teacher_thres_skip(new_margin)
                    self.reward_model.set_teacher_thres_equal(new_margin)
                    
                    # first learn reward
                    self.learn_reward(first_flag=1)
                    
                    # relabel buffer
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    
                    # reset Q due to unsuperivsed exploration
                    self.agent.reset_critic()
                    
                    # update agent
                    self.agent.update_after_reset(
                        self.replay_buffer, self.step, 
                        gradient_update=self.cfg.reset_update, 
                        policy_update=True)
                    
                    # reset interact_count
                    interact_count = 0
                elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                    # update reward function
                    if self.total_feedback < self.cfg.max_feedback:
                        if interact_count == self.cfg.num_interact:
                            # update schedule
                            if self.cfg.reward_schedule == 1:
                                frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                                if frac == 0:
                                    frac = 0.01
                            elif self.cfg.reward_schedule == 2:
                                frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                            else:
                                frac = 1
                            self.reward_model.change_batch(frac)
                            
                            # update margin --> not necessary / will be updated soon
                            new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                            self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                            self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                            
                            # corner case: new total feed > max feed
                            if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                                self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                                
                            self.learn_reward()

                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                            interact_count = 0

                    if (self.total_feedback < self.cfg.max_feedback):
                        size = min(self.cfg.max_reward_buffer_size, len(self.reward_model.inputs)-1)
                        self.agent.update_onpolicy_sample(self.replay_buffer, self.step, size, 1, self.cfg.her_ratio)
                    else:
                        self.agent.update(self.replay_buffer, self.step, 1)

                # unsupervised exploration
                elif self.step > self.cfg.num_seed_steps:
                    self.agent.update_state_ent(self.replay_buffer, self.step, 
                                                gradient_update=1, K=self.cfg.topK)
                    
                next_obs, reward, done, extra = self.env.step(action)
                reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

                # allow infinite bootstrap
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
                episode_reward += reward_hat
                true_episode_reward += reward
                
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                    
                # adding data to the reward training data
                self.reward_model.add_data(obs, action, reward, done)
                self.replay_buffer.add(
                    obs, action, reward_hat, 
                    next_obs, done, done_no_max)

                obs = next_obs
                episode_step += 1
                self.step += 1
                interact_count += 1
                pbar.update(1)
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        
@hydra.main(config_path='config/train_QPA.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()