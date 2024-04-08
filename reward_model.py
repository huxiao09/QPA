import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KCenterGreedy(obs, full_obs, num_new_sample, device):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs, device)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs, device):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, device,
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 mu=1,
                 weight_factor=1.0,
                 adv_mu=2,
                 path=None,
                 data_aug_ratio=1):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.device = device
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.path = path
        self.data_aug_ratio = data_aug_ratio
        self.count = 0
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        #self.buffer_seg1 = np.empty((self.capacity), dtype='O')
        #self.buffer_seg2 = np.empty((self.capacity), dtype='O')

        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_mask = np.ones((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.CEloss_ = nn.CrossEntropyLoss(reduction='none')
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.mu = mu
        self.weight_factor = weight_factor
        self.adv_mu = adv_mu
        self.obs_l = 0
        self.action_l = 0
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_rank_discriminator(self, x_1, x_2, disc):
        r_1 = self.r_hat_batch(x_1)
        r_2 = self.r_hat_batch(x_2)
        r_1 = torch.from_numpy(np.sum(r_1, axis=1)).float().to(self.device)
        r_2 = torch.from_numpy(np.sum(r_2, axis=1)).float().to(self.device)
        # r_hat = torch.cat([r_1, r_2], axis=-1)
        labels = 1*(r_1 < r_2)
        snip1 = x_1.reshape(x_1.shape[0], -1)
        snip1 = torch.from_numpy(snip1).float().to(self.device)
        snip2 = x_2.reshape(x_2.shape[0], -1)
        snip2 = torch.from_numpy(snip2).float().to(self.device)
        p = disc(snip1, snip2, labels)

        return p.cpu().detach().numpy()
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]

    def get_p_value(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.abs(np.mean(probs, axis=0) - 0.5)
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat_member_ndarray(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](x)

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def r_hat_disagreement(self, x):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member_ndarray(x, member=member).detach())
        r_hats = torch.cat(r_hats, axis=-1)

        return torch.mean(r_hats, axis=-1), torch.std(r_hats, axis=-1)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2


    def get_queries_part(self, mb_size=20, part=10):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), part
        img_t_1, img_t_2 = None, None
        
        # get train traj
        if len(self.inputs[-1]) < len_traj:
            train_inputs = np.array(self.inputs[-part-1:-1])
            train_targets = np.array(self.targets[-part-1:-1])
        else:
            train_inputs = np.array(self.inputs[-part:])
            train_targets = np.array(self.targets[-part:])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            if self.buffer_seg1.dtype == 'O':
                for i in range(sa_t_1.shape[0]):
                    self.buffer_seg1[self.buffer_index+i] = sa_t_1[i]
                for i in range(sa_t_2.shape[0]):
                    self.buffer_seg2[self.buffer_index+i] = sa_t_2[i]
            else:
                np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
                np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
    
    def put_unlabel_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            self.buffer_mask[self.buffer_index:self.capacity] = 0

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                self.buffer_mask[0:remain] = 0

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_mask[self.buffer_index:next_index] = 0
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1

        if self.path:
            sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
            r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
            sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
            r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
            np.save(sa_t_1_path, sa_t_1)
            np.save(r_t_1_path, r_t_1)
            np.save(sa_t_2_path, sa_t_2)
            np.save(r_t_2_path, r_t_2)
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
    def uniform_sampling(self, explore=False):
        # get queries
        if not explore:
            sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
                mb_size=self.mb_size)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
                mb_size=int(self.mb_size*explore))
            sa_t_1_, sa_t_2_, r_t_1_, r_t_2_ = self.get_queries_part(
                mb_size=int(self.mb_size*(1-explore)))
            sa_t_1 = np.concatenate([sa_t_1, sa_t_1_], axis=0)
            sa_t_2 = np.concatenate([sa_t_2, sa_t_2_], axis=0)
            r_t_1 = np.concatenate([r_t_1, r_t_1_], axis=0)
            r_t_2 = np.concatenate([r_t_2, r_t_2_], axis=0)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def unlabel_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, _, _ =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:int(self.mb_size*self.mu)]
        sa_t_1 = sa_t_1[top_k_index]
        sa_t_2 = sa_t_2[top_k_index]

        r_t_1 = self.r_hat_batch(sa_t_1)
        r_t_2 = self.r_hat_batch(sa_t_2)      
        
        # get labels
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
     
        if len(labels) > 0:
            self.put_unlabel_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, np.mean(ensemble_losses)

    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        for i in range(w):
            B, S, _ = r_hat1.shape
            length = np.random.randint(S-15, S-5+1, size=B)
            start_index_1 = np.random.randint(0, S+1-length)
            start_index_2 = np.random.randint(0, S+1-length)
            mask_1 = torch.zeros((B,S,1)).to(self.device)
            mask_2 = torch.zeros((B,S,1)).to(self.device)
            for b in range(B):
                mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
                mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1
            mask_1_.append(mask_1)
            mask_2_.append(mask_2)

        return torch.cat(mask_1_), torch.cat(mask_2_)

    def train_reward_iter(self, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = self.shuffle_dataset(max_len)
        
        total = 0
        
        start_index = 0
        for epoch in range(num_iters):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][start_index:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                if self.data_aug_ratio:
                    labels = labels.repeat(self.data_aug_ratio)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                if self.data_aug_ratio:
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                    r_hat1 = r_hat1.repeat(self.data_aug_ratio,1,1)
                    r_hat2 = r_hat2.repeat(self.data_aug_ratio,1,1)
                    r_hat1 = (mask_1*r_hat1).sum(axis=1)
                    r_hat2 = (mask_2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()

            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break;
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, np.mean(ensemble_losses)

    def reshape_input(self, sa_t_1):
        x = []
        for i in range(sa_t_1.shape[0]):
            x.append(sa_t_1[i])
        return np.concatenate(x)
    
    def compute_r(self, r_hat1, t_len):
        x = []
        index = 0
        for i in range(len(t_len)):
            x.append(r_hat1[index:index+t_len[i]].sum().reshape(1))
            index += t_len[i]
        return torch.cat(x).reshape(-1,1)
    
    def train_scl_reward(self, disc):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        num_label_ = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        loss_d_ = []
        ps = []
        masks_ = []
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss_r = 0.0
            loss_d = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                masks = self.buffer_mask[idxs] 
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                loss_A = self.CEloss_(r_hat, labels)
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                true_index  = np.where(masks == 1)[0]
                correct = (predicted[true_index] == labels[true_index]).sum().item()
                ensemble_acc[member] += correct

                snip1 = sa_t_1.reshape(sa_t_1.shape[0], -1)
                snip1 = torch.from_numpy(snip1).float().to(self.device)
                snip2 = sa_t_2.reshape(sa_t_2.shape[0], -1)
                snip2 = torch.from_numpy(snip2).float().to(self.device)
                p = disc(snip1, snip2, r_hat, loss_A.clone().detach())
                soft_mask = masks
                soft_mask[soft_mask > 0.9] = 0.9
                soft_mask[soft_mask < 0.1] = 0.1
                soft_mask = torch.from_numpy(soft_mask).float().to(self.device)
                loss_B = torch.nn.BCELoss(reduction='mean')(p, soft_mask.to(self.device))

                p1 = p.clone().detach().squeeze()
                ps.append(p1.cpu().numpy())
                masks_.append(masks.squeeze())

                sample_weights = torch.zeros_like(p1, dtype=torch.float, device=self.device)
                num_label = (masks > 0.5).sum()
                num_label_[member] += num_label
                num_unlabel = (masks < 0.5).sum()
                sample_weights[masks.squeeze() > 0.5] = (1 + self.weight_factor / p1[masks.squeeze() > 0.5]) / num_label
                sample_weights[masks.squeeze() < 0.5] = (1 - self.weight_factor * 1 / (1 - p1[masks.squeeze() < 0.5])) / num_unlabel
                loss_C = (sample_weights * loss_A).sum() / sample_weights.sum()
                ensemble_losses[member].append(loss_C.item())

                loss_r += loss_C
                loss_d += loss_B
                
            loss_r.backward()
            self.opt.step()

            disc.disc_optimizer.zero_grad()
            loss_d.backward()
            disc.disc_optimizer.step()

            loss_d_.append(loss_d.item())
        
        ensemble_acc = ensemble_acc / num_label_

        ps = np.concatenate(ps)
        masks_ = np.concatenate(masks_)
        label_ps = ps[masks_ > 0.5]
        unlabel_ps = ps[masks_ < 0.5]
        
        return ensemble_acc, np.mean(ensemble_losses), np.mean(loss_d_), np.mean(label_ps), np.mean(unlabel_ps)
    
    def relabel_unlabel(self):
        unlabel_index = np.where(self.buffer_mask[:self.buffer_index] == 0)[0]
        sa_t_1 = self.buffer_seg1[unlabel_index]
        sa_t_2 = self.buffer_seg2[unlabel_index]
        r_1 = self.r_hat_batch(sa_t_1)
        r_2 = self.r_hat_batch(sa_t_2)
        r_1 = np.sum(r_1, axis=1)
        r_2 = np.sum(r_2, axis=1)

        labels = 1*(r_1 < r_2)
        
        self.buffer_label[unlabel_index] = labels
    
    def get_s_a_l(self, index=1):
        # label_index = np.where(self.buffer_mask[:self.buffer_index] == 1)[0]
        # sa_t_1 = self.buffer_seg1[label_index]
        # sa_t_2 = self.buffer_seg2[label_index]
        # sa_t_1 = self.buffer_seg1[label_index]
        # sa_t_2 = self.buffer_seg2[label_index]
        sa_t_1 = self.buffer_seg1.reshape(-1, sa_t_1.shape[-1])
        sa_t_2 = self.buffer_seg2.reshape(-1, sa_t_2.shape[-1])
        obs_l_1, action_l_1 = np.hsplit(sa_t_1, [index])
        obs_l_2, action_l_2 = np.hsplit(sa_t_2, [index])
        self.obs_l = np.concatenate([obs_l_1, obs_l_2], axis=0)
        self.action_l = np.concatenate([action_l_1, action_l_2], axis=0)
    
    def sample(self, batch_size):
        sa_t_1 = self.buffer_seg1.reshape(-1, sa_t_1.shape[-1])
        sa_t_2 = self.buffer_seg2.reshape(-1, sa_t_2.shape[-1])
        sa_l = np.concatenate([sa_t_1, sa_t_2], axis=0)
        idxs = np.random.randint(0, self.sa_l.shape[0], size=batch_size)
        
        # obs_ls = torch.as_tensor(self.obs_l[idxs], device=self.device).float()
        # action_ls = torch.as_tensor(self.action_l[idxs], device=self.device)
        sa_l = torch.as_tensor(sa_l[idxs], device=self.device)

        return sa_l