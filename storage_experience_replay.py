import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBatch(object):
    def __init__(self):
        self.observations = None
        self.states = None
        self.rewards = None
        self.value_preds = None
        self.returns = None
        self.action_log_probs = None
        self.masks = None
        self.actions = None

class RolloutStorage(object):
    def __init__(self, buffer_size, num_steps, num_processes, obs_shape, action_space, state_size):
        
        # obs_shape = envs.observation_space.shape 
        self.observations = torch.zeros(buffer_size, num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(buffer_size, num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(buffer_size, num_steps, num_processes, 1)
        self.value_preds = torch.zeros(buffer_size, num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(buffer_size, num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(buffer_size, num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(buffer_size, num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(buffer_size, num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.memory_size = 0
        self.buffer_size = buffer_size
        self.exceed_buffer = False

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert_new_trajectory(self):
        if self.memory_size+1 > self.buffer_size:
            self.exceed_buffer = True
        self.memory_size = (self.memory_size+1)%self.buffer_size
        
    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        
        self.observations[self.memory_size, self.step + 1].copy_(current_obs)
        self.states[self.memory_size, self.step + 1].copy_(state)
        self.actions[self.memory_size, self.step].copy_(action)
        self.action_log_probs[self.memory_size, self.step].copy_(action_log_prob)
        self.value_preds[self.memory_size, self.step].copy_(value_pred)
        self.rewards[self.memory_size, self.step].copy_(reward)
        self.masks[self.memory_size, self.step + 1].copy_(mask)
        
        self.step = (self.step + 1) % self.num_steps

    def sample(self, batch_size, prioritized = False):
        if not self.exceed_buffer:
            max_batch_size = self.memory_size + 1
            if batch_size > max_batch_size:
                batch_size = max_batch_size                
            partial_return = self.returns[:max_batch_size]
            
            
        else:
            max_batch_size = self.buffer_size
            if batch_size > max_batch_size:
                batch_size = max_batch_size
            partial_return = self.returns
            
        if prioritized:
            all_return = partial_return.sum()
            trajectory_return = partial_return.view(max_batch_size , -1).sum(dim = 1)            
            prob = trajectory_return/all_return
            idx = np.random.choice(range(max_batch_size), batch_size, replace=False, p = prob.cpu().numpy())
            
        else:   
            idx = np.random.choice(range(max_batch_size), batch_size, replace=False)
        

        
        rollouts_batch = RolloutBatch()
        rollouts_batch.observations = self.observations[idx]
        rollouts_batch.states = self.states[idx]
        rollouts_batch.actions = self.actions[idx]
        rollouts_batch.action_log_probs = self.action_log_probs[idx]
        rollouts_batch.value_preds = self.value_preds[idx]
        rollouts_batch.rewards = self.rewards[idx]
        rollouts_batch.masks = self.masks[idx]
        rollouts_batch.returns = self.returns[idx]  
        
        return rollouts_batch
        
    '''
    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
    '''
    
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[self.memory_size, step, :, :] + gamma * self.value_preds[self.memory_size, step+1, :, :] * self.masks[self.memory_size, step+1, :, :] - self.value_preds[self.memory_size, step, :, :]
                gae = delta + gamma * tau * self.masks[self.memory_size, step+1, :, :] * gae
                self.returns[self.memory_size, step, :, :] = gae + self.value_preds[self.memory_size, step, :, :]
        else:
            self.returns[self.memory_size, -1, :, :] = next_value
            for step in reversed(range(self.rewards.size(1))):
                self.returns[self.memory_size, step, :, :] = self.returns[self.memory_size, step+1, :, :] * \
                    gamma * self.masks[self.memory_size, step+1, :, :] + self.rewards[self.memory_size, step, :, :]

