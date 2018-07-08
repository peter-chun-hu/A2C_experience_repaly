import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer


class A2C_ACKTR(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 weight_decay = 0,
                 max_grad_norm=None,
                 acktr=False,
                ):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha , weight_decay = weight_decay)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]

        action_shape = rollouts.actions.size()[-1]

        '''
        print('observations')
        print(rollouts.observations[:-1].shape)
        print(rollouts.observations.shape)
        print('states')
        print(rollouts.states[0].shape)
        print(rollouts.states.shape)
        print('masks')
        print(rollouts.masks[:-1].shape)
        print(rollouts.masks.shape)
        print('action')
        print(rollouts.actions.shape)
        print(rollouts.actions.view(-1, action_shape).shape)
        '''
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))
        
    
        
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

            self.optimizer.step()


        return value_loss.item(), action_loss.item(), dist_entropy.item()

    
    def update_experience_replay(self, rollouts_batch):
        obs_shape = rollouts_batch.observations.size()[3:]

        action_shape = rollouts_batch.actions.size()[-1]

        batch_size, num_steps, num_processes, _ = rollouts_batch.rewards.size()

        values = None
        action_log_probs = None
        dist_entropy = None
        states = None
        for i in range(batch_size):
            tmp_values, tmp_action_log_probs, tmp_dist_entropy, tmp_states = self.actor_critic.evaluate_actions(
                rollouts_batch.observations[i,:-1,:,:].view(-1, *obs_shape),
                rollouts_batch.states[i,0,:,:].view(-1, self.actor_critic.state_size),
                rollouts_batch.masks[i,:-1,:,:].view(-1, 1),
                rollouts_batch.actions[i,:,:,:].view(-1, action_shape))
            
            tmp_values = tmp_values.unsqueeze(0)
            tmp_action_log_probs = tmp_action_log_probs.unsqueeze(0)
            tmp_dist_entropy = tmp_dist_entropy.unsqueeze(0)
            tmp_states = tmp_states.unsqueeze(0)
            
            if values is None:
                values, action_log_probs, dist_entropy, states = tmp_values, tmp_action_log_probs, tmp_dist_entropy, tmp_states
            else:
                values =  torch.cat([values, tmp_values], 0)
                action_log_probs =  torch.cat([action_log_probs, tmp_action_log_probs], 0)
                dist_entropy =  torch.cat([dist_entropy, tmp_dist_entropy], 0)
                states =  torch.cat([states, tmp_states], 0)
                
                


        values = values.view(batch_size, num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(batch_size, num_steps, num_processes, 1)


        advantages = rollouts_batch.returns[:,:-1] - values
        
        value_loss = advantages.pow(2)

        value_loss = value_loss.view(batch_size,-1).mean()

        action_loss = -(advantages.detach() * action_log_probs)

        action_loss = action_loss.view(batch_size,-1).mean()

        
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        dist_entropy = dist_entropy.mean()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()    