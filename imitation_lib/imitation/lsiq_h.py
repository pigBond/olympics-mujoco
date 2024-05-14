from copy import deepcopy
import torch
import numpy as np
from .lsiq import LSIQ
import torch.nn.functional as F
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from mushroom_rl.utils.torch import to_float_tensor


class LSIQ_H(LSIQ):

    def __init__(self, H_params=None, clip_expert_entropy_to_policy_max=True ,
                 max_H_policy_tau_down = 1e-4, max_H_policy_tau_up = 1e-2, **kwargs):

        # call parent
        super().__init__(**kwargs)
        
        # define the H function with the target
        target_H_params = deepcopy(H_params)
        self._H_approximator = Regressor(TorchApproximator,
                                         **H_params)
        self._target_H_approximator = Regressor(TorchApproximator,
                                                **target_H_params)
        self._clip_expert_entropy_to_policy_max = clip_expert_entropy_to_policy_max
        self._max_H_policy = None
        self._max_H_policy_tau_down = max_H_policy_tau_down
        self._max_H_policy_tau_up = max_H_policy_tau_up

        # define the optimizer for the H function
        net_params = self._H_approximator.model.network.parameters()
        self._H_optimizer = H_params["optimizer"]["class"](net_params, **H_params["optimizer"]["params"])

    def _lossQ_iq_like(self, obs, act, next_obs, absorbing, is_expert):
        
        # update Q according to lsiq_update
        loss_term1, loss_term2, chi2_loss = super()._lossQ_iq_like(obs, act, next_obs, absorbing, is_expert)

        # update the H function
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        self.update_H_function(obs, act, next_obs, absorbing, gamma.detach(), is_expert)

        return loss_term1, loss_term2, chi2_loss
    
    def _lossQ_sqil_like(self, obs, act, next_obs, absorbing, is_expert):

        # update Q according to lsiq_update
        loss_term1, loss_term2, chi2_loss = super(LSIQ_H, self)._lossQ_sqil_like(obs, act, next_obs, absorbing, is_expert)
        
        # update the H function
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        self.update_H_function(obs, act, next_obs, absorbing, gamma.detach(), is_expert)

        return loss_term1, loss_term2, chi2_loss

    def update_H_function(self, obs, action, next_obs, absorbing, gamma, is_expert):
        H = self._H_approximator(obs, action, output_tensor=True)
        with torch.no_grad():
            next_action, log_pi = self.policy.compute_action_and_log_prob_t(next_obs)

        # restrict the target H of the expert to the maximum one of the policy
        neg_log_pi = -log_pi
        if self._clip_expert_entropy_to_policy_max:
            if self._max_H_policy is None:
                self._max_H_policy = torch.max(neg_log_pi[~is_expert])
            else:
                curr_max_H_policy = torch.max(neg_log_pi[~is_expert])
                if curr_max_H_policy > self._max_H_policy:
                    self._max_H_policy = (1 - self._max_H_policy_tau_up) * self._max_H_policy + \
                                         self._max_H_policy_tau_up * curr_max_H_policy
                else:
                    self._max_H_policy = (1 - self._max_H_policy_tau_down) * self._max_H_policy + \
                                          self._max_H_policy_tau_down * curr_max_H_policy
            neg_log_pi[is_expert] = torch.clip(neg_log_pi[is_expert], self._max_H_policy, 100000)
        
        next_H = (self._target_H_approximator(next_obs, next_action, output_tensor=True).detach() +
                                              self._alpha.detach() * torch.unsqueeze(neg_log_pi, 1))
        target_H = (1 - absorbing) * gamma * next_H

        # clip the target for numerical stability
        target_H = torch.clip(target_H, -10000, 1000)
        loss_H = F.mse_loss(H, target_H)

        self._H_optimizer.zero_grad()
        loss_H.backward()
        self._H_optimizer.step()

        H = H.detach().cpu().numpy()
        log_pi = log_pi.detach().cpu().numpy()

        # do some additional logging
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('H function/Loss', loss_H, self._iter)
            self.sw_add_scalar('H function/H', np.mean(H), self._iter)
            self.sw_add_scalar('H function/H plcy', np.mean(H[~is_expert]), self._iter)
            self.sw_add_scalar('H function/H expert', np.mean(H[is_expert]), self._iter)
            self.sw_add_scalar('H function/H_step', np.mean(-log_pi), self._iter)
            self.sw_add_scalar('H function/H_step plcy', np.mean(-log_pi[~is_expert]), self._iter)
            self.sw_add_scalar('H function/H_step expert', np.mean(-log_pi[is_expert]), self._iter)

        return loss_H, H, log_pi

    def _actor_loss(self, state, action_new, log_prob):
        q = self._critic_approximator(state, action_new, output_tensor=True)
        H = self._H_approximator(state, action_new, output_tensor=True)
        soft_q = q + H
        return (self._alpha.detach() * log_prob - soft_q).mean()

    def getV(self, obs):
        with torch.no_grad():
            action, _ = self.policy.compute_action_and_log_prob_t(obs)
        current_V = self._critic_approximator(obs, action.detach().cpu().numpy(), output_tensor=True)
        return current_V

    def get_targetV(self, obs):
        with torch.no_grad():
            action, _ = self.policy.compute_action_and_log_prob_t(obs)
        target_V = self._target_critic_approximator(obs, action.detach().cpu().numpy(), output_tensor=True)
        return target_V

    def _update_all_targets(self):
        self._update_target(self._critic_approximator,
                            self._target_critic_approximator)
        self._update_target(self._H_approximator,
                            self._target_H_approximator)
