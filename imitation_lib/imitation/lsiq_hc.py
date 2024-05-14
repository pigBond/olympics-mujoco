from copy import deepcopy
import torch
import numpy as np
from .lsiq_h import LSIQ_H
import torch.nn.functional as F
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from mushroom_rl.utils.torch import to_float_tensor


class LSIQ_HC(LSIQ_H):

    def __init__(self, H_tau, H_loss_mode="Huber", **kwargs):

        # call parent
        super().__init__(**kwargs)

        self._H_tau = to_parameter(H_tau)
        self._H_loss_mode = H_loss_mode     # either MSE or Huber

    def update_H_function(self, obs, action, next_obs, absorbing, gamma, is_expert):

        # calculate the squared reward of the current Q
        H = self._H_approximator(obs, action, output_tensor=True)
        with torch.no_grad():
            next_action, log_pi = self.policy.compute_action_and_log_prob_t(next_obs)
            Q_plcy = self._target_critic_approximator(obs, action, output_tensor=True)
            V_plcy = self.get_targetV(obs)
            y = (1 - absorbing) * gamma.detach() * torch.clip(V_plcy, self._Q_min,
                                                                                                 self._Q_max)

            reward_non_abs = torch.square(torch.clip(Q_plcy - y, -1/self._reg_mult, 1/self._reg_mult)).detach()
            reward_abs = torch.square(torch.clip(Q_plcy - y, self._Q_min, self._Q_max)).detach()

            squared_reg_reward_plcy = (1 - absorbing) * self._reg_mult * reward_non_abs \
                                      + absorbing * (1.0 - gamma.detach()) * self._reg_mult * reward_abs

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

        # calculate the target for the HC-function
        next_H = (self._target_H_approximator(next_obs, next_action, output_tensor=True).detach() +
                                              self._alpha.detach() * torch.unsqueeze(neg_log_pi, 1))
        target_H = squared_reg_reward_plcy + (1 - absorbing) * gamma * next_H

        # clip the target for numerical stability
        Q2_max = (1.0/self._reg_mult)**2 / (1 - gamma.detach())
        target_H = torch.clip(target_H, -1000, Q2_max+100)

        if self._H_loss_mode == "Huber":
            loss_H = F.huber_loss(H, target_H)
        elif self._H_loss_mode == "MSE":
            loss_H = F.mse_loss(H, target_H)
        else:
            raise ValueError("Unsupported H_loss %s" % self._H_loss_mode)

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

    def _update_all_targets(self):
        self._update_target(self._critic_approximator,
                            self._target_critic_approximator)
        self._update_target_H(self._H_approximator,
                              self._target_H_approximator)

    def _update_target_H(self, online, target):
        for i in range(len(target)):
            weights = self._H_tau() * online[i].get_weights()
            weights += (1 - self._H_tau.get_value()) * target[i].get_weights()
            target[i].set_weights(weights)
