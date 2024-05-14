import torch
import numpy as np
from .iq_sac import IQ_SAC
import torch.nn.functional as F

from mushroom_rl.utils.torch import to_float_tensor


class LSIQ(IQ_SAC):

    def __init__(self, Q_max=1.0, Q_min =-1.0, loss_mode_exp="fix", Q_exp_loss=None,
                 treat_absorbing_states=False, target_clipping=True, lossQ_type="iq_like", **kwargs):

        # call parent
        super(LSIQ, self).__init__(**kwargs)

        self._Q_max = Q_max
        self._Q_min = Q_min
        self._loss_mode_exp = loss_mode_exp # or bootstrap
        self._Q_exp_loss = Q_exp_loss  
        self._treat_absorbing_states = treat_absorbing_states
        self._target_clipping = target_clipping
        self._lossQ_type = lossQ_type

    def _lossQ(self, obs, act, next_obs, absorbing, is_expert):
        if self._lossQ_type == "sqil_like":
            return self._lossQ_sqil_like(obs, act, next_obs, absorbing, is_expert)
        elif self._lossQ_type == "iq_like":
            return self._lossQ_iq_like(obs, act, next_obs, absorbing, is_expert)
        else:
            raise ValueError("Unsupported lossQ type %s" % self._lossQ_type)
        
    def _lossQ_iq_like(self, obs, act, next_obs, absorbing, is_expert):

        # 1st expert term of loss
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()
        absorbing = torch.unsqueeze(absorbing, 1)

        if self._target_clipping:
            y = (1 - absorbing) * gamma.detach() * torch.clip(next_v, self._Q_min, self._Q_max)
        else:
            y = (1 - absorbing) * gamma.detach() * next_v

        reward = (current_Q - y)
        exp_reward = reward[is_expert]

        if self._loss_mode_exp == "bootstrap":
            loss_term1 = - exp_reward.mean()
        elif self._loss_mode_exp == "fix":
            if self._Q_exp_loss == "MSE":
                loss_term1 = F.mse_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            elif self._Q_exp_loss == "Huber":
                loss_term1 = F.huber_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            elif self._Q_exp_loss is None:
                raise ValueError("If you choose loss_mode_exp == fix, you have to specify Q_exp_loss. Setting it to"
                                 "None is not valid.")
            else:
                raise ValueError(
                    "Choosen Q_exp_loss %s is not supported. Choose either MSE or Huber." % self._Q_exp_loss)

        # do the logging
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        # 2nd policy term for our loss
        V = self.getV(obs)
        value = (V - y)
        self.sw_add_scalar('V for policy on all states', V.mean(), self._iter)
        value_loss = value
        if self._plcy_loss_mode == "value":
            loss_term2 = value_loss.mean()
        elif self._plcy_loss_mode == "value_expert":
            value_loss_exp = value_loss[is_expert]
            loss_term2 = value_loss_exp.mean()
        elif self._plcy_loss_mode == "value_policy":
            value_loss_plcy = value_loss[~is_expert]
            loss_term2 = value_loss_plcy.mean()
        elif self._plcy_loss_mode == "q_old_policy":
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean()
        elif self._plcy_loss_mode == "value_q_old_policy":
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean() + value_loss.mean()
        elif self._plcy_loss_mode == "v0":
            value_loss_v0 = (1-gamma.detach()) * self.getV(obs[is_expert])
            loss_term2 = value_loss_v0.mean()
        elif self._plcy_loss_mode == "off":
            loss_term2 = 0.0
        else:
            raise ValueError("Undefined policy loss mode: %s" % self._plcy_loss_mode)

        # regularize
        chi2_loss = self.regularizer_loss(absorbing, reward, gamma, is_expert, treat_absorbing_states=self._treat_absorbing_states)

        loss_Q = loss_term1 + loss_term2 + chi2_loss
        self.update_Q_parameters(loss_Q)

        if self._iter % self._logging_iter == 0:
            grads = []
            for param in self._critic_approximator.model.network.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            norm = grads.norm(dim=0, p=2)
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, chi2_loss

    def _lossQ_sqil_like(self, obs, act, next_obs, absorbing, is_expert):
        
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()
        absorbing = torch.unsqueeze(absorbing, 1)
        if self._target_clipping:
            y = (1 - absorbing) * gamma.detach() * torch.clip(next_v, self._Q_min, self._Q_max)
        else:
            y = (1 - absorbing) * gamma.detach() * next_v

        # define the rewards
        if self._treat_absorbing_states:
            r_max = (1 - absorbing) * ((1 / self._reg_mult)) \
                    + absorbing * (1 / (1 - gamma.detach())) * ((1 / self._reg_mult))
            r_min = (1 - absorbing) * (-(1 / self._reg_mult))\
                    + absorbing * (1 / (1 - gamma.detach())) * (-(1 / self._reg_mult))
        else:
            r_max = torch.ones_like(absorbing) * ((1 / self._reg_mult))
            r_min = torch.ones_like(absorbing) * (-(1 / self._reg_mult))

        r_max = r_max[is_expert]
        r_min = r_min[~is_expert]

        # expert part
        if self._loss_mode_exp == "bootstrap":
            if self._Q_exp_loss == "MSE":
                loss_term1 = torch.mean(torch.square(current_Q[is_expert] - (r_max + y[is_expert])))
            elif self._Q_exp_loss == "Huber":
                loss_term1 = F.huber_loss(current_Q[is_expert], (r_max + y[is_expert]))
            else:
                raise ValueError("Unknown loss.")
        elif self._loss_mode_exp == "fix":
            if self._Q_exp_loss == "MSE":
                loss_term1 = F.mse_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            elif self._Q_exp_loss == "Huber":
                loss_term1 = F.huber_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            else:
                raise ValueError("Unknown loss.")
        else:
            raise ValueError("Unknown expert loss mode.")

        # policy part
        if self._plcy_loss_mode == "value":
            value = self.getV(obs)
            target = y
            r_min = torch.concat([r_min, torch.ones_like(r_min) * (-(1 / self._reg_mult))])
        elif self._plcy_loss_mode == "value_plcy":
            value = self.getV(obs[~is_expert])
            target = y[~is_expert]
        elif self._plcy_loss_mode == "q_old_policy":
            value = current_Q[~is_expert]
            target = y[~is_expert]

        if self._Q_exp_loss == "MSE":
            loss_term2 = torch.mean(torch.square(value - (r_min + target)))
        elif self._Q_exp_loss == "Huber":
            loss_term2 = F.huber_loss(value, (r_min + target))
        else:
            raise ValueError("Unknown loss.")

        # do the logging
        reward = (current_Q - y)
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        loss_Q = loss_term1 + loss_term2
        self.update_Q_parameters(loss_Q)

        grads = []
        for param in self._critic_approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, 0.0