import torch
import numpy as np
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from imitation_lib.imitation.iq_sac import IQ_SAC


class IQ_Offline(IQ_SAC):

    def __init__(self, **kwargs):

        if "regularizer_mode" in kwargs.keys():
            if kwargs["regularizer_mode"] != "exp":
                raise ValueError("This is the offline implementation of IQ, which expects the regularizer to take only"
                                 "samples from the expert.")
        else:
            kwargs["regularizer_mode"] = "exp"
        if "plcy_loss_mode" in kwargs.keys():
            if kwargs["plcy_loss_mode"] != "v0":
                raise ValueError("This is the offline implementation of IQ, which expects: plcy_loss_mode=\"v0\".")
        else:
            kwargs["plcy_loss_mode"] = "v0"

        super(IQ_Offline, self).__init__(**kwargs)

    def fit(self, dataset):
        raise AttributeError("This is the offline implementation of IQ, it is not supposed to use the fit function. "
                             "Use the fit_offline function instead.")

    def fit_offline(self, n_steps):

        for i in range(n_steps):

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            assert self._act_mask.size > 0, "IQ-Learn needs demo actions!"
            demo_obs, demo_act, demo_nobs, demo_absorbing = next(minibatch_generator(self._batch_size(),
                                                                 self._demonstrations["states"],
                                                                 self._demonstrations["actions"],
                                                                 self._demonstrations["next_states"],
                                                                 self._demonstrations["absorbing"]))

            # prepare data for IQ update
            input_states = to_float_tensor(demo_obs.astype(np.float32)[:, self._state_mask])
            input_actions = to_float_tensor(demo_act.astype(np.float32))
            input_n_states = to_float_tensor(demo_nobs.astype(np.float32)[:, self._state_mask])
            input_absorbing = to_float_tensor(demo_absorbing.astype(np.float32))
            is_expert = torch.ones(len(demo_obs), dtype=torch.bool)

            # make IQ update
            self.iq_update(input_states, input_actions, input_n_states, input_absorbing, is_expert)

            self._iter += 1
            self.policy.iter += 1

    def _lossQ(self, obs, act, next_obs, absorbing, is_expert):
        """
        Main contribution of the IQ-learn paper. This function is based on the repository of the paper:
        https://github.com/Div99/IQ-Learn
        """
        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()

        y = (1 - torch.unsqueeze(absorbing, 1)) * gamma.detach() * self._Q_Q_multiplier * next_v

        reward = (self._Q_Q_multiplier * current_Q - y)
        exp_reward = reward[is_expert]
        loss_term1 = -exp_reward.mean()

        # do the logging
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        V = self._Q_Q_multiplier * self.getV(obs)
        value = (V - y)
        self.sw_add_scalar('V for policy on all states', self._Q_Q_multiplier * V.mean(), self._iter)
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
            reward = (current_Q - y)
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean()
        elif self._plcy_loss_mode == "v0":
            value_loss_v0 = (1 - gamma.detach()) * self.getV(obs[is_expert])
            loss_term2 = value_loss_v0.mean()
        else:
            raise ValueError("Undefined policy loss mode: %s" % self._plcy_loss_mode)

        # regularize
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()
        # WARNING: TURNED OFF absorbing in regularization TODO: check if this works, if not go back
        # y = (1 - torch.unsqueeze(absorbing, 1)) * gamma.detach() * self._Q_Q_multiplier * next_v
        abs_mult = 1.0 if self._reg_no_absorbing else (1 - torch.unsqueeze(absorbing, 1))
        y = abs_mult * gamma.detach() * self._Q_Q_multiplier * next_v
        current_Q = self._Q_Q_multiplier * self._critic_approximator(obs, act, output_tensor=True)
        if self._turn_off_reg_absorbing:
            reward = (1 - torch.unsqueeze(absorbing, 1)) * (current_Q - y)
        else:
            reward = current_Q - y

        reg_multiplier = (1.0 / (1 - gamma.detach())) if self._normalized_val_func else 1.0
        if self._regularizer_mode == "exp_and_plcy":
            chi2_loss = torch.tensor(self._reg_mult) * reg_multiplier * (torch.square(reward)).mean()
        elif self._regularizer_mode == "exp":
            chi2_loss = torch.tensor(self._reg_mult) * reg_multiplier * (torch.square(reward[is_expert])).mean()
        elif self._regularizer_mode == "plcy":
            chi2_loss = torch.tensor(self._reg_mult) * reg_multiplier * (torch.square(reward[~is_expert])).mean()
        elif self._regularizer_mode == "value":
            V = self._Q_Q_multiplier * self.getV(obs)
            value = (V - y)
            chi2_loss = torch.tensor(self._reg_mult) * reg_multiplier * (torch.square(value)).mean()
        elif self._regularizer_mode == "exp_and_plcy_and_value":
            V = self._Q_Q_multiplier * self.getV(obs[is_expert])
            value = (V - y[is_expert])
            reward = torch.concat([reward, value])
            chi2_loss = torch.tensor(self._reg_mult) * reg_multiplier * (torch.square(reward)).mean()
        elif self._regularizer_mode == "off":
            chi2_loss = 0.0
        else:
            raise ValueError("Undefined regularizer mode %s." % (self._regularizer_mode))

        # Add Q penalty TODO: maybe remove, since it did not work that great
        if self._use_Q_regularizer:
            loss_Q_pen = self._Q_reg_mult * torch.mean(
                torch.square(current_Q - torch.ones_like(current_Q) * self._Q_reg_target))
        else:
            loss_Q_pen = 0.0

        # add gradient penalty if needed
        if self._gp_lambda > 0:
            with torch.no_grad():
                act_plcy, _ = self.policy.compute_action_and_log_prob_t(obs[is_expert])
            loss_gp = self._gradient_penalty(obs[is_expert], act[is_expert],
                                             obs[is_expert], act_plcy, self._gp_lambda)
        else:
            loss_gp = 0.0

        loss_Q = loss_term1 + loss_term2 + chi2_loss + loss_Q_pen + loss_gp
        self.update_Q_parameters(loss_Q)

        grads = []
        for param in self._critic_approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, chi2_loss

    def iq_update(self, input_states, input_actions, input_n_states, input_absorbing, is_expert):

        # update Q function
        if self._iter % self._delay_Q == 0:
            self.update_Q_function(input_states, input_actions, input_n_states, input_absorbing, is_expert)

        # update policy
        if self._iter % self._delay_pi == 0:
            self.update_policy(input_states, is_expert)

        if self._iter % self._delay_Q == 0:
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def logging_loss(self, current_Q, y, reward, is_expert, obs, act, absorbing):

        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Action-Value/Q for expert', self._Q_Q_multiplier * current_Q[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for expert', self._Q_Q_multiplier * torch.square(current_Q[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Expert', reward[is_expert].mean(), self._iter)

            Q_exp = current_Q[is_expert]
            Q_plcy = current_Q[~is_expert]
            abs_exp = absorbing[is_expert].bool()
            abs_plcy = absorbing[~is_expert].bool()
            self.sw_add_scalar('Action-Value/Q Absorbing state exp', torch.mean(Q_exp[abs_exp]), self._iter)
            self.sw_add_scalar('Action-Value/Q Absorbing state plcy', torch.mean(Q_plcy[abs_plcy]), self._iter)

            # norm
            w = self._critic_approximator.get_weights()
            self.sw_add_scalar("Action-Value/Norm of Q net: ",np.linalg.norm(w), self._iter)
            self.sw_add_scalar('Targets/expert data', y[is_expert].mean(), self._iter)
            # log mean squared action
            self.sw_add_scalar('Actions/mean squared action expert (from data)', torch.square(act[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Actions/mean squared action expert (from policy)', np.square(self.policy.draw_action(obs[is_expert])).mean(), self._iter)

            # log mean of each action
            n_actions = len(act[0])
            for i in range(n_actions):
                self.sw_add_scalar('All Actions means/action %d expert' % i, act[is_expert, i].mean(),
                                   self._iter)
                self.sw_add_scalar('All Actions variances/action %d expert' % i, torch.var(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions maxs/action %d expert' % i, torch.max(act[is_expert, i]),
                                   self._iter)
