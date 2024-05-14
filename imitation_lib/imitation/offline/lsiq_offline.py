import torch
from torch.functional import F
import numpy as np
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from imitation_lib.imitation.lsiq import LSIQ


class LSIQ_Offline(LSIQ):

    def __init__(self, loss_mode_exp="fix", regularizer_mode="off", **kwargs):

        if "plcy_loss_mode" in kwargs.keys():
            if kwargs["plcy_loss_mode"] != "v0":
                raise ValueError("This is the offline implementation of IQ, which expects: plcy_loss_mode=\"v0\".")
        else:
            kwargs["plcy_loss_mode"] = "v0"

        super().__init__(loss_mode_exp=loss_mode_exp, regularizer_mode=regularizer_mode, **kwargs)

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
            input_states = to_float_tensor(demo_obs.astype(np.float32)[:, self._state_mask], self._use_cuda)
            input_actions = to_float_tensor(demo_act.astype(np.float32), self._use_cuda)
            input_n_states = to_float_tensor(demo_nobs.astype(np.float32)[:, self._state_mask], self._use_cuda)
            input_absorbing = to_float_tensor(demo_absorbing.astype(np.float32), self._use_cuda)
            is_expert = torch.ones(len(demo_obs), dtype=torch.bool).cuda() if self._use_cuda else torch.ones(len(demo_obs), dtype=torch.bool)

            # make IQ update
            self.iq_update(input_states, input_actions, input_n_states, input_absorbing, is_expert)

            self._iter += 1
            self.policy.iter += 1

    def _lossQ(self, obs, act, next_obs, absorbing, is_expert):

        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()
        absorbing = torch.unsqueeze(absorbing, 1)
        y = (1 - absorbing) * gamma.detach() * self._Q_Q_multiplier * torch.clip(next_v, self._Q_min, self._Q_max)

        reward = (self._Q_Q_multiplier*current_Q - y)
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
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean()
        elif self._plcy_loss_mode == "value_q_old_policy":
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean() + value_loss.mean()
        elif self._plcy_loss_mode == "v0":
            #value_loss_v0 = (1-gamma.detach()) * self.getV(obs[is_expert])
            value_loss_v0 = (1-gamma.detach()) * self.getV(obs[is_expert])
            loss_term2 = value_loss_v0.mean()
        elif self._plcy_loss_mode == "off":
            loss_term2 = 0.0
        else:
            raise ValueError("Undefined policy loss mode: %s" % self._plcy_loss_mode)

        # regularize
        if self._regularizer_mode == "exp_and_plcy":
            chi2_loss = ((1 - absorbing) * torch.tensor(self._reg_mult) * torch.square(reward)
                         + self._abs_mult * absorbing * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward))).mean()
        elif self._regularizer_mode == "exp":
            chi2_loss = ((1 - absorbing[is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[is_expert])
                         + self._abs_mult * absorbing[is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[is_expert]))).mean()
        elif self._regularizer_mode == "plcy":
            chi2_loss = ((1 - absorbing[~is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[~is_expert])
                         + self._abs_mult * absorbing[~is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[~is_expert]))).mean()
        elif self._regularizer_mode == "value":
            V = self._Q_Q_multiplier * self.getV(obs)
            value = (V - y)
            chi2_loss = torch.tensor(self._reg_mult) * (torch.square(value)).mean()
        elif self._regularizer_mode == "exp_and_plcy_and_value":
            V = self._Q_Q_multiplier * self.getV(obs[is_expert])
            value = (V - y[is_expert])
            reward = torch.concat([reward, value])
            chi2_loss = torch.tensor(self._reg_mult) * (torch.square(reward)).mean()
        elif self._regularizer_mode == "off":
            chi2_loss = 0.0
        else:
            raise ValueError("Undefined regularizer mode %s." % (self._regularizer_mode))

        # add gradient penalty if needed
        if self._gp_lambda > 0:
            with torch.no_grad():
                act_plcy, _ = self.policy.compute_action_and_log_prob_t(obs[is_expert])
            loss_gp = self._gradient_penalty(obs[is_expert], act[is_expert],
                                             obs[is_expert], act_plcy, self._gp_lambda)
        else:
            loss_gp = 0.0

        loss_Q = loss_term1 + loss_term2 + chi2_loss + loss_gp
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
            self.update_policy(input_states, input_actions, is_expert)

        if self._iter % self._delay_Q == 0:
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def update_policy(self, input_states, input_actions, is_expert):

        if self._train_policy_only_on_own_states:
            policy_training_states = input_states[~is_expert]
        else:
            policy_training_states = input_states
        action_new, log_prob = self.policy.compute_action_and_log_prob_t(policy_training_states)
        loss = self._actor_loss(policy_training_states, action_new, log_prob)

        self._optimize_actor_parameters(loss)
        grads = []
        for param in self.policy._approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Gradients/Norm2 Gradient Q wrt. Pi-parameters', norm,
                               self._iter)
            self.sw_add_scalar('Actor/Loss', loss, self._iter)
            _, log_prob = self.policy.compute_action_and_log_prob_t(input_states)
            self.sw_add_scalar('Actor/Entropy Expert States', torch.mean(-log_prob[is_expert]).detach().item(),
                               self._iter)
            self.sw_add_scalar('Actor/Entropy Policy States', torch.mean(-log_prob[~is_expert]).detach().item(),
                               self._iter)
            _, logsigma = self.policy.get_mu_log_sigma(input_states[~is_expert])
            ent_gauss = self.policy.entropy_from_logsigma(logsigma)
            e_lb = self.policy.get_e_lb()
            self.sw_add_scalar('Actor/Entropy from Gaussian Policy States', torch.mean(ent_gauss).detach().item(),
                               self._iter)
            self.sw_add_scalar('Actor/Entropy Lower Bound', e_lb, self._iter)
            _, logsigma = self.policy.get_mu_log_sigma(input_states[is_expert])
            ent_gauss = self.policy.entropy_from_logsigma(logsigma)
            self.sw_add_scalar('Actor/Entropy from Gaussian Expert States', torch.mean(ent_gauss).detach().item(),
                               self._iter)
        if self._learnable_alpha:
            self._update_alpha(log_prob.detach())

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
