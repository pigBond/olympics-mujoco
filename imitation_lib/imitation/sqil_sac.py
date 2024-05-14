import numpy as np
import torch
from .iq_sac import IQ_SAC
from mushroom_rl.utils.torch import to_float_tensor


class SQIL(IQ_SAC):

    def __init__(self, R_min=0.0, R_max=1.0, plcy_loss_mode="plcy", **kwargs):

        super(SQIL, self).__init__(plcy_loss_mode=plcy_loss_mode, **kwargs)

        self._R_min = R_min
        self._R_max = R_max

    def iq_update(self, input_states, input_actions, input_n_states, input_absorbing, is_expert):
        """ This function overrides the respective function from iq. It makes only slight changes. """
        if self._iter % self._delay_Q == 0:
            lossQ = self._lossQ(input_states, input_actions, input_n_states, input_absorbing,
                                                  is_expert)
            if self._iter % self._logging_iter == 0:
                self.sw_add_scalar('IQ-Loss/LossQ', lossQ, self._iter)

        # update policy
        if self._replay_memory.size > self._warmup_transitions() and self._iter % self._delay_pi == 0:
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

        if self._iter % self._delay_Q == 0:
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _lossQ(self, obs, act, next_obs, absorbing, is_expert):
        """
        This function overrides the iq-loss and replaces it with the sqil loss.
        """
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()

        y = (1 - torch.unsqueeze(absorbing, 1)) * gamma.detach() * next_v

        # expert part of loss
        loss_Q = 0.5 * torch.mean(torch.square(current_Q[is_expert] - (self._R_max + y[is_expert])))

        # plcy part of loss
        if self._plcy_loss_mode == "value":
            V = self.getV(obs)
            loss_Q += 0.5 * torch.mean(torch.square(V - (self._R_min + y)))
        elif self._plcy_loss_mode == "value_plcy":
            V = self.getV(obs)
            loss_Q += 0.5 * torch.mean(torch.square(V[~is_expert] - (self._R_min + y[~is_expert])))
        elif self._plcy_loss_mode == "plcy":    # this is the true sqil objective for the policy.
            loss_Q += 0.5 * torch.mean(torch.square(current_Q[~is_expert] - (self._R_min + y[~is_expert])))

        loss_Q *= self._reg_mult

        if self._iter % self._logging_iter == 0:
            reward = (current_Q - y)
            self.sw_add_scalar('Action-Value/Q for expert', current_Q[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for expert', torch.square(current_Q[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q for policy', current_Q[~is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for policy',  torch.square(current_Q[~is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward', reward.mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Expert', reward[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Policy', reward[~is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/R_min', self._R_min, self._iter)
            # norm
            w = self._critic_approximator.get_weights()
            self.sw_add_scalar("Action-Value/Norm of Q net: ",np.linalg.norm(w), self._iter)
            self.sw_add_scalar('Targets/expert data', y[is_expert].mean(), self._iter)
            self.sw_add_scalar('Targets/policy data', y[~is_expert].mean(), self._iter)
            # log mean squared action
            self.sw_add_scalar('Actions/mean squared action expert (from data)', torch.square(act[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Actions/mean squared action expert (from policy)', np.square(self.policy.draw_action(obs[is_expert])).mean(), self._iter)
            self.sw_add_scalar('Actions/mean squared action policy', torch.square(act[~is_expert]).mean(), self._iter)
            self.sw_add_scalar('Actions/mean squared action both', torch.square(act).mean(), self._iter)

            # log mean of each action
            n_actions = len(act[0])
            for i in range(n_actions):
                self.sw_add_scalar('All Actions means/action %d expert' % i, act[is_expert, i].mean(),
                                   self._iter)
                self.sw_add_scalar('All Actions means/action %d policy' % i, act[~is_expert, i].mean(),
                                   self._iter)
                self.sw_add_scalar('All Actions variances/action %d expert' % i, torch.var(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions variances/action %d policy' % i, torch.var(act[~is_expert, i]),
                                   self._iter)

        self.update_Q_parameters(loss_Q)

        grads = []
        for param in self._critic_approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_Q
