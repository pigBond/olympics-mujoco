from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy


class IQ_Learn_Policy(Policy):
    def __init__(self, approximator, min_a, max_a, log_std_min, log_std_max, use_entropy_projec=False,
                 ent_projec_mode="linear", e_reduc=1e-5, e_thres=2.0):
        """
        Constructor.

        Args:
            approximator (Regressor): a regressor computing mean in given a
                state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.
            ent_projec_mode (str): entropy projection mode, which is either linearily decreasing or fixed.

        """
        self.use_mean = False   # if true the mean action is taken instead of sampling from Gaussian
        self._approximator = approximator
        self._output_shape = self._approximator.model.network.output_shape
        self._half_out_shape = self._output_shape // 2
        assert type(self._output_shape) == int, "Output shape needs to be an integer."
        assert 2 * self._half_out_shape == self._output_shape, "Output shape needs to be an even number."

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self.iter = 0
        self._use_entropy_projec = use_entropy_projec
        self._ent_projec_mode = ent_projec_mode
        self._e_reduc = e_reduc
        self._e_thresh = e_thres

        self._eps_log_prob = 1e-6

        use_cuda = self._approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            use_mean='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _eps_log_prob='primitive',
            _use_entropy_projec='primitive',
            _ent_projec_mode='primitive',
            _e_reduc='primitive',
            _e_thresh='primitive',
            iter='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state)
        if self.use_mean:
            a_raw = dist.mean
        else:
            a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._eps_log_prob).sum(dim=1)
            return a_true, log_prob
        else:
            return a_true

    def get_mu_log_sigma(self, state):
        mu_log_sigma = self._approximator.predict(state, output_tensor=True)
        if len(mu_log_sigma.size()) == 1:
            mu = mu_log_sigma[0: self._half_out_shape]
            log_sigma = mu_log_sigma[self._half_out_shape:self._output_shape]
        elif len(mu_log_sigma.size()) == 2:
            mu = mu_log_sigma[:, 0: self._half_out_shape]
            log_sigma = mu_log_sigma[:, self._half_out_shape:self._output_shape]
        else:
            raise ValueError("The shape of mu_log_sigma needs to be either one or two-dimensional, found ",
                             mu_log_sigma.size())

        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        # Entropy projection
        if self._use_entropy_projec:
            log_sigma = self.project_entropy(log_sigma)

        return mu, log_sigma

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu, log_sigma = self.get_mu_log_sigma(state)

        return torch.distributions.Normal(mu, log_sigma.exp())

    def get_e_lb(self):
        if self._ent_projec_mode == "linear":
            return self._e_thresh - self.iter * self._e_reduc
        elif self._ent_projec_mode == "fixed":
            return self._e_thresh
        else:
            raise ValueError("Unknown entropy projection mode: %s. Use either fixed or linear." % self._ent_projec_mode)

    def set_ent_lower_bound(self, e_lb):
        self._e_thresh = e_lb

    def project_entropy(self, log_sigma):
        ent = self.entropy_from_logsigma(log_sigma)
        e_lb = self.get_e_lb()
        projec_log_sigma = torch.ones_like(ent)
        projec_log_sigma[ent >= e_lb] = 0
        a_dim = self._half_out_shape
        return (log_sigma.T + projec_log_sigma * (e_lb - ent) / a_dim).T

    def entropy_from_logsigma(self, log_sigma):
        return self._half_out_shape * (0.5 + 0.5 * np.log(2 * np.pi)) + torch.sum(log_sigma, axis=-1)

    def sampled_entropy_t(self, state):
        _, log_prob = self.compute_action_and_log_prob_t(state, compute_log_prob=True)
        return torch.mean(-log_prob)

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """

        return torch.mean(self.distribution(state).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        weights = weights[:self._approximator.weights_size]
        self._approximator.set_weights(weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        return  self._approximator.get_weights()

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return self._approximator.model.network.parameters()

    def get_central_delta(self):
        return self._central_a,  self._delta_a


class IQ_SAC(DeepAC):
    """
    Implementation of IQ-Learn based on SAC.

    "IQ-Learn: Inverse soft-Q Learning for Imitation"
    Divyansh Garg, Shuvam Chakraborty,  Chris Cundy, Jiaming Song and Stefano Ermon (2021)

    The implementation of the critic update is based on the repository of the paper authors:
    https://github.com/Div99/IQ-Learn/blob/main/iq_learn
    """
    def __init__(self, mdp_info, actor_params,
                 actor_optimizer, critic_params, batch_size, sw, use_target,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, delay_pi=1, delay_Q=1,
                 reg_mult=(1 / (4 * 0.5)), log_std_min=-20, log_std_max=2, target_entropy=None, ext_normalizer=None,
                 critic_fit_params=None, demonstrations=None, state_mask=None,
                 learnable_alpha=False, init_alpha=0.001, plcy_loss_mode="value", regularizer_mode="exp_and_plcy",
                 logging_iter=1, gradient_penalty_lambda=0.0, n_fits=1,  train_policy_only_on_own_states=False,
                 use_cuda=False, treat_absorbing_states=False):

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] <= 1 # here it differs from sac, as we only take one critic

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_approximator = Regressor(TorchApproximator,
                                       **actor_params)
        self._iter = 1
        policy = IQ_Learn_Policy(actor_approximator,
                                 mdp_info.action_space.low,
                                 mdp_info.action_space.high,
                                 log_std_min,
                                 log_std_max)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        policy_parameters = actor_approximator.model.network.parameters()

        ### End of SAC constructor, start of IQ_Learn's

        # define the optimizer
        net_params = self._critic_approximator.model.network.parameters()
        self._critic_optimizer = critic_params["optimizer"]["class"](net_params, **critic_params["optimizer"]["params"])

        self._demonstrations = demonstrations
        assert demonstrations is not None, "No demonstrations have been loaded"

        self._state_mask = np.arange(demonstrations["states"].shape[1]) \
            if state_mask is None else np.array(state_mask, dtype=np.int64)

        # use target for critic update
        self._use_target = use_target

        # check if alpha should be learnable or not
        self._learnable_alpha = learnable_alpha
        self._log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        self._plcy_loss_mode = plcy_loss_mode
        self._regularizer_mode = regularizer_mode
        self._gp_lambda = gradient_penalty_lambda
        self._reg_mult = reg_mult
        self._use_cuda = use_cuda
        self._delay_pi = delay_pi
        self._delay_Q = delay_Q
        self._train_policy_only_on_own_states = train_policy_only_on_own_states
        self._n_fits = n_fits
        self._treat_absorbing_states = treat_absorbing_states

        self._logging_iter = logging_iter

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        self._epoch_counter = 1

        self.ext_normalizer = ext_normalizer
        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch',
            _reg_mult='primitive',
            ext_normalizer='pickle',
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):

        # add to replay memory
        self._replay_memory.add(dataset)

        if self._replay_memory.initialized:

            for i in range(self._n_fits):

                # sample batch from policy replay buffer
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._batch_size())

                # sample batch of same size from expert replay buffer and concatenate with samples from own policy
                demo_obs, demo_act, demo_nobs, demo_absorbing = next(minibatch_generator(state.shape[0],
                                                                     self._demonstrations["states"],
                                                                     self._demonstrations["actions"],
                                                                     self._demonstrations["next_states"],
                                                                     self._demonstrations["absorbing"]))

                # prepare data for IQ update
                input_states = to_float_tensor(np.concatenate([state, demo_obs.astype(np.float32)[:, self._state_mask]]))
                input_actions = to_float_tensor(np.concatenate([action, demo_act.astype(np.float32)]))
                input_n_states = to_float_tensor(np.concatenate([next_state,
                                                                 demo_nobs.astype(np.float32)[:, self._state_mask]]))
                input_absorbing = to_float_tensor(np.concatenate([absorbing, demo_absorbing.astype(np.float32)]))
                is_expert = torch.concat([torch.zeros(len(state), dtype=torch.bool),
                                          torch.ones(len(state), dtype=torch.bool)])

                # make IQ update
                self.iq_update(input_states, input_actions, input_n_states, input_absorbing, is_expert)

        self._iter += 1
        self.policy.iter += 1

    def iq_update(self, input_states, input_actions, input_n_states, input_absorbing, is_expert):

        # update Q function
        if self._iter % self._delay_Q == 0:
            self.update_Q_function(input_states, input_actions, input_n_states, input_absorbing, is_expert)

        # update policy
        if self._replay_memory.size > self._warmup_transitions() and self._iter % self._delay_pi == 0:
            self.update_policy(input_states, is_expert)

        if self._iter % self._delay_Q == 0:
            self._update_all_targets()

    def update_Q_function(self, input_states, input_actions, input_n_states, input_absorbing, is_expert):

        loss1, loss2, chi2_loss = self._lossQ(input_states, input_actions, input_n_states, input_absorbing,
                                              is_expert)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('IQ-Loss/Loss1', loss1, self._iter)
            self.sw_add_scalar('IQ-Loss/Loss2', loss2, self._iter)
            self.sw_add_scalar('IQ-Loss/Chi2 Loss', chi2_loss, self._iter)
            self.sw_add_scalar('IQ-Loss/Alpha', self._alpha, self._iter)

    def update_policy(self, input_states, is_expert):

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

        y = (1 - torch.unsqueeze(absorbing, 1)) * gamma.detach() * next_v

        reward = (current_Q - y)
        exp_reward = reward[is_expert]
        loss_term1 = -exp_reward.mean()

        # do the logging
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
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
            reward = (current_Q - y)
            reward_plcy = reward[~is_expert]
            loss_term2 = reward_plcy.mean()
        elif self._plcy_loss_mode == "v0":
            value_loss_v0 = (1-gamma.detach()) * self.getV(obs[is_expert])
            loss_term2 = value_loss_v0.mean()
        else:
            raise ValueError("Undefined policy loss mode: %s" % self._plcy_loss_mode)

        # regularize
        absorbing = torch.unsqueeze(absorbing, 1)
        chi2_loss = self.regularizer_loss(absorbing, reward, gamma, is_expert,
                                          treat_absorbing_states=self._treat_absorbing_states)

        # add gradient penalty if needed
        if self._gp_lambda > 0:
            loss_gp = self._gradient_penalty(obs[is_expert], act[is_expert],
                                             obs[~is_expert], act[~is_expert], self._gp_lambda)
        else:
            loss_gp = 0.0

        loss_Q = loss_term1 + loss_term2 + chi2_loss  + loss_gp
        self.update_Q_parameters(loss_Q)

        grads = []
        for param in self._critic_approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, chi2_loss

    def regularizer_loss(self, absorbing, reward, gamma, is_expert, treat_absorbing_states=False):
        # choose whether to treat absorbing states or not
        if treat_absorbing_states:
            reg_absorbing = absorbing
        else:
            reg_absorbing = torch.zeros_like(absorbing)

        if self._regularizer_mode == "exp_and_plcy":
            chi2_loss = ((1 - reg_absorbing) * torch.tensor(self._reg_mult) * torch.square(reward)
                         + reg_absorbing * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward))).mean()
        elif self._regularizer_mode == "exp":
            chi2_loss = ((1 - reg_absorbing[is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[is_expert])
                         + reg_absorbing[is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[is_expert]))).mean()
        elif self._regularizer_mode == "plcy":
            chi2_loss = ((1 - reg_absorbing[~is_expert]) * torch.tensor(self._reg_mult) * torch.square(reward[~is_expert])
                         + reg_absorbing[~is_expert] * (1.0 - gamma.detach()) * torch.tensor(self._reg_mult)
                         * (torch.square(reward[~is_expert]))).mean()
        elif self._regularizer_mode == "off":
            chi2_loss = 0.0
        else:
            raise ValueError("Undefined regularizer mode %s." % (self._regularizer_mode))

        return chi2_loss

    def update_Q_parameters(self, loss):
        loss = loss.mean()
        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()

    def getV(self, obs):
        with torch.no_grad():
            action, log_prob = self.policy.compute_action_and_log_prob_t(obs)
            log_prob = torch.unsqueeze(log_prob, 1)
        current_Q = self._critic_approximator(obs, action.detach().cpu().numpy(), output_tensor=True)
        current_V = current_Q - self._alpha.detach() * log_prob.detach()
        return current_V

    def get_targetV(self, obs):
        with torch.no_grad():
            action, log_prob = self.policy.compute_action_and_log_prob_t(obs)
            log_prob = torch.unsqueeze(log_prob, 1)
        target_Q = self._target_critic_approximator(obs, action.detach().cpu().numpy(), output_tensor=True)
        target_V = target_Q - self._alpha.detach() * log_prob.detach()
        return target_V

    def _actor_loss(self, state, action_new, log_prob):
        q = self._critic_approximator(state, action_new, output_tensor=True)
        return (self._alpha.detach() * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _gradient_penalty(self, obs1, action1, obs2, action2, lambda_=1.0):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1).cuda() if self._use_cuda else torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [len(obs1[0]), len(action1[0])], dim=1)
        q = self._critic_approximator(interpolated_state, interpolated_action, output_tensor=True)
        ones = torch.ones(q.size()).cuda() if self._use_cuda else torch.ones(q.size())
        gradient = torch.autograd.grad(outputs=q,
                                       inputs=interpolated,
                                       grad_outputs=[ones, ones],
                                       create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]

        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def logging_loss(self, current_Q, y, reward, is_expert, obs, act, absorbing):

        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Action-Value/Q for expert', current_Q[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for expert', torch.square(current_Q[is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q for policy', current_Q[~is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Q^2 for policy', torch.square(current_Q[~is_expert]).mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward', reward.mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Expert', reward[is_expert].mean(), self._iter)
            self.sw_add_scalar('Action-Value/Reward_Policy', reward[~is_expert].mean(), self._iter)

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
            self.sw_add_scalar('Targets/policy data', y[~is_expert].mean(), self._iter)
            # log mean squarred action
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
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d policy' % i, torch.min(act[~is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d expert' % i, torch.min(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions mins/action %d policy' % i, torch.min(act[~is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions maxs/action %d expert' % i, torch.max(act[is_expert, i]),
                                   self._iter)
                self.sw_add_scalar('All Actions maxs/action %d policy' % i, torch.max(act[~is_expert, i]),
                                   self._iter)

    def _update_all_targets(self):
        self._update_target(self._critic_approximator,
                            self._target_critic_approximator)

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    def sw_add_scalar(self, name, val, iter):
        if self._iter % self._logging_iter == 0:
            self._sw.add_scalar(name, val, self._iter)

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
