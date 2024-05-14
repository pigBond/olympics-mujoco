import numpy as np
import torch
from torch.nn import GaussianNLLLoss
import torch.nn.functional as F
from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.approximators.parametric import TorchApproximator
from imitation_lib.imitation.iq_sac import IQ_Learn_Policy


class BehavioralCloning(Agent):

    def __init__(self, mdp_info, actor_params, actor_optimizer, demonstrations, log_std_min=-20,
                 log_std_max=2, use_cuda=False, logging_iter=1, batch_size=32, sw=None):

        actor_approximator = Regressor(TorchApproximator,
                                       **actor_params)
        policy = IQ_Learn_Policy(actor_approximator,
                                 mdp_info.action_space.low,
                                 mdp_info.action_space.high,
                                 log_std_min,
                                 log_std_max)

        policy_parameters = actor_approximator.model.network.parameters()

        self._demonstrations = demonstrations
        self._optimizer = actor_optimizer['class'](policy_parameters, **actor_optimizer['params'])
        self._actor_loss = GaussianNLLLoss()
        self._use_cuda = use_cuda
        self._iter = 0
        self._batch_size = batch_size
        self._logging_iter = logging_iter

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None)    # dont need to be copyable, causes pickle error otherwise

        super(BehavioralCloning, self).__init__(mdp_info, policy)

    def fit(self, dataset):
        raise AttributeError("This is a behavior cloning algorithms, which is meant to run offline. It is not supposed"
                             "to use the fit function. Use the fit_offline function instead.")

    def fit_offline(self, n_steps):

        for i in range(n_steps):

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            demo_obs, demo_act, demo_nobs, demo_absorbing = next(minibatch_generator(self._batch_size,
                                                                 self._demonstrations["states"],
                                                                 self._demonstrations["actions"],
                                                                 self._demonstrations["next_states"],
                                                                 self._demonstrations["absorbing"]))

            # prepare tensors
            states = to_float_tensor(demo_obs, self._use_cuda) \
                if self._use_cuda else to_float_tensor(demo_obs)
            target_actions = to_float_tensor(demo_act, self._use_cuda) \
                if self._use_cuda else to_float_tensor(demo_act)

            # do unsquashing of target actions
            central, delta = self.policy.get_central_delta()
            target_actions = torch.clip((target_actions - central) / delta, -1.0 + 1e-7, 1.0 - 1e-7)
            target_actions = torch.arctanh(target_actions)

            # predict mu and log_sigma
            mu, log_sigma = self.policy.get_mu_log_sigma(states)

            # calculate loss and do an optimizer step
            loss = self._actor_loss(input=mu, target=target_actions, var=torch.square(log_sigma.exp()))
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # make some logging
            self.logging(states, target_actions, loss, mu, log_sigma)

            self._iter += 1

    def logging(self, states, target_actions, loss, mu, log_sigma):
        # log some useful information
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar("GaussianNLLLoss", np.mean(loss.detach().cpu().numpy()))

            gauss_ent = self.policy.entropy_from_logsigma(log_sigma)
            self.sw_add_scalar("Squashed Gaussian Entropy", np.mean(gauss_ent.detach().cpu().numpy()))
            act, log_prob = self.policy.compute_action_and_log_prob(states)
            squashed_gauss_ent = -np.mean(log_prob)
            self.sw_add_scalar("Squashed Gaussian Entropy (Empirical)", squashed_gauss_ent)

            mse_loss = F.mse_loss(mu, target_actions)
            self.sw_add_scalar("MSELoss (between mean & target actions)", np.mean(mse_loss.detach().cpu().numpy()))

    def sw_add_scalar(self, name, val):
        if self._iter % self._logging_iter == 0:
            self._sw.add_scalar(name, val, self._iter)
