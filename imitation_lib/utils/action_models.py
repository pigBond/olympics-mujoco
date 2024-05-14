from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from mushroom_rl.core import Serializable
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.parameters import to_parameter

from imitation_lib.utils.distributions import InverseGamma

from itertools import chain


class SingleTensorGaussianNLLLoss(torch.nn.Module):
    """
    This class is based on torch's GaussianNLLLoss, yet takes a single tensor as input, which is divided during call.
    """

    def __init__(self, half_output_shape, output_shape):
        super().__init__()
        self._half_out_shape = half_output_shape
        self._output_shape = output_shape
        self._loss = torch.nn.GaussianNLLLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Divides the input tensor into mu and var and calculated the Gaussian negative log-likelihood.
        """
        if len(inputs.size()) == 1:
            mu = inputs[0: self._half_out_shape]
            log_sigma = inputs[self._half_out_shape:self._output_shape]
        elif len(inputs.size()) == 2:
            mu = inputs[:, 0: self._half_out_shape]
            log_sigma = inputs[:, self._half_out_shape:self._output_shape]
        else:
            raise ValueError("The shape of mu_log_sigma needs to be either one or two-dimensional, found ",
                             inputs.size())
        loss = self._loss(input=mu, var=torch.square(log_sigma.exp()), target=targets).mean()

        return loss


class MAP_Learnable_Var(torch.nn.Module):
    """
    This class is based on torch's GaussianNLLLoss, yet takes a single tensor as input, which is divided during call.
    """

    def __init__(self, mu_0, lam, alpha, beta, use_cuda, use_arctanh=True):
        super().__init__()
        self._mu_0 = to_float_tensor(mu_0, use_cuda)
        self._lam = to_float_tensor(lam, use_cuda)
        self._alpha = to_float_tensor(alpha, use_cuda)
        self._beta = to_float_tensor(beta, use_cuda)
        self._eps_log_prob = 1e-6
        self._use_arctanh = use_arctanh
        self._loss = torch.nn.GaussianNLLLoss()

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor, targets: torch.Tensor):
        """
        Divides the input tensor into mu and var and calculated the Gaussian negative log-likelihood.
        """
        # divide the mu's and sigma's into ones from policy and expert
        n_samples = len(targets)
        assert n_samples % 2 == 0, "The number of targets and inputs needs to be dividable by 2."
        mu_plcy = mu[0:n_samples//2, :]
        mu_exp = mu[n_samples//2:, :]
        log_sigma_plcy = log_sigma[0:n_samples//2, :]
        log_sigma_exp = log_sigma[n_samples//2:, :]
        targets_plcy = targets[0:n_samples//2, :]
        targets_exp = targets[n_samples//2:, :]     # these are not used, as they are only dummy actions

        #loss = self._loss(input=mu_plcy, var=torch.square(log_sigma_plcy.exp()), target=targets_plcy).mean()

        # get the likelihood log_prob
        targets_plcy = torch.clip(targets_plcy, -1+1e-6, 1-1e-6)
        targets_plcy = torch.arctanh(targets_plcy) if self._use_arctanh else targets_plcy
        L = torch.distributions.Normal(mu_plcy, log_sigma_plcy.exp()).log_prob(targets_plcy).sum(dim=1)
        #L -= torch.log(1. - targets_plcy.pow(2) + self._eps_log_prob).sum(dim=1)

        # get the probability on the sigma²
        p_sigma = InverseGamma(self._alpha, self._beta).log_prob(torch.square(log_sigma_exp.exp())).sum(dim=1)

        # get the probability on the mu
        mu_0 = torch.tile(self._mu_0, (mu_exp.shape[0], 1))
        p_mu = torch.distributions.Normal(mu_0, log_sigma_exp.exp()/torch.sqrt(self._lam)).log_prob(mu_exp).sum(dim=1)

        #return -(L.mean() + p_mu.mean() + p_sigma.mean())/3
        return -L.mean()


class MAP(torch.nn.Module):
    """
    This class is based on torch's GaussianNLLLoss, yet takes a single tensor as input, which is divided during call.
    """

    def __init__(self, mu_0, lam, alpha, beta, use_cuda, use_arctanh=True):
        super().__init__()

        self._mu_0 = to_float_tensor(mu_0, use_cuda)
        self._lam = to_float_tensor(lam, use_cuda)
        self._alpha = to_float_tensor(alpha, use_cuda)
        self._beta = to_float_tensor(beta, use_cuda)
        self._eps_log_prob = 1e-6
        self._use_arctanh = use_arctanh
        self._loss = torch.nn.GaussianNLLLoss()

    def forward_divided(self, mu: torch.Tensor, log_sigma, targets: torch.Tensor):
        """
        Divides the input tensor into mu and var and calculated the Gaussian negative log-likelihood.
        """
        # divide the mu's and sigma's into ones from policy and expert
        n_samples = len(targets)
        assert n_samples % 2 == 0, "The number of targets and inputs needs to be dividable by 2."
        mu_plcy = mu[0:n_samples//2, :]
        mu_exp = mu[n_samples//2:, :]
        log_sigma_plcy = log_sigma[0:n_samples//2, :]
        log_sigma_exp = log_sigma[n_samples//2:, :]
        targets_plcy = targets[0:n_samples//2, :]
        targets_exp = targets[n_samples//2:, :]     # these are not used, as they are only dummy actions

        #loss = self._loss(input=mu_plcy, var=torch.square(log_sigma_plcy.exp()), target=targets_plcy).mean()

        # get the likelihood log_prob
        targets_plcy = torch.clip(targets_plcy, -1+1e-6, 1-1e-6)
        targets_plcy = torch.arctanh(targets_plcy) if self._use_arctanh else targets_plcy
        L = torch.distributions.Normal(mu_plcy, log_sigma_plcy.exp()).log_prob(targets_plcy).sum(dim=1)
        #L -= torch.log(1. - targets_plcy.pow(2) + self._eps_log_prob).sum(dim=1)

        # get the probability on the sigma²
        p_sigma = InverseGamma(self._alpha, self._beta).log_prob(torch.square(log_sigma_exp.exp())).sum(dim=1)

        # get the probability on the mu
        mu_0 = torch.tile(self._mu_0, (mu_exp.shape[0], 1))
        p_mu = torch.distributions.Normal(mu_0, log_sigma_exp.exp()/torch.sqrt(self._lam)).log_prob(mu_exp).sum(dim=1)

        # # get the prior probability on the mu
        # mu_0 = torch.tile(self._mu_0, (mu_exp.shape[0], 1))
        # sigma_0 = torch.tile(self._log_sig_0.exp(), (mu_exp.shape[0], 1))
        # p_mu = torch.distributions.Normal(mu_0, sigma_0).log_prob(mu_exp).sum(dim=1)
        #
        # # get the prior probability on the sigma
        # p_sig = torch.distributions.Gamma(self._alpha, self._beta).log_prob(log_sigma_exp.exp()).sum(dim=1)

        return -(L.mean() + p_mu.mean() + p_sigma.mean())/3

    def forward(self, mu: torch.Tensor, log_sigma, targets: torch.Tensor):

        # get the likelihood log_prob
        targets = torch.arctanh(targets) if self._use_arctanh else targets
        L = torch.distributions.Normal(mu, log_sigma.exp()).log_prob(targets).sum(dim=1)
        #L -= torch.log(1. - targets_plcy.pow(2) + self._eps_log_prob).sum(dim=1)

        # get the probability on the sigma²
        p_sigma = InverseGamma(self._alpha, self._beta).log_prob(torch.square(log_sigma.exp())).sum(dim=1)

        # get the probability on the mu
        mu_0 = torch.tile(self._mu_0, (mu.shape[0], 1))
        p_mu = torch.distributions.Normal(mu_0, log_sigma.exp()/torch.sqrt(self._lam)).log_prob(mu).sum(dim=1)

        # # get the prior probability on the mu
        # mu_0 = torch.tile(self._mu_0, (mu_exp.shape[0], 1))
        # sigma_0 = torch.tile(self._log_sig_0.exp(), (mu_exp.shape[0], 1))
        # p_mu = torch.distributions.Normal(mu_0, sigma_0).log_prob(mu_exp).sum(dim=1)
        #
        # # get the prior probability on the sigma
        # p_sig = torch.distributions.Gamma(self._alpha, self._beta).log_prob(log_sigma_exp.exp()).sum(dim=1)

        return -(L.mean() + p_mu.mean() + p_sigma.mean())/3


class GCP(torch.nn.Module):

    def forward(self, mus, lams, alphas, betas, ys):
        # calculate the targets
        mus_tar = (lams*mus + ys) / (lams + 1.0)
        lams_tar = lams + 1
        alphas_tar = alphas + 0.5
        betas_tar = betas + (lams / (lams+1)) * ((ys - mus)**2 / 2)

        # calculate the KL
        kl = self.calc_KL_NIG(mus, lams, alphas, betas, mus_tar, lams_tar, alphas_tar, betas_tar)

        return torch.mean(kl)

    # def calc_KL_NIG_wrong(self, mus, lams, alphas, betas, mus_tar, lams_tar, alphas_tar, betas_tar):
    #     return 0.5 * (alphas_tar / betas_tar) * ((mus_tar - mus)**2) * lams\
    #            + 0.5 * (lams / lams_tar)\
    #            - 0.5 \
    #            + alphas * torch.log(betas_tar / betas)\
    #            - torch.lgamma(alphas_tar) + torch.lgamma(alphas)\
    #            + (alphas_tar - alphas) * torch.digamma(alphas_tar) \
    #            - (betas_tar - betas) * (alphas_tar / betas_tar)

    @staticmethod
    def calc_KL_NIG(mus, lams, alphas, betas, mus_tar, lams_tar, alphas_tar, betas_tar):
        return 0.5 * (alphas / betas) * ((mus_tar - mus)**2) * lams_tar \
               + 0.5 * (lams_tar / lams) \
               - 0.5 * torch.log(lams_tar / lams)\
               - 0.5\
               - alphas * torch.log(betas_tar / betas)\
               + torch.lgamma(alphas_tar) - torch.lgamma(alphas)\
               - (alphas_tar - alphas) * torch.digamma(alphas) \
               + (betas_tar - betas) * (alphas / betas)

    def prior_loss(self, mus, lams, alphas, betas, mus_old, lams_old, alphas_old, betas_old):
        # here we use reverse kl
        kl = self.calc_KL_NIG(mus, lams, alphas, betas, mus_old, lams_old, alphas_old, betas_old)
        return 1e-0*torch.mean(kl)


class DeepEvidentialLoss(torch.nn.Module):

    def forward(self, mus, lams, alphas, betas, ys, coeff=0):
        nll = self.StudentT_NLL(mus, lams, alphas, betas, ys)
        reg = self.NIG_Reg(mus, lams, alphas, betas, ys)
        return nll + coeff * reg

    def StudentT_NLL(self, mu, lam, alpha, beta, y, coeff=1e-2):
        twoBlambda = 2 * beta * (1 + lam)

        nll = 0.5 * torch.log(np.pi / lam) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(lam * (y - mu) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        return torch.mean(nll)

    def NIG_Reg(self, mu, lam, alpha, beta, y):
        # error = tf.stop_gradient(tf.abs(y-gamma))
        error = torch.abs(y - mu)
        evi = 2 * lam + (alpha)
        reg = error * evi
        return torch.mean(reg)

    def calc_KL_NIG(self, mus, lams, alphas, betas, mus_tar, lams_tar, alphas_tar, betas_tar):
        return 0.5 * (alphas_tar / betas_tar) * (mus_tar - mus)**2 * lams + 0.5 * (lams / lams_tar) - 0.5 \
               + alphas * torch.log(betas_tar / betas) - torch.lgamma(alphas_tar)\
               + torch.lgamma(alphas) + (alphas_tar - alphas) * torch.digamma(alphas_tar) \
               - (betas_tar - betas) * (alphas_tar / betas_tar)

    def calc_KL_NIG_wrong(self, mus, lams, alphas, betas, mus_tar, lams_tar, alphas_tar, betas_tar):
        return 0.5 * (alphas_tar / betas_tar) * (mus - mus_tar)**2 * lams + 0.5 * (lams / lams_tar) \
               - 0.5 * torch.log(lams / lams_tar) - 0.5 - alphas * torch.log(betas / betas_tar) + torch.lgamma(alphas)\
               - torch.lgamma(alphas_tar) - (alphas - alphas_tar) * torch.digamma(alphas_tar) \
               + (betas - betas_tar) * (alphas_tar / betas_tar)

    def prior_loss(self, mus, lams, alphas, betas, mus_old, lams_old, alphas_old, betas_old):
        kl = self.calc_KL_NIG(mus, lams, alphas, betas, mus_old, lams_old, alphas_old, betas_old)
        return 1e-3*torch.mean(kl)


class InvActionModel(Serializable):
    """
    Interface for different inverse action models.

    """
    def __init__(self, loss=None, init_target_approximator=False, tau=0.0, **params):

        self._approximator = Regressor(TorchApproximator, loss=loss, **params)
        if init_target_approximator:
            target_params = deepcopy(params)
            self._target_approximator = Regressor(TorchApproximator, loss=deepcopy(loss), **target_params)
            self._init_target(self._approximator, self._target_approximator)
        else:
            self._target_approximator = None

        self._tau = tau

        self._use_cuda = params["use_cuda"]
        self._loss = loss

        self._add_save_attr(
            _approximator='mushroom'
        )

    def draw_action(self, state, n_state):
        raise NotImplementedError

    @property
    def use_cuda(self):
        """
        True if the action model is using cuda_tensors.
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

    def fit(self, state, n_state, action, **fit_params):
        assert self._loss is not None, "In order to call the base fit method, you need to specify a loss!"
        state_nstate = np.concatenate([state, n_state], axis=1)
        self._approximator.fit(state_nstate, action, **fit_params)
        pred = self._approximator(state_nstate)
        loss = self._loss(to_float_tensor(pred, self._use_cuda), to_float_tensor(action, self._use_cuda))
        return loss

    @staticmethod
    def _init_target(online, target):
        for i in range(len(target)):
            target[i].set_weights(online[i].get_weights())

    def _update_target(self):
        for i in range(len(self._target_approximator)):
            weights = self._tau * self._approximator[i].get_weights()
            weights += (1 - self._tau) * self._target_approximator[i].get_weights()
            self._target_approximator[i].set_weights(weights)


class GaussianInvActionModel(InvActionModel):

    def __init__(self, min_a, max_a, loss_params, optimizer, log_std_min=-5, log_std_max=2, **params):
        """
        Constructor.

        Args:
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the model log std;
            log_std_max ([float, Parameter]): max value for the model log std.

        """
        # call parent constructor
        super().__init__(**params)

        self._action_model_loss = loss_params["type"](**loss_params["params"])
        self._action_model_optimizer = optimizer["class"](params=self._approximator.model.network.parameters(),
                                                          **optimizer["params"])
        self._output_shape = params["output_shape"][0]
        self._half_out_shape = self._output_shape // 2
        assert type(self._output_shape) == int, "Output shape needs to be an integer."
        assert 2 * self._half_out_shape == self._output_shape, "Output shape needs to be an even number."
        self._use_cuda = params["use_cuda"]

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self.use_mean = False  # if true the mean action is taken instead of sampling from Gaussian

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        if self.use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            use_mean='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _eps_log_prob='primitive'
        )

    def __call__(self, state, n_state):
        return self.draw_action(state, n_state)

    def draw_action(self, state, n_state):
        return self.compute_action_and_log_prob_t(
            state, n_state, compute_log_prob=False).detach().cpu().numpy()

    def fit(self, state, n_state, action, **fit_params):
        if "n_epochs" in fit_params:
            n_epochs = fit_params["n_epochs"]
        else:
            n_epochs = 1

        # do unsquashing
        action = torch.tensor(action).cuda().cuda() if self._use_cuda else torch.tensor(action)
        action = torch.clip((action-self._central_a)/self._delta_a, -1.0 + 1e-7, 1.0 - 1e-7)
        action = torch.arctanh(action)
        losses = []
        for epoch in range(n_epochs):
            mu, log_sigma = self.get_mu_log_sigma(state, n_state)
            loss = self._action_model_loss(input=mu, target=action, var=torch.square(log_sigma.exp()))

            self._action_model_optimizer.zero_grad()
            loss.backward()
            self._action_model_optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        return np.mean(losses)

    def compute_action_and_log_prob(self, state, n_state, use_target=False):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.
            n_state (np.ndarray): the next state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state, n_state, use_target=use_target)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, n_state, compute_log_prob=True, use_target=False):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            n_state (np.ndarray): the next state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state, n_state, use_target=use_target)
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

    def get_mu_log_sigma(self, state, n_state, use_target=False):

        state_nstate = to_float_tensor(np.concatenate([state, n_state], axis=1))
        if not use_target:
            mu_log_sigma = self._approximator.predict(state_nstate, output_tensor=True)
        else:
            mu_log_sigma = self._target_approximator.predict(state_nstate, output_tensor=True)

        # divide mu_log_sigma
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

        return mu, log_sigma

    def distribution(self, state, n_state, use_target=False):
        """
        Compute the distribution for the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.
            n_state (np.ndarray): the set of next states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu, log_sigma = self.get_mu_log_sigma(state, n_state, use_target=use_target)
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state, n_state):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.
            n_state (np.ndarray): the set of next states to consider.

        Returns:
            The value of the entropy of the policy.

        """
        return torch.mean(self.distribution(state, n_state).entropy()).detach().cpu().numpy().item()

    def get_mse(self, state, n_state, action):
        state_nstate = np.concatenate([state, n_state], axis=1)
        mu_log_sigma = self._approximator.predict(state_nstate, output_tensor=True)
        mu, log_sigma = self.divide_mu_log_var_t(mu_log_sigma)
        loss = F.mse_loss(mu, to_float_tensor(action))
        return loss, torch.mean(log_sigma.exp())

    def get_std(self, state, n_state):
        state_nstate = np.concatenate([state, n_state], axis=1)
        mu_log_sigma = self._approximator.predict(state_nstate, output_tensor=True)
        mu, log_sigma = self.divide_mu_log_var_t(mu_log_sigma)
        return log_sigma.exp()

    def divide_mu_log_var_t(self, mu_log_sigma):
        if len(mu_log_sigma.size()) == 1:
            mu = mu_log_sigma[0: self._half_out_shape]
            log_sigma = mu_log_sigma[self._half_out_shape:self._output_shape]
        elif len(mu_log_sigma.size()) == 2:
            mu = mu_log_sigma[:, 0: self._half_out_shape]
            log_sigma = mu_log_sigma[:, self._half_out_shape:self._output_shape]
        else:
            raise ValueError("The shape of mu_log_sigma needs to be either one or two-dimensional, found ",
                             mu_log_sigma.size())
        return mu, log_sigma


class KLGaussianInvActionModel(GaussianInvActionModel):

    def __init__(self, update_mode="hard", n_hard=1, reg_const=1e-5, **params):
        super(KLGaussianInvActionModel, self).__init__(init_target_approximator=True, **params)

        self._n_hard = n_hard
        if update_mode == "hard":
            self._update_target_method = self._update_target_hard
        elif update_mode == "soft":
            self._update_target_method = self._update_target
        else:
            raise ValueError("Unknown target update mode %s." % update_mode)

        self._demonstrations = params["demonstrations"]

        self._reg_const = reg_const
        self._iter = 1

    def fit(self, state, n_state, action, **fit_params):
        if "n_epochs" in fit_params:
            n_epochs = fit_params["n_epochs"]
        else:
            n_epochs = 1

        # sample batch of same size from expert replay buffer and concatenate with samples from own policy
        demo_obs, demo_nobs = next(minibatch_generator(state.shape[0],
                                                       self._demonstrations["states"],
                                                       self._demonstrations["next_states"]))

        # do unsquashing
        action = torch.tensor(action).cuda().cuda() if self._use_cuda else torch.tensor(action)
        action = torch.clip((action-self._central_a)/self._delta_a, -1.0 + 1e-7, 1.0 - 1e-7)
        action = torch.arctanh(action)

        losses = []
        for epoch in range(n_epochs):

            # calculate main loss
            mu, log_sigma = self.get_mu_log_sigma(state, n_state)
            loss = self._action_model_loss(input=mu, target=action, var=torch.square(log_sigma.exp()))

            # calculate KL regularization on expert states
            dist_cur = self.distribution(demo_obs, demo_nobs)
            dist_tar = self.distribution(demo_obs, demo_nobs, use_target=True)
            loss_exp = torch.distributions.kl.kl_divergence(dist_cur, dist_tar)
            loss += self._reg_const * torch.mean(loss_exp)

            # update target networks
            self._update_target_method()

            # update online action model params
            self._action_model_optimizer.zero_grad()
            loss.backward()
            self._action_model_optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            self._iter += 1

        return np.mean(losses)

    def _update_target_hard(self):
        if self._iter % self._n_hard == 0:
            self._init_target(self._approximator, self._target_approximator)


class LearnableVarGaussianInvActionModel(InvActionModel):
    """
    Torch action model implementing a Gaussian distribution with trainable standard
    deviation. The standard deviation is not state-dependent.

    """
    def __init__(self, std_0=1.0, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            std_0 (float, 1.): initial standard deviation;
            params (dict): parameters used by the network constructor.

        """
        self._input_shape = params["input_shape"][0]
        self._output_shape = params["output_shape"][0]
        self._action_dim = self._output_shape

        log_sigma_init = (torch.ones(self._action_dim) * np.log(std_0)).float()

        self._use_cuda = params["use_cuda"]
        if self._use_cuda:
            log_sigma_init = log_sigma_init.cuda()

        self._log_sigma = torch.nn.Parameter(log_sigma_init)

        loss_params = params["loss_params"]
        loss = loss_params["type"]
        if loss == SingleTensorGaussianNLLLoss:
            train_loss = loss(half_output_shape=self._half_out_shape, output_shape=self._output_shape)
        elif loss == MAP_Learnable_Var:
            alpha = loss_params["alpha"]
            beta = loss_params["beta"]
            mu_0 = loss_params["mu_0"]
            lam = loss_params["lam"]
            train_loss = loss(mu_0=mu_0, lam=lam, alpha=alpha, beta=beta, use_cuda=self._use_cuda)
        else:
            raise ValueError("Unexpected class %s for loss." % loss)

        # call parent constructor
        super().__init__(train_loss, **params)

        self._add_save_attr(
            _action_dim='primitive',
            _mu='mushroom',
            _predict_params='pickle',
            _log_sigma='torch'
        )

    def draw_action(self, state, nstate):
        return self.draw_action_t(state, nstate).cpu().numpy()

    def draw_action_t(self, state, nstate):
        return self.distribution_t(state, nstate).sample().detach()

    def log_prob_t(self, state, nstate, action):
        return self.distribution_t(state, nstate).log_prob(action)[:, None]

    def entropy_t(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def distribution_t(self, state, nstate):
        mu, sigma = self.get_mean_and_covariance(state, nstate)
        #return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        log_sigma = torch.tile(self._log_sigma, (mu.shape[0], 1))
        return torch.distributions.Normal(mu, log_sigma.exp())

    def get_mean_and_covariance(self, state, nstate):
        state_nstate = to_float_tensor(np.concatenate([state, nstate], axis=1))
        return self._approximator(state_nstate, output_tensor=True), torch.diag(torch.exp(2 * self._log_sigma))

    def get_mu_log_sigma(self, state, nstate):
        state_nstate = to_float_tensor(np.concatenate([state, nstate], axis=1))
        mu = self._approximator(state_nstate, output_tensor=True)
        log_sigma = torch.tile(self._log_sigma, (mu.shape[0], 1))
        return mu, log_sigma

    def set_weights(self, weights):
        log_sigma_data = torch.from_numpy(weights[-self._action_dim:])
        if self.use_cuda:
            log_sigma_data = log_sigma_data.cuda()
        self._log_sigma.data = log_sigma_data

        self._approximator.set_weights(weights[:-self._action_dim])

    def get_weights(self):
        mu_weights = self._approximator.get_weights()
        sigma_weights = self._log_sigma.data.detach().cpu().numpy()

        return np.concatenate([mu_weights, sigma_weights])

    def parameters(self):
        return chain(self._approximator.model.network.parameters(), [self._log_sigma])


class FixedVarGaussianInvActionModel(InvActionModel):
    """
    Fixed variance action model. NOTE: By default the std is zero resulting in a deterministic model!
    """
    def __init__(self, noise_std=0.0, noise_clip=None, **params):
        self._noise_std = noise_std
        self._noise_clip = noise_clip
        self._output_shape = params["output_shape"][0]
        loss = torch.nn.MSELoss()

        # call parent constructor
        super().__init__(loss, **params)

        self._add_save_attr(
            _noise_std="primitive",
            _noise_clip="primitive",
        )

    def __call__(self, state, n_state):
        return self.draw_action(state, n_state)

    def draw_action(self, state, n_state):
        state_nstate = to_float_tensor(np.concatenate([state, n_state], axis=1))
        act = self._approximator(state_nstate)
        noise = np.random.normal(loc=0.0, scale=self._noise_std,
                                 size=np.size(act)).reshape(act.shape)
        noise = np.clip(noise, -self._noise_clip, self._noise_clip) \
            if self._noise_clip is not None else noise
        act += noise
        return act

    def entropy(self, state, n_state):
        if self._noise_std > 0.0:
            D = self._output_shape
            det_cov = np.power(np.square(self._noise_std), D)
            ent = (D/2) * (1+np.log(2 * np.pi)) + 0.5 * np.log(det_cov)
            return ent
        else:
            return -10000


class GCPActionModel(InvActionModel):

    def __init__(self, min_a, max_a, loss_params, optimizer, log_std_min=-5, log_std_max=2,
                 lam_norm_const=1e-3, alpha_norm_const=1e-3, beta_norm_const=1e-3, reg_const=1e-5, **params):
        """
        Constructor.

        Args:
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the model log std;
            log_std_max ([float, Parameter]): max value for the model log std.

        """
        # call parent constructor
        super().__init__(init_target_approximator=True, **params)

        self._action_model_loss = loss_params["type"](**loss_params["params"])
        self._action_model_optimizer = optimizer["class"](params=self._approximator.model.network.parameters(),
                                                          **optimizer["params"])

        self._demonstrations = params["demonstrations"]

        self._use_cuda = params["use_cuda"]

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self.use_mean = False  # if true the mean action is taken instead of sampling from Gaussian

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6
        self._alpha_norm_const = alpha_norm_const
        self._beta_norm_const = beta_norm_const
        self._lam_norm_const = lam_norm_const
        self._reg_const = reg_const

        if self.use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            use_mean='primitive',
            _output_shape='primitive',
            _half_out_shape='primitive',
            _eps_log_prob='primitive'
        )

    def __call__(self, state, n_state):
        return self.draw_action(state, n_state)

    def fit(self, state, n_state, action, **fit_params):
        if "n_epochs" in fit_params:
            n_epochs = fit_params["n_epochs"]
        else:
            n_epochs = 1

        # sample batch of same size from expert replay buffer and concatenate with samples from own policy
        demo_obs, demo_nobs = next(minibatch_generator(state.shape[0],
                                                       self._demonstrations["states"],
                                                       self._demonstrations["next_states"]))

        a_exp = self.draw_raw_action(demo_obs, demo_nobs)
        a_exp = torch.tensor(a_exp).cuda() if self._use_cuda else torch.tensor(a_exp)
        action = torch.tensor(action).cuda().cuda() if self._use_cuda else torch.tensor(action)
        action = torch.clip((action-self._central_a)/self._delta_a, -1.0+1e-7, 1.0-1e-7)
        action = torch.arctanh(action)

        losses = []
        for epoch in range(n_epochs):
            mu, lam, alpha, beta = self.get_prior_params(state, n_state)
            loss = self._action_model_loss(mu, lam, alpha, beta, action)
            mu_exp, lam_exp, alpha_exp, beta_exp = self.get_prior_params(demo_obs, demo_nobs, use_target=False)
            loss_exp = self._action_model_loss(mu_exp, lam_exp, alpha_exp, beta_exp, a_exp)
            loss += self._reg_const * loss_exp
            loss += self._alpha_norm_const * torch.norm(alpha)
            loss += self._beta_norm_const * torch.norm(beta)
            loss += self._lam_norm_const * torch.norm(lam)

            self._action_model_optimizer.zero_grad()
            loss.backward()
            self._action_model_optimizer.step()

            self._update_target()

            losses.append(loss.detach().cpu().numpy())

        return np.mean(losses)

    def draw_action(self, state, n_state, corrected_var=True, use_target=False):
        return self.compute_action_and_log_prob_t(
            state, n_state, compute_log_prob=False, corrected_var=corrected_var,
            use_target=use_target, raw_action=False).detach().cpu().numpy()

    def draw_raw_action(self, state, n_state, corrected_var=True, use_target=False):
        return self.compute_action_and_log_prob_t(
            state, n_state, compute_log_prob=False, corrected_var=corrected_var,
            use_target=use_target, raw_action=True).detach().cpu().numpy()

    def compute_action_and_log_prob(self, state, n_state, corrected_var, use_target, raw_action=False):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.
            n_state (np.ndarray): the next state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state, n_state, corrected_var=corrected_var,
                                                         use_target=use_target, raw_action=raw_action)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, n_state, compute_log_prob=True, corrected_var=True, use_target=False,
                                      raw_action=False):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            n_state (np.ndarray): the next state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state, n_state, corrected_var, use_target)
        if self.use_mean:
            a_raw = dist.mean
        else:
            a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._eps_log_prob).sum(dim=1)
            return a_true, log_prob
        else:
            return a_raw if raw_action else a_true

    def get_prior_params(self, state, n_state, use_target=False):
        state_nstate = to_float_tensor(np.concatenate([state, n_state], axis=1))
        state_nstate = state_nstate.cuda() if self.use_cuda else state_nstate
        if not use_target:
            return self._approximator.model.network.get_prior_params(state_nstate)
        else:
            return self._target_approximator.model.network.get_prior_params(state_nstate)

    def distribution(self, state, n_state, corrected_var, use_target):
        """
        Compute the predictive distribution for the given states (Student's T distribution).

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.
            n_state (np.ndarray): the set of next states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu, lam, alpha, beta = self.get_prior_params(state, n_state, use_target)
        if corrected_var:
            var = self.get_corrected_pred_var(lam, alpha, beta)
        else:
            var = self.get_pred_var(lam, alpha, beta)
        return torch.distributions.StudentT(loc=mu, scale=var, df=2*alpha)

    def get_pred_var(self, lam, alpha, beta):
        return ((beta * (1+lam))/(lam*alpha))

    def get_corrected_pred_var(self, lam, alpha, beta):
        A = (2 * alpha) / (2*alpha + 3)
        return ((beta * (1+lam))/(lam*(alpha - A)))

    def entropy(self, state, n_state):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.
            n_state (np.ndarray): the set of next states to consider.

        Returns:
            The value of the entropy of the policy.

        """

        return torch.mean(self.distribution(state, n_state, corrected_var=True,
                                            use_target=False).entropy()).detach().cpu().numpy().item()


class KLGCPActionModel(GCPActionModel):

    def __init__(self, update_mode="hard", n_hard=1, lam_mode="vanilla", lam_delta=1, lam_mult=1.0, **params):
        super(KLGCPActionModel, self).__init__(**params)
        self._n_hard = n_hard
        if update_mode == "hard":
            self._update_target_method = self._update_target_hard
        elif update_mode == "soft":
            self._update_target_method = self._update_target
        else:
            raise ValueError("Unknown target update mode %s." % update_mode)

        self.lam_counter = 0
        self._lam_delta = lam_delta
        self._lam_mult = lam_mult
        assert lam_mode in ["vanilla", "counter", "mult"], "Unknown mode for lambda %s" % lam_mode
        self._lam_mode = lam_mode

        self._iter = 1

    def fit(self, state, n_state, action, **fit_params):
        if "n_epochs" in fit_params:
            n_epochs = fit_params["n_epochs"]
        else:
            n_epochs = 1

        # sample batch of same size from expert replay buffer and concatenate with samples from own policy
        demo_obs, demo_nobs = next(minibatch_generator(state.shape[0],
                                                       self._demonstrations["states"],
                                                       self._demonstrations["next_states"]))

        action = torch.tensor(action).cuda().cuda() if self._use_cuda else torch.tensor(action)
        action = torch.clip((action-self._central_a)/self._delta_a, -1.0+1e-7, 1.0-1e-7)
        action = torch.arctanh(action)
        self.raise_if_nan_or_if(action, "action")

        losses = []
        for epoch in range(n_epochs):

            # calculate main loss
            mu, lam, alpha, beta = self.get_prior_params(state, n_state)
            loss = self._action_model_loss(mu, lam, alpha, beta, action)

            # calculate KL regularization on expert states
            mu_cur, lam_cur, alpha_cur, beta_cur = self.get_prior_params(demo_obs, demo_nobs)
            mu_tar, lam_tar, alpha_tar, beta_tar = self.get_prior_params(demo_obs, demo_nobs, use_target=True)
            loss_exp = GCP.calc_KL_NIG(mu_cur, lam_cur, alpha_cur, beta_cur, mu_tar, lam_tar, alpha_tar, beta_tar)
            loss += self._reg_const * torch.mean(loss_exp)

            # add some regularization to alpha and beta
            loss += self._alpha_norm_const * torch.norm(alpha)
            loss += self._beta_norm_const * torch.norm(beta)
            #loss += self._lam_norm_const * torch.norm(lam)
            #loss += self._alpha_norm_const * self._reg_const * torch.norm(alpha_cur)
            #loss += self._beta_norm_const * self._reg_const * torch.norm(beta_cur)
            #loss += self._lam_norm_const * self._reg_const * torch.norm(lam_cur)
            self.raise_if_nan_or_if(loss, "loss")

            # update target networks
            self._update_target_method()

            # update lambda bias if needed
            if self._lam_mode == "counter":
                self.lam_counter += self._lam_delta

            # update online action model params
            self._action_model_optimizer.zero_grad()
            loss.backward()
            self._action_model_optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            self._iter += 1

        return np.mean(losses)

    def get_prior_params(self, state, n_state, use_target=False):
        state_nstate = to_float_tensor(np.concatenate([state, n_state], axis=1))
        state_nstate = state_nstate.cuda() if self.use_cuda else state_nstate
        if not use_target:
            mu, lam, alpha, beta = self._approximator.model.network.get_prior_params(state_nstate)
            if self._lam_mode == "mult":
                lam *= self._lam_mult
            else:
                lam = torch.ones_like(lam) * self.lam_counter + lam
            self.raise_if_nan_or_if(mu, "mu")
            self.raise_if_nan_or_if(lam, "lam")
            self.raise_if_nan_or_if(alpha, "alpha")
            self.raise_if_nan_or_if(beta, "beta")
            return mu, lam, alpha, beta
        else:
            mu, lam, alpha, beta = self._target_approximator.model.network.get_prior_params(state_nstate)
            if self._lam_mode == "mult":
                lam *= self._lam_mult
            else:
                lam = torch.ones_like(lam) * self.lam_counter + lam
            self.raise_if_nan_or_if(mu, "mu")
            self.raise_if_nan_or_if(lam, "lam")
            self.raise_if_nan_or_if(alpha, "alpha")
            self.raise_if_nan_or_if(beta, "beta")
            return mu, lam, alpha, beta

    @staticmethod
    def raise_if_nan_or_if(t, name):
        if torch.any(torch.isnan(t)):
            raise ValueError("%s is Nan!" % name)
        elif torch.any(torch.isinf(t)):
            raise ValueError("%s is inf!" % name)

    def _update_target_hard(self):
        if self._iter % self._n_hard == 0:
            self._init_target(self._approximator, self._target_approximator)

