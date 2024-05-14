import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F

from mushroom_rl.utils.preprocessors import RunningStandardization


class BiasedTanh(torch.nn.Module):

    def __init__(self, mult=0.5, bias=0.5):
        super(BiasedTanh, self).__init__()
        self._bias = bias
        self._mult = mult

    def forward(self, input):
        return self._mult * torch.tanh(input) + self._bias


def reparameterize(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)
    return mu + std * eps


class IQInitializer:

    def __call__(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight.data)
            if hasattr(layer.bias, 'data'):
                layer.bias.data.fill_(0.0)


class NormcInitializer:

    def __init__(self, std=1.0):
        self._std = std

    def __call__(self, tensor):
        with torch.no_grad():
            tensor.normal_(std=self._std)
            tensor /= torch.sqrt(torch.sum(torch.square(tensor)))
            return tensor


class Standardizer(nn.Module):

    def __init__(self, alpha=1e-32, use_cuda=False):
        # call base constructor
        super(Standardizer, self).__init__()

        self._sum = 0.0
        self._sumsq = 1e-2
        self._count = 1e-2
        self._use_cuda = use_cuda

        self.mean = 0.0
        self.std = 1.0

    # def forward(self, inputs):
    #     self.update_mean_std(inputs.detach().cpu().numpy())
    #     mean = torch.tensor(self.mean).cuda() if self._use_cuda else torch.tensor(self.mean)
    #     std = torch.tensor(self.std).cuda() if self._use_cuda else torch.tensor(self.std)
    #     return (inputs - mean) / std
        
    def forward(self, inputs):
        # 更新mean和std，先将inputs移动到CPU
        self.update_mean_std(inputs.detach().cpu().numpy())
        # 确保mean和std在正确的设备上
        mean = torch.tensor(self.mean).to(inputs.device)
        std = torch.tensor(self.std).to(inputs.device)
        return (inputs - mean) / std
    
    def update_mean_std(self, x):
        self._sum += x.sum(axis=0).ravel()
        self._sumsq += np.square(x).sum(axis=0).ravel()
        self._count += np.array([len(x)])
        self.mean = self._sum / self._count
        self.std = np.sqrt(np.maximum((self._sumsq / self._count) - np.square(self.mean), 1e-2 ))


# This class is for backwards compatability only
class Standardizerv2(Standardizer):

    def forward(self, inputs):
        self.update_mean_std(inputs.detach().cpu().numpy())
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)
        return (inputs - mean) / std


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, activations, activations_params=None,
                 initializers=None, squeeze_out=False, standardizer=None, **kwargs):
        """
        This class implements a simple fully-connected feedforward network using torch.
        Args:
            input_shape (Tuple): Shape of the input (only 1-Dim) allowed.
            output_shape (Tuple): Shape of the output (only 1-Dim) allowed.
            n_features (List): Number of dimensions of the hidden layers,
            activations (List): List containing the activation names for each layer.
                                NOTE: len(dims_layers)-1 = len(activations)
            activations_params (List): List of dicts containing the parameters for the activations for each layer.

        """

        # call base constructor
        super().__init__()

        assert len(input_shape) == len(output_shape) == 1

        self.input_shape = input_shape[0]
        self.output_shape = output_shape[0]
        dims_network = [self.input_shape] + n_features + [self.output_shape]
        assert len(activations) == len(dims_network) - 1

        # construct the linear layers
        self._linears = ModuleList([nn.Linear(dims_network[i], dims_network[i+1]) for i in range(len(dims_network)-1)])

        # add activations
        if activations_params:
            self._activations = ModuleList([self.activation_function(name, params) for name, params in zip(activations, activations_params)])
        else:
            self._activations = ModuleList([self.activation_function(name) for name in activations])

        self._stand = standardizer
        self._squeeze_out = squeeze_out

        # make initialization
        if initializers is None:
            for layer, activation in zip(self._linears, activations):
               try:
                   nn.init.xavier_uniform_(layer.weight,
                                           gain=nn.init.calculate_gain(activation))
               except:
                   nn.init.xavier_uniform_(layer.weight)
        else:
            for layer, initializer in zip(self._linears, initializers):
                initializer(layer.weight)

    def forward(self, *inputs, dim=1):
        inputs = torch.squeeze(torch.cat(inputs, dim=dim), 1)
        if self._stand is not None:
            inputs = self._stand(inputs)
        # define forward pass
        z = inputs.float()
        for layer, activation in zip(self._linears, self._activations):
            z = activation(layer(z))

        if self._squeeze_out:
            out = torch.squeeze(z)
        else:
            out = z

        return out

    @staticmethod
    def activation_function(activation_name, params=None):
        """
        This functions returns the torch activation function.
        Args:
            activation_name (String): Name of the activation function.
            params (dict): Parameters for the activation function.

        """
        if activation_name == 'sigmoid':
            return torch.nn.Sigmoid()
            return torch.nn.Sigmoid()
        elif activation_name == 'tanh':
            return torch.nn.Tanh()
        elif activation_name == 'multtanh':
            return MultTanh(**params)
        elif activation_name == 'biased_tanh':
            return BiasedTanh(**params) if params is not None else BiasedTanh()
        elif activation_name == 'relu':
            return torch.nn.ReLU()
        elif activation_name == 'leaky_relu':
            return torch.nn.LeakyReLU(**params) if params is not None else torch.nn.LeakyReLU()
        elif activation_name == 'selu':
            return torch.nn.SELU()
        elif activation_name == 'identity':
            return torch.nn.Identity()
        elif activation_name == 'softplus':
            return torch.nn.Softplus()
        elif activation_name == 'softplustransformed':
            return SoftPlusTransformed(**params)
        else:
            raise ValueError('The activation %s in not supported.' % activation_name)


class DiscriminatorNetwork(FullyConnectedNetwork):
    def __init__(self, input_shape, output_shape, n_features, activations,  initializers=None,
                 squeeze_out=True, standardizer=None, use_actions=True, use_next_states=False, **kwargs):
        # call base constructor
        super(DiscriminatorNetwork, self).__init__(input_shape=input_shape, output_shape=output_shape, n_features=n_features,
                                                   activations=activations, initializers=initializers, squeeze_out=squeeze_out,
                                                   standardizer=standardizer, **kwargs)

        assert not (use_actions and use_next_states), "Discriminator with states, actions and next states as" \
                                                      "input currently not supported."

        self.use_actions = use_actions
        self.use_next_states = use_next_states

    def forward(self, *inputs):
        inputs = self.preprocess_inputs(*inputs)
        # define forward pass
        z = inputs.float()
        for layer, activation in zip(self._linears, self._activations):
            z = activation(layer(z))
        return z

    def preprocess_inputs(self, *inputs):
        if self.use_actions:
            states, actions = inputs
        elif self.use_next_states:
            states, next_states = inputs
        else:
            states = inputs[0]
        # normalize states
        if self._stand is not None:
            states = self._stand(states)
            if self.use_next_states:
                next_states = self._stand(next_states)
        if self.use_actions:
            inputs = torch.squeeze(torch.cat([states, actions], dim=1), 1)
        elif self.use_next_states:
            inputs = torch.squeeze(torch.cat([states, next_states], dim=1), 1)
        else:
            inputs = states
        return inputs

class VariationalNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, z_size, encoder_net: FullyConnectedNetwork,
                 decoder_net: FullyConnectedNetwork, standardizer=None, use_actions=True, use_next_states=False, **kwargs):
        # call base constructor
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.mu_out = torch.nn.Linear(encoder_net.output_shape, z_size)
        self.logvar_out = torch.nn.Linear(encoder_net.output_shape, z_size)

        # initialize mu and logvar
        init = NormcInitializer(std=0.01)
        init(self.mu_out.weight)
        init(self.logvar_out.weight)

        self._stand = standardizer
        self._use_actions = use_actions
        self._use_next_states = use_next_states

    def forward(self, states, *argv):
        if self._use_actions:
            actions = argv[0]
        if self._use_next_states:
            next_states = argv[0]
        # standardize states only
        inputs = []
        if self._stand is not None:
            inputs.append(self._stand(states))
            if self._use_actions:
                inputs.append(actions)
            if self._use_next_states:
                inputs.append(self._stand(next_states))
        else:
            inputs.append(states)
            if self._use_actions:
                inputs.append(actions)
            if self._use_next_states:
                inputs.append(next_states)

        inputs = torch.squeeze(torch.cat(inputs, dim=1), 1)
        enc_out = self.encoder_net(inputs)
        mu = self.mu_out(enc_out)
        logvar = self.logvar_out(enc_out)
        z = reparameterize(mu, logvar)
        out = self.decoder_net(z)
        return out, mu, logvar


class ShapedRewardNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, base_net: FullyConnectedNetwork, shaping_net: FullyConnectedNetwork,
                 standardizer, gamma=0.995, use_action=False,
                 use_next_state=False, use_done=False, vairl=False, **kwargs):
        # call base constructor
        super(ShapedRewardNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.base_net = base_net
        self.shaping_net = shaping_net
        self._stand = standardizer
        self._gamma = gamma
        self._use_action = use_action
        self._use_next_state = use_next_state
        self._use_done = use_done
        self._vairl= vairl

    def forward(self, states, actions, next_states, dones, log_policy_act_prob):
        # standardize states only
        if self._stand is not None:
            states = self._stand(states)
            next_states = self._stand(next_states)
        if self._vairl:
            shaped_reward, mus, logvars = self.get_shaped_reward(states, actions, next_states, dones, standardize=False)
        else:
            shaped_reward = self.get_shaped_reward(states, actions, next_states, dones, standardize=False)
        # create logits, where the expert demo is high
        # NOTE: This is equal logits(exp(r)/(exp(r) + pi)), which are the logits trained in AIRL
        logits = shaped_reward - log_policy_act_prob - torch.ones_like(shaped_reward)*20.2
        if self._vairl:
            mus = torch.concat(mus)
            logvars = torch.concat(logvars)
            return logits, mus, logvars
        else:
            return logits

    def get_shaped_reward(self, states, actions, next_states, dones, standardize=True):
        # standardize states only
        if self._stand is not None and standardize:
            states = self._stand(states)
            next_states = self._stand(next_states)
        inputs = []
        inputs.append(states)
        if self._use_action:
            inputs.append(actions)
        if self._use_next_state:
            inputs.append(next_states)

        if self._vairl:
            rewards, mu_r, logvar_r = self.base_net(*inputs)
            Vs, mu_vs, logvar_vs = self.shaping_net(states)
            Vss, mu_vss, logvar_vss = self.shaping_net(next_states)
            mus = [mu_r, mu_vs, mu_vss]
            logvars = [logvar_r, logvar_vs, logvar_vss]
        else:
            rewards = self.base_net(*inputs)
            Vs = self.shaping_net(states)
            Vss = self.shaping_net(next_states)

        if self._use_done:
            Vss = (1 - dones) * Vss
        if self._vairl:
            return rewards + self._gamma * Vss - Vs, mus, logvars
        else:
            return rewards + self._gamma * Vss - Vs,

    def get_base_reward(self, states, actions, next_states):
        # standardize states only
        if self._stand is not None:
            states = self._stand(states)
            next_states = self._stand(next_states)
        inputs = []
        inputs.append(states)
        if self._use_action:
            inputs.append(actions)
        if self._use_next_state:
            inputs.append(next_states)

        inputs = torch.concat(inputs, dim=1)

        return self.base_net(inputs)


class DoubleActionModel(torch.nn.Module):

    def __init__(self, input_shape, output_shape, first_net: FullyConnectedNetwork,
                 second_net: FullyConnectedNetwork, use_cuda, **kwargs):

        # call base constructor
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.first_net = first_net
        self.second_net = second_net
        self._discrim = torch.nn.Linear(input_shape[0], 1)
        self._use_cuda = use_cuda

        # initialize discrim
        init = NormcInitializer(std=0.01)
        init(self._discrim.weight)

    def forward(self, state_nstate):

        state_nstate = torch.tensor(state_nstate, dtype=torch.float32)
        first_out = self.first_net(state_nstate)
        second_out = self.second_net(state_nstate)

        discrim = self._discrim(state_nstate)
        discrim = torch.special.expit(10*discrim)

        return discrim * first_out - (1 - discrim) * second_out


class DoubleGaussianNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, net_mu: FullyConnectedNetwork,
                 net_log_sigma: FullyConnectedNetwork, use_cuda, **kwargs):

        # call base constructor
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._net_mu = net_mu
        self._net_log_sigma = net_log_sigma
        self._use_cuda = use_cuda

    def forward(self, inputs):
        mus = self._net_mu(inputs)
        log_sigmas = self._net_log_sigma(inputs)
        return torch.concat([mus, log_sigmas], dim=1)


class GCPNet(torch.nn.Module):

    def __init__(self, input_shape, output_shape, net_mu: FullyConnectedNetwork,
                 net_lambda: FullyConnectedNetwork, net_alpha: FullyConnectedNetwork, net_beta: FullyConnectedNetwork,
                 use_cuda, **kwargs):

        # call base constructor
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._net_mu = net_mu
        self._net_lambda = net_lambda
        self._net_alpha = net_alpha
        self._net_beta = net_beta
        self._use_cuda = use_cuda

    def forward(self, inputs):
        mus = self._net_mu(inputs)
        lambdas = self._net_lambda(inputs)
        alphas = self._net_alpha(inputs)
        betas = self._net_beta(inputs)
        return torch.concat([mus, lambdas, alphas, betas], dim=1)

    def get_prior_params(self, inputs):
        mus = self._net_mu(inputs)
        lambdas = self._net_lambda(inputs)
        alphas = self._net_alpha(inputs)
        betas = self._net_beta(inputs)
        return mus, lambdas, alphas, betas


class GCPNetSmall(FullyConnectedNetwork):

    def __init__(self, input_shape, output_shape, lam_i, lam_b, alpha_i, alpha_b, beta_i, beta_b, **kwargs):

        # call base constructor
        super().__init__(input_shape, output_shape, **kwargs)

        assert self.output_shape % 4 == 0, "The output shape needs to be dividable by 4."
        self._n_actions = self.output_shape // 4

        assert type(self._activations[-1]) == torch.nn.Identity, "Last activation has to be identity in order to not interfere with" \
                                                                 "the individual activations of lam, alpha, and beta. mu activation" \
                                                                 "is always supposed to be identity."
        self._lam_activation = SoftPlusTransformed(intercept_ordinate=lam_i, bias=lam_b)
        self._alpha_activation = SoftPlusTransformed(intercept_ordinate=alpha_i, bias=alpha_b)
        self._beta_activation = SoftPlusTransformed(intercept_ordinate=beta_i, bias=beta_b)

    def get_prior_params(self, inputs):
        out = super(GCPNetSmall, self).forward(inputs, dim=0)
        mus, lams, alphas, betas = self.divide_to_prior_parms(out)
        lams = self._lam_activation(lams)
        alphas = self._alpha_activation(alphas)
        betas = self._beta_activation(betas)
        return mus, lams, alphas, betas

    def divide_to_prior_parms(self, out):
        i = self._n_actions
        mus = out[:, 0:i]
        lams = out[:, i:2*i]
        alphas = out[:, 2*i:3*i]
        betas = out[:, 3*i:4*i]
        return mus, lams, alphas, betas


class SoftPlusTransformed(torch.nn.Module):

    def __init__(self, intercept_ordinate=0.6931, bias=0.0, threshold=20):
        super(SoftPlusTransformed, self).__init__()
        assert intercept_ordinate > bias, "The ordinate intercept is not allowed to be smaller" \
                                          "than or equal to the bias!"
        self._beta = np.log(2) / (intercept_ordinate - bias)
        self._bias = bias
        self._threshold = threshold

    def forward(self, input):
        out = F.softplus(input, self._beta, self._threshold) + torch.ones_like(input) * self._bias
        return out

class MultTanh(torch.nn.Module):

    def __init__(self, mult=1.0):
        super(MultTanh, self).__init__()
        self._mult = mult

    def forward(self, input):
        return F.tanh(input) * self._mult


def pp_divide_output_in_half(x: torch.Tensor, output_shape: int):
    half_out_shape = output_shape // 2
    assert type(output_shape) == int, "Output shape needs to be an integer."
    assert 2*half_out_shape == output_shape, "Output shape needs to be an even number."
    assert len(x.size()) == 2, "x needs to be two-dimensional."
    return (x[:, 0: half_out_shape], x[:, half_out_shape:output_shape])


if __name__ == '__main__':
    net = FullyConnectedNetwork(input_shape=[40], output_shape=[17],
                                n_features=[512, 256], activations=('relu', 'relu', 'identity'))
