from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import get_gradient, zero_grad, to_float_tensor
from mushroom_rl.utils.dataset import parse_dataset, compute_J, arrays_as_dataset, compute_episodes_length
from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.trpo import TRPO
from mushroom_rl.utils.minibatches import minibatch_generator

from imitation_lib.utils import GailDiscriminatorLoss, to_float_tensors


class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning(GAIL) implementation.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params, sw,
                 discriminator_params, critic_params, trpo_standardizer=None, D_standardizer=None,
                 train_D_n_th_epoch=3, n_epochs_discriminator=1, ext_normalizer=None,
                 ent_coeff=0., max_kl=.01, lam=0.97,
                 n_epochs_line_search=10, n_epochs_cg=10,
                 cg_damping=1e-1, cg_residual_tol=1e-10,
                 demonstrations=None, env_reward_frac=0.0,
                 state_mask=None, act_mask=None, use_next_states=False, use_noisy_targets=False,
                 critic_fit_params=None, discriminator_fit_params=None, loss=GailDiscriminatorLoss()):

        # initialize PPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, critic_params,
                                   ent_coeff, max_kl, lam, n_epochs_line_search,
                                   n_epochs_cg, cg_damping, cg_residual_tol,
                                   critic_fit_params=critic_fit_params)
        # standardizers
        self._trpo_standardizer = trpo_standardizer
        self._D_standardizer = D_standardizer

        # discriminator params
        self._discriminator_fit_params = (dict() if discriminator_fit_params is None
                                          else discriminator_fit_params)

        self._loss = loss
        discriminator_params.setdefault("loss", deepcopy(self._loss))
        self._D = Regressor(TorchApproximator, **discriminator_params)
        self._train_D_n_th_epoch = train_D_n_th_epoch
        self._n_epochs_discriminator = n_epochs_discriminator

        self._env_reward_frac = env_reward_frac
        self._demonstrations = demonstrations   # should be: dict(states=np.array, actions=(np.array/None))
        assert 0.0 <= env_reward_frac <= 1.0, "Environment reward must be between [0,1]"
        assert demonstrations is not None or env_reward_frac == 1.0, "No demonstrations have been loaded"

        # select which observations / actions to discriminate
        if not "actions" in demonstrations:
            act_mask = []

        self._state_mask = np.arange(demonstrations["states"].shape[1]) \
            if state_mask is None else np.array(state_mask, dtype=np.int64)

        self._act_mask = np.arange(demonstrations["actions"].shape[1]) \
            if act_mask is None else np.array(act_mask, dtype=np.int64)

        self._use_next_state = use_next_states

        self._epoch_counter = 1

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None) # dont need to be copyable, causes pickle error otherwise
        else:
            self._sw = None

        self._use_noisy_targets = use_noisy_targets
        self.ext_normalizer = ext_normalizer

        self._add_save_attr(
            discriminator_fit_params='pickle',
            _loss='torch',
            _train_n_th_epoch ='pickle',
            _D='mushroom',
            _env_reward_frac='pickle',
            _demonstrations='pickle!',
            _act_mask='pickle',
            _state_mask='pickle',
            _use_next_state='pickle',
            _use_noisy_targets='pickle',
            _trpo_standardizer='pickle',
            _D_standardizer='pickle',
            _train_D_n_th_epoch='pickle',
            ext_normalizer='pickle',
        )

    def load_demonstrations(self, demonstrations):
        self._demonstrations = demonstrations

    def fit(self, dataset, **info):
        state, action, reward, next_state, absorbing, last = parse_dataset(dataset)
        x = state.astype(np.float32)
        u = action.astype(np.float32)
        r = reward.astype(np.float32)
        xn = next_state.astype(np.float32)

        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        # update running mean and std if neccessary
        if self._trpo_standardizer is not None:
            self._trpo_standardizer.update_mean_std(x)

        # create reward
        if self._env_reward_frac < 1.0:

            # create reward from the discriminator(can use fraction of environment reward)
            r_disc = self.make_discrim_reward(x, u, xn)
            r = r * self._env_reward_frac + r_disc * (1 - self._env_reward_frac)

        v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last,
                                       self.mdp_info.gamma, self._lambda())
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = to_float_tensor(np_adv, self.policy.use_cuda)

        # Policy update
        self._old_policy = deepcopy(self.policy)
        old_pol_dist = self._old_policy.distribution_t(obs)
        old_log_prob = self._old_policy.log_prob_t(obs, act).detach()

        zero_grad(self.policy.parameters())
        loss = self._compute_loss(obs, act, adv, old_log_prob)

        prev_loss = loss.item()

        # Compute Gradient
        loss.backward()
        g = get_gradient(self.policy.parameters())

        # Compute direction through conjugate gradient
        stepdir = self._conjugate_gradient(g, obs, old_pol_dist)

        # Line search
        self._line_search(obs, act, adv, old_log_prob, old_pol_dist, prev_loss, stepdir)

        # VF update
        if self._trpo_standardizer is not None:
            for i in range(self._critic_fit_params["n_epochs"]):
                self._trpo_standardizer.update_mean_std(x)  # update running mean
        self._V.fit(x, v_target, **self._critic_fit_params)

        # fit discriminator
        self._fit_discriminator(x, u, xn)

        # Print fit information
        # create dataset with discriminator reward
        new_dataset = arrays_as_dataset(x, u, r, xn, absorbing, last)
        self._logging_sw(dataset, new_dataset, x, v_target, old_pol_dist)
        #self._log_info(dataset, x, v_target, old_pol_dist)
        self._iter += 1

    def _fit_discriminator(self, plcy_obs, plcy_act, plcy_n_obs):
        plcy_obs = plcy_obs[:, self._state_mask]
        plcy_act = plcy_act[:, self._act_mask]
        plcy_n_obs = plcy_n_obs[:, self._state_mask]

        if self._iter % self._train_D_n_th_epoch == 0:

            for epoch in range(self._n_epochs_discriminator):

                # get batch of data to discriminate
                if self._use_next_state and not self._act_mask.size > 0:
                    demo_obs, demo_n_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                                    self._demonstrations["states"],
                                                                    self._demonstrations["next_states"]))
                    demo_obs = demo_obs[:, self._state_mask]
                    demo_n_obs = demo_n_obs[:, self._state_mask]
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    input_next_states = np.concatenate([plcy_n_obs, demo_n_obs.astype(np.float32)])
                    inputs = (input_states, input_next_states)
                elif self._act_mask.size > 0 and not self._use_next_state:
                    demo_obs, demo_act = next(minibatch_generator(plcy_obs.shape[0],
                                                                  self._demonstrations["states"],
                                                                  self._demonstrations["actions"]))
                    demo_obs = demo_obs[:, self._state_mask]
                    demo_act = demo_act[:, self._act_mask]
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    input_actions = np.concatenate([plcy_act, demo_act.astype(np.float32)])
                    inputs = (input_states, input_actions)
                elif self._act_mask.size > 0 and self._use_next_state:
                    raise ValueError("Discriminator with states, actions and next states as input currently not supported.")
                else:
                    demo_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                        self._demonstrations["states"]))[0]
                    demo_obs = demo_obs[:, self._state_mask]
                    input_states = np.concatenate([plcy_obs, demo_obs.astype(np.float32)])
                    inputs = (input_states,)

                # update running mean if neccessary
                if self._D_standardizer is not None:
                    self._D_standardizer.update_mean_std(np.concatenate([plcy_obs, demo_obs.astype(np.float32)]))

                # create label targets
                if self._use_noisy_targets:
                    demo_target = np.random.uniform(low=0.80, high=0.99, size=(plcy_obs.shape[0], 1)).astype(np.float32)
                    plcy_target = np.random.uniform(low=0.01, high=0.10, size=(plcy_obs.shape[0], 1)).astype(np.float32)
                else:
                    plcy_target = np.zeros(shape=(plcy_obs.shape[0], 1)).astype(np.float32)
                    demo_target = np.ones(shape=(plcy_obs.shape[0], 1)).astype(np.float32)

                targets = np.concatenate([plcy_target, demo_target])

                self._D.fit(*inputs, targets, **self._discriminator_fit_params)

                self._discriminator_logging(inputs, targets)

    def _discriminator_logging(self, inputs, targets):
        if self._sw:
            plcy_inputs, demo_inputs = self.divide_data_to_demo_and_plcy(inputs)
            loss = deepcopy(self._loss)
            loss_eval = loss.forward(to_float_tensors(self._D(*inputs)), torch.tensor(targets))
            self._sw.add_scalar('DiscrimLoss', loss_eval, self._iter // 3)

            # calculate the accuracies
            dout_exp = torch.sigmoid(torch.tensor(self.discrim_output(*demo_inputs, apply_mask=False)))
            dout_plcy = torch.sigmoid(torch.tensor(self.discrim_output(*plcy_inputs, apply_mask=False)))
            accuracy_exp = np.mean((dout_exp > 0.5).numpy())
            accuracy_gen = np.mean((dout_plcy < 0.5).numpy())
            self._sw.add_scalar('D_Generator_Accuracy', accuracy_gen, self._iter // 3)
            self._sw.add_scalar('D_Out_Generator', np.mean(dout_plcy.numpy()), self._iter // 3)
            self._sw.add_scalar('D_Expert_Accuracy', accuracy_exp, self._iter // 3)
            self._sw.add_scalar('D_Out_Expert', np.mean(dout_exp.numpy()), self._iter // 3)

            # calculate individual losses
            bernoulli_ent = torch.mean(loss.logit_bernoulli_entropy(torch.tensor(self.discrim_output(*inputs, apply_mask=False))))
            neg_bernoulli_ent_loss = -loss.entcoeff * bernoulli_ent
            plcy_target = targets[0:len(targets)//2]
            demo_target = targets[len(targets)//2:]
            loss_exp = loss.forward(to_float_tensors(self._D(*demo_inputs)), torch.tensor(demo_target)) / 2
            loss_gen = loss.forward(to_float_tensors(self._D(*plcy_inputs)), torch.tensor(plcy_target)) / 2
            self._sw.add_scalar('Bernoulli Ent.', bernoulli_ent, self._iter // 3)
            self._sw.add_scalar('Neg. Bernoulli Ent. Loss (incl. in DiscrimLoss)', neg_bernoulli_ent_loss, self._iter // 3)
            self._sw.add_scalar('Generator_loss', loss_gen, self._iter // 3)
            self._sw.add_scalar('Expert_Loss', loss_exp, self._iter // 3)
            
    def _logging_sw(self, dataset, new_data_set, x, v_target, old_pol_dist):
        if self._iter % self._train_D_n_th_epoch == 0 and self._sw:
            logging_verr = []
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            v_pred = torch.tensor(self._V(x), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)

            logging_ent = self.policy.entropy(x)
            new_pol_dist = self.policy.distribution(x)
            logging_kl = torch.mean(
                torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist)
            )
            avg_rwd = np.mean(compute_J(dataset))
            avg_rwd_new = np.mean(compute_J(new_data_set))
            L = int(np.round(np.mean(compute_episodes_length(dataset))))

            self._sw.add_scalar('EpTrueRewMean', avg_rwd, self._iter // 3)
            self._sw.add_scalar('EpRewMean', avg_rwd_new, self._iter // 3)
            self._sw.add_scalar('EpLenMean', L, self._iter // 3)
            self._sw.add_scalar('vf_loss', v_err, self._iter // 3)
            self._sw.add_scalar('entropy', logging_ent, self._iter // 3)
            self._sw.add_scalar('kl', logging_kl, self._iter // 3)

    def divide_data_to_demo_and_plcy(self, inputs):
        if self._act_mask.size > 0:
            input_states, input_actions = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_act = input_actions[0:len(input_actions)//2]
            plcy_inputs = (plcy_obs, plcy_act)
            demo_obs = input_states[len(input_states)//2:]
            demo_act = input_actions[len(input_actions)//2:]
            demo_inputs = (demo_obs, demo_act)
        elif self._use_next_state:
            input_states, input_next_states = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_n_obs = input_next_states[0:len(input_next_states)//2]
            plcy_inputs = (plcy_obs, plcy_n_obs)
            demo_obs = input_states[len(input_states)//2:]
            demo_n_obs = input_next_states[len(input_next_states)//2:]
            demo_inputs = (demo_obs, demo_n_obs)
        else:
            input_states = inputs[0]
            plcy_inputs = (input_states[0:len(input_states)//2],)
            demo_inputs = (input_states[len(input_states)//2:],)
        return plcy_inputs, demo_inputs

    def prepare_discrim_inputs(self, inputs, apply_mask=True):
        if self._use_next_state and not self._act_mask.size > 0:
            states, next_states = inputs
            states = states[:, self._state_mask] if apply_mask else states
            next_states = next_states[:, self._state_mask] if apply_mask else next_states
            inputs = (states, next_states)
        elif self._act_mask.size > 0 and not self._use_next_state:
            states, actions = inputs
            states = states[:, self._state_mask] if apply_mask else states
            actions = actions[:, self._act_mask] if apply_mask else actions
            inputs = (states, actions)
        elif self._act_mask.size > 0 and self._use_next_state:
            raise ValueError("Discriminator with states, actions and next states as input currently not supported.")
        else:
            states = inputs[0][:, self._state_mask] if apply_mask else inputs[0]
            inputs = (states,)
        return inputs

    def discrim_output(self, *inputs, apply_mask=True):
        inputs = self.prepare_discrim_inputs(inputs, apply_mask=apply_mask)
        d_out = self._D(*inputs)
        return d_out

    @torch.no_grad()
    def make_discrim_reward(self, state, action, next_state, apply_mask=True):
        if self._use_next_state:
            d = self.discrim_output(state, next_state, apply_mask=apply_mask)
        else:
            d = self.discrim_output(state, action, apply_mask=apply_mask)
        plcy_prob = 1/(1 + np.exp(-d))     # sigmoid
        return np.squeeze(-np.log(1 - plcy_prob + 1e-8)).astype(np.float32)
