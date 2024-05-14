import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from mushroom_rl.core import Serializable
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from imitation_lib.imitation.lsiq import LSIQ
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from imitation_lib.utils.action_models import GaussianInvActionModel, LearnableVarGaussianInvActionModel,\
    GCPActionModel, KLGCPActionModel, KLGaussianInvActionModel

from imitation_lib.utils.distributions import InverseGamma


class LSIQfO(LSIQ):

    def __init__(self, action_model, action_model_params, action_model_fit_params=None, action_model_noise_std=0.0,
                 action_model_noise_clip=None, add_noise_to_obs=False, ext_normalizer_action_model=None,
                 interpolate_expert_states=False, interpolation_coef=1.0, **kwargs):

        super().__init__(**kwargs)

        if action_model == GaussianInvActionModel or action_model == GCPActionModel \
                or action_model == KLGCPActionModel or action_model == KLGaussianInvActionModel:
            action_model_params.setdefault("min_a", self.mdp_info.action_space.low)
            action_model_params.setdefault("max_a", self.mdp_info.action_space.high)
            action_model_params.setdefault("use_cuda", self._use_cuda)
        elif action_model == LearnableVarGaussianInvActionModel:
            action_model_params.setdefault("use_cuda", self._use_cuda)

        # setup the action model
        self._action_model = action_model(**action_model_params, demonstration=self._demonstrations)

        self._action_model_fit_params = dict(fits_per_step=1, init_epochs=0, )\
            if action_model_fit_params is None else action_model_fit_params
        self._action_model_initialized = True if self._action_model_fit_params["init_epochs"] > 0 else False
        self._action_model_batch_size = action_model_params["batch_size"]

        self._action_model_noise_std = action_model_noise_std
        self._action_model_noise_clip = action_model_noise_clip
        self.ext_normalizer_action_model = ext_normalizer_action_model
        self._add_noise_to_obs = add_noise_to_obs
        self._interpolate_expert_states = interpolate_expert_states
        self._interpolation_coef = interpolation_coef

        self._add_save_attr(
            _action_model='mushroom',
            _action_model_fit_params='pickle',
            _action_model_noise_std='primitive',
            _action_model_noise_clip='primitive',
            ext_normalizer_action_model='pickle',
            _add_noise_to_obs='primitive'
        )

    def fit(self, dataset):

        # add to replay memory
        self._replay_memory.add(dataset)

        if self._replay_memory.initialized:

            # train the action model
            if not self._action_model_initialized:
                self.train_action_model(init=True)
                self._action_model_initialized = True
            else:
                self.train_action_model()

            # sample batch from policy replay buffer
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            demo_obs, demo_nobs, demo_absorbing = next(minibatch_generator(state.shape[0],
                                                                           self._demonstrations["states"],
                                                                           self._demonstrations["next_states"],
                                                                           self._demonstrations["absorbing"]))

            # predict the actions for our expert dataset
            demo_obs_act = demo_obs.astype(np.float32)[:, self._state_mask]
            demo_nobs_act = demo_nobs.astype(np.float32)[:, self._state_mask]
            demo_act = self._action_model.draw_action(to_float_tensor(demo_obs_act),
                                                      to_float_tensor(demo_nobs_act))

            # clip predicted action to action range
            demo_act = np.clip(demo_act, self.mdp_info.action_space.low, self.mdp_info.action_space.high)

            if self._add_noise_to_obs:
                assert self.ext_normalizer_action_model is not None, "Normalizer is needed to be defined."

                demo_obs = self.ext_normalizer_action_model(demo_obs)
                demo_nobs = self.ext_normalizer_action_model(demo_nobs)
                demo_obs += self._get_noise(demo_obs)
                demo_nobs += self._get_noise(demo_nobs)
                demo_obs = self.ext_normalizer_action_model.inv(demo_obs)
                demo_nobs = self.ext_normalizer_action_model.inv(demo_nobs)

            # make interpolation if needed
            if self._interpolate_expert_states:
                demo_obs = self.interpolate(demo_obs[:, self._state_mask], state[:, self._state_mask],
                                            mixing_coef=self._interpolation_coef)
                demo_act = self.interpolate(demo_act, action,
                                            mixing_coef=self._interpolation_coef)

            # prepare data for IQ update
            input_states = to_float_tensor(np.concatenate([state,
                                                           demo_obs.astype(np.float32)[:, self._state_mask]]))
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

    def _get_noise(self, x):
        noise = np.random.normal(loc=0.0, scale=self._action_model_noise_std,
                                 size=np.size(x)).reshape(x.shape)
        noise = np.clip(noise, -self._action_model_noise_clip, self._action_model_noise_clip) \
            if self._action_model_noise_clip is not None else noise
        return noise

    def interpolate(self, expert_data, policy_data, mixing_coef=None):
        interpolated = mixing_coef * expert_data + (1 - mixing_coef) * policy_data
        return interpolated

    def train_action_model(self, init=False):

        if init and self._action_model_fit_params["init_epochs"] > 0:
            n_epochs = self._action_model_fit_params["init_epochs"]
            # initialize the model
            state, action, _, next_state, _, _ = self._replay_memory.get(self._replay_memory.size)
            state = self.ext_normalizer_action_model(state) if self.ext_normalizer_action_model else state
            next_state = self.ext_normalizer_action_model(next_state) if self.ext_normalizer_action_model else next_state
            state_train = state[0:int(len(state)*0.9), :]
            state_val = state[int(len(state)*0.9):, :]
            next_state_train = next_state[0:int(len(next_state)*0.9), :]
            next_state_val = next_state[int(len(next_state)*0.9):, :]
            action_train = action[0:int(len(next_state)*0.9), :]
            action_val = action[int(len(next_state)*0.9):, :]
            state_nstate_train = np.concatenate([state_train, next_state_train], axis=1)
            state_nstate_val = np.concatenate([state_val, next_state_val], axis=1)

            # make eval before training
            action_pred = self._action_model(state_nstate_val)
            loss = F.mse_loss(to_float_tensor(action_pred), to_float_tensor(action_val))
            self._sw.add_scalar('Action-Model/Loss', loss, self._iter)
            print("Action Model Validation Loss before training: ", loss)
            action_pred = self._action_model(state_nstate_train)
            loss = F.mse_loss(to_float_tensor(action_pred), to_float_tensor(action_train))
            print("Action Model Training Loss before training: ", loss)
            w = self._action_model.get_weights()
            norm = np.linalg.norm(w)
            self.sw_add_scalar("Action-Model/Norm", norm, self._iter)

            # make training
            self._action_model.fit(state_nstate_train, action_train, n_epochs=n_epochs)

            # make eval after training
            action_pred = self._action_model(state_nstate_val)
            loss = F.mse_loss(to_float_tensor(action_pred), to_float_tensor(action_val))
            self._sw.add_scalar('Action-Model/Loss', loss, self._iter)
            print("Action Model Validation Loss After training: ", loss)
            action_pred = self._action_model(state_nstate_train)
            loss = F.mse_loss(to_float_tensor(action_pred), to_float_tensor(action_train))
            print("Action Model Validation Loss After training: ", loss)

        else:
            state_nstates = []
            actions = []
            for i in range(self._action_model_fit_params["fits_per_step"]):
                # sample batch from policy replay buffer
                state, action, reward, next_state, absorbing, _ = \
                    self._replay_memory.get(self._action_model_batch_size)

                state = self.ext_normalizer_action_model(state) if self.ext_normalizer_action_model else state
                next_state = self.ext_normalizer_action_model(next_state) if self.ext_normalizer_action_model else next_state
                self._action_model.fit(state, next_state, action)

                state_nstates.append([state, next_state])
                actions.append(action)

            if self._iter % self._logging_iter == 0:

                # sample batch from policy replay buffer
                states, actions, rewards, next_states, absorbings, _ = \
                    self._replay_memory.get(self._action_model_batch_size)

                # we need to check if we have a dataset with expert actions available or not
                try:
                    exp_states, exp_next_states, exp_actions = next(
                        minibatch_generator(self._action_model_batch_size,
                                            self._demonstrations["states"],
                                            self._demonstrations["next_states"],
                                            self._demonstrations["actions"]))
                except KeyError:
                    exp_states, exp_next_states = next(minibatch_generator(self._action_model_batch_size,
                                                                           self._demonstrations["states"],
                                                                           self._demonstrations["next_states"]))
                    exp_actions = None

                # log mse
                action_pred = self._action_model(states[:, self._state_mask], next_states[:, self._state_mask])
                mse = F.mse_loss(to_float_tensor(action_pred), to_float_tensor(actions))
                self.sw_add_scalar('Action-Model/Loss Policy', mse, self._iter)
                if exp_actions is not None:
                    action_pred_exp = self._action_model(exp_states[:, self._state_mask],
                                                         exp_next_states[:, self._state_mask])
                    mse_exp = F.mse_loss(to_float_tensor(action_pred_exp), to_float_tensor(exp_actions))
                    self.sw_add_scalar('Action-Model/Loss Exp', mse_exp, self._iter)

                # log entropy
                ent_plcy = self._action_model.entropy(states[:, self._state_mask],
                                                      next_states[:, self._state_mask])
                ent_exp = self._action_model.entropy(exp_states[:, self._state_mask],
                                                     exp_next_states[:, self._state_mask])
                self.sw_add_scalar('Action-Model/Entropy Plcy', ent_plcy,
                                    self._iter)
                self.sw_add_scalar('Action-Model/Entropy Exp', ent_exp,
                                    self._iter)

                # log mu, lam, alpha, beta
                if type(self._action_model) == GCPActionModel or type(self._action_model) == KLGCPActionModel:
                    mu, lam, alpha, beta = self._action_model.get_prior_params(states[:, self._state_mask],
                                                                               next_states[:, self._state_mask])
                    self.sw_add_scalar('Action-Model/Mu', np.mean(mu.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Lambda', np.mean(lam.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Lambda Counter', self._action_model.lam_counter, self._iter)
                    self.sw_add_scalar('Action-Model/Alpha', np.mean(alpha.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Beta', np.mean(beta.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Var',
                                       np.mean(self._action_model.get_corrected_pred_var(lam,
                                                                                         alpha,
                                                                                         beta).detach().cpu().numpy()),
                                       self._iter)
                    mu_exp, lam_exp, alpha_exp, beta_exp = \
                        self._action_model.get_prior_params(exp_states[:, self._state_mask],
                                                            exp_next_states[:, self._state_mask])
                    self.sw_add_scalar('Action-Model/Mu Exp', np.mean(mu_exp.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Lambda Exp', np.mean(lam_exp.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Alpha Exp', np.mean(alpha_exp.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Beta Exp', np.mean(beta_exp.detach().cpu().numpy()), self._iter)
                    self.sw_add_scalar('Action-Model/Var Exp',
                                       np.mean(self._action_model.get_corrected_pred_var(lam_exp,
                                                                                         alpha_exp,
                                                                                         beta_exp).detach().cpu().numpy()),
                                       self._iter)
                elif type(self._action_model) == GaussianInvActionModel or \
                        type(self._action_model) == KLGaussianInvActionModel:
                    mu, log_sigma = self._action_model.get_mu_log_sigma(state[:, self._state_mask],
                                                                        next_state[:, self._state_mask])
                    mu_exp, log_sigma_exp = self._action_model.get_mu_log_sigma(exp_states.astype(np.float32)[:, self._state_mask],
                                                                                exp_next_states.astype(np.float32)[:, self._state_mask])

                    self._sw.add_scalar('Action-Model/Std Exp', torch.mean(torch.exp(log_sigma_exp)), self._iter)
                    self._sw.add_scalar('Action-Model/Std', torch.mean(torch.exp(log_sigma)), self._iter)
                    self._sw.add_scalar('Action-Model/Mu Exp', torch.mean(mu_exp), self._iter)
                    self._sw.add_scalar('Action-Model/Mu', torch.mean(mu), self._iter)
