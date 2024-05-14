from copy import deepcopy
import torch
from torch.functional import F
import numpy as np
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from imitation_lib.imitation.lsiq import LSIQ


class LSIQ_Offline_DM(LSIQ):

    def __init__(self, dynamics_model_params, dynamics_model_init_epochs=250,
                 random_demonstrations=None, loss_mode_exp="fix", regularizer_mode="off",  **kwargs):

        super().__init__(loss_mode_exp=loss_mode_exp, regularizer_mode=regularizer_mode, **kwargs)

        self._dynamics_model = Regressor(TorchApproximator, **dynamics_model_params)
        self._dynamics_model_init_epochs = dynamics_model_init_epochs
        self._dynamics_model_initialized = False

        expert_demonstrations = deepcopy(kwargs["demonstrations"])
        if random_demonstrations is not None:
            self._dynamics_model_training_data = dict()
            for key, value in expert_demonstrations.items():
                if key != "episode_starts":
                    self._dynamics_model_training_data[key] = np.concatenate([value, random_demonstrations[key]])
                self.add_dataset_to_replay_memory(random_demonstrations)
        else:
            self._dynamics_model_training_data = expert_demonstrations

        low, high = self.mdp_info.observation_space.low.copy(),\
                    self.mdp_info.observation_space.high.copy()
        self.norm_act_mean = (high + low) / 2.0
        self.norm_act_delta = (high - low) / 2.0

        self._state = None
        self._idx_state = 0
        self._max_traj_len = 200

    def fit(self, dataset):
        raise AttributeError("This is the offline implementation of IQ, it is not supposed to use the fit function. "
                             "Use the fit_offline function instead.")

    def fit_offline(self, n_steps):

        if not self._dynamics_model_initialized:
            self.fit_dynamics_model(self._dynamics_model_init_epochs)
            self.predict_trajectories_and_add_to_replay_buffer(100, 100)
            self._dynamics_model_initialized = True
        #else:
        #     self.fit_dynamics_model(1)


        for i in range(n_steps):

            self.add_next_step_to_buffer()

            # sample batch from policy replay buffer
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            assert self._act_mask.size > 0, "IQ-Learn needs demo actions!"
            demo_obs, demo_act, demo_nobs, demo_absorbing = next(minibatch_generator(state.shape[0],
                                                                                     self._demonstrations["states"],
                                                                                     self._demonstrations["actions"],
                                                                                     self._demonstrations[
                                                                                         "next_states"],
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

        #if self._loss_mode_exp == "bootstrap": # todo: remove this was just for testing
        #    loss_term1 = F.mse_loss(current_Q[is_expert],
        #                            torch.ones_like(current_Q[is_expert]) * (1.0/self._reg_mult) + gamma.detach() * current_Q[is_expert].detach().cpu())
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
            self.update_policy(input_states, is_expert)

        if self._iter % self._delay_Q == 0:
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def fit_dynamics_model(self, n_epochs=1):

        states = self._dynamics_model_training_data["states"]
        actions = self._dynamics_model_training_data["actions"]
        inputs = np.concatenate([states, actions], axis=1)
        targets = self._dynamics_model_training_data["next_states"]

        # normalize targets
        targets = (targets - self.norm_act_mean) / self.norm_act_delta

        self._dynamics_model.fit(inputs, targets, n_epochs=n_epochs)

        preds = self._dynamics_model.predict(inputs)
        loss = F.mse_loss(to_float_tensor(preds), to_float_tensor(targets))
        self.sw_add_scalar("Forward_DM/Loss", torch.mean(loss), self._iter)
        print("Loss", torch.mean(loss).detach().cpu().numpy())

    def add_next_step_to_buffer(self):

        if self._idx_state >= self._max_traj_len or self._state is None:
            init_state_idx = np.random.randint(len(self._dynamics_model_training_data["states"])*0.8)
            self._state = self._dynamics_model_training_data["states"][init_state_idx]
            self._idx_state = 0

        action = self.policy.draw_action(self._state)
        action = np.clip(action, self.mdp_info.action_space.low, self.mdp_info.action_space.high)
        inputs = np.concatenate([self._state, action])
        next_state = self._dynamics_model.predict(inputs)

        # unnormalize next state
        next_state = (next_state * self.norm_act_delta) + self.norm_act_mean

        self._replay_memory.add([[self._state, action, 0.0, next_state, 0, 0]])

        self._state = next_state
        self._idx_state += 1


    def predict_trajectories_and_add_to_replay_buffer(self, n_trajec, trajec_len):

        for i in range(n_trajec):
            # get initial state
            init_state_idx = np.random.randint(len(self._dynamics_model_training_data["states"])*0.8)
            state = self._dynamics_model_training_data["states"][init_state_idx]
            for j in range(trajec_len):
                action = self.policy.draw_action(state)
                action = np.clip(action, self.mdp_info.action_space.low, self.mdp_info.action_space.high)
                inputs = np.concatenate([state, action])
                next_state = self._dynamics_model.predict(inputs)

                # unnormalize next state
                next_state = (next_state * self.norm_act_delta) + self.norm_act_mean

                self._replay_memory.add([[state, action, 0.0, next_state, 0, 0]])

                state = next_state

    def add_dataset_to_replay_memory(self, dataset):

        for i in range(len(dataset["states"])):
            self._replay_memory.add([[dataset["states"][i], dataset["actions"][i], dataset["rewards"][i],
                                    dataset["next_states"][i], dataset["absorbing"][i], dataset["last"][i]]])
