import torch
import torch.nn.functional as F
import numpy as np
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from imitation_lib.imitation.iq_sac import IQ_SAC
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor


class IQfO_ORIG(IQ_SAC):


    def fit(self, dataset):

        # add to replay memory
        self._replay_memory.add(dataset)

        if self._replay_memory.initialized:

            # sample batch from policy replay buffer
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            demo_obs, demo_nobs, demo_absorbing = next(minibatch_generator(state.shape[0],
                                                                           self._demonstrations["states"],
                                                                           self._demonstrations["next_states"],
                                                                           self._demonstrations["absorbing"]))

            # the action by the expert is predicted by the policy
            with torch.no_grad():
                demo_act, _ = self.policy.compute_action_and_log_prob_t(demo_obs)
                demo_act = demo_act.detach().numpy()

            # prepare data for IQ update
            input_states = to_float_tensor(np.concatenate([state, demo_obs.astype(np.float32)]))
            input_actions = to_float_tensor(np.concatenate([action, demo_act.astype(np.float32)]))
            input_n_states = to_float_tensor(np.concatenate([next_state, demo_nobs.astype(np.float32)]))
            input_absorbing = to_float_tensor(np.concatenate([absorbing, demo_absorbing.astype(np.float32)]))
            is_expert = torch.concat([torch.zeros(len(state), dtype=torch.bool),
                                      torch.ones(len(state), dtype=torch.bool)])
            # make IQ update
            loss1, loss2, chi2_loss = self._lossQ(input_states, input_actions, input_n_states, input_absorbing,
                                                  is_expert)
            self._sw.add_scalar('IQ-Loss/Loss1', loss1, self._iter)
            self._sw.add_scalar('IQ-Loss/Loss2', loss2, self._iter)
            self._sw.add_scalar('IQ-Loss/Chi2 Loss', chi2_loss, self._iter)
            self._sw.add_scalar('IQ-Loss/Alpha', self._alpha, self._iter)

            # update policy
            if self._replay_memory.size > self._warmup_transitions() and self._iter % self._delay_pi == 0:
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(input_states)
                loss = self._actor_loss(input_states, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                grads = []
                for param in self.policy._approximator.model.network.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                norm = grads.norm(dim=0, p=2)
                self._sw.add_scalar('Gradients/Norm2 Gradient Q wrt. Pi-parameters', norm,
                                    self._iter)
                self._sw.add_scalar('Actor/Loss', loss, self._iter)
                self._sw.add_scalar('Actor/Entropy', torch.mean(-log_prob).detach().item(), self._iter)
                if self._learnable_alpha:
                    self._update_alpha(log_prob.detach())

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

        self._iter += 1

