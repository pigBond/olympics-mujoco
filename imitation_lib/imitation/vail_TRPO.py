from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from imitation_lib.imitation import GAIL_TRPO
from imitation_lib.utils import to_float_tensors


class VAIL(GAIL_TRPO):

    def __init__(self, **kwargs):

        # call base constructor
        super(VAIL, self).__init__(**kwargs)

    def discrim_output(self, *inputs, apply_mask=True):
        inputs = self.prepare_discrim_inputs(inputs, apply_mask=apply_mask)
        d_out,_ ,_ = self._D(*inputs)
        return d_out

    def _discriminator_logging(self, inputs, targets):
        super(VAIL, self)._discriminator_logging(inputs, targets)
        if self._sw:
            # calculate bottleneck loss
            loss = deepcopy(self._loss)
            d, mu, logvar = to_float_tensors(self._D(*inputs))
            bottleneck_loss = loss.bottleneck_loss(mu, logvar)
            self._sw.add_scalar('Bottleneck_Loss', bottleneck_loss, self._iter // 3)
            self._sw.add_scalar('Beta', loss._beta, self._iter // 3)
            self._sw.add_scalar('Bottleneck_Loss_times_Beta', loss._beta * bottleneck_loss, self._iter // 3)

