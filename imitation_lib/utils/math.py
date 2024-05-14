from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from mushroom_rl.utils.angles import euler_to_quat


from mushroom_rl.utils.torch import to_float_tensor


class GailDiscriminatorLoss(torch.nn.modules.BCEWithLogitsLoss):

    def __init__(self, entcoeff=1e-3, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None) -> None:

        super(GailDiscriminatorLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)

        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.entcoeff = entcoeff

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # overrides original BCELoss
        # from tensorflow max(x, 0) - x * z + log(1 + exp(-abs(x)))
        bce_loss = torch.maximum(input, torch.zeros_like(input)) - input * target + torch.log(1 + torch.exp(-torch.abs(input)))
        bce_loss = torch.mean(bce_loss)
        
        bernoulli_ent = self.entcoeff * torch.mean(self.logit_bernoulli_entropy(input))
        return bce_loss - bernoulli_ent

    def logit_bernoulli_entropy(self, logits):
        """
        Adapted from:
        https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
        """
        return (1. - self.sigmoid(logits)) * logits - self.logsigmoid(logits)


class VDBLoss(GailDiscriminatorLoss):

    def __init__(self, info_constraint, lr_beta, use_bernoulli_ent=False, entcoeff=1e-3, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None) -> None:

        # call base constructor
        super().__init__(entcoeff, weight, size_average, reduce, reduction, pos_weight)

        self._use_bernoulli_ent = use_bernoulli_ent
        self._info_constr = info_constraint
        self._lr_beta = lr_beta
        self._beta = 0.1

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, mu, logvar = input

        # bottleneck loss
        bottleneck_loss = self.bottleneck_loss(mu, logvar)

        # binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(torch.squeeze(logits), torch.squeeze(target), self.weight,
                                                      pos_weight=self.pos_weight,
                                                      reduction=self.reduction)

        # optional, additional bernoulli entropy (as in gail, but this was not used in the paper)
        bernoulli_ent = self.logit_bernoulli_entropy(logits) if self._use_bernoulli_ent else torch.zeros_like(bce_loss)

        # overall vdb loss
        vdb_loss = bce_loss + self._beta * bottleneck_loss + bernoulli_ent

        # update beta
        self._update_beta(bottleneck_loss)

        return vdb_loss

    def bottleneck_loss(self, mu, logvar):
        kld = self.kl_divergence(mu, logvar).mean()
        bottleneck_loss = kld - self._info_constr
        return bottleneck_loss

    @torch.no_grad()
    def _update_beta(self, bottleneck_loss):
        self._beta = max(0, self._beta + self._lr_beta * bottleneck_loss)

    @staticmethod
    def kl_divergence(mu, logvar):
        kl_div = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1, dim=1)
        return kl_div


def to_float_tensors(inputs):
    """ Takes a list or tuple of of numpy arrays and converts them to a list of torch tensors. If only an array is
        provided, it returns a torch tensor."""
    if type(inputs) is not tuple and type(inputs) is not list:
        return to_float_tensor(inputs)
    else:
        out = []
        for elem in inputs:
            out.append(to_float_tensor(elem))
        return out




