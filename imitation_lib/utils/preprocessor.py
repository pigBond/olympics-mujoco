from mushroom_rl.core import Serializable


class MaskingPreprocessor(Serializable):

    def __init__(self, mask):
        self._mask = mask
        self._add_save_attr(_mask='primitive')
    
    def __call__(self, obs):
        masked_obs = obs[self._mask]
        return masked_obs
