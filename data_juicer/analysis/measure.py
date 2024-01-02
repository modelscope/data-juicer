import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class Measure(object):
    """Base class for Measure distribution.
    """
    name = 'base'

    def measure(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)

    def _convert_to_tensor(self, p):
        """
        Convert input data to torch tensor.
        :param p: input data, now support
            [`scalar`,`list`, `tuple`, `torch binary file`, and `Categorical`].
        :return: torch tensor
        """
        if isinstance(p, Tensor):
            return p
        elif isinstance(p, Categorical):
            return p.probs
        elif isinstance(p, str):
            return torch.load(p)
        else:
            return torch.tensor(p)

    def _convert_to_categorical(self, p):
        """
        Convert input data to torch Categorical.
        :param p: input data, now support
            [`scalar`,`list`, `tuple`, `torch binary file`, and `Categorical`].
        :return: torch Categorical
        """
        if isinstance(p, Categorical):
            return p
        elif isinstance(p, Tensor):
            return Categorical(p)
        elif isinstance(p, str):
            return Categorical(torch.load(p))
        else:
            return Categorical(torch.tensor(p))


class KLDivMeasure(Measure):
    """
    Measure Kullback-Leibler divergence.
    """
    name = 'kl_divergence'

    def measure(self, p, q):
        p = self._convert_to_categorical(p)
        q = self._convert_to_categorical(q)
        assert p.probs.shape == q.probs.shape, \
            'The two inputs have different shape:' \
            f'{p.probs.shape} != {q.probs.shape} in {self.name}'
        return F.kl_div(q.logits, p.probs, log_target=False, reduction='sum')


class JSDivMeasure(Measure):
    """
    Measure Jensen-Shannon divergence.
    """
    name = 'js_divergence'

    def measure(self, p, q):
        p = self._convert_to_tensor(p)
        q = self._convert_to_tensor(q)
        assert p.shape == q.shape,  \
            'The two inputs have different shape:' \
            f'{p.shape} != {q.shape} in {self.name}'

        m = 0.5 * (p + q)
        kl_p = KLDivMeasure()(p, m)
        kl_q = KLDivMeasure()(q, m)
        js = 0.5 * (kl_p + kl_q)
        return js


class CrossEntropyMeasure(Measure):
    """
    Measure Cross-Entropy.
    """
    name = 'cross_entropy'

    def measure(self, p, q):
        p = self._convert_to_categorical(p)
        q = self._convert_to_categorical(q)
        assert p.probs.shape == q.probs.shape, \
            'The two inputs have different shape: '\
            f'{p.probs.shape} != {q.probs.shape} in {self.name}'
        return F.cross_entropy(q.logits, p.probs, reduction='sum')


class EntropyMeasure(Measure):
    """
    Measure Entropy.
    """
    name = 'entropy'

    def measure(self, p):
        p = self._convert_to_categorical(p)
        return p.entropy()
