# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from .special_red_blue import SpecialRedBlueMove

__all__ = ["WalkMove"]


class WalkMove(SpecialRedBlueMove):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param s: (optional)
        The number of helper walkers to use. By default it will use all the
        walkers in the complement.

    """
    def __init__(self, s=None, **kwargs):
        self.s = s
        super(WalkMove, self).__init__(**kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        q = np.empty((Ns, ndim), dtype=np.float64)
        s0 = Nc if self.s is None else self.s
        for i in range(Ns):
            inds = random.choice(Nc, s0, replace=False)
            cov = np.atleast_2d(np.cov(c[inds], rowvar=0))
            try:
                q[i] = random.multivariate_normal(s[i], cov)
            except ValueError:
                import pdb; pdb.set_trace()
        return q, np.zeros(Ns, dtype=np.float64)
