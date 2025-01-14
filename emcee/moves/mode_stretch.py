# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .mode_red_blue import ModeRedBlueMove

__all__ = ["StretchMove"]


class ModeStretchMove(ModeRedBlueMove):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """
    def __init__(self, nwalkers, long_inject, a=2.0, **kwargs):
        self.a = a
        self.nwalkers = nwalkers
        ModeRedBlueMove.__init__(self, **kwargs)

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        zz = ((self.a - 1.) * random.rand(Ns) + 1) ** 2. / self.a
        factors = (ndim - 1.) * np.log(zz)
        rint = random.randint(Nc, size=(Ns,))
        return c[rint] - (c[rint] - s) * zz[:, None], factors
