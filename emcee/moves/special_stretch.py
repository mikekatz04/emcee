# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .special_red_blue import SpecialRedBlueMove

__all__ = ["StretchMove"]


class SpecialStretchMove(SpecialRedBlueMove):
    """
    A `Goodman & Weare (2010)
    <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <http://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """
    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(SpecialStretchMove, self).__init__(**kwargs)

    def get_proposal(self, current_walker, other_walkers, random):
        nwalkers_other = other_walkers.shape[0]
        ndim = other_walkers.shape[1]
        draw_ind = random.randint(nwalkers_other)
        zz = ((self.a - 1.) * random.rand() + 1) ** 2. / self.a
        factors = (ndim - 1.) * np.log(zz)
        new_pos = other_walkers[draw_ind] + zz*(current_walker - other_walkers[draw_ind])
        return new_pos, factors
