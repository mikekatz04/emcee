# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move
from ..state import State

__all__ = ["AboveBelowMove"]


class AboveBelowMove(Move):
    """
    A move for a walker to go above and below the LISA plane.
    Performs correct transformation of LISA frame paramters:
    beta, iota, psi.
    beta' = -beta
    inc' = pi - inc
    psi' = pi - psi

    :param inds:
        Dictionary with indices for beta, inc, psi.

    """
    def __init__(self, nwalkers, ndim, inds, cov, prob_jump_above_below=0.5, **kwargs):
        self.inds = inds
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.buffer = np.zeros((nwalkers, ndim))
        self.prob_jump_above_below = prob_jump_above_below
        self.cov = cov
        self.means = np.zeros(nwalkers)
        super(AboveBelowMove, self).__init__(**kwargs)

    def transform(self, c):
        c[:,self.inds['beta']] = -c[:,self.inds['beta']]
        c[:,self.inds['inc']] = np.pi - c[:,self.inds['inc']]
        c[:,self.inds['psi']] = np.pi - c[:,self.inds['psi']]
        return c

    def get_proposal(self, coords, random):
        import pdb; pdb.set_trace()
        jump = np.choice([0,1], size=self.nwalkers, replace=True)
        new_coords = self.transform(coords)*(jump==1) + coords*(jump==0)
        new_coords = new_coords + np.random.multivariate_normal(self.means, self.cov, size=new_coords.shape[0])
        return new_coords, np.zeros(coords.shape[0])

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check to make sure that the dimensions match.
        nwalkers, ndim = state.coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")

        self.buffer[:] = state.coords[:]
        # Get the move-specific proposal.
        q, factors = self.get_proposal(self.buffer, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted
