# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move

__all__ = ["AboveBelowMove"]


class AboveBelowMove(Move):
    """
    A move for a walker to go above and below the LISA plane.
    Performs correct transformation of LISA frame paramters:
    beta, iota, psi.
    beta' = -beta
    iota' = pi - iota
    psi' = pi - iota

    :param inds:
        Dictionary with indices for beta, iota, psi.

    """
    def __init__(self, ndim, inds, **kwargs):
        self.inds = inds
        self.ndim = ndim
        super(AboveBelowMove, self).__init__(**kwargs)

    def transform(self, c):
        c[:,self.inds['beta']] = -c[:,self.inds['beta']]
        c[:,self.inds['iota']] = np.pi - c[:,self.inds['iota']]
        c[:,self.inds['psi']] = np.pi - c[:,self.inds['psi']]
        return c

    def get_proposal(self, coords, random):
        return self.transform(coords), np.zeros(coords.shape[0])

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

        # Get the move-specific proposal.
        q, factors = self.get_proposal(state.coords, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted
