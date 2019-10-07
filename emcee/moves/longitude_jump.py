# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move
from ..state import State

__all__ = ["LongitudeMove"]


class LongitudeMove(Move):
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
    def __init__(self, nwalkers, ndim, inds, hop_longtitude=True, **kwargs):
        self.inds = inds
        self.ndim = ndim
        self.buffer = np.zeros((nwalkers, ndim))
        self.hop_longtitude = hop_longtitude
        self.nsplits = kwargs.get('nsplits', 2)
        self.nwalkers = nwalkers
        self.inds_split = np.split(np.arange(nwalkers), 2)
        super(LongitudeMove, self).__init__(**kwargs)

    def transform(self, c):
        jump_val = np.pi/2.*np.random.choice([0., 1., 2., 3,],
            replace=True, size=c.shape[0])
        c[:,self.inds['lambda']] = (c[:,self.inds['lambda']] + jump_val) % (2*np.pi)
        c[:, self.inds['psi']] = (c[:, self.inds['psi']] + jump_val) % np.pi
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

        self.buffer[:] = state.coords[:]
        # Get the move-specific proposal.
        q, factors = self.get_proposal(self.buffer, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs = np.zeros(self.nwalkers)
        for split in self.inds_split:
            new_log_probs[split], new_blobs = model.compute_log_prob_fn(q[split])


        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted
