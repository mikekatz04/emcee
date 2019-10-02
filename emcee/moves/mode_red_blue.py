# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .move import Move
from ..state import State

__all__ = ["ModeRedBlueMove"]


class ModeRedBlueMove(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <http://arxiv.org/abs/1202.3665>`_.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.

        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.

    """
    def __init__(self,
                 nwalkers=None,
                 nsplits=2,
                 randomize_split=True,
                 live_dangerously=False):

        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split
        self.long_mode = np.zeros(self.nwalkers, dtype=int)
        self.lat_mode = np.zeros(self.nwalkers, dtype=int)
        self.psi_mode = np.zeros(self.nwalkers, dtype=int)
        self.inc_mode = np.zeros(self.nwalkers, dtype=int)

    def setup(self, coords):
        self.long_mode = coords[:, 7] // (np.pi/2.)
        self.lat_mode = np.sign(coords[:, 8]).astype(int)
        self.psi_mode = np.sign(coords[:, 9] - np.pi/2).astype(int)
        self.inc_mode = np.sign(coords[:, 6]).astype(int)

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by "
                                  "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError("It is unadvisable to use a red-blue move "
                               "with fewer walkers than twice the number of "
                               "dimensions.")

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        stop = False

        for split in range(self.nsplits):
            S1 = inds == split

            state.coords[:, 7] = state.coords[:,7] % (np.pi/2.)
            state.coords[:, 8] = np.abs(state.coords[:, 8])
            state.coords[:, 9] = self.psi_mode*(state.coords[:, 9] - np.pi/2.)
            state.coords[:, 6] = np.abs(state.coords[:, 6])

            # Get the two halves of the ensemble.
            sets = [state.coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = sets[:split] + sets[split+1:]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)

            q[:, 7] = (self.long_mode[S1]*q[:, 7]) % (2*np.pi)
            q[:, 8] = self.lat_mode[S1]*q[:, 8]
            q[:, 9] = (self.psi_mode[S1]*q[:, 9]) + np.pi/2.
            q[:, 6] = self.inc_mode[S1]*q[:, 6]

            state.coords[:, 7] = state.coords[:,7]*self.long_mode
            state.coords[:, 8] = state.coords[:, 8]*self.lat_mode
            state.coords[:, 9] = np.pi/2. + (self.psi_mode*state.coords[:, 9])
            state.coords[:, 6] = state.coords[:, 6]*self.inc_mode

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp) in enumerate(zip(
                    all_inds[S1], factors, new_log_probs)):
                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1)

        return state, accepted
