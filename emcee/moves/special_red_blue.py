# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from .special_move import Move
from ..state import State

__all__ = ["RedBlueMove"]


class SpecialRedBlueMove(Move):
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
                 nsplits=2,
                 randomize_split=True,
                 live_dangerously=False,
                 ind_beta=8,
                 hop_mode_prob=0.02,
                 inds={'beta': 8, 'inc': 6, 'psi': 9}):
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split
        self.ind_beta = ind_beta
        self.hmp = hop_mode_prob
        self.inds = inds

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by "
                                  "subclasses")

    def transform(self, c):
        c[:,self.inds['beta']] = -c[:,self.inds['beta']]
        c[:,self.inds['inc']] = np.pi - c[:,self.inds['inc']]
        c[:,self.inds['psi']] = np.pi - c[:,self.inds['psi']]
        return c

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

        orig_temp = state.coords.copy()
        if self.hmp > 0.0:
            import pdb; pdb.set_trace()
            jump = np.random.choice([0,1], p=[1-self.hmp, self.hmp], size=self.nwalkers, replace=True)
            if any(jump==1):
                orig_temp[jump==1] = self.transform(orig_temp[jump==1])

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds_mode = (0*(orig_temp[:, self.ind_beta] >= 0.0)
                     + 1*(orig_temp[:, self.ind_beta] < 0.0))

        for split in range(self.nsplits):
            S1 = np.where(inds_mode == split)[0]
            if len(S1) < 2:
                continue

            model.random.shuffle(S1)
            nwalkers_temp = len(S1)
            for num, ind in enumerate(S1):
                temp_walkers = orig_temp[S1].copy()

                # Get the move-specific proposal.
                walkers_comp = np.asarray([temp_walkers[kk] for kk in range(nwalkers_temp)
                                         if kk != num])

                q, factors = self.get_proposal(temp_walkers[num], walkers_comp, model.random)

                # Compute the lnprobs of the proposed position.
                new_log_probs, new_blobs = model.compute_log_prob_fn(np.array([q]))

                # Loop over the walkers and update them accordingly.
                lnpdiff = factors + new_log_probs.item() - state.log_prob[ind]
                af = np.log(model.random.rand())
                #if np.isinf(lnpdiff) and ind ==0:
                #    import pdb; pdb.set_trace()
                #    self.get_proposal(temp_walkers[num], walkers_comp, model.random)
                if lnpdiff > af:
                    accepted[ind] = True
                    new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
                    state = self.update(state, new_state, accepted, ind)


        return state, accepted
