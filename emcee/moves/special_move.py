# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

__all__ = ["Move"]


class Move(object):

    def tune(self, state, accepted):
        pass

    def update(self,
               old_state,
               new_state,
               accepted,
               subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        Args:
            coords: The original ensemble coordinates.
            log_probs: The original log probabilities of the walkers.
            blobs: The original blobs.
            new_coords: The proposed coordinates.
            new_log_probs: The proposed log probabilities.
            new_blobs: The proposed blobs.
            accepted: A vector of booleans indicating which walkers were
                accepted.
            subset (Optional): A boolean mask indicating which walkers were
                included in the subset. This can be used, for example, when
                updating only the primary ensemble in a :class:`RedBlueMove`.

        """
        old_state.coords[subset] = new_state.coords
        old_state.log_prob[subset] = new_state.log_prob

        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_prob, "
                    "you also need to provide the current list of "
                    "blobs at that position.")
            old_state.blobs[subset] = new_state.blobs

        return old_state
