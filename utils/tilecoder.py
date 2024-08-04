"""From https://github.com/AmiiThinks/rltoolkit/blob/rlglue_gridworld/RLtoolkit/tilecoding.py"""

import numpy as np
from typing import Sequence, Optional


class TileCoder:

    def __init__(self,
                 tiling_dims: Sequence[int],
                 limits_per_dim: Sequence[Sequence[float]],
                 num_tilings: int,
                 wrap: Optional[Sequence[bool]] = None,
                 offset=lambda n: 2 * np.arange(n) + 1,
                 style="indices"):

        # check style
        assert style in ["indices", "vector"]
        self.style = style

        # normalize the ith input float to be between 0 and tile_dims[i]
        self._limits = np.array(limits_per_dim)
        self._norm_dims = (np.array(tiling_dims)
                           / (self._limits[:, 1] - self._limits[:, 0]))

        # wrapping means adding 1 to the ith dim or not
        if wrap is None:
            wrap = np.ones(len(tiling_dims), dtype=bool)
        else:
            wrap = ~np.array(wrap, dtype=bool)
        self.wrap = not bool(wrap.sum())
        self.tiling_dims = np.array(tiling_dims, dtype=np.int) + wrap

        # displacement matrix; default is assymetric displacement a la Parks
        # and Militzer https://doi.org/10.1016/S1474-6670(17)54222-6
        offset_vec = offset(len(tiling_dims))
        self._offsets = (offset_vec
                         * np.repeat([np.arange(num_tilings)], len(tiling_dims), 0).T
                         / float(num_tilings)
                         % 1)

        # these send each displaced float to the proper index
        self._tiling_loc = np.arange(num_tilings) * np.prod(self.tiling_dims)
        self._tile_loc = np.array([np.prod(self.tiling_dims[0:i])
                                   for i in range(len(tiling_dims))])

        # the total number of indices needed
        self._n_tiles = num_tilings * np.prod(self.tiling_dims)

    @property
    def n_tiles(self):
        return self._n_tiles

    def getitem(self, x):

        if self.wrap:
            # wrapping means modding by dim[i] instead of dim[i] + 1
            off_coords = (((x - self._limits[:, 0])
                           * self._norm_dims
                           + self._offsets) % self.tiling_dims).astype(int)
        else:
            # don't need to mod here, because dim[i] + 1 is bigger than the
            # displaced floats
            off_coords = (((x - self._limits[:, 0])
                           * self._norm_dims
                           + self._offsets)).astype(int)

        ones = self._tiling_loc + np.dot(off_coords, self._tile_loc)

        if self.style == "indices":
            return ones
        else:
            vec = np.zeros(self._n_tiles)
            vec[ones] = 1

            return vec


def test_tilecoder():
    tc = TileCoder(tiling_dims=[2, 2], limits_per_dim=[[0, 1], [0, 1]], num_tilings=4, style='vector')
    # tc.
    # tc.__getitem__([0.5, 0.5])