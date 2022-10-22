#!/bin/python
# Small script to generate isotropically distributed non-overlapping prisms
# Used with molecular-dna in Geant4 to generate test geometries
#
# For usage, type python prisms.py help
#
# (c) Nathanael Lampe, 2016

from typing import List, Set, Union

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d  # NOQA
from tqdm import tqdm


class Prism(object):
    """ """

    def __init__(
        self, center: np.array, size: np.array, axis: np.array, rotation: float = 0
    ):
        """
        Prism(center, size, axis, rotation=0)

        Make a prism object
        x-axis is translated to point along the axis vector.
        rotation is the rotation of the shape around the x-axis
        """
        assert center.shape == (3,), "center should be a 3-vector"
        assert size.shape == (3,), "size should be a 3-vector"
        assert axis.shape == (3,), "axis should be a 3-vector"
        self.center = center
        self.size = size
        self.axis = axis / np.linalg.norm(axis)
        self.angx = rotation
        self.angy = -np.arctan2(axis[2], (axis[0] ** 2 + axis[1] ** 2) ** 0.5)
        self.angz = np.arctan2(axis[1], axis[0])
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.angx), -np.sin(self.angx)],
                [0, np.sin(self.angx), np.cos(self.angx)],
            ]
        )
        ry = np.array(
            [
                [np.cos(self.angy), 0, np.sin(self.angy)],
                [0, 1, 0],
                [-np.sin(self.angy), 0, np.cos(self.angy)],
            ]
        )
        rz = np.array(
            [
                [np.cos(self.angz), -np.sin(self.angz), 0],
                [np.sin(self.angz), np.cos(self.angz), 0],
                [0, 0, 1],
            ]
        )
        self.rotation = np.dot(rz, np.dot(ry, rx))
        self.norm1 = np.dot(self.rotation, np.array([0, 0, 1]))
        self.norm2 = np.dot(self.rotation, np.array([0, -1, 0]))

        corners = [
            0.5 * np.array([size[0], size[1], size[2]]),
            0.5 * np.array([size[0], size[1], -size[2]]),
            0.5 * np.array([size[0], -size[1], size[2]]),
            0.5 * np.array([size[0], -size[1], -size[2]]),
            0.5 * np.array([-size[0], size[1], size[2]]),
            0.5 * np.array([-size[0], size[1], -size[2]]),
            0.5 * np.array([-size[0], -size[1], size[2]]),
            0.5 * np.array([-size[0], -size[1], -size[2]]),
        ]
        self.corners = [
            self.center + np.dot(self.rotation, corner) for corner in corners
        ]

    def to_text(self) -> str:
        """
        Describe prism according to the following text specification:
        POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI
        """
        euler = get_euler_angles(self.rotation)
        return " ".join(
            map(
                str,
                [
                    self.center[0],
                    self.center[1],
                    self.center[2],
                    euler[0],
                    euler[1],
                    euler[2],
                ],
            )
        )

    def to_series(self) -> pd.Series:
        """Return the prism as a pandas series object."""
        euler = get_euler_angles(self.rotation)
        ss = pd.Series(
            {
                "TYPE": "prism",
                "POS_X": self.center[0],
                "POS_Y": self.center[1],
                "POS_Z": self.center[2],
                "EUL_PSI": euler[0],
                "EUL_THETA": euler[1],
                "EUL_PHI": euler[2],
            }
        )
        return ss

    def to_plot(self, n: int = 200, ax: plt.Axes = None, arrows: bool = False):
        """
        Prism.to_plot(n)

        plot in 3d using n points
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        x = []
        y = []
        z = []
        for ii in range(n):
            p = self.get_point()
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        ax.scatter(x, y, z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if arrows is True:
            for (axis, sz) in zip([self.axis, self.norm1, self.norm2], self.size):
                a = Arrow3D(
                    self.center,
                    axis * sz + self.center,
                    mutation_scale=20,
                    lw=1,
                    arrowstyle="-|>",
                    color="k",
                )
                ax.add_artist(a)

        if ax is None:
            return fig
        else:
            return None

    def does_overlap(self, other):
        """
        Evaluate if the two prisms overlap using a separating axis test
        This is reasonably easy with prisms, as we only need to compare to
        the axes of each wall, using all the corners.
        """
        assert isinstance(other, type(self)), "argument must be a object"

        d = sum((self.center - other.center) ** 2)
        sum_rads = sum(self.size**2) + sum(other.size**2)

        if d > sum_rads:
            return False
        else:
            # do a separating axis test
            test_axes = [
                self.axis,
                self.norm1,
                self.norm2,
                other.axis,
                other.norm1,
                other.norm2,
            ]

            overlaps = [
                is_separating_axis(ax, self.corners, other.corners) for ax in test_axes
            ]

            separating_axis_exists = True in overlaps

            return not separating_axis_exists

    def get_point(self):
        """
        Return a point inside the prism
        """
        random = np.random.rand(3) - 0.5
        point = self.center + np.dot(self.rotation, random * self.size)
        return point

    def contains_point(self, point):
        """
        Prism.contains_point(point)

        Returns True if point is in prism
        """
        overlaps = [
            is_separating_axis(ax, self.corners, point)
            for ax in [self.axis, self.norm1, self.norm2]
        ]
        return True not in overlaps


class PrismList(object):
    """
    A List of Non-Overlapping Rectangular Prisms

    This class facilitates generation of a set of prisms so that when a new
    prism is added, it:

    a) does not overlap other prisms and
    b) only checks for overlaps against nearby prisms

    A 'prism grid' is constructed which divides the space within
    '+/- extent' of the grid centre into cells with side length
    specified by 'resolution'. This helper grid is used so that
    the routine only needs to see if the prism being placed overlaps
    local prisms.


    :param center: XYZ-position where the prism should be centered
    :param extent: XYZ-half lengths for the maximum bounding cube
                   of all prisms to be placed.
    :param resolution: The size of a 'cell' in the prism grid.

    """

    def __init__(self, center: List, extent: List, resolution: float):
        if len(center) != 3:
            raise ValueError("center must be a 3-vector")
        if len(extent) != 3:
            raise ValueError("extent must be a 3-vector")

        self.center = np.array(center)
        self.extent = np.array(extent)
        self.resolution = resolution

        # Calculate the maximum number of divisions needed along the
        # longest axis of the prism.
        self.divisions = 2 * max(self.extent) // self.resolution + 1

        # Make a list to hold lists of all prisms potentially in a grid cell.
        # If we denote a cell in the x direction by i, a cell in the y direction
        # by j, and a cell in the z-direction by k, the unique index in the
        # prism grid of the ijk-th cell is int(i + j*divisions + k*divisions^2)
        self.prism_grid = [[] for ii in range(int(self.divisions**3))]
        self.prisms = []

    def append(self, prism: Prism) -> bool:
        """Append a new prism to the Prism list

        :param prism: prism to be appended.
        :returns: True if a prism is placed, else False
        """
        if not isinstance(prism, Prism):
            raise ValueError("Expected a Prism object")

        # get prism grid cells to check
        indices = self.get_prism_grid_indices(prism)

        for index in indices:
            for test_prism in self.prism_grid[index]:
                if prism.does_overlap(test_prism):
                    return False

        # if no overlaps, append the prism to the prism grid
        for index in indices:
            self.prism_grid[index].append(prism)
        self.prisms.append(prism)

        return True

    def to_frame(self, suppress_hash: bool = False) -> pd.DataFrame:
        """Generate a Data Frame describing the prisms

        :param suppress_hash: Hide the hash in front of the 'IDX' column
            which is kept for compatibiilty with the Geant4
            DNA simulation format

        :returns: Voxelised fractal as a data frame
        """
        series_list = []
        hash_option = "" if suppress_hash else "#"
        for ii, prism in enumerate(self.prisms):
            ss = prism.to_series()
            ss[hash_option + "IDX"] = ii
            series_list.append(ss)
        df = pd.DataFrame(series_list)
        df = df[
            [
                hash_option + "IDX",
                "TYPE",
                "POS_X",
                "POS_Y",
                "POS_Z",
                "EUL_PSI",
                "EUL_THETA",
                "EUL_PHI",
            ]
        ]
        return df

    def get_prism_grid_indices(self, prism: Prism) -> Set[int]:
        """Get the prism grid indices"""
        # get the corners of the incoming prism
        ijks = [self._position_to_ijk(corner) for corner in prism.corners]
        ijks = np.array(ijks)

        # Get the x, y and z co-ordinates spanned by the prism
        ii = np.arange(min(ijks[:, 0]), max(ijks[:, 0] + 1), 1)
        jj = np.arange(min(ijks[:, 1]), max(ijks[:, 1] + 1), 1)
        kk = np.arange(min(ijks[:, 2]), max(ijks[:, 2] + 1), 1)
        indices = set(
            [
                self._ijk_to_index(np.array([i_, j_, k_]))
                for i_ in ii
                for j_ in jj
                for k_ in kk
            ]
        )
        return indices

    def _position_to_ijk(self, position: np.array) -> np.array:
        """Convert a position to ijk-coordinates in the prism grid"""
        # transform the local position onto the grid (which is +ve values only)
        local = position - self.center + self.extent
        ijk = local // self.resolution
        return ijk

    def _index_to_ijk(self, index: int) -> np.array:
        """Convert an index value in the prism grid back to ijk coords

        This reverses the equation:
        index = int(i + j*divisions + k*divisions^2)

        where i,j,k < divisions
        """
        ii = index % self.divisions
        jj = ((index - ii) % self.divisions**2) / self.divisions
        kk = (index - ii - self.divisions * jj) / self.divisions**2
        return np.array([ii, jj, kk])

    def _ijk_to_index(self, ijk: Union[np.array, List]) -> int:
        """Convert an ijk reference in the 'prism grid' to a single index:

        The ijk-th cell is in the grid is int(i + j*divisions + k*divisions^2)
        """
        return int(
            ijk[0] + self.divisions * ijk[1] + self.divisions**2 * ijk[2]
        )  # NOQA


def get_euler_angles(rotmatrix: np.array) -> np.array:
    """Get Euler Angles from Rotation Matrix

    :param rotmatrix: Rotation Matrix
    :returns: Euler Angles
    """
    sintheta = rotmatrix[2, 0]
    if abs(sintheta) != 1:
        theta = -np.arcsin(rotmatrix[2, 0])
        costheta = np.cos(theta)
        psi = np.arctan2(rotmatrix[2, 1] / costheta, rotmatrix[2, 2] / costheta)
        phi = np.arctan2(rotmatrix[1, 0] / costheta, rotmatrix[0, 0] / costheta)
    else:
        phi = 0
        if sintheta < 0:  # Positive case
            theta = np.pi / 2.0
            psi = phi + np.arctan2(rotmatrix[0, 1], rotmatrix[0, 2])
        else:
            theta = -np.pi / 2.0
            psi = -phi + np.arctan2(-rotmatrix[0, 1], -rotmatrix[0, 2])

    return np.array([psi, theta, phi])


class Arrow3D(FancyArrowPatch):
    """Helper class to plot arrows"""

    def __init__(self, start, end, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = ([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def random_point_in_ball(rad=1) -> np.array:
    """
    random_point_in_ball(rad=1)

    Return a point uniformly distributed in the radius 'rad' ball

    :param rad: Radius of ball
    :returns: 3-vector array
    """
    p = 2 * (np.random.rand(3) - 0.5)
    while sum(p**2) > 1:
        p = 2 * (np.random.rand(3) - 0.5)
    return rad * p


def random_direction() -> np.array:
    """
    Generate a random 3-vector

    :returns: euler angles for a random direction
    """
    theta = 2 * np.pi * np.random.rand()
    u = 2 * (np.random.rand() - 0.5)
    k = (1 - u * u) ** 0.5
    return np.array([k * np.cos(theta), k * np.sin(theta), u])


def is_separating_axis(axis, points1, points2):
    """
    test_projection(axis, points1, points2)

    Test if two sets of points in 3-D overlap or not after projection onto an
    axis
    """
    if type(points1) != list:
        points1 = [points1]
    if type(points2) != list:
        points2 = [points2]
    distances1 = np.array([np.dot(axis, point) for point in points1])
    distances2 = np.array([np.dot(axis, point) for point in points2])
    grouped = [(distance, distances2) for distance in distances1]

    d1_smaller = [np.all(d < dists2) for (d, dists2) in grouped]
    d1_bigger = [np.all(d > dists2) for (d, dists2) in grouped]

    is_d1_smaller = False if False in d1_smaller else True
    is_d1_bigger = False if False in d1_bigger else True

    can_be_separated = is_d1_smaller or is_d1_bigger
    # return True if there is an overlap (both are false)
    return can_be_separated


def generate_non_overlapping_prisms(
    n_prisms: int,
    size: Union[np.array, List],
    rad: float,
    early_exit: int = -1,
    verbose: bool = False,
) -> PrismList:
    """Generate a collection of non-overlapping prisms

    Candidate prisms are generated at random and placed in a spherical volume.
    Overlaps are then checked and a new candidate is chosen if there is an overlap.
    The algorithm will exit early if early_exit is set to a number greater than 0.

    :param n_prisms: Number of prisms to generate
    :param size: size of prism in x, y, z directions
    :param rad: radius of ball in which to generate prisms
    :param early_exit: stop attempting to place 'early_exit' prisms due to too many overlaps (-1 to disable)
    :param verbose: Display progress of algorithm
    """
    size = np.asarray(size)
    if size.shape != (3,):
        raise ValueError(f"Size should be a 3-vector like np.array([1, 2, 3])")

    def new_prism(rad=rad):
        position = random_point_in_ball(rad)
        axis = random_direction()
        rotation = np.random.rand() * 2 * np.pi
        return Prism(position, size, axis, rotation)

    mag = np.sqrt(np.sum(size * size))
    extent = np.ones(3) * (mag + rad)
    n_prisms_placed = 0
    prisms = PrismList(np.zeros(3), extent, (mag + rad) / 6)
    prisms.append(new_prism(rad))
    attempts = 0
    if verbose is True:
        pbar = tqdm(iterable=None, total=n_prisms)
    while n_prisms_placed < n_prisms:
        if attempts >= early_exit and early_exit >= 0:
            sys.stderr.write(
                f"Too many failed prism placements, stopped after {n_prisms_placed} placements."
            )
            break
        this_prism = new_prism()

        if prisms.append(this_prism) is True:
            if verbose is True:
                pbar.update(1)
            attempts = 0
            n_prisms_placed += 1
        else:
            attempts += 1
    if verbose is True:
        pbar.close()
    return prisms


# if __name__ == "__main__":
#     usage = """
#     prisms.py num N size X Y Z rad R
#     num  N: number of prisms to generate
#     size X Y Z: dimensions of prisms to place
#     rad R: radius of sphere in which to place prisms
#     """
#     if "help" in sys.argv:
#         print(usage)
#         sys.exit(0)

#     err1 = (len(sys .argv) != 9)
#     err2 = ("num" not in sys.argv) or ("size" not in sys.argv) or\
#            ("rad" not in sys.argv)

#     if err1 or err2:
#         print(usage)
#     else:
#         try:
#             n_idx = sys.argv.index("num")
#             n = int(sys.argv[n_idx + 1])
#             size_idx = sys.argv.index("size")
#             sx = float(sys.argv[size_idx + 1])
#             sy = float(sys.argv[size_idx + 2])
#             sz = float(sys.argv[size_idx + 3])
#             size = np.array([sx, sy, sz])
#             rad_idx = sys.argv.index("rad")
#             rad = float(sys.argv[rad_idx + 1])
#         except:
#             print(usage)
#             sys.exit(0)

#         prisms = non_overlapping_prisms(n, size, rad, verbose=False)
#         header = "# IDX TYPE POS_X POS_Y POS_Z EUL_PSI EUL_THETA EUL_PHI\n"
#         outstring = "\n".join(["{} prism ".format(ii) + p.to_text()
#                                for ii, p in enumerate(prisms.prisms)])
#         sys.stdout.write(header + outstring)
