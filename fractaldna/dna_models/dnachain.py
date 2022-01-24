"""
Class description of a DNA chain built of base pairs
"""

from typing import List, Tuple, Union

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from scipy.interpolate import interp1d

try:
    from mayavi import mlab

    maya_imported = True
except ImportError:
    maya_imported = False
    print("Could not import mayavi libraries, 3d plotting is disabled")

from fractaldna.dna_models import basepair
from fractaldna.utils import rotations as r
from fractaldna.utils.constants import BP_ROTATION, BP_SEPARATION


class PlottableSequence:
    """
    This is an inheritable class that gives DNA chains plotting methods and
    output methods.
    """

    def to_text(self, seperator: str = " ") -> str:
        """
        Return a description of the molecules in the chain as text

        :param seperator: column seperator
        """
        key = (
            "#NAME SHAPE CHAIN_ID STRAND_ID BP_INDEX "
            + "SIZE_X SIZE_Y SIZE_Z POS_X "
            + "POS_Y POS_Z ROT_X ROT_Y ROT_Z\n"
        )
        output = [key.replace(" ", seperator)]
        for pair in self.basepairs:
            output.append(pair.to_text(seperator=seperator))

        return "".join(output)

    def to_frame(self) -> pd.DataFrame:
        """
        Return the molecules as a pandas data frame

        :return: Pandas data frame with molecule information
        """
        return pd.concat(
            [pair.to_frame() for pair in self.basepairs], ignore_index=False, sort=False
        )

    def to_plot(
        self, plot_p: bool = True, plot_b: bool = True, plot_s: bool = True
    ) -> matplotlib.figure.Figure:
        """
        Return a matplotlib.Figure instance with molecules plotted

        :param plot_p: Show Phosphates in plot
        :param plot_b: Show Bases in plot
        :param plot_s: Show sugars in plot

        :return: Matplotlib Figure
        """
        sugars = []
        triphosphates = []
        bases = []
        bps = ["guanine", "adenine", "thymine", "cytosine"]
        for pair in self.basepairs:
            for (name, molecule) in pair.iterMolecules():
                if molecule.name.lower() == "sugar":
                    sugars.append(molecule.position)
                elif molecule.name.lower() == "phosphate":
                    triphosphates.append(molecule.position)
                elif molecule.name.lower() in bps:
                    bases.append(molecule.position)

        # Plotting
        empty = [[], [], []]
        bases = [ii for ii in zip(*map(list, bases))] if plot_b else empty
        triphosphates = (
            [ii for ii in zip(*map(list, triphosphates))] if plot_p else empty
        )
        sugars = [ii for ii in zip(*map(list, sugars))] if plot_s else empty

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(bases[0], bases[1], bases[2], c="0.6", s=20)
        ax.scatter(triphosphates[0], triphosphates[1], triphosphates[2], c="y", s=20)
        ax.scatter(sugars[0], sugars[1], sugars[2], c="r", s=20)

        return fig

    def to_surface_plot(self) -> matplotlib.figure.Figure:
        """
        Plot the surfaces of each molecule in the chain.
        Avoid this with large chains, this assumes each molecule is an ellipse

        :return: Matplotlib figure (contour plot)
        """

        def ellipse_xyz(center, extent, rotation=np.zeros([3])):
            rmatrix = r.eulerMatrix(*rotation)
            [a, b, c] = extent
            u, v = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 5j]
            x = a * np.cos(u) * np.sin(v) + center[0]
            y = b * np.sin(u) * np.sin(v) + center[1]
            z = c * np.cos(v) + center[2]
            for ii in range(0, len(x)):
                for jj in range(0, len(x[ii])):
                    row = np.array([x[ii][jj], y[ii][jj], z[ii][jj]]) - center
                    xp, yp, zp = np.dot(rmatrix, row.transpose())
                    x[ii][jj] = xp + center[0]
                    y[ii][jj] = yp + center[1]
                    z[ii][jj] = zp + center[2]
            return x, y, z

        sugars = []
        triphosphates = []
        bases = []
        bps = ["guanine", "adenine", "thymine", "cytosine"]
        for pair in self.basepairs:
            for (name, molecule) in pair.iterMolecules():
                if molecule.name.lower() == "sugar":
                    sugars.append(
                        (molecule.position, molecule.dimensions, molecule.rotation)
                    )
                elif molecule.name.lower() == "phosphate":
                    triphosphates.append(
                        (molecule.position, molecule.dimensions, molecule.rotation)
                    )
                elif molecule.name.lower() in bps:
                    bases.append(
                        (molecule.position, molecule.dimensions, molecule.rotation)
                    )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for base in bases:
            x, y, z = ellipse_xyz(base[0], base[1], rotation=base[2])
            ax.plot_wireframe(x, y, z, color="0.6")

        for phosphate in triphosphates:
            x, y, z = ellipse_xyz(phosphate[0], phosphate[1], rotation=phosphate[2])
            ax.plot_wireframe(x, y, z, color="y")

        for sugar in sugars:
            x, y, z = ellipse_xyz(sugar[0], sugar[1], rotation=sugar[2])
            ax.plot_wireframe(x, y, z, color="r")

        return fig

    def to_line_plot(self, size: Tuple[int, int] = (400, 350)):
        """
        Return a mayavi figure instance with histone and linkers shown

        :param size: Figure size (width, height)

        :return: mayavi figure

        :raises ImportError: MayaVi likely Not installed
        """
        if not maya_imported:
            raise ImportError("MayaVi could not be imported")

        if maya_imported is True:
            fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=size)
            if hasattr(self, "histones"):
                # ax = fig.add_subplot(111, projection='3d')
                histones = []
                for histone in self.histones:
                    pos = np.array([bp.position for bp in histone.basepairs])
                    mlab.plot3d(
                        pos[:, 0],
                        pos[:, 1],
                        pos[:, 2],
                        color=(1.0, 0.8, 0),
                        tube_radius=11.5,
                    )
                    histones.append(histone.position)

                histones = np.array(histones)
                mlab.points3d(
                    histones[:, 0],
                    histones[:, 1],
                    histones[:, 2],
                    color=(0, 0, 1.0),
                    opacity=0.4,
                    scale_factor=70,
                )

                for linker in self.linkers:
                    pos = np.array([bp.position for bp in linker.basepairs])
                    mlab.plot3d(
                        pos[:, 0],
                        pos[:, 1],
                        pos[:, 2],
                        color=(0, 0.8, 0),
                        tube_radius=11.5,
                    )

            else:
                chains = set([bp.chain for bp in self.basepairs])
                for chain in chains:
                    pos = np.array(
                        [
                            bp.position
                            for bp in filter(lambda x: x.chain == chain, self.basepairs)
                        ]
                    )
                    mlab.plot3d(
                        pos[:, 0],
                        pos[:, 1],
                        pos[:, 2],
                        color=(1.0, 0, 0),
                        tube_radius=11.5,
                    )

            return fig
        else:
            print("MayaVi not imporrted, cannot produce this plot")
            return None

    def to_strand_plot(self, plot_p=True, plot_b=True, plot_s=True, plot_bp=False):
        """
        Return a mayavi figure instance with strands plotted

        :param plot_p : plot phosphate strands
        :param plot_s : plot sugar strands
        :param plot_b : plot base strands
        :param plot_bp : join base pairs together

        :return: Mayavi Figure

        :raises ImportError: MayaVi not imported
        """
        if not maya_imported:
            raise ImportError("MayaVi Not Imported")

        if maya_imported is True:
            fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
            chains = set([bp.chain for bp in self.basepairs])
            for chain in chains:
                basepairs = [
                    bp for bp in filter(lambda x: x.chain == chain, self.basepairs)
                ]
                sugar_l = []
                sugar_r = []
                phosphate_l = []
                phosphate_r = []
                base_l = []
                base_r = []
                bps = ["guanine", "adenine", "thymine", "cytosine"]
                for pair in basepairs:
                    for (name, molecule) in pair.iterMolecules():
                        if molecule.name.lower() == "sugar":
                            if molecule.strand == 0:
                                sugar_l.append(molecule.position)
                            elif molecule.strand == 1:
                                sugar_r.append(molecule.position)
                        elif molecule.name.lower() == "phosphate":
                            if molecule.strand == 0:
                                phosphate_l.append(molecule.position)
                            elif molecule.strand == 1:
                                phosphate_r.append(molecule.position)
                        elif molecule.name.lower() in bps:
                            if molecule.strand == 0:
                                base_l.append(molecule.position)
                            elif molecule.strand == 1:
                                base_r.append(molecule.position)
                # Plotting
                base_l = [ii for ii in zip(*map(list, base_l))]
                base_r = [ii for ii in zip(*map(list, base_r))]
                phosphate_l = [ii for ii in zip(*map(list, phosphate_l))]
                phosphate_r = [ii for ii in zip(*map(list, phosphate_r))]
                sugar_l = [ii for ii in zip(*map(list, sugar_l))]
                sugar_r = [ii for ii in zip(*map(list, sugar_r))]

                if plot_b:
                    mlab.plot3d(
                        base_l[0],
                        base_l[1],
                        base_l[2],
                        color=(0.6, 0.6, 0.6),
                        tube_radius=1,
                    )
                    mlab.plot3d(
                        base_r[0],
                        base_r[1],
                        base_r[2],
                        color=(0.6, 0.6, 0.6),
                        tube_radius=1,
                    )
                if plot_s:
                    mlab.plot3d(
                        sugar_l[0],
                        sugar_l[1],
                        sugar_l[2],
                        color=(1.0, 0, 0),
                        tube_radius=1,
                    )
                    mlab.plot3d(
                        sugar_r[0],
                        sugar_r[1],
                        sugar_r[2],
                        color=(1.0, 0, 0),
                        tube_radius=1,
                    )
                if plot_p:
                    mlab.plot3d(
                        phosphate_l[0],
                        phosphate_l[1],
                        phosphate_l[2],
                        color=(1, 1, 0),
                        tube_radius=1,
                    )
                    mlab.plot3d(
                        phosphate_r[0],
                        phosphate_r[1],
                        phosphate_r[2],
                        color=(1, 1, 0),
                        tube_radius=1,
                    )

                if plot_bp:
                    # plot bars joining base pairs
                    for ii in range(0, len(base_l[0])):
                        xs = (
                            phosphate_l[0][ii],
                            sugar_l[0][ii],
                            base_l[0][ii],
                            base_r[0][ii],
                            sugar_r[0][ii],
                            phosphate_r[0][ii],
                        )
                        ys = (
                            phosphate_l[1][ii],
                            sugar_l[1][ii],
                            base_l[1][ii],
                            base_r[1][ii],
                            sugar_r[1][ii],
                            phosphate_r[1][ii],
                        )
                        zs = (
                            phosphate_l[2][ii],
                            sugar_l[2][ii],
                            base_l[2][ii],
                            base_r[2][ii],
                            sugar_r[2][ii],
                            phosphate_r[2][ii],
                        )
                        mlab.plot3d(xs, ys, zs, color=(1, 1, 1), tube_radius=0.5)

            return fig
        else:
            print("MayaVi not imporrted, cannot produce this plot")
            return None


class SplineLinker(PlottableSequence):
    """
    *Inherits from PlottableSequence*

    Link two histones together via a cubic spline

    linker = SplineLinker(bp1, bp2, bp3, bp4, curviness=1., zrot=None)

    Create base pairs that link to sections of DNA as follows:
    bp1 bp2 <==== LINKER =====> bp3 bp4
    Two base pairs on either side of the linker are needed to build splines
    low curviness = straighter
    high_curviness = smoother

    startkey and stopkey act as keyframes for rotations

    :param zrot:
        Describes the twist of bp3 relative to bp2
            - None: Determine automatically (experimental)
            - double: rotation in radians (mod 2*pi)

    :param startkey:
        Starting keyframe for rotations
            - key = i will start/stop rotations after the i-th base pair
            - key = -i will start/stop rotations after the i-th last base pair

    :param stopkey:
        Ending keyframe for rotations
            - key = i will start/stop rotations after the i-th base pair
            - key = -i will start/stop rotations after the i-th last base pair

    :param method:
        Method to handle rotational interpolation
            - "quaternion": Full use of quaternions
            - "corrected_quaternion": Full use of quaternions, with a correction
              to check that the base pair is aligned.
              This method is recommended
            - "matrix": Experimental method that doesn't use
              quaternions. Currently incorrect.

    :param chain:
        Chain index assigned to linker and basepairs created therein
    """

    linker_rotation = BP_ROTATION  # rad, default screw rotation of dna
    linker_bp_spacing = BP_SEPARATION  # angstrom, default spacing between bps

    def __init__(
        self,
        bp1: basepair.BasePair,
        bp2: basepair.BasePair,
        bp3: basepair.BasePair,
        bp4: basepair.BasePair,
        curviness: float = 1.0,
        zrot: float = None,
        startkey: int = None,
        stopkey: int = None,
        method: str = "corrected_quaternion",
        chain: int = 0,
    ):
        """
        Constructor
        """
        assert method in [
            "quaternion",
            "matrix",
            "corrected_quaternion",
        ], "Invalid interpolation method"
        self.basepairs = []
        self.chain = chain
        points = np.array([bp1.position, bp2.position, bp3.position, bp4.position])
        # start_x = bp2.rmatrix[:, 0]
        # end_x = bp3.rmatrix[:, 0]
        # this line is incorrect. We need to transfer the two rmatrices into
        # the same frame
        # relative_angle = np.arccos(np.sum(start_x*end_x) /
        #                            np.sum(start_x**2)**.5 /
        #                            np.sum(end_x**2)**.5)
        # d = np.sum((bp2.position - bp3.position)**2)**.5
        if curviness <= 0:
            curviness = 1e-200
        diff = 3.4 / 180 / curviness
        t = np.array([1 - diff, 1, 2, 2 + diff])
        x_interp = interp1d(t, points[:, 0], kind="cubic")
        y_interp = interp1d(t, points[:, 1], kind="cubic")
        z_interp = interp1d(t, points[:, 2], kind="cubic")
        self.x_interp = x_interp
        self.y_interp = y_interp
        self.z_interp = z_interp

        # calculate length
        tt = np.linspace(1, 2, 1000)
        xx = x_interp(tt)
        yy = y_interp(tt)
        zz = z_interp(tt)
        dx = xx[1:] - xx[: (len(xx) - 1)]
        dy = yy[1:] - yy[: (len(yy) - 1)]
        dz = zz[1:] - zz[: (len(zz) - 1)]
        length = sum((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5)
        n = length // self.linker_bp_spacing
        self.spacing = length / n
        # print(self.spacing)
        tt = np.linspace(1, 2, int(n))
        xx = x_interp(tt[1 : len(tt)])
        yy = y_interp(tt[1 : len(tt)])
        zz = z_interp(tt[1 : len(tt)])
        interpolator = lambda t: np.array(
            [x_interp(t + 1), y_interp(t + 1), z_interp(t + 1)]
        )

        if startkey is None:
            startkey = 0
        elif startkey < 0:
            startkey = len(tt) - abs(startkey) - 1
        if stopkey is None:
            stopkey = len(tt) - 1
        elif stopkey < 0:
            stopkey = len(tt) - abs(stopkey)

        # total rotation that the BP undergoes, relative to initial
        rotation_unmodified = (-n * self.linker_rotation) % (2 * np.pi)

        if zrot is None:
            zrot = 0

        # desired rotation relative to initial
        zrot = zrot % (2 * np.pi)
        diff = zrot - rotation_unmodified
        if diff < -np.pi:
            diff += 2 * np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi
        rot_angle = self.linker_rotation + diff / n
        # print("\n", n, diff*180/np.pi, rot_angle*180/np.pi)
        # print("Desired rotation", zrot*180/np.pi)
        # print("Default rotation", rotation_unmodified*180/np.pi)
        # print("Final rotation", ((rot_angle*n) % (2*np.pi))*180/np.pi)

        # Run one loop to generate a series of rotation matrices
        for (ii, (_x, _y, _z)) in enumerate(zip(xx, yy, zz)):
            if ii != len(xx) - 1:
                pos = np.array([_x, _y, _z])
                bp = basepair.BasePair(
                    np.random.choice(["G", "A", "T", "C"]),
                    chain=chain,
                    position=[0, 0, 0],
                    rotation=[0, 0, 0],
                    index=ii,
                )

                if method in ["quaternion", "corrected_quaternion"]:
                    start_quaternion = r.quaternion_from_matrix(bp2.rmatrix)
                    if ii < startkey:
                        ll = 0
                    elif ii >= stopkey:
                        ll = 1
                    else:
                        ll = (ii - startkey + 1.0) / float(stopkey - startkey)
                    if method == "corrected_quaternion":
                        # get interpolated rotation matrix
                        end_quaternion = r.quaternion_from_matrix(bp3.rmatrix)
                        quat = r.quaternion_slerp(
                            start_quaternion, end_quaternion, ll, shortestpath=True
                        )
                        rmat = r.quaternion_matrix(quat)
                        # correct z
                        z = interpolator((ii + 1) / len(xx)) - interpolator(
                            ii / len(xx)
                        )
                        z = -z / np.linalg.norm(z)
                        z_current = rmat[:, 2]
                        perp = np.cross(z_current, z)
                        angle = np.arccos(
                            np.dot(z, z_current)
                            / (np.linalg.norm(z) * np.linalg.norm(z_current))
                        )
                        r2 = r.rot_ax_angle(perp, angle)
                        rmat = np.dot(r2, rmat)
                        # new_z = bp3.rmatrix[:, 2]
                        # old_x = bp2.rmatrix[:, 0]
                        # old_y = bp2.rmatrix[:, 1]
                        # a = 1
                        # b = -np.dot(new_z, old_y) / np.dot(new_z, old_x)
                        # perp_x = a*old_x + b*old_y
                        # perp_x /= np.linalg.norm(perp_x)
                        # perp_y = np.cross(new_z, perp_x)
                        # end_matrix =\
                        #     np.array([perp_x, perp_y, new_z]).transpose()
                        # end_quaternion = r.quaternion_from_matrix(end_matrix)
                    else:
                        end_quaternion = r.quaternion_from_matrix(bp3.rmatrix)
                        quat = r.quaternion_slerp(
                            start_quaternion, end_quaternion, ll, shortestpath=True
                        )
                        rmat = r.quaternion_matrix(quat)
                elif method == "matrix":
                    ll = ii / len(xx)
                    rmat = r.matrix_interpolate(
                        bp2.rmatrix, bp3.rmatrix, interpolator, ll, precision=0.01
                    )
                bp.rotate(rmat)
                spin = rot_angle * (ii + 1)
                bp.rotate(r.rot_ax_angle(rmat[:, 2], spin))
                bp.translate(pos)
                self.basepairs.append(bp)

        return None

    def translate(self, translation):
        """Translate the histone spatially

        :param translation: 3-vector for translation
        """
        for bp in self.basepairs:
            bp.translate(translation)
        return None

    def setChain(self, chainIdx):
        """Set the Chain Index of all base pairs in the histone

        :param chainIdx: Index for Chain
        """
        self.chain = chainIdx
        for bp in self.basepairs:
            bp.setNewChain(chainIdx)
        return None


class Histone(PlottableSequence):
    """
    *Inherits from PlottableSequence*

    This class defines a histone.

    :param position: 3-vector for histone position
    :param rotation: 3-vector for histone rotation (euler angles)
    :param genome: string defining the genome for the histone
    :param chain: Chain index for histone and basepairs therein
    :param histone_index: An index for the histone (by default, order in the solenoid)
    """

    radius_histone = 25  # radius of histone, angstrom
    pitch_dna = 23.9  # 23.9  # pitch of DNA helix, angstrom
    radius_dna = 41.8  # radius of DNA wrapping, angstrom
    histone_bps = 146  # number of bps in histone
    histone_turns = 1.65 * 2 * np.pi  # angular turn around histone, radians
    height = 27 * 1.65
    z_offset = -height / 2.0  # distance between first bp and xy-plane, angstrom
    # separation of bps around histone, angstrom
    hist_bp_separation = histone_turns * radius_dna / histone_bps
    hist_bp_rotation = BP_ROTATION  # screw rotation of bp, radians
    z_per_bp = height / histone_bps
    turn_per_bp = histone_turns / histone_bps
    z_angle = np.arctan(1.0 / pitch_dna)
    histone_start_bp_rot = 0  # radians, rotation of bp at start of histone
    histone_end_bp_rot = histone_start_bp_rot + histone_bps * hist_bp_rotation
    histone_total_twist = (histone_bps * hist_bp_separation) % (2 * np.pi)

    def __init__(
        self,
        position: Union[List, np.array],
        rotation: Union[List, np.array],
        genome: str = None,
        chain: int = 0,
        histone_index: int = 0,
    ):
        """Create a Histone"""
        assert len(position) == 3, "position is length 3 array"
        assert len(rotation) == 3, "position is length 3 array"
        if genome is None:
            genome = "".join(
                [
                    np.random.choice(["G", "A", "T", "C"])
                    for ii in range(self.histone_bps)
                ]
            )
        assert len(genome) == self.histone_bps, "genome should be {} base pairs".format(
            self.histone_bps
        )
        self.histone_index = histone_index
        self.position = np.array(position)
        self.rotation = np.array(rotation)
        self.chain = chain
        self.basepairs = []
        theta = -0.5 * (self.histone_turns - 3 * np.pi)
        z = self.z_offset
        for ii, char in enumerate(genome):
            bp = basepair.BasePair(
                char,
                chain=chain,
                position=np.array([0, 0, 0]),
                rotation=np.array([0, 0, 0]),
                index=ii,
            )
            # make rotation matrix

            rmatrix = r.rotx(np.pi / 2.0 + self.z_angle)
            rmatrix = np.dot(r.rotz(theta), rmatrix)
            bp.rotate(rmatrix)
            bp.rotate(
                np.dot(
                    rmatrix,
                    np.dot(
                        r.rotz(self.histone_start_bp_rot + ii * self.hist_bp_rotation),
                        np.linalg.inv(rmatrix),
                    ),
                )
            )
            # bp.rotate(np.array([np.pi/2., 0., 0]))
            # bp.rotate(np.array([0, 0, theta]))
            # bp.rotate(np.array([ii*self.turn_per_bp, 0, 0]))
            x = self.radius_dna * np.cos(theta)
            y = self.radius_dna * np.sin(theta)
            position = np.array([x, y, z])
            bp.translate(position)
            theta += self.turn_per_bp
            z += self.z_per_bp
            self.basepairs.append(bp)

        for bp in self.basepairs:
            bp.rotate(self.rotation, about_origin=True)
            bp.translate(self.position)
        return None

    def as_series(self) -> pd.Series:
        """Express the histone as a single molecule in a pandas series

        :returns: Pandas Series for Histone
        """
        return pd.Series(
            {
                "name": "Histone",
                "shape": "sphere",
                "chain_idx": self.chain,
                # "strand_idx": -1,
                "histone_idx": self.histone_index,
                "size_x": self.radius_histone,
                "size_y": self.radius_histone,
                "size_z": self.radius_histone,
                "pos_x": self.position[0],
                "pos_y": self.position[1],
                "pos_z": self.position[2],
                "rot_x": self.rotation[0],
                "rot_y": self.rotation[1],
                "rot_z": self.rotation[2],
            }
        )

    def translate(self, translation: Union[List, np.array]) -> None:
        """Translate the histone spatially

        :param translation: 3-vector for translation
        """
        for bp in self.basepairs:
            bp.translate(translation)
        self.position += translation
        return None

    def setChain(self, chainIdx: int) -> None:
        """Set the Chain Index of all base pairs in the histone

        :param chainIdx: Index for Chain
        """
        self.chain = chainIdx
        for bp in self.basepairs:
            bp.setNewChain(chainIdx)
        return None


class Solenoid(PlottableSequence):
    """
    *Inherits from PlottableSequence*

    Define Solenoidal DNA in a voxel (basically a box).

    This method works by placing histones around the z-axis (≈6 histones
    per rotation) and then joining them together using SplineLinkers

    :param voxelheight: Height of 'voxel' in angstrom
    :param radius: Radius from Solenoid centre to histone centre
    :param nhistones: Number of histones to place
    :param histone_angle: tilt of histones from axis in degrees
    :param twist: whether the DNA exiting the final spine should be
        rotated an extra pi/2.
    :param chain: Chain index for solenoid and basepairs therein

    """

    def __init__(
        self,
        voxelheight: float = 750,
        radius: float = 100,
        nhistones: int = 38,
        histone_angle: float = 50,
        twist: bool = False,
        chain: int = 0,
    ):
        self.radius = radius
        self.voxelheight = voxelheight
        self.nhistones = nhistones
        self.chain = chain
        self.tilt = histone_angle * np.pi / 180.0
        self.zshift = (self.voxelheight - 4.0 * Histone.radius_histone) / self.nhistones
        self.height = (self.nhistones - 1) * self.zshift  # length of the fibre
        prev_bp1 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, -1 * BP_SEPARATION]),
            index=-2,
        )
        prev_bp2 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, -0 * BP_SEPARATION]),
            index=-1,
        )
        rot = np.array([0, 0, np.pi / 2.0]) if twist is True else np.zeros(3)
        next_bp3 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, self.voxelheight + 0.0 * BP_SEPARATION]),
            rotation=rot,
            index=1000,
        )
        next_bp4 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, self.voxelheight + 1.0 * BP_SEPARATION]),
            rotation=rot,
            index=1001,
        )
        self.basepairs = []
        self.positions = [
            np.array([0, -self.radius, 0.5 * (self.voxelheight - self.height)])
        ]
        rm = r.eulerMatrix(np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0)
        rm = np.dot(r.roty(self.tilt), rm)
        self.rotations = [r.getEulerAngles(rm)]
        for ii in range(self.nhistones - 1):
            last = self.positions[-1]
            this = np.dot(r.rotz(np.pi / 3.0), last)
            this[2] = last[2] + self.zshift
            self.positions.append(this)
            last = self.rotations[-1]
            this = np.array([last[0], last[1], last[2] + np.pi / 3.0])
            self.rotations.append(this)
        self.histones = []
        self.linkers = (
            []
        )  # the BPs in the linkers array are also in the basepairs array
        for ii, (pos, rot) in enumerate(zip(self.positions, self.rotations)):
            h = Histone(pos, rot, chain=chain, histone_index=ii)
            self.histones.append(h)
            if len(self.histones) > 1:
                bp1 = self.histones[-2].basepairs[-2]
                bp2 = self.histones[-2].basepairs[-1]
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                zr = -Histone.histone_total_twist - np.pi / 3 + Histone.hist_bp_rotation
                l = SplineLinker(
                    bp1,
                    bp2,
                    bp3,
                    bp4,
                    curviness=1,
                    zrot=zr,
                    method="corrected_quaternion",
                    chain=chain,
                )
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            else:
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                l = SplineLinker(
                    prev_bp1,
                    prev_bp2,
                    bp3,
                    bp4,
                    curviness=1,
                    zrot=0,
                    method="corrected_quaternion",
                    chain=chain,
                )
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            self.basepairs.extend(h.basepairs)

        # Add final linker
        bp1 = self.histones[-1].basepairs[-2]
        bp2 = self.histones[-1].basepairs[-1]
        zr = -Histone.histone_total_twist - np.pi / 3 * (self.nhistones % 6)
        if twist is True:
            zr += np.pi / 2.0
        l = SplineLinker(
            bp1,
            bp2,
            next_bp3,
            next_bp4,
            curviness=1,
            zrot=zr,
            method="corrected_quaternion",
            chain=chain,
        )
        self.linkers.append(l)
        self.basepairs.extend(l.basepairs)
        # reset bp indices
        for ii, bp in enumerate(self.basepairs):
            bp.set_bp_index(ii)
        return None

    def translate(self, translation: Union[List, np.array]) -> None:
        """Translate the solenoid spatially

        :param translation: 3-vector for translation
        """
        for histone in self.histones:
            histone.translate(translation)
        for linker in self.linkers:
            linker.translate(translation)
        return None

    def setChain(self, chainIdx: int) -> None:
        """Set the Chain Index of all base pairs in the solenoid

        :param chainIdx: Index for Chain
        """
        self.chain = chainIdx
        for histone in self.histones:
            histone.setChain(chainIdx)
        for linker in self.linkers:
            linker.setChain(chainIdx)
        for basepair in self.basepairs:
            basepair.setNewChain(chainIdx)
        return None

    def histones_to_frame(self) -> pd.DataFrame:
        """Get Histones in Solenoid as a dataframe of their positions

        :return: DataFrame of Histones
        """
        return pd.DataFrame([histone.as_series() for histone in self.histones])


class TurnedSolenoid(Solenoid):
    """
    *Inherits from Solenoid*

    Define Solenoidal DNA in a voxel (basically a box). This Solenoid
    will turn 90 degrees through the box

    This method works by placing histones around the z-axis (≈6 histones
    per rotation) and then joining them together using SplineLinkers

    :param voxelheight: Height of 'voxel' in angstrom
    :param radius: Radius of circle the solenoid is turning around (angstrom)
    :param radius: Radius from Solenoid centre to histone centre
    :param nhistones: Number of histones to place
    :param histone_angle: tilt of histones from axis in degrees
    :param twist: whether the DNA exiting the final spine should be
        rotated an extra pi/2.
    :param chain: Chain index for solenoid and basepairs therein
    """

    def __init__(
        self,
        voxelheight: float = 750,
        radius: float = 100,
        nhistones: int = 38,
        histone_angle: float = 50,
        twist: bool = False,
        chain: int = 0,
    ):
        """
        Constructor
        """
        self.nhistones = int(nhistones / 2 ** 0.5)
        self.box_width = voxelheight / 2.0
        self.radius = radius
        self.chain = chain
        self.strand_length = voxelheight / 2 ** 0.5
        self.zshift = (
            self.strand_length - 4.0 * Histone.radius_histone
        ) / self.nhistones
        self.height = (self.nhistones - 1) * self.zshift
        self.tilt = histone_angle * np.pi / 180.0
        prev_bp1 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, -1 * BP_SEPARATION]),
            index=-2,
        )
        prev_bp2 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([0, 0, -0 * BP_SEPARATION]),
            index=-1,
        )
        rot = np.array([0, 0, np.pi / 2.0]) if twist is True else np.zeros(3)
        next_bp3 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([self.box_width + 0 * BP_SEPARATION, 0, self.box_width]),
            rotation=rot,
            index=1000,
        )
        next_bp4 = basepair.BasePair(
            np.random.choice(["G", "A", "T", "C"]),
            chain=chain,
            position=np.array([self.box_width + 1 * BP_SEPARATION, 0, self.box_width]),
            rotation=rot,
            index=1001,
        )
        self.basepairs = []
        # print(self.height, self.zshift, self.nhistones)
        pos1 = np.array([0, -self.radius, 0.5 * (self.strand_length - self.height)])
        self.positions = [np.dot(r.rotz(0 * np.pi / 3.0), pos1)]  # start at 2pi/3
        rm = r.eulerMatrix(np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0 + 0 * np.pi / 3.0)
        rm = np.dot(r.roty(self.tilt), rm)
        self.rotations = [r.getEulerAngles(rm)]
        for ii in range(self.nhistones - 1):
            last = self.positions[-1]
            this = np.dot(r.rotz(np.pi / 3.0), last)
            this[2] = last[2] + self.zshift
            self.positions.append(this)
            last = self.rotations[-1]
            this = np.array([last[0], last[1], last[2] + np.pi / 3.0])
            self.rotations.append(this)
        # Rotate positions/rotations through pi/2.
        for ii in range(0, len(self.positions)):
            pos = self.positions[ii]
            old_x = pos[0]
            old_z = pos[2]
            ang_histone = old_z / self.strand_length * np.pi / 2.0
            ang_pos = old_z / self.strand_length * np.pi / 4.0

            # need to do two rotations to eliminate shear
            ref_x = old_z * np.sin(ang_pos)
            ref_z = old_z * np.cos(ang_pos)

            new_x1 = old_z * np.sin(ang_pos) + old_x * np.cos(ang_pos)
            new_z1 = old_z * np.cos(ang_pos) - old_x * np.sin(ang_pos)

            new_x2 = new_x1 - ref_x
            new_z2 = new_z1 - ref_z

            new_x = ref_x + new_z2 * np.sin(ang_pos) + new_x2 * np.cos(ang_pos)
            new_z = ref_z + new_z2 * np.cos(ang_pos) - new_x2 * np.sin(ang_pos)

            self.positions[ii] = [new_x, pos[1], new_z]

            rot = self.rotations[ii]
            rm = r.eulerMatrix(*rot)
            rm = np.dot(r.roty(ang_histone), rm)
            self.rotations[ii] = r.getEulerAngles(rm)

        self.histones = []
        self.linkers = []
        for ii, (pos, rot) in enumerate(zip(self.positions, self.rotations)):
            h = Histone(pos, rot, chain=chain, histone_index=ii)
            self.histones.append(h)
            if len(self.histones) > 1:
                bp1 = self.histones[-2].basepairs[-2]
                bp2 = self.histones[-2].basepairs[-1]
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                zr = -Histone.histone_total_twist - np.pi / 3 + Histone.hist_bp_rotation
                l = SplineLinker(
                    bp1,
                    bp2,
                    bp3,
                    bp4,
                    curviness=1,
                    zrot=zr,
                    method="corrected_quaternion",
                    chain=chain,
                )
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            else:
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                l = SplineLinker(
                    prev_bp1,
                    prev_bp2,
                    bp3,
                    bp4,
                    curviness=1,
                    zrot=0,
                    method="corrected_quaternion",
                    chain=chain,
                )
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            self.basepairs.extend(h.basepairs)

        # Add final linker
        bp1 = self.histones[-1].basepairs[-2]
        bp2 = self.histones[-1].basepairs[-1]
        zr = -Histone.histone_total_twist - np.pi / 3 * (self.nhistones % 6)
        if twist is True:
            zr += np.pi / 2.0
        l = SplineLinker(
            bp1,
            bp2,
            next_bp3,
            next_bp4,
            curviness=1.0,
            zrot=zr,
            method="corrected_quaternion",
            chain=chain,
        )
        self.linkers.append(l)
        self.basepairs.extend(l.basepairs)
        # reset bp indices
        for ii, bp in enumerate(self.basepairs):
            bp.set_bp_index(ii)
        return None


class MultiSolenoidVolume(PlottableSequence):
    """
    Class to build placement volumes that contain multiple solenoidal DNA
    strands.

    Constructor:

    MultiSolenoidVolume(voxelheight=1500., separation=400, twist=False,
                        turn=False)

    voxelheight: size of placement volume
    separation: separation between DNA strands

    Try:
    dna = MultiSolenoidVolume()
    dna.to_line_plot()
    dna.to_text()
    """

    def __init__(
        self,
        voxelheight: float = 1500.0,
        separation: float = 400,
        twist: bool = False,
        turn: bool = False,
        chains: List = list(range(9)),
    ):
        if not (len(chains) == len(set(chains))):
            raise ValueError("The same chain cannot be generated twice")
        if not set(chains).issubset(set(range(9))):
            raise ValueError(f"Valid Chains are {set(range(9))} and must be ints")
        self.voxelheight = voxelheight
        self.radius = 100
        self.nhistones = int(38 * voxelheight / 750.0)
        self.histone_angle = 50
        self.sep = separation
        self.twist = twist

        self.turn = turn

        self.basepairs = []
        self.histones = []
        self.linkers = []

        if turn is True:
            big_height = 2 * (self.voxelheight / 2.0 + self.sep)
            little_height = 2 * (self.voxelheight / 2.0 - self.sep)
            lengths = [
                self.voxelheight,
                little_height,
                self.voxelheight,
                big_height,
                self.voxelheight,
                little_height,
                big_height,
                big_height,
                little_height,
            ]
        else:
            lengths = [self.voxelheight] * 8

        translations = [
            np.array([0, 0, 0]),
            np.array([self.sep, 0, 0]),
            np.array([0, self.sep, 0]),
            np.array([-self.sep, 0, 0]),
            np.array([0, -self.sep, 0]),
            np.array([self.sep, self.sep, 0]),
            np.array([-self.sep, self.sep, 0]),
            np.array([-self.sep, -self.sep, 0]),
            np.array([self.sep, -self.sep, 0]),
        ]
        solenoids = []
        for ii in chains:
            if self.turn is True:
                nhistones = int(self.nhistones * lengths[ii] / self.voxelheight)
                s = TurnedSolenoid(
                    voxelheight=lengths[ii],
                    radius=self.radius,
                    nhistones=nhistones,
                    histone_angle=self.histone_angle,
                    twist=self.twist,
                    chain=ii,
                )
            else:
                s = Solenoid(
                    voxelheight=lengths[ii],
                    radius=self.radius,
                    nhistones=self.nhistones,
                    histone_angle=self.histone_angle,
                    twist=self.twist,
                    chain=ii,
                )
            # s.setChain(ii)
            s.translate(translations[ii])
            solenoids.append(s)

        for s in solenoids:
            self.basepairs.extend(s.basepairs)
            self.linkers.extend(s.linkers)
            self.histones.extend(s.histones)

        return None


class DNAChain(PlottableSequence):
    """
    *Inherits from PlottableSequence*
    A single DNA Chain built from a genome specified.


    :param genome: A string specifying the genome, e.g. 'AGTATC'
    :param chain: The Chain index to label this strand
    """

    def __init__(self, genome: str, chain: int = 0):
        """
        Construct a DNA chain from a genome specified from a string
        """
        self.basepairs_chain0 = self._makeFromGenome(genome, chain=chain)
        self.basepairs = self.basepairs_chain0
        self.center_in_z()

    @staticmethod
    def _makeFromGenome(genome: str, chain: int = 0):
        """
        :param genome: String of the genome, e.g. "GATTACA"
        :param chain: Integer to set as the chain index
        :return: DNA Chain object
        """
        dnachain = []
        position = np.array([0, 0, 0], dtype=float)
        rotation = np.array([0, 0, 0], dtype=float)
        index = 0
        for char in genome:
            # print("Appending " + char)
            dnachain.append(
                basepair.BasePair(
                    char, chain=chain, position=position, rotation=rotation, index=index
                )
            )
            position += np.array([0.0, 0.0, BP_SEPARATION])
            rotation += np.array([0.0, 0.0, BP_ROTATION])
            index += 1
        return dnachain

    @staticmethod
    def _turnAndTwistChain(chain, twist=0.0):
        zmax = 0
        zmin = 0
        for pair in chain:
            # for (name, mol) in pair.iterMolecules():
            if pair.position[2] < zmin:
                zmin = pair.position[2]
            elif pair.position[2] > zmax:
                zmax = pair.position[2]

        zrange = zmax - zmin
        radius = 2.0 * zrange / np.pi
        # print(radius)

        for pair in chain:
            # Translation of the frame - new center position
            theta = np.pi / 2.0 * (pair.position[2] - zmin) / zrange
            new_origin = np.array(
                [radius * (1 - np.cos(theta)), 0.0, radius * np.sin(theta) - radius]
            )
            # rotation of the frame
            # oldframe = np.array([mol.position[0], mol.position[1], 0])
            yang = np.pi / 2.0 * (pair.position[2] - zmin) / zrange
            pair.rotate(np.array([0, yang, 0]), about_origin=True)

            xang = twist * (pair.position[2] - zmin) / zrange
            chain_z_axis = pair.rmatrix[:, 2]
            rmatrix = r.rot_ax_angle(chain_z_axis, xang)
            pair.rotate(rmatrix, about_origin=True)

            pair.translate(new_origin - pair.position)
        return chain

    def center_in_z(self):
        """
        Center the molecule around the z=0 plane
        """
        minz = 0
        maxz = 0
        for bp in self.basepairs:
            for (name, mol) in bp.iterMolecules():
                if mol.position[2] < minz:
                    minz = mol.position[2]
                elif mol.position[2] > maxz:
                    maxz = mol.position[2]

        ztrans = (minz - maxz) / 2.0 - minz
        translation = np.array([0.0, 0.0, ztrans])

        for bp in self.basepairs:
            bp.translate(translation)

        return None


class TurnedDNAChain(DNAChain):
    """
    *Inherits from DNAChain*

    TurnedDNAChain(genome)
    Construct a single turned, twisted DNA chaiun

    :param genome: string of GATC specifying genome order
    """

    def __init__(self, genome):
        """
        TurnedDNAChain(genome)

        Construct a DNA chain from a genome of GATC that turns 90 degrees
        """
        super().__init__(genome)
        self.turnDNA()

    def turnDNA(self):
        self.basepairs = DNAChain._turnAndTwistChain(self.basepairs)
        return None


class TurnedTwistedDNAChain(DNAChain):
    """
    *Inherits from DNAChain*

    TurnedTwistedDNAChain(genome)
    Construct a single turned, twisted DNA chaiun

    :param genome: string of GATC specifying genome order
    """

    def __init__(self, genome):
        """
        TurnedDNAChain(genome)

        Construct a DNA chain from a genome of GATC that turns 90 degrees
        """
        super().__init__(genome)
        self.turnAndTwistDNA()

    def turnAndTwistDNA(self):
        self.basepairs = DNAChain._turnAndTwistChain(self.basepairs, twist=np.pi / 2.0)
        return None


class DoubleDNAChain(DNAChain):
    """
    *Inherits from DNAChain*

    DoubleDNAChain(genome, separation)
    Construct four straight DNA chains
    Chain indices are assigned anticlockwise starting from the +y strand.

    :param genome: string of GATC specifying genome order
    :param separation: separation of each strand from the center in angstroms
    """

    def __init__(self, genome, separation):
        """
        DoubleDNAChain(genome, separation)

        Construct two parallel straight DNA chains
        """
        super().__init__(genome)
        self._duplicateDNA(separation)

    def _duplicateDNA(self, separation):
        translation = np.array([0.0, separation / 2.0, 0.0], dtype=float)
        self.basepairs_chain1 = deepcopy(self.basepairs_chain0)

        for bp in self.basepairs_chain0:
            bp.translate(translation)
            bp.setNewChain(1)

        for bp in self.basepairs_chain1:
            bp.translate(-1 * translation)
            bp.setNewChain(2)

        self.basepairs = self.basepairs_chain0 + self.basepairs_chain1


class TurnedDoubleDNAChain(TurnedDNAChain, DoubleDNAChain):
    def __init__(self, genome, separation):
        self._makeFromGenome(genome)
        self._duplicateDNA(separation=separation)
        self._turnDNA()


class TurnedTwistedDoubleDNAChain(TurnedTwistedDNAChain, DoubleDNAChain):
    def __init__(self, genome, separation):
        self._makeFromGenome(genome)
        self._duplicateDNA(separation=separation)
        self._turnAndTwistDNA()


class FourStrandDNAChain(DNAChain):
    """
    *Inherits from DNAChain*

    FourStrandDNAChain(genome, separation)
    Construct four straight DNA chains
    Chain indices are assigned anticlockwise starting from the +y strand.

    :param genome: string of GATC specifying genome order
    :param separation: separation of each strand from the center in angstroms
    """

    def __init__(self, genome: str, separation: float):
        """
        constructor
        """
        super().__init__(genome)
        self.makeFourStrands(separation)

    def makeFourStrands(self, separation):
        translation_y = np.array([0.0, separation / 2.0, 0.0], dtype=float)
        translation_x = np.array([separation / 2.0, 0.0, 0.0], dtype=float)
        self.basepairs_chain1 = deepcopy(self.basepairs_chain0)
        self.basepairs_chain2 = deepcopy(self.basepairs_chain0)
        self.basepairs_chain3 = deepcopy(self.basepairs_chain0)

        for bp in self.basepairs_chain0:
            bp.translate(translation_y)
            bp.setNewChain(0)

        for bp in self.basepairs_chain1:
            bp.translate(-1 * translation_x)
            bp.setNewChain(1)

        for bp in self.basepairs_chain2:
            bp.translate(-1 * translation_y)
            bp.setNewChain(2)

        for bp in self.basepairs_chain3:
            bp.translate(1 * translation_x)
            bp.setNewChain(3)

        self.basepairs = (
            self.basepairs_chain0
            + self.basepairs_chain1
            + self.basepairs_chain2
            + self.basepairs_chain3
        )


class FourStrandTurnedDNAChain(DNAChain):
    """
    *Inherits from DNAChain*

    FourStrandTurnedDNAChain(genome, separation)
    Construct four DNA chains that turn 90 degrees.
    Chain indices are assigned anticlockwise starting from the +y strand.

    :param genome: string of GATC specifying genome order
    :param separation: separation of each strand from the center in angstroms
    :param twist: boolean, add a 90 deg twist to each chain
    """

    def __init__(self, genome: str, separation: float, twist: bool = False):
        """
        Constructor
        """
        DNAChain.__init__(self, genome)
        translation_y = np.array([0.0, separation / 2.0, 0.0], dtype=float)
        translation_x = np.array(
            [separation / 2.0, 0.0, -separation / 2.0], dtype=float
        )
        ang = np.pi / 2.0 if twist is True else 0

        radiusC0 = len(self.basepairs_chain0) * BP_SEPARATION * 2 / np.pi
        radiusC3 = radiusC0 - separation / 2.0
        radiusC1 = radiusC0 + separation / 2.0

        self.basepairs_chain2 = DNAChain(genome, chain=2).basepairs

        lengthC3 = int(np.floor(radiusC3 / radiusC0 * len(genome)))
        lengthC1 = int(np.floor(radiusC1 / radiusC0 * len(genome)))
        longGenome = genome * int(np.ceil(radiusC1 / radiusC0))

        genome_chain3 = genome[:lengthC3]
        self.basepairs_chain3 = DNAChain(genome_chain3, chain=3).basepairs
        genome_chain1 = longGenome[:lengthC1]
        self.basepairs_chain1 = DNAChain(genome_chain1, chain=1).basepairs

        chains = [
            self.basepairs_chain0,
            self.basepairs_chain1,
            self.basepairs_chain2,
            self.basepairs_chain3,
        ]
        transforms = [+translation_y, -translation_x, -translation_y, +translation_x]
        angles = [
            ang + (2 * np.pi - BP_ROTATION * len(c) % (2 * np.pi)) for c in chains
        ]

        for (ii, (c, t, a)) in enumerate(zip(chains, transforms, angles)):
            c = self._turnAndTwistChain(c, twist=a)
            for bp in c:
                bp.translate(t)
            chains[ii] = c

        self.basepairs = []
        for c in chains:
            self.basepairs.extend(c)

        return None


class EightStrandDNAChain(DNAChain):
    """
    Construct eight DNA chains that can turn 90 degrees if turn=True

    Chain indices are assigned anticlockwise starting from the +y strand,
    first to the inner four strands, then two the outer four strands.
    i.e.::

        ____________________
        |                  |
        |        4         |
        |        0         |
        |  5  1     3  7   |
        |        2         |
        |        6         |
        |__________________|

    Strands 1 and 3, 0 and 2 are separated by sep1
    Strands 4 and 6, 5 and 7 are separated by sep2

    :param genome: string of GATC specifying genome order
    :param sep1: separation of inner strands from the center in angstroms
    :param sep2: separation of outer strands from the center in angstroms
    :param turn: boolean, turn strands 90 degrees along box
    :param twist: boolean, add a 90 deg twist to each chain
    """

    def __init__(
        self,
        genome: str,
        sep1: float,
        sep2: float,
        turn: bool = False,
        twist: bool = False,
    ):
        """
        EightStrandTurnedDNAChain(genome, sep1, sep2, turn=False, twist=False)
        """
        DNAChain.__init__(self, genome)
        v1 = -sep1 / 2.0 if turn is True else 0
        v2 = -sep2 / 2.0 if turn is True else 0
        trans_y1 = np.array([0.0, sep1 / 2.0, 0.0], dtype=float)
        trans_x1 = np.array([sep1 / 2.0, 0.0, v1], dtype=float)
        trans_y2 = np.array([0.0, sep2 / 2.0, 0.0], dtype=float)
        trans_x2 = np.array([sep2 / 2.0, 0.0, v2], dtype=float)
        ang = np.pi / 2.0 if twist is True else 0

        # centrally aligned strands
        self.basepairs_chain2 = DNAChain(genome, chain=2).basepairs
        self.basepairs_chain4 = DNAChain(genome, chain=4).basepairs
        self.basepairs_chain6 = DNAChain(genome, chain=6).basepairs

        radiusC0 = len(self.basepairs_chain0) * BP_SEPARATION * 2 / np.pi
        if turn is True:
            radiusC1 = radiusC0 + sep1 / 2.0
            radiusC3 = radiusC0 - sep1 / 2.0
            radiusC5 = radiusC0 + sep2 / 2.0
            radiusC7 = radiusC0 - sep2 / 2.0
        else:
            radiusC1 = radiusC0
            radiusC3 = radiusC0
            radiusC5 = radiusC0
            radiusC7 = radiusC0

        lengthC1 = int(np.floor(radiusC1 / radiusC0 * len(genome)))
        lengthC3 = int(np.floor(radiusC3 / radiusC0 * len(genome)))
        lengthC5 = int(np.floor(radiusC5 / radiusC0 * len(genome)))
        lengthC7 = int(np.floor(radiusC7 / radiusC0 * len(genome)))
        longGenome = genome * int(np.ceil(radiusC5 / radiusC0))

        self.basepairs_chain1 = DNAChain(longGenome[:lengthC1], chain=1).basepairs
        self.basepairs_chain3 = DNAChain(longGenome[:lengthC3], chain=3).basepairs
        self.basepairs_chain5 = DNAChain(longGenome[:lengthC5], chain=5).basepairs
        self.basepairs_chain7 = DNAChain(longGenome[:lengthC7], chain=7).basepairs

        chains = [
            self.basepairs_chain0,
            self.basepairs_chain1,
            self.basepairs_chain2,
            self.basepairs_chain3,
            self.basepairs_chain4,
            self.basepairs_chain5,
            self.basepairs_chain6,
            self.basepairs_chain7,
        ]
        transforms = [
            +trans_y1,
            -trans_x1,
            -trans_y1,
            +trans_x1,
            +trans_y2,
            -trans_x2,
            -trans_y2,
            +trans_x2,
        ]
        angles = [
            ang + (2 * np.pi - BP_ROTATION * len(c) % (2 * np.pi)) for c in chains
        ]
        # print(angles)

        for (ii, (c, t, a)) in enumerate(zip(chains, transforms, angles)):
            if turn is True:
                c = self._turnAndTwistChain(c, twist=a)
            for bp in c:
                bp.translate(t)
            chains[ii] = c

        self.basepairs = []
        for c in chains:
            self.basepairs.extend(c)

        return None
