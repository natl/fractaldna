"""
Class description of a DNA chain built of base pairs
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from copy import deepcopy
from scipy.interpolate import interp1d

try:
    from mayavi import mlab
    maya_imported = True
except ImportError:
    maya_imported = False
    print("Could not import mayavi libraries, 3d plotting is disabled")
    print("MayaVi may need Python2")

from utils import rotations as r
from utils import basepair
from utils import BP_ROTATION, BP_SEPARATION


class PlottableSequence(object):
    """
    This is an inheritable class that gives DNA chains plotting methods and
    output methods.
    """
    def to_text(self, seperator=" "):
        """
        Return a description of the molecules in the chain as text
        """
        key = "# NAME SHAPE CHAIN_ID STRAND_ID BP_INDEX " +\
              "SIZE_X SIZE_Y SIZE_Z POS_X " +\
              "POS_Y POS_Z ROT_X ROT_Y ROT_Z\n"
        output = [key]
        for pair in self.basepairs:
            output.append(pair.to_text(seperator=seperator))

        return "".join(output)

    def to_plot(self, plot_p=True, plot_b=True, plot_s=True):
        """
        Return a matplotlib.Figure instance with molecules plotted
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
        bases = [ii for ii in zip( * map(list, bases))] if plot_b else empty
        triphosphates = [ii for ii in zip( * map(list, triphosphates))]\
            if plot_p else empty
        sugars = [ii for ii in zip( * map(list, sugars))] if plot_s else empty

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(bases[0], bases[1], bases[2], c="0.6", s=20)
        ax.scatter(triphosphates[0], triphosphates[1], triphosphates[2], c="y",
                   s=20)
        ax.scatter(sugars[0], sugars[1], sugars[2], c="r", s=20)

        return fig

    def to_surface_plot(self):
        """
        Plot the surfaces of each molecule in the chain.
        Avoid this with large chains, this assumes each molecule is an ellipse
        """

        def ellipse_xyz(center, extent, rotation=np.zeros([3])):
            rmatrix = r.eulerMatrix(*rotation)
            [a, b, c] = extent
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
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
                    sugars.append((molecule.position, molecule.dimensions,
                                   molecule.rotation))
                elif molecule.name.lower() == "phosphate":
                    triphosphates.append((molecule.position,
                                          molecule.dimensions,
                                          molecule.rotation))
                elif molecule.name.lower() in bps:
                    bases.append((molecule.position, molecule.dimensions,
                                  molecule.rotation))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for base in bases:
            x, y, z = ellipse_xyz(base[0], base[1], rotation=base[2])
            ax.plot_wireframe(x, y, z, color="0.6")

        for phosphate in triphosphates:
            x, y, z = ellipse_xyz(phosphate[0], phosphate[1],
                                  rotation=phosphate[2])
            ax.plot_wireframe(x, y, z, color="y")

        for sugar in sugars:
            x, y, z = ellipse_xyz(sugar[0], sugar[1], rotation=sugar[2])
            ax.plot_wireframe(x, y, z, color="r")

        return fig

    def to_line_plot(self):
        """
        Return a matplotlib.Figure instance with histone and linkers shown
        """
        if maya_imported is True:
            fig = mlab.figure()
            if hasattr(self, "histones"):
                # ax = fig.add_subplot(111, projection='3d')
                histones = []
                for histone in self.histones:
                    pos = np.array([bp.position for bp in histone.basepairs])
                    mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2],
                                color=(1., .8, 0),
                                tube_radius = 11.5)
                    histones.append(histone.position)

                for linker in self.linkers:
                    pos = np.array([bp.position for bp in linker.basepairs])
                    mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2],
                                color=(0, .8, 0),
                                tube_radius = 11.5)

                histones = np.array(histones)
                mlab.points3d(histones[:, 0], histones[:, 1], histones[:, 2],
                              color=(0, 0, 1.), opacity=.4, scale_factor=70)

            else:
                pos = np.array([bp.position for bp in self.basepairs])
                mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2], color=(1., 0, 0),
                            tube_radius = 11.5)

            return fig
        else:
            print("MayaVi not imporrted, cannot produce this plot")
            return None

    def to_strand_plot(self, plot_p=True, plot_b=True, plot_s=True,
                       plot_bp=False):
        """
        to_strand_plot(plot_p=True, plot_b=True, plot_s=True, plot_bp=False)
        Return a mayav1 figure instance with strands plotted

        plot_p : plot phosphate strands
        plot_s : plot sugar strands
        plot_b : plot base strands
        plot_bp : join base pairs together
        """
        if maya_imported is True:
            sugar_l = []
            sugar_r = []
            phosphate_l = []
            phosphate_r = []
            base_l = []
            base_r = []
            bps = ["guanine", "adenine", "thymine", "cytosine"]
            for pair in self.basepairs:
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
            base_l = [ii for ii in zip( * map(list, base_l))]
            base_r = [ii for ii in zip( * map(list, base_r))]
            phosphate_l = [ii for ii in zip( * map(list, phosphate_l))]
            phosphate_r = [ii for ii in zip( * map(list, phosphate_r))]
            sugar_l = [ii for ii in zip( * map(list, sugar_l))]
            sugar_r = [ii for ii in zip( * map(list, sugar_r))]
            fig = mlab.figure()

            if plot_b:
                mlab.plot3d(base_l[0], base_l[1], base_l[2],
                            color=(0.6, 0.6, 0.6), tube_radius = 1)
                mlab.plot3d(base_r[0], base_r[1], base_r[2],
                            color=(0.6, 0.6, 0.6), tube_radius = 1)
            if plot_s:
                mlab.plot3d(sugar_l[0], sugar_l[1], sugar_l[2],
                            color=(1., 0, 0), tube_radius = 1)
                mlab.plot3d(sugar_r[0], sugar_r[1], sugar_r[2],
                            color=(1., 0, 0), tube_radius = 1)
            if plot_p:
                mlab.plot3d(phosphate_l[0], phosphate_l[1], phosphate_l[2],
                            color=(1, 1, 0), tube_radius = 1)
                mlab.plot3d(phosphate_r[0], phosphate_r[1], phosphate_r[2],
                            color=(1, 1, 0), tube_radius = 1)

            if plot_bp:
                # plot bars joining base pairs
                for ii in range(0, len(base_l[0])):
                    xs = (phosphate_l[0][ii], sugar_l[0][ii], base_l[0][ii],
                          base_r[0][ii], sugar_r[0][ii], phosphate_r[0][ii])
                    ys = (phosphate_l[1][ii], sugar_l[1][ii], base_l[1][ii],
                          base_r[1][ii], sugar_r[1][ii], phosphate_r[1][ii])
                    zs = (phosphate_l[2][ii], sugar_l[2][ii], base_l[2][ii],
                          base_r[2][ii], sugar_r[2][ii], phosphate_r[2][ii])
                    mlab.plot3d(xs, ys, zs, color=(1, 1, 1), tube_radius=.5)

            return fig
        else:
            print("MayaVi not imporrted, cannot produce this plot")
            return None


class SplineLinker(PlottableSequence):
    """
    """
    linker_rotation = BP_ROTATION  # rad, default screw rotation of dna
    linker_bp_spacing = BP_SEPARATION  # angstrom, default spacing between bps

    def __init__(self, bp1, bp2, bp3, bp4, curviness=1., zrot=None,
                 startkey=None, stopkey=None, method="corrected_quaternion"):
        """
        Create a smooth linker based on splines.

        linker = SplineLinker(bp1, bp2, bp3, bp4, curviness=1., zrot=None)

        Create base pairs that link to sections of DNA as follows:
        bp1 bp2 <==== LINKER =====> bp3 bp4
        Two base pairs on either side of the linker are needed to build splines
        low curviness = straighter
        high_curviness = smoother

        zrot describes the twist of bp3 relative to bp2
            None: Determine automatically (experimental)
            double: rotation in radians (mod 2*pi)

        startkey and stopkey act as keyframes for rotations
            key = i will start/stop rotations after the i-th base pair
            key = -i will start/stop rotations after the i-th last base pair

        method: method to handle rotational interpolation
            "quaternion":           Full use of quaternions
            "corrected_quaternion": Full use of quaternions, with a correction
                                    to check that the base pair is aligned.
                                    This method is recommended
            "matrix":               Experimental method that doesn't use
                                    quaternions. Currently incorrect.
        """
        assert method in ["quaternion", "matrix", "corrected_quaternion"],\
            "Invalid interpolation method"
        self.basepairs = []
        points = np.array([bp1.position, bp2.position,
                           bp3.position, bp4.position])
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
        diff = 3.4/180/curviness
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
        dx = xx[1:] - xx[:(len(xx) - 1)]
        dy = yy[1:] - yy[:(len(yy) - 1)]
        dz = zz[1:] - zz[:(len(zz) - 1)]
        length = sum((dx**2 + dy**2 + dz**2)**.5)
        n = length // self.linker_bp_spacing
        self.spacing = length / n
        # print(self.spacing)
        tt = np.linspace(1, 2, n)
        xx = x_interp(tt[1:len(tt)])
        yy = y_interp(tt[1:len(tt)])
        zz = z_interp(tt[1:len(tt)])
        interpolator = lambda t: np.array([x_interp(t + 1), y_interp(t + 1),
                                           z_interp(t + 1)])

        if startkey is None:
            startkey = 0
        elif startkey < 0:
            startkey = len(tt) - abs(startkey) - 1
        if stopkey is None:
            stopkey = len(tt) - 1
        elif stopkey < 0:
            stopkey = len(tt) - abs(stopkey)

        # total rotation that the BP undergoes, relative to initial
        rotation_unmodified = (-n*self.linker_rotation) % (2*np.pi)

        if zrot is None:
            zrot = 0

        # desired rotation relative to initial
        zrot = zrot % (2*np.pi)
        diff = zrot - rotation_unmodified
        if diff < -np.pi:
            diff += 2*np.pi
        elif diff > np.pi:
            diff -= 2*np.pi
        rot_angle = (self.linker_rotation + diff/n)
        # print("\n", n, diff*180/np.pi, rot_angle*180/np.pi)
        # print("Desired rotation", zrot*180/np.pi)
        # print("Default rotation", rotation_unmodified*180/np.pi)
        # print("Final rotation", ((rot_angle*n) % (2*np.pi))*180/np.pi)

        # Run one loop to generate a series of rotation matrices
        for (ii, (_x, _y, _z)) in enumerate(zip(xx, yy, zz)):
            if ii != len(xx) - 1:
                pos = np.array([_x, _y, _z])
                bp = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                       chain=0,
                                       position=[0, 0, 0],
                                       rotation=[0, 0, 0],
                                       index=ii)

                if method in ["quaternion", "corrected_quaternion"]:
                    start_quaternion = r.quaternion_from_matrix(bp2.rmatrix)
                    if ii < startkey:
                        ll = 0
                    elif ii >= stopkey:
                        ll = 1
                    else:
                        ll = (ii - startkey + 1.)/float(stopkey - startkey)
                    if method == "corrected_quaternion":
                        # get interpolated rotation matrix
                        end_quaternion = r.quaternion_from_matrix(bp3.rmatrix)
                        quat = r.quaternion_slerp(start_quaternion,
                                                  end_quaternion,
                                                  ll, shortestpath=True)
                        rmat = r.quaternion_matrix(quat)
                        # correct z
                        z = interpolator((ii + 1)/len(xx)) -\
                            interpolator(ii/len(xx))
                        z = -z/np.linalg.norm(z)
                        z_current = rmat[:, 2]
                        perp = np.cross(z_current, z)
                        angle = np.arccos(np.dot(z, z_current) /
                                          (np.linalg.norm(z) *
                                           np.linalg.norm(z_current)))
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
                        quat = r.quaternion_slerp(start_quaternion,
                                                  end_quaternion,
                                                  ll, shortestpath=True)
                        rmat = r.quaternion_matrix(quat)
                elif method == "matrix":
                    ll = ii/len(xx)
                    rmat = r.matrix_interpolate(bp2.rmatrix, bp3.rmatrix,
                                                interpolator, ll,
                                                precision=.01)
                bp.rotate(rmat)
                spin = rot_angle * (ii + 1)
                bp.rotate(r.rot_ax_angle(rmat[:, 2], spin))
                bp.translate(pos)
                self.basepairs.append(bp)

        pass


class Histone(PlottableSequence):
    """
    """
    radius_histone = 25  # radius of histone, angstrom
    pitch_dna = 23.9  # 23.9  # pitch of DNA helix, angstrom
    radius_dna = 41.8  # radius of DNA wrapping, angstrom
    histone_bps = 146  # number of bps in histone
    histone_turns = 1.65 * 2 * np.pi  # angular turn around histone, radians
    height = 27*1.65
    z_offset = -height / 2.  # distance between first bp and xy-plane, angstrom
    # separation of bps around histone, angstrom
    hist_bp_separation = histone_turns * radius_dna / histone_bps
    hist_bp_rotation = BP_ROTATION  # screw rotation of bp, radians
    z_per_bp = height / histone_bps
    turn_per_bp = histone_turns / histone_bps
    z_angle = np.arctan(1./pitch_dna)
    histone_start_bp_rot = 0  # radians, rotation of bp at start of histone
    histone_end_bp_rot = histone_start_bp_rot + histone_bps*hist_bp_rotation
    histone_total_twist = (histone_bps * hist_bp_separation) % (2*np.pi)

    def __init__(self, position, rotation, genome=None):
        """
        """
        assert len(position) == 3, "position is length 3 array"
        assert len(rotation) == 3, "position is length 3 array"
        if genome is None:
            genome = "".join([np.random.choice(["G", "A", "T", "C"])
                              for ii in range(self.histone_bps)])
        assert len(genome) == self.histone_bps,\
            "genome should be {} base pairs".format(self.histone_bps)
        self.position = np.array(position)
        self.rotation = np.array(rotation)
        self.basepairs = []
        theta = -0.5 * (self.histone_turns - 3 * np.pi)
        z = self.z_offset
        for ii, char in enumerate(genome):
            bp = basepair.BasePair(char, chain=0,
                                   position=np.array([0, 0, 0]),
                                   rotation=np.array([0, 0, 0]),
                                   index=ii)
            # make rotation matrix

            rmatrix = r.rotx(np.pi/2. + self.z_angle)
            rmatrix = np.dot(r.rotz(theta), rmatrix)
            bp.rotate(rmatrix)
            bp.rotate(np.dot(rmatrix,
                      np.dot(r.rotz(self.histone_start_bp_rot +
                                    ii*self.hist_bp_rotation),
                             np.linalg.inv(rmatrix))))
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


class Solenoid(PlottableSequence):
    voxelheight = 750
    radius = 100  # angstroms, radius from center to place histones
    # tilt = 20*np.pi/180.  # tilt chromosomes 20 deg following F98
    # zshift = 18.3  # angstrom, z shift per chromosome following F98
    nhistones = 38
    tilt = 50*np.pi/180.
    zshift = (voxelheight - 4. * Histone.radius_histone)/nhistones
    height = (nhistones - 1) * zshift  # length of the fibre

    def __init__(self, turn=False):
        prev_bp1 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        -1*BP_SEPARATION]),
                                     index=-2)
        prev_bp2 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        -0*BP_SEPARATION]),
                                     index=-1)
        rot = np.array([0, 0, np.pi/2.]) if turn is True else np.zeros(3)
        next_bp3 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        self.voxelheight +
                                                        0.*BP_SEPARATION]),
                                     rotation=rot,
                                     index=1000)
        next_bp4 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        self.voxelheight +
                                                        1.*BP_SEPARATION]),
                                     rotation=rot,
                                     index=1001)
        self.basepairs = []
        self.positions = [np.array([0, -self.radius,
                                    .5*(self.voxelheight - self.height)])]
        rm = r.eulerMatrix(np.pi/2., -np.pi/2., np.pi/2.)
        rm = np.dot(r.roty(self.tilt), rm)
        self.rotations = [r.getEulerAngles(rm)]
        for ii in range(self.nhistones - 1):
            last = self.positions[-1]
            this = np.dot(r.rotz(np.pi/3.), last)
            this[2] = last[2] + self.zshift
            self.positions.append(this)
            last = self.rotations[-1]
            this = np.array([last[0], last[1], last[2] + np.pi/3.])
            self.rotations.append(this)
        self.histones = []
        self.linkers = []
        for pos, rot in zip(self.positions, self.rotations):
            h = Histone(pos, rot)
            self.histones.append(h)
            if len(self.histones) > 1:
                bp1 = self.histones[-2].basepairs[-2]
                bp2 = self.histones[-2].basepairs[-1]
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                zr = - Histone.histone_total_twist - np.pi/3 +\
                    Histone.hist_bp_rotation
                l = SplineLinker(bp1, bp2, bp3, bp4, curviness=1,
                                 zrot=zr, method="corrected_quaternion")
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            else:
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                l = SplineLinker(prev_bp1, prev_bp2, bp3, bp4,
                                 curviness=1,
                                 zrot=0,
                                 method="corrected_quaternion")
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            self.basepairs.extend(h.basepairs)

        # Add final linker
        bp1 = self.histones[-1].basepairs[-2]
        bp2 = self.histones[-1].basepairs[-1]
        zr = -Histone.histone_total_twist - np.pi/3 * (self.nhistones % 6)
        if turn is True:
            zr += np.pi/2.
        l = SplineLinker(bp1, bp2, next_bp3, next_bp4, curviness=1,
                         zrot=zr, method="corrected_quaternion")
        self.linkers.append(l)
        self.basepairs.extend(l.basepairs)
        # reset bp indices
        for ii, bp in enumerate(self.basepairs):
            bp.set_bp_index(ii)
        return None


class TurnedSolenoid(Solenoid):
    nhistones = int(Solenoid.nhistones/2**.5)
    box_width = Solenoid.voxelheight/2.
    strand_length = Solenoid.voxelheight/2**.5
    zshift = (strand_length - 4. * Histone.radius_histone)/nhistones
    height = (nhistones - 1) * zshift
    tilt = 50*np.pi/180.  # tilt chromosomes 20 deg following F98

    def __init__(self, turn=False):
        prev_bp1 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        -1*BP_SEPARATION]),
                                     index=-2)
        prev_bp2 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array([0, 0,
                                                        -0*BP_SEPARATION]),
                                     index=-1)
        rot = np.array([0, 0, np.pi/2.]) if turn is True else np.zeros(3)
        next_bp3 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array(
                                     [self.box_width + 0*BP_SEPARATION,
                                      0, self.box_width]),
                                     rotation=rot,
                                     index=1000)
        next_bp4 = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                     chain=0,
                                     position=np.array(
                                     [self.box_width + 1*BP_SEPARATION,
                                      0, self.box_width]),
                                     rotation=rot,
                                     index=1001)
        self.basepairs = []
        print(self.height, self.zshift, self.nhistones)
        pos1 = np.array([0, -self.radius,
                         .5*(self.strand_length - self.height)])
        self.positions = [np.dot(r.rotz(0*np.pi/3.), pos1)]  # start at 2pi/3
        rm = r.eulerMatrix(np.pi/2., -np.pi/2., np.pi/2. + 0*np.pi/3.)
        rm = np.dot(r.roty(self.tilt), rm)
        self.rotations = [r.getEulerAngles(rm)]
        for ii in range(self.nhistones - 1):
            last = self.positions[-1]
            this = np.dot(r.rotz(np.pi/3.), last)
            this[2] = last[2] + self.zshift
            self.positions.append(this)
            last = self.rotations[-1]
            this = np.array([last[0], last[1], last[2] + np.pi/3.])
            self.rotations.append(this)
        # Rotate positions/rotations through pi/2.
        for ii in range(0, len(self.positions)):
            pos = self.positions[ii]
            old_x = pos[0]
            old_z = pos[2]
            ang_histone = old_z/self.strand_length*np.pi/2.
            ang_pos = old_z/self.strand_length*np.pi/4.

            # need to do two rotations to eliminate shear
            ref_x = old_z*np.sin(ang_pos)
            ref_z = old_z*np.cos(ang_pos)

            new_x1 = old_z*np.sin(ang_pos) + old_x*np.cos(ang_pos)
            new_z1 = old_z*np.cos(ang_pos) - old_x*np.sin(ang_pos)

            new_x2 = new_x1 - ref_x
            new_z2 = new_z1 - ref_z

            new_x = ref_x + new_z2*np.sin(ang_pos) + new_x2*np.cos(ang_pos)
            new_z = ref_z + new_z2*np.cos(ang_pos) - new_x2*np.sin(ang_pos)

            self.positions[ii] = [new_x, pos[1], new_z]

            rot = self.rotations[ii]
            rm = r.eulerMatrix(*rot)
            rm = np.dot(r.roty(ang_histone), rm)
            self.rotations[ii] = r.getEulerAngles(rm)

        self.histones = []
        self.linkers = []
        for pos, rot in zip(self.positions, self.rotations):
            h = Histone(pos, rot)
            self.histones.append(h)
            if len(self.histones) > 1:
                bp1 = self.histones[-2].basepairs[-2]
                bp2 = self.histones[-2].basepairs[-1]
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                zr = - Histone.histone_total_twist - np.pi/3 +\
                    Histone.hist_bp_rotation
                l = SplineLinker(bp1, bp2, bp3, bp4, curviness=1,
                                 zrot=zr, method="corrected_quaternion")
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            else:
                bp3 = self.histones[-1].basepairs[0]
                bp4 = self.histones[-1].basepairs[1]
                l = SplineLinker(prev_bp1, prev_bp2, bp3, bp4,
                                 curviness=1,
                                 zrot=0,
                                 method="corrected_quaternion")
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            self.basepairs.extend(h.basepairs)

        # Add final linker
        bp1 = self.histones[-1].basepairs[-2]
        bp2 = self.histones[-1].basepairs[-1]
        zr = -Histone.histone_total_twist - np.pi/3 * (self.nhistones % 6)
        if turn is True:
            zr += np.pi/2.
        l = SplineLinker(bp1, bp2, next_bp3, next_bp4, curviness=1.,
                         zrot=zr, method="corrected_quaternion")
        self.linkers.append(l)
        self.basepairs.extend(l.basepairs)
        # reset bp indices
        for ii, bp in enumerate(self.basepairs):
            bp.set_bp_index(ii)
        return None


class DNAChain(PlottableSequence):

    def __init__(self, genome, chain=0):
        """
        DNAChain(genome)

        Construct a DNA chain from a genome of GATC
        """
        self.basepairs_chain0 = self.makeFromGenome(genome, chain=chain)
        self.basepairs = self.basepairs_chain0
        self.center_in_z()

    @staticmethod
    def makeFromGenome(genome, chain=0):
        dnachain = []
        position = np.array([0, 0, 0], dtype=float)
        rotation = np.array([0, 0, 0], dtype=float)
        index = 0
        for char in genome:
            print("Appending " + char)
            dnachain.append(
                basepair.BasePair(char, chain=chain, position=position,
                                  rotation=rotation, index=index))
            position += np.array([0., 0., BP_SEPARATION])
            rotation += np.array([0., 0., BP_ROTATION])
            index += 1
        return dnachain

    @staticmethod
    def turnAndTwistChain(chain, twist=0.):
        zmax = 0
        zmin = 0
        for pair in chain:
            # for (name, mol) in pair.iterMolecules():
            if pair.position[2] < zmin:
                zmin = pair.position[2]
            elif pair.position[2] > zmax:
                zmax = pair.position[2]

        zrange = zmax - zmin
        radius = 2. * zrange / np.pi
        print(radius)

        for pair in chain:
            # Translation of the frame - new center position
            theta = np.pi / 2. * (pair.position[2] - zmin) / zrange
            new_origin = np.array([radius * (1 - np.cos(theta)),
                                   0.,
                                   radius * np.sin(theta) - radius])
            # rotation of the frame
            # oldframe = np.array([mol.position[0], mol.position[1], 0])
            yang = np.pi / 2. * (pair.position[2] - zmin) / zrange
            pair.rotate(np.array([0, yang, 0]), about_origin=True)

            xang = twist * (pair.position[2] - zmin) / zrange
            chain_z_axis = pair.rmatrix[:, 2]
            rmatrix = r.rot_ax_angle(chain_z_axis, xang)
            pair.rotate(rmatrix, about_origin=True)

            pair.translate(new_origin - pair.position)
        return chain

    def center_in_z(self):
        """
        DNAChain.center_in_z()
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

        ztrans = (minz - maxz)/2. - minz
        translation = np.array([0., 0., ztrans])

        for bp in self.basepairs:
            bp.translate(translation)

        return None

    # def to_text(self, seperator=" "):
    #     """
    #     Return a description of the molecules in the chain as text
    #     """
    #     key = "# NAME SHAPE CHAIN_ID STRAND_ID BP_INDEX " +\
    #           "SIZE_X SIZE_Y SIZE_Z POS_X " +\
    #           "POS_Y POS_Z ROT_X ROT_Y ROT_Z\n"
    #     output = [key]
    #     for pair in self.basepairs:
    #         output.append(pair.to_text(seperator=seperator))
    #
    #     return "".join(output)
    #
    # def to_plot(self):
    #     """
    #     Return a matplotlib.Figure instance with molecules plotted
    #     """
    #     sugars = []
    #     triphosphates = []
    #     bases = []
    #     bps = ["guanine", "adenine", "thymine", "cytosine"]
    #     for pair in self.basepairs:
    #         for (name, molecule) in pair.iterMolecules():
    #             if molecule.name.lower() == "sugar":
    #                 sugars.append(molecule.position)
    #             elif molecule.name.lower() == "phosphate":
    #                 triphosphates.append(molecule.position)
    #             elif molecule.name.lower() in bps:
    #                 bases.append(molecule.position)
    #
    #     # Plotting
    #     bases = zip( * map(list, bases))
    #     triphosphates = zip( * map(list, triphosphates))
    #     sugars = zip( * map(list, sugars))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     ax.scatter(bases[0], bases[1], bases[2], c="0.6", s=20)
    #     ax.scatter(triphosphates[0], triphosphates[1], triphosphates[2],
    #                 c="y", s=20)
    #     ax.scatter(sugars[0], sugars[1], sugars[2], c="r", s=20)
    #
    #     return fig
    #
    # def to_surface_plot(self):
    #     """
    #     Plot the surfaces of each molecule in the chain.
    #     Avoid this with large chains, this assumes each molecule is an ellipse  # NOQA
    #     """
    #
    #     def ellipse_xyz(center, extent, rotation=np.zeros([3])):
    #         rmatrix = r.eulerMatrix(*rotation)
    #         [a, b, c] = extent
    #         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    #         x = a * np.cos(u) * np.sin(v) + center[0]
    #         y = b * np.sin(u) * np.sin(v) + center[1]
    #         z = c * np.cos(v) + center[2]
    #         for ii in range(0, len(x)):
    #             for jj in range(0, len(x[ii])):
    #                 row = np.array([x[ii][jj], y[ii][jj], z[ii][jj]]) - center  # NOQA
    #                 xp, yp, zp = np.dot(rmatrix, row.transpose())
    #                 x[ii][jj] = xp + center[0]
    #                 y[ii][jj] = yp + center[1]
    #                 z[ii][jj] = zp + center[2]
    #         return x, y, z
    #
    #     sugars = []
    #     triphosphates = []
    #     bases = []
    #     bps = ["guanine", "adenine", "thymine", "cytosine"]
    #     for pair in self.basepairs:
    #         for (name, molecule) in pair.iterMolecules():
    #             if molecule.name.lower() == "sugar":
    #                 sugars.append((molecule.position, molecule.dimensions,
    #                                molecule.rotation))
    #             elif molecule.name.lower() == "phosphate":
    #                 triphosphates.append((molecule.position,
    #                                       molecule.dimensions,
    #                                       molecule.rotation))
    #             elif molecule.name.lower() in bps:
    #                 bases.append((molecule.position, molecule.dimensions,
    #                               molecule.rotation))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     for base in bases:
    #         x, y, z = ellipse_xyz(base[0], base[1], rotation=base[2])
    #         ax.plot_wireframe(x, y, z, color="0.6")
    #
    #     for phosphate in triphosphates:
    #         x, y, z = ellipse_xyz(phosphate[0], phosphate[1],
    #                               rotation=phosphate[2])
    #         ax.plot_wireframe(x, y, z, color="y")
    #
    #     for sugar in sugars:
    #         x, y, z = ellipse_xyz(sugar[0], sugar[1], rotation=sugar[2])
    #         ax.plot_wireframe(x, y, z, color="r")
    #
    #     return fig


class TurnedDNAChain(DNAChain):
    """
    """
    def __init__(self, genome):
        """
        TurnedDNAChain(genome)

        Construct a DNA chain from a genome of GATC that turns 90 degrees
        """
        super(TurnedDNAChain, self).__init__(genome)
        self.turnDNA()

    def turnDNA(self):
        self.basepairs = DNAChain.turnAndTwistChain(self.basepairs)
        return None


class TurnedTwistedDNAChain(DNAChain):
    """
    """
    def __init__(self, genome):
        """
        TurnedDNAChain(genome)

        Construct a DNA chain from a genome of GATC that turns 90 degrees
        """
        super(TurnedTwistedDNAChain, self).__init__(genome)
        self.turnAndTwistDNA()

    def turnAndTwistDNA(self):
        self.basepairs =\
            DNAChain.turnAndTwistChain(self.basepairs, twist=np.pi/2.)
        return None


class DoubleDNAChain(DNAChain):
    """
    """
    def __init__(self, genome, separation):
        """
        DoubleDNAChain(genome, separation)

        Construct two parallel straight DNA chains

        args:
            genome: string of GATC specifying genome order
            separation: separation of each strand from the center in angstroms
        """
        super(DoubleDNAChain, self).__init__(genome)
        self.duplicateDNA(separation)

    def duplicateDNA(self, separation):
        translation = np.array([0., separation / 2., 0.], dtype=float)
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
        self.makeFromGenome(genome)
        self.duplicateDNA(separation=separation)
        self.turnDNA()


class TurnedTwistedDoubleDNAChain(TurnedTwistedDNAChain, DoubleDNAChain):
    def __init__(self, genome, separation):
        self.makeFromGenome(genome)
        self.duplicateDNA(separation=separation)
        self.turnAndTwistDNA()


class FourStrandDNAChain(DNAChain):
    def __init__(self, genome, separation):
        """
        FourStrandDNAChain(genome, separation)

        Construct four parallel straight DNA chains

        args:
            genome: string of GATC specifying genome order
            separation: separation of each strand from the center in angstroms
        """
        super(FourStrandDNAChain, self).__init__(genome)
        self.makeFourStrands(separation)

    def makeFourStrands(self, separation):
        translation_y = np.array([0., separation / 2., 0.], dtype=float)
        translation_x = np.array([separation / 2., 0., 0.], dtype=float)
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

        self.basepairs = self.basepairs_chain0 + self.basepairs_chain1 + \
            self.basepairs_chain2 + self.basepairs_chain3


class FourStrandTurnedDNAChain(DNAChain):
    def __init__(self, genome, separation, twist=False):
        """
        FourStrandTurnedDNAChain(genome, separation)

        Construct four DNA chains that turn 90 degrees.

        Chain indices are assigned anticlockwise starting from the +y strand.

        args:
            genome: string of GATC specifying genome order
            separation: separation of each strand from the center in angstroms

        kwargs:
            twist: boolean, add a 90 deg twist to each chain
        """
        DNAChain.__init__(self, genome)
        translation_y = np.array([0., separation / 2., 0.], dtype=float)
        translation_x = np.array([separation / 2., 0., -separation / 2.],
                                 dtype=float)
        ang = np.pi/2. if twist is True else 0

        radiusC0 =\
            len(self.basepairs_chain0) * BP_SEPARATION * 2 / np.pi
        radiusC3 = (radiusC0 - separation / 2.)
        radiusC1 = (radiusC0 + separation / 2.)

        self.basepairs_chain2 = DNAChain(genome, chain=2).basepairs

        lengthC3 = \
            int(np.floor(radiusC3 / radiusC0 * len(genome)))
        lengthC1 = \
            int(np.floor(radiusC1 / radiusC0 * len(genome)))
        longGenome = genome * int(np.ceil(radiusC1 /
                                          radiusC0))

        genome_chain3 = genome[:lengthC3]
        self.basepairs_chain3 = DNAChain(genome_chain3, chain=3).basepairs
        genome_chain1 = longGenome[:lengthC1]
        self.basepairs_chain1 = DNAChain(genome_chain1, chain=1).basepairs

        chains = [self.basepairs_chain0,
                  self.basepairs_chain1,
                  self.basepairs_chain2,
                  self.basepairs_chain3]
        transforms = [+translation_y,
                      -translation_x,
                      -translation_y,
                      +translation_x]
        angles = [ang + (2*np.pi - BP_ROTATION*len(c) % (2*np.pi))
                  for c in chains]

        for (ii, (c, t, a)) in enumerate(zip(chains, transforms, angles)):
            c = self.turnAndTwistChain(c, twist=a)
            for bp in c:
                bp.translate(t)
            chains[ii] = c

        self.basepairs = []
        for c in chains:
            self.basepairs.extend(c)

        return None


class EightStrandDNAChain(DNAChain):
    def __init__(self, genome, sep1, sep2, turn=False, twist=False):
        """
        FourStrandTurnedDNAChain(genome, sep1, sep2, turn=False, twist=False)

        Construct eight DNA chains that can turn 90 degrees if turn=True

        Chain indices are assigned anticlockwise starting from the +y strand,
            first to the inner four strands, then two the outer four strands.
            ie.
                      4           Strands 1 and 3, 0 and 2 are separated by
                      0           sep1
                5  1     3  7     Strands 4 and 6, 5 and 7 are separated by
                      2           sep2
                      6


        args:
            genome: string of GATC specifying genome order
            sep1: separation of inner strands from the center in angstroms
            sep1: separation of outer strands from the center in angstroms

        kwargs:
            turn: boolean, turn strands 90degrees along box
            twist: boolean, add a 90 deg twist to each chain
        """
        DNAChain.__init__(self, genome)
        v1 = -sep1 / 2. if turn is True else 0
        v2 = -sep2 / 2. if turn is True else 0
        trans_y1 = np.array([0., sep1 / 2., 0.], dtype=float)
        trans_x1 = np.array([sep1 / 2., 0., v1],  dtype=float)
        trans_y2 = np.array([0., sep2 / 2., 0.], dtype=float)
        trans_x2 = np.array([sep2 / 2., 0., v2],  dtype=float)
        ang = np.pi/2. if twist is True else 0

        # centrally aligned strands
        self.basepairs_chain2 = DNAChain(genome, chain=2).basepairs
        self.basepairs_chain4 = DNAChain(genome, chain=4).basepairs
        self.basepairs_chain6 = DNAChain(genome, chain=6).basepairs

        radiusC0 = len(self.basepairs_chain0) * BP_SEPARATION * 2 / np.pi
        if turn is True:
            radiusC1 = (radiusC0 + sep1 / 2.)
            radiusC3 = (radiusC0 - sep1 / 2.)
            radiusC5 = (radiusC0 + sep2 / 2.)
            radiusC7 = (radiusC0 - sep2 / 2.)
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

        self.basepairs_chain1 = DNAChain(longGenome[:lengthC1],
                                         chain=1).basepairs
        self.basepairs_chain3 = DNAChain(longGenome[:lengthC3],
                                         chain=3).basepairs
        self.basepairs_chain5 = DNAChain(longGenome[:lengthC5],
                                         chain=5).basepairs
        self.basepairs_chain7 = DNAChain(longGenome[:lengthC7],
                                         chain=7).basepairs

        chains = [self.basepairs_chain0,
                  self.basepairs_chain1,
                  self.basepairs_chain2,
                  self.basepairs_chain3,
                  self.basepairs_chain4,
                  self.basepairs_chain5,
                  self.basepairs_chain6,
                  self.basepairs_chain7]
        transforms = [+trans_y1,
                      -trans_x1,
                      -trans_y1,
                      +trans_x1,
                      +trans_y2,
                      -trans_x2,
                      -trans_y2,
                      +trans_x2]
        angles = [ang + (2*np.pi - BP_ROTATION*len(c) % (2*np.pi))
                  for c in chains]
        print(angles)

        for (ii, (c, t, a)) in enumerate(zip(chains, transforms, angles)):
            if turn is True:
                c = self.turnAndTwistChain(c, twist=a)
            for bp in c:
                bp.translate(t)
            chains[ii] = c

        self.basepairs = []
        for c in chains:
            self.basepairs.extend(c)

        return None
