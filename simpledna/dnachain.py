"""
Class description of a DNA chain built of base pairs
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from copy import deepcopy
from scipy.interpolate import interp1d

from utils import rotations as r
from utils import basepair
from utils import BP_ROTATION, BP_SEPARATION


class PlottableSequence(object):
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

    def to_plot(self):
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
        bases = [ii for ii in zip( * map(list, bases))]
        triphosphates = [ii for ii in zip( * map(list, triphosphates))]
        sugars = [ii for ii in zip( * map(list, sugars))]

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


class Solenoid(PlottableSequence):
    radius = 80  # angstroms, radius from center to place histones
    tilt = 20*np.pi/180.  # tilt chromosomes 20 deg following F98
    zshift = 18.3  # angstrom, z shift per chromosome following F98
    histones = 6

    def __init__(self):
        self.basepairs = []
        self.positions = [np.array([0, -self.radius, 0])]
        rm = r.eulerMatrix(np.pi/2., -np.pi/2., np.pi/2.)
        rm = np.dot(r.roty(self.tilt), rm)
        self.rotations = [r.getEulerAngles(rm)]
        for ii in range(self.histones - 1):
            last = self.positions[-1]
            this = np.dot(r.rotz(np.pi/3.), last)
            this[2] = last[2] + self.zshift
            self.positions.append(this)
            last = self.rotations[-1]
            this = np.array([last[0], last[1], last[2] + np.pi/3.])
            self.rotations.append(this)
        print(self.rotations)
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
                l = SplineLinker(bp1, bp2, bp3, bp4, curviness=1)
                self.linkers.append(l)
                self.basepairs.extend(l.basepairs)
            self.basepairs.extend(h.basepairs)


class SplineLinker(PlottableSequence):
    """
    """
    linker_rotation = BP_ROTATION  # rad, default screw rotation of dna
    linker_bp_spacing = BP_SEPARATION  # angstrom, default spacing between bps

    def __init__(self, bp1, bp2, bp3, bp4, curviness=1.):
        """
        Create a smooth linker based on splines.

        linker = SplineLinker(bp1, bp2, bp3, bp4, curviness=1.)

        Create base pairs that link to sections of DNA as follows:
        bp1 bp2 <==== LINKER =====> bp3 bp4
        Two base pairs on either side of the linker are needed to build splines
        low curviness = straighter
        high_curviness = smoother
        """
        self.basepairs = []
        points = np.array([bp1.position, bp2.position,
                           bp3.position, bp4.position])
        start_x = bp2.rmatrix[:, 0]
        end_x = bp3.rmatrix[:, 0]
        relative_angle = np.arccos(np.sum(start_x*end_x) /
                                   np.sum(start_x**2)**.5 /
                                   np.sum(end_x**2)**.5)
        d = np.sum((bp2.position - bp3.position)**2)**.5
        if curviness <= 0:
            curviness = 1e-200
        diff = 3.4/180/curviness
        print(d)
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
        tt = np.linspace(1, 2, n)
        xx = x_interp(tt[1:len(tt)])
        yy = y_interp(tt[1:len(tt)])
        zz = z_interp(tt[1:len(tt)])

        estim_rel_angle = ((n*self.linker_rotation) % (2*np.pi)) - np.pi
        diff = relative_angle - estim_rel_angle
        rot_angle = self.linker_rotation - estim_rel_angle/n
        # rot_start = bp2.rotation
        # rot_stop = bp3.rotation
        for (ii, (_x, _y, _z)) in enumerate(zip(xx, yy, zz)):
            if ii == 0:
                prev_bp = bp2
            else:
                prev_bp = self.basepairs[-1]

            if ii != len(xx) - 1:
                new_z = np.array([xx[ii + 1] - xx[ii],
                                  yy[ii + 1] - yy[ii],
                                  zz[ii + 1] - zz[ii]])
                old_rotation = prev_bp.rmatrix
                old_x = old_rotation[:, 0]
                old_z = old_rotation[:, 2]
                # rotate about local z axis by 34 deg.
                rz = r.rot_ax_angle(old_z, rot_angle)
                # rotate about a chosen eigenvector so new z axis aligns
                # with normal. Choose local x axis to be eigenvector
                dot_product = new_z[0]*old_z[0] + new_z[1]*old_z[1] +\
                    new_z[2]*old_z[2]
                mag_old = np.sum(old_z**2)**.5
                mag_new = np.sum(new_z**2)**.5
                angle = np.arccos(dot_product/mag_old/mag_new)

                rx = r.rot_ax_angle(old_x, angle)
                rots = r.getEulerAngles(np.dot(np.dot(rx, rz), old_rotation))
                bp = basepair.BasePair(np.random.choice(["G", "A", "T", "C"]),
                                       chain=0,
                                       position=np.array([_x, _y, _z]),
                                       rotation=np.array(rots),
                                       index=ii)
                self.basepairs.append(bp)

        pass


class Histone(PlottableSequence):
    """
    """
    radius_histone = 25  # radius of histone, angstrom
    pitch_dna = 23.9  # pitch of DNA helix, angstrom
    radius_dna = 41.8  # radius of DNA wrapping, angstrom
    histone_bps = 146  # number of bps in histone
    histone_turns = 1.65 * 2 * np.pi  # angular turn around histone, radians
    height = histone_turns * radius_dna / pitch_dna
    z_offset = -height / 2  # distance between first bp and xy-plane, angstrom
    # separation of bps around histone, angstrom
    hist_bp_separation = histone_turns * radius_dna / histone_bps
    hist_bp_rotation = 34.3 / 180. * np.pi  # screw rotation of bp, radians
    z_per_bp = 2 * height / histone_bps
    turn_per_bp = histone_turns / histone_bps
    z_angle = np.arctan(1./pitch_dna)
    histone_start_bp_rot = 0  # radians, rotation of bp at start of histone
    histone_end_bp_rot = histone_start_bp_rot + histone_bps*hist_bp_rotation

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


class DNAChain(object):

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
            for (name, mol) in pair.iterMolecules():
                if mol.position[2] < zmin:
                    zmin = mol.position[2]
                elif mol.position[2] > zmax:
                    zmax = mol.position[2]

        zrange = zmax - zmin
        radius = 2. * zrange / np.pi

        for pair in chain:
            for (name, mol) in pair.iterMolecules():
                # Translation of the frame - new center position
                theta = np.pi / 2. * (mol.position[2] - zmin) / zrange
                neworigin = np.array([radius * (1 - np.cos(theta)),
                                      0.,
                                      radius * np.sin(theta) - radius])
                # rotation of the frame
                oldframe = np.array([mol.position[0], mol.position[1], 0])
                yang = np.pi / 2. * (mol.position[2] - zmin) / zrange
                xang = twist * (mol.position[2] - zmin) / zrange

                newframe = np.dot(r.rotx(xang), np.dot(r.roty(yang), oldframe))

                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]
                mol.rotate(np.array([xang, yang, 0]))
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

    def to_plot(self):
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
        bases = zip( * map(list, bases))
        triphosphates = zip( * map(list, triphosphates))
        sugars = zip( * map(list, sugars))

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
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
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
