"""
Class description of a DNA chain built of base pairs
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from copy import deepcopy

from utils import rotations as r
from utils import basepair
from utils import BP_ROTATION, BP_SEPARATION

# BP_SEPARATION = 3.32  # Angstrom
# BP_ROTATION = 34.3 / 180. * np.pi  # degrees


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

    # @staticmethod
    # def turnAndTwistChain(chain):
    #     zmax = 0
    #     zmin = 0
    #     for pair in chain:
    #         for (name, mol) in pair.iterMolecules():
    #             if mol.position[2] < zmin:
    #                 zmin = mol.position[2]
    #             elif mol.position[2] > zmax:
    #                 zmax = mol.position[2]
    #
    #     zrange = zmax - zmin
    #     radius = 2. * zrange / np.pi
    #
    #     for pair in chain:
    #         for (name, mol) in pair.iterMolecules():
    #             # Translation of the frame - new center position
    #             theta = np.pi / 2. * (mol.position[2] - zmin) / zrange
    #             neworigin = np.array([radius * (1 - np.cos(theta)),
    #                                   0.,
    #                                   radius * np.sin(theta) - radius])
    #             # rotation of the frame
    #             oldframe = np.array([mol.position[0], mol.position[1], 0])
    #             yrotation = np.pi / 2. * (mol.position[2] - zmin) / zrange
    #
    #             newframe = np.dot(r.roty(yrotation), oldframe)
    #             mol.position[0] = neworigin[0] + newframe[0]
    #             mol.position[1] = neworigin[1] + newframe[1]
    #             mol.position[2] = neworigin[2] + newframe[2]
    #             mol.rotate(np.array([0, yrotation, 0]))
    #     return chain

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
        angles = [ang + (2*np.pi - BP_ROTATION*len(c)%(2*np.pi)) for c in chains]

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
        angles = [ang + (2*np.pi - BP_ROTATION*len(c)%(2*np.pi)) for c in chains]
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
