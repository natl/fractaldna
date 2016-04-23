"""
Class description of a DNA chain built of base pairs
"""
from __future__ import division, unicode_literals, print_function

import basepair

import numpy as np
import matplotlib.pyplot as plt
import rotations as r
from mpl_toolkits.mplot3d import Axes3D  # NOQA
from copy import deepcopy

import rotations as rot

BP_SEPARATION = 3.32  # Angstrom
BP_ROTATION = 36 / 180. * np.pi  # degrees


class DNAChain(object):

    def __init__(self, genome):
        """
        DNAChain(genome)

        Construct a DNA chain from a genome of GATC
        """
        self.basepairs_chain0 = self.makeFromGenome(genome)
        self.basepairs = self.basepairs_chain0
        self.center_in_z()

    @staticmethod
    def makeFromGenome(genome):
        chain = []
        position = np.array([0, 0, 0], dtype=float)
        rotation = np.array([0, 0, 0], dtype=float)
        index = 0
        for char in genome:
            print("Appending " + char)
            chain.append(
                basepair.BasePair(char, chain=0, position=position,
                                  rotation=rotation, index=index))
            position += np.array([0., 0., BP_SEPARATION])
            rotation += np.array([0., 0., BP_ROTATION])
            index += 1
        return chain

    @staticmethod
    def turnChain(chain):
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
                yrotation = np.pi / 2. * (mol.position[2] - zmin) / zrange

                newframe = np.dot(r.roty(yrotation), oldframe)
                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]
                mol.rotate(np.array([0, yrotation, 0]))
        return chain

    @staticmethod
    def turnAndTwistChain(chain):
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
                xang = np.pi / 2. * (mol.position[2] - zmin) / zrange

                print(mol.position[2])

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
                if molecule.name.lower() == "dnasugar":
                    sugars.append(molecule.position)
                elif molecule.name.lower() == "triphosphate":
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
            rmatrix = rot.eulerMatrix(*rotation)
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
                if molecule.name.lower() == "dnasugar":
                    sugars.append((molecule.position, molecule.dimensions,
                                   molecule.rotation))
                elif molecule.name.lower() == "triphosphate":
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
        self.basepairs = DNAChain.turnChain(self.basepairs)
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
        self.basepairs = DNAChain.turnAndTwistChain(self.basepairs)
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
            bp.translate(-1 * translation_y)
            bp.setNewChain(1)

        for bp in self.basepairs_chain2:
            bp.translate(translation_x)
            bp.setNewChain(2)

        for bp in self.basepairs_chain3:
            bp.translate(-1 * translation_x)
            bp.setNewChain(3)

        self.basepairs = self.basepairs_chain0 + self.basepairs_chain1 + \
            self.basepairs_chain2 + self.basepairs_chain3


class FourStrandTurnedDNAChain(DNAChain):
    def __init__(self, genome, separation, twist=False):
        """
        FourStrandTurnedDNAChain(genome, separation)

        Construct four DNA chains that turn 90 degrees

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
        if twist is True:
            transform = self.turnAndTwistChain
        else:
            transform = self.turnChain
        radiusMiddleChain =\
            len(self.basepairs_chain0) * BP_SEPARATION * 2 / np.pi
        radiusInnerChain = (radiusMiddleChain - separation / 2.)
        radiusOuterChain = (radiusMiddleChain + separation / 2.)

        self.basepairs_chain1 = DNAChain(genome, chain=1).basepairs

        chain2Length = \
            int(np.floor(radiusInnerChain / radiusMiddleChain * len(genome)))
        chain3Length = \
            int(np.floor(radiusOuterChain / radiusMiddleChain * len(genome)))
        longGenome = genome * int(np.ceil(radiusOuterChain /
                                          radiusMiddleChain))

        genome_chain2 = genome[:chain2Length]
        self.basepairs_chain2 = DNAChain(genome_chain2, chain=2).basepairs
        genome_chain3 = longGenome[:chain3Length]
        self.basepairs_chain3 = DNAChain(genome_chain3, chain=3).basepairs
        # pdb.set_trace()

        self.basepairs_chain0 = transform(self.basepairs_chain0)
        for bp in self.basepairs_chain0:
            bp.translate(translation_y)

        self.basepairs_chain1 = transform(self.basepairs_chain1)
        for bp in self.basepairs_chain1:
            bp.translate(-1 * translation_y)

        self.basepairs_chain2 = transform(self.basepairs_chain2)
        for bp in self.basepairs_chain2:
            bp.translate(translation_x)

        self.basepairs_chain3 = transform(self.basepairs_chain3)
        for bp in self.basepairs_chain3:
            bp.translate(-1 * translation_x)

        self.basepairs = self.basepairs_chain0 + self.basepairs_chain1 + \
            self.basepairs_chain2 + self.basepairs_chain3

        return None
