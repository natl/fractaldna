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


BP_SEPARATION = 34  # Angstrom
BP_ROTATION = 36. / 180. * np.pi  # degrees


class DNAChain(object):

    def __init__(self, genome):
        """
        DNAChain(genome)

        Construct a DNA chain from a genome of GATC
        """
        self.basepairs_chain0 = self.makeFromGenome(genome)
        self.basepairs = self.basepairs_chain0

    @staticmethod
    def makeFromGenome(genome):
        chain = []
        position = np.array([0, 0, 0], dtype=float)
        rotation = np.array([0, 0, 0], dtype=float)
        for char in genome:
            print("Appending " + char)
            chain.append(
                basepair.BasePair(char, chain=0, position=position,
                                  rotation=rotation))
            position += np.array([0., 0., BP_SEPARATION])
            rotation += np.array([0., 0., BP_ROTATION])
        return chain

    @staticmethod
    def turnChain(chain):
        zmax = len(chain) * BP_SEPARATION
        radius = 2. * zmax / np.pi
        for pair in chain:
            for (name, mol) in pair.iterMolecules():
                # Translation of the frame - new center position
                theta = np.pi / 2. * mol.position[2] / zmax
                neworigin = np.array([radius * (1 - np.cos(theta)),
                                      0.,
                                      radius * np.sin(theta)])
                # rotation of the frame
                oldframe = np.array([mol.position[0], mol.position[1], 0])
                yrotation = np.pi / 2. * mol.position[2] / zmax

                newframe = np.dot(r.roty(yrotation), oldframe)
                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]
        return chain

    @staticmethod
    def turnAndTwistChain(chain):
        zmax = len(chain) * BP_SEPARATION
        radius = 2. * zmax / np.pi
        for pair in chain:
            for (name, mol) in pair.iterMolecules():
                # Translation of the frame - new center position
                theta = np.pi / 2. * mol.position[2] / zmax
                neworigin = np.array([radius * (1 - np.cos(theta)),
                                      0.,
                                      radius * np.sin(theta)])
                # rotation of the frame
                oldframe = np.array([mol.position[0], mol.position[1], 0])
                yang = np.pi / 2. * mol.position[2] / zmax
                xang = np.pi / 2. * mol.position[2] / zmax

                print(mol.position[2])

                newframe = np.dot(r.rotx(xang), np.dot(r.roty(yang), oldframe))

                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]
        return chain

    def to_text(self, seperator=" "):
        """
        Return a description of the molecules in the chain as text
        """
        key = "# NAME SHAPE CHAIN_ID STRAND_ID SIZE_X SIZE_Y SIZE_Z POS_X " +\
              "POS_Y POS_Z ROT_X ROT_Y ROT_Z\n"
        output = [key]
        for pair in self.basepairs:
            output.append(pair.toText(seperator=seperator))

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
        zmax = len(self.basepairs_chain0) * BP_SEPARATION
        radius = 2. * zmax / np.pi
        for pair in self.basepairs:
            for (name, mol) in pair.iterMolecules():
                # Translation of the frame - new center position
                theta = np.pi / 2. * mol.position[2] / zmax
                neworigin = np.array([radius * (1 - np.cos(theta)),
                                      0.,
                                      radius * np.sin(theta)])
                # rotation of the frame
                oldframe = np.array([mol.position[0], mol.position[1], 0])
                yrotation = np.pi / 2. * mol.position[2] / zmax

                newframe = np.dot(r.roty(yrotation), oldframe)
                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]


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
        zmax = len(self.basepairs_chain0) * BP_SEPARATION
        radius = 2. * zmax / np.pi
        for pair in self.basepairs:
            for (name, mol) in pair.iterMolecules():
                # Translation of the frame - new center position
                theta = np.pi / 2. * mol.position[2] / zmax
                neworigin = np.array([radius * (1 - np.cos(theta)),
                                      0.,
                                      radius * np.sin(theta)])
                # rotation of the frame
                oldframe = np.array([mol.position[0], mol.position[1], 0])
                yang = np.pi / 2. * mol.position[2] / zmax
                xang = np.pi / 2. * mol.position[2] / zmax

                print(mol.position[2])

                newframe = np.dot(r.rotx(xang), np.dot(r.roty(yang), oldframe))

                mol.position[0] = neworigin[0] + newframe[0]
                mol.position[1] = neworigin[1] + newframe[1]
                mol.position[2] = neworigin[2] + newframe[2]


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
        translation_x = np.array([separation / 2., 0., 0.], dtype=float)
        if twist is True:
            transform = self.turnAndTwistChain
        else:
            transform = self.turnChain
        radiusMiddleChain = len(self.basepairs_chain0) * BP_SEPARATION
        radiusInnerChain = (radiusMiddleChain - separation)
        radiusOuterChain = (radiusMiddleChain + separation)

        self.basepairs_chain1 = deepcopy(self.basepairs_chain0)

        chain2Length = \
            int(np.floor(radiusInnerChain / radiusMiddleChain * len(genome)))
        chain3Length = \
            int(np.floor(radiusOuterChain / radiusMiddleChain * len(genome)))
        longGenome = genome * int(np.ceil(radiusOuterChain /
                                          radiusMiddleChain))

        genome_chain2 = genome[:chain2Length]
        genome_chain3 = longGenome[:chain3Length]
        # pdb.set_trace()

        self.basepairs_chain0 = transform(self.basepairs_chain0)
        for bp in self.basepairs_chain0:
            bp.translate(translation_y)
            bp.setNewChain(0)

        self.basepairs_chain1 = transform(self.basepairs_chain1)
        for bp in self.basepairs_chain1:
            bp.translate(-1 * translation_y)
            bp.setNewChain(1)

        self.basepairs_chain2 = transform(self.makeFromGenome(genome_chain2))
        for bp in self.basepairs_chain2:
            bp.translate(translation_x)
            bp.setNewChain(2)

        self.basepairs_chain3 = transform(self.makeFromGenome(genome_chain3))
        for bp in self.basepairs_chain3:
            bp.translate(-1 * translation_x)
            bp.setNewChain(3)

        self.basepairs = self.basepairs_chain0 + self.basepairs_chain1 + \
            self.basepairs_chain2 + self.basepairs_chain3

        return None
