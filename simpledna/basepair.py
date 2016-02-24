"""
basepair.py

Define base pairs in a DNA chain
"""
from __future__ import division, unicode_literals, print_function

import molecules
import numpy as np
from copy import deepcopy
from rotations import eulerMatrix

PHOSPHATE_POS = np.array([2, 2, 2], dtype=float)  # Angstrom
SUGAR_POS = np.array([2, 0, 0], dtype=float)  # Angstrom
BASEPAIR_POS = np.array([1, 0, 0], dtype=float)  # Angstrom


class BasePair(object):
    """
    Defines a base pair
    """
    pairings = {"G": [molecules.Guanine, molecules.Cytosine],
                "A": [molecules.Adenine, molecules.Thymine],
                "T": [molecules.Thymine, molecules.Adenine],
                "C": [molecules.Cytosine, molecules.Guanine]}
    moleculeNames = ["leftPhosphate", "leftSugar", "leftBasePair",
                     "rightPhosphate", "rightSugar", "rightBasePair"]

    def __init__(self, base, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        """
        BasePair(base, chain=-1, position=np.zeros(3), rotation=np.zeros(3))

        Create a base pair defined by a particular base GTAC (in the 5' dirn)

        args:
            base:   defining base (GATC)

        kwargs:
            chain:    int to indicate the id of the current DNA macromolecule
            position: position of base pair relative to (0, 0, 0)
            rotation: rotation euler angles of base pair relative to local xyz
        """
        assert base in ["G", "A", "T", "C"], "base must be either G, A, T, C"
        position = deepcopy(position)
        rotation = deepcopy(rotation)

        leftPhosphateRot = np.zeros(3)
        leftPhosphatePos = -1*PHOSPHATE_POS
        self.leftPhosphate = molecules.Triphosphate(strand=0, chain=chain,
                                                    position=leftPhosphatePos,
                                                    rotation=leftPhosphateRot)
        leftSugarRot = np.zeros(3)
        leftSugarPos = -1*SUGAR_POS
        self.leftSugar = molecules.DNASugar(strand=0, chain=chain,
                                            position=leftSugarPos,
                                            rotation=leftSugarRot)

        leftBasePairRot = np.zeros(3)
        leftBasePairPos = -1*BASEPAIR_POS

        rightPhosphateRot = np.zeros(3)
        rightPhosphatePos = +1*PHOSPHATE_POS
        self.rightPhosphate = \
            molecules.Triphosphate(strand=1, chain=chain,
                                   position=rightPhosphatePos,
                                   rotation=rightPhosphateRot)

        rightSugarRot = np.zeros(3)
        rightSugarPos = +1*SUGAR_POS
        self.rightSugar = molecules.DNASugar(strand=1, chain=chain,
                                             position=rightSugarPos,
                                             rotation=rightSugarRot)

        rightBasePairRot = np.zeros(3)
        rightBasePairPos = +1*BASEPAIR_POS

        bases = self.pairings[base]
        self.leftBasePair = bases[0](strand=0, chain=chain,
                                     position=leftBasePairPos,
                                     rotation=leftBasePairRot)
        self.rightBasePair = bases[1](strand=1, chain=chain,
                                      position=rightBasePairPos,
                                      rotation=rightBasePairRot)
        self.molecules = [self.leftPhosphate, self.leftSugar,
                          self.leftBasePair, self.rightPhosphate,
                          self.rightSugar, self.rightBasePair]

        self.moleculeDict = dict(zip(self.moleculeNames, self.molecules))

        self.rotate(rotation)
        self.translate(position)

    def translate(self, translation):
        """
        BasePair.translate(translation)

        translate the base pair by the specified 3-vector array
        """
        for molecule in self.moleculeDict.itervalues():
            molecule.translate(translation)

        return None

    def rotate(self, rotation):
        """
        BasePair.rotation(rotation)

        Rotate elements in base pair
        """
        rmatrix = eulerMatrix(rotation[0], rotation[1], rotation[2])
        for molecule in self.moleculeDict.itervalues():
            molecule.position = np.dot(rmatrix, molecule.position)
            molecule.rotation += rotation

        return None

    def setNewChain(self, chainIdx):
        """
        BasePair.setNewChain(chainIdx)

        Reset the chain index for all molecules
        """
        for molecule in self.moleculeDict.itervalues():
            molecule.chain = chainIdx

        return None

    def toText(self, seperator=" "):
        """
        Return a description of the molecules in the base pair as text
        """
        output = ""
        for molecule in self.moleculeDict.itervalues():
            output += molecule.toText(seperator=seperator)
            output += "\n"

        return output

    def __getitem__(self, key):
        return self.moleculeDict[key]

    def iterMolecules(self):
        return self.moleculeDict.iteritems()
