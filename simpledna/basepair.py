"""
basepair.py

Define base pairs in a DNA chain
"""
from __future__ import division, unicode_literals, print_function

import molecules
import dnapositions as dpos
import numpy as np
from copy import deepcopy
from rotations import eulerMatrix
from collections import OrderedDict

# Positions in Angstroms
PHOSPHATE_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.PHOSPHATE).find_center()
SUGAR_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.DEOXYRIBOSE).find_center()
GUANINE_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.GUANINE).find_center()
ADENINE_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.ADENINE).find_center()
THYMINE_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.THYMINE).find_center()
CYTOSINE_POS =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.CYTOSINE).find_center()


PHOSPHATE_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.PHOSPHATE,
                                            inverse=True).find_center()
SUGAR_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.DEOXYRIBOSE,
                                            inverse=True).find_center()
GUANINE_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.GUANINE,
                                            inverse=True).find_center()
ADENINE_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.ADENINE,
                                            inverse=True).find_center()
THYMINE_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.THYMINE,
                                            inverse=True).find_center()
CYTOSINE_POS_OPP =\
    dpos.MoleculeFromAtoms.from_cylindrical(dpos.CYTOSINE,
                                            inverse=True).find_center()


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
                 rotation=np.zeros(3), index=0):
        """
        BasePair(base, chain=-1, position=np.zeros(3), rotation=np.zeros(3))

        Create a base pair defined by a particular base GTAC (in the 5' dirn)

        args:
            base:   defining base (GATC)

        kwargs:
            chain:    int to indicate the id of the current DNA macromolecule
            position: position of base pair relative to (0, 0, 0)
            rotation: rotation euler angles of base pair relative to local xyz
            index:    index of base pair in chain
        """
        assert base in ["G", "A", "T", "C"], "base must be either G, A, T, C"
        position = deepcopy(position)
        rotation = deepcopy(rotation)

        leftPhosphateRot = np.zeros(3)
        leftPhosphatePos = deepcopy(PHOSPHATE_POS)
        self.leftPhosphate = molecules.Triphosphate(strand=0, chain=chain,
                                                    position=leftPhosphatePos,
                                                    rotation=leftPhosphateRot,
                                                    index=index)
        leftSugarRot = np.zeros(3)
        leftSugarPos = deepcopy(SUGAR_POS)
        self.leftSugar = molecules.DNASugar(strand=0, chain=chain,
                                            position=leftSugarPos,
                                            rotation=leftSugarRot,
                                            index=index)

        rightPhosphateRot = np.zeros(3)
        rightPhosphatePos = deepcopy(PHOSPHATE_POS_OPP)
        self.rightPhosphate = \
            molecules.Triphosphate(strand=1, chain=chain,
                                   position=rightPhosphatePos,
                                   rotation=rightPhosphateRot,
                                   index=index)

        rightSugarRot = np.zeros(3)
        rightSugarPos = deepcopy(SUGAR_POS_OPP)
        self.rightSugar = molecules.DNASugar(strand=1, chain=chain,
                                             position=rightSugarPos,
                                             rotation=rightSugarRot,
                                             index=index)

        if base == "G":
            leftpos = deepcopy(GUANINE_POS)
            rightpos = deepcopy(CYTOSINE_POS_OPP)
        elif base == "C":
            leftpos = deepcopy(CYTOSINE_POS)
            rightpos = deepcopy(GUANINE_POS_OPP)
        elif base == "T":
            leftpos = deepcopy(ADENINE_POS)
            rightpos = deepcopy(THYMINE_POS_OPP)
        elif base == "A":
            leftpos = deepcopy(THYMINE_POS)
            rightpos = deepcopy(ADENINE_POS_OPP)

        leftBasePairRot = np.zeros(3)
        leftBasePairPos = leftpos

        rightBasePairRot = np.zeros(3)
        rightBasePairPos = rightpos

        bases = self.pairings[base]
        self.leftBasePair = bases[0](strand=0, chain=chain,
                                     position=leftBasePairPos,
                                     rotation=leftBasePairRot,
                                     index=index)
        self.rightBasePair = bases[1](strand=1, chain=chain,
                                      position=rightBasePairPos,
                                      rotation=rightBasePairRot,
                                      index=index)
        self.molecules = [self.leftPhosphate, self.leftSugar,
                          self.leftBasePair, self.rightPhosphate,
                          self.rightSugar, self.rightBasePair]

        self.moleculeDict =\
            OrderedDict(zip(self.moleculeNames, self.molecules))

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
            molecule.rotate(rotation)

        return None

    def getCenter(self):
        """
        basepair.getCenter()

        Get the geometric center of the base-pair
        (non-weighted average molecule position)
        """
        positions = [mol.position for mol in self.moleculeDict.values()]
        return np.mean(np.array(positions), axis=1)

    def setNewChain(self, chainIdx):
        """
        BasePair.setNewChain(chainIdx)

        Reset the chain index for all molecules
        """
        for molecule in self.moleculeDict.itervalues():
            molecule.chain = chainIdx

        return None

    def to_text(self, seperator=" "):
        """
        BasePair.to_text(self, seperator=" ")
        Return a description of the molecules in the base pair as text
        """
        output = []
        for molecule in self.moleculeDict.itervalues():
            output.append(molecule.to_text(seperator=seperator))

        return "".join(output)

    def __getitem__(self, key):
        return self.moleculeDict[key]

    def iterMolecules(self):
        return self.moleculeDict.iteritems()
