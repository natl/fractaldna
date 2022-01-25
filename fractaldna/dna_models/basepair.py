"""
basepair.py

Define base pairs in a DNA chain
"""

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from fractaldna.dna_models import dnapositions as dpos
from fractaldna.dna_models import molecules
from fractaldna.utils.rotations import eulerMatrix

# Positions in Angstroms
PHOSPHATE_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.PHOSPHATE).find_center()
SUGAR_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.DEOXYRIBOSE).find_center()
GUANINE_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.GUANINE).find_center()
ADENINE_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.ADENINE).find_center()
THYMINE_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.THYMINE).find_center()
CYTOSINE_POS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.CYTOSINE).find_center()


PHOSPHATE_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.PHOSPHATE, inverse=True
).find_center()
SUGAR_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.DEOXYRIBOSE, inverse=True
).find_center()
GUANINE_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.GUANINE, inverse=True
).find_center()
ADENINE_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.ADENINE, inverse=True
).find_center()
THYMINE_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.THYMINE, inverse=True
).find_center()
CYTOSINE_POS_OPP = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.CYTOSINE, inverse=True
).find_center()


class BasePair:
    """
    Defines a base pair

    BasePair(base, chain=-1, position=np.zeros(3), rotation=np.zeros(3))

    Create a base pair defined by a particular base GTAC (in the 5' dirn)

    :param base:   defining base (GATC)
    :param chain:    int to indicate the id of the current DNA macromolecule
    :param position: position of base pair relative to (0, 0, 0)
    :param rotation: rotation euler angles of base pair relative to local xyz
    :param index:    index of base pair in chain
    """

    pairings = {
        "G": [molecules.Guanine, molecules.Cytosine],
        "A": [molecules.Adenine, molecules.Thymine],
        "T": [molecules.Thymine, molecules.Adenine],
        "C": [molecules.Cytosine, molecules.Guanine],
    }
    moleculeNames = [
        "leftPhosphate",
        "leftSugar",
        "leftBasePair",
        "rightPhosphate",
        "rightSugar",
        "rightBasePair",
    ]

    def __init__(
        self,
        base: str,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """Constructor"""
        if base not in ["G", "A", "T", "C"]:
            raise ValueError("base must be either G, A, T, C")

        # start building at 0, so we can rotate in a neutral frame
        # before translating
        self.position = np.zeros(3)
        self.rmatrix = np.eye(3)
        self.chain = chain
        position = np.array(deepcopy(position), dtype=float)
        rotation = np.array(deepcopy(rotation), dtype=float)

        leftPhosphateRot = np.zeros(3)
        leftPhosphatePos = deepcopy(PHOSPHATE_POS)
        self.leftPhosphate = molecules.Triphosphate(
            strand=0,
            chain=chain,
            position=leftPhosphatePos,
            rotation=leftPhosphateRot,
            index=index,
        )
        leftSugarRot = np.zeros(3)
        leftSugarPos = deepcopy(SUGAR_POS)
        self.leftSugar = molecules.DNASugar(
            strand=0,
            chain=chain,
            position=leftSugarPos,
            rotation=leftSugarRot,
            index=index,
        )

        rightPhosphateRot = np.zeros(3)
        rightPhosphatePos = deepcopy(PHOSPHATE_POS_OPP)
        self.rightPhosphate = molecules.Triphosphate(
            strand=1,
            chain=chain,
            position=rightPhosphatePos,
            rotation=rightPhosphateRot,
            index=index,
        )

        rightSugarRot = np.zeros(3)
        rightSugarPos = deepcopy(SUGAR_POS_OPP)
        self.rightSugar = molecules.DNASugar(
            strand=1,
            chain=chain,
            position=rightSugarPos,
            rotation=rightSugarRot,
            index=index,
        )

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
        self.leftBasePair = bases[0](
            strand=0,
            chain=chain,
            position=leftBasePairPos,
            rotation=leftBasePairRot,
            index=index,
        )
        self.rightBasePair = bases[1](
            strand=1,
            chain=chain,
            position=rightBasePairPos,
            rotation=rightBasePairRot,
            index=index,
        )
        self.molecules = [
            self.leftPhosphate,
            self.leftSugar,
            self.leftBasePair,
            self.rightPhosphate,
            self.rightSugar,
            self.rightBasePair,
        ]

        self.moleculeDict = OrderedDict(zip(self.moleculeNames, self.molecules))

        self.rotate(rotation, about_origin=False)
        self.translate(position)  # sets self.position

    def translate(self, translation: np.array) -> None:
        """
        BasePair.translate(translation)

        translate the base pair by the specified 3-vector array
        """
        for molecule in self.moleculeDict.values():
            molecule.translate(translation)
        self.position = self.position + np.array(translation)
        return None

    def rotate(self, rotation: np.array, about_origin: bool = False) -> None:
        """
        BasePair.rotation(rotation, about_origin=False)

        Rotate elements in base pair

        If about_origin is True, the rotation is with respect to (0, 0, 0)
        If about_origin is False, the rotation is with respect to the base pair position
        """
        if rotation.size == 3:
            rmatrix = eulerMatrix(rotation[0], rotation[1], rotation[2])
        elif rotation.shape == (3, 3):
            rmatrix = rotation
        else:
            return NotImplementedError("The rotation was invalid")

        self.rmatrix = np.dot(rmatrix, self.rmatrix)
        if about_origin is True:
            self.position = np.dot(rmatrix, self.position)

        for molecule in self.moleculeDict.values():
            if about_origin is True:
                molecule.position = np.dot(rmatrix, molecule.position)
            else:
                local_pos = molecule.position - self.position
                molecule.position = np.dot(rmatrix, local_pos) + self.position
            molecule.rotate(rotation)

        return None

    def getCenter(self) -> np.array:
        """
        basepair.getCenter()

        Get the geometric center of the base-pair
        (non-weighted average molecule position)
        """
        positions = [mol.position for mol in self.moleculeDict.values()]
        return np.mean(np.array(positions), axis=1)

    def setNewChain(self, chainIdx: int) -> None:
        """
        BasePair.setNewChain(chainIdx)

        Reset the chain index for all molecules and the base pair
        """
        self.chain = chainIdx
        for molecule in self.moleculeDict.values():
            molecule.chain = chainIdx

        return None

    def set_bp_index(self, bp_idx: int) -> None:
        """
        BasePair.setNewChain(chainIdx)

        Reset the chain index for all molecules
        """
        for molecule in self.moleculeDict.values():
            molecule.index = bp_idx

        return None

    def to_text(self, seperator: str = " ") -> str:
        """
        BasePair.to_text(self, seperator=" ")
        Return a description of the molecules in the base pair as text
        """
        output = []
        for molecule in self.moleculeDict.values():
            output.append(molecule.to_text(seperator=seperator))

        return "".join(output)

    def to_frame(self) -> pd.DataFrame:
        """Return base pair as data frame of molecules"""
        return pd.DataFrame([mol.to_series() for mol in self.moleculeDict.values()])

    def __getitem__(self, key: str) -> molecules.Molecule:
        return self.moleculeDict[key]

    def iterMolecules(self):
        return self.moleculeDict.items()
