"""
molecules.py

Define the molecules that make up a simple DNA chain
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from fractaldna.dna_models import dnapositions as dpos
from fractaldna.utils import rotations as rot

# Physical parameters, all in Angstrom
GUANINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.GUANINE
).find_equivalent_half_lengths()
ADENINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.ADENINE
).find_equivalent_half_lengths()
THYMINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.THYMINE
).find_equivalent_half_lengths()
CYTOSINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.CYTOSINE
).find_equivalent_half_lengths()

SUGAR_RADIUS = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.DEOXYRIBOSE
).find_equivalent_radius()
PHOSPHATE_RADIUS = dpos.MoleculeFromAtoms.from_cylindrical(
    dpos.PHOSPHATE
).find_equivalent_radius()


class Molecule:
    """Create a molecule

    :param name: molecule name
    :param shape: shape of molecule
    :param dimensions: dimensions of molecule shape
        (3-vector for xyz-dims, int/float for radius)
    :param strand: int to hold an ID related to a localised strand
    :param chain: int to hold an ID related to a macromolecule/protein or
        DNA chain
    :param position: 3-vector for molecule position relative to global axis
    :param rotation: 3-vector of euler angles (radians) for molecule
        rotation relative to the global xyz axis
    :param index: index of base pair along chain
    """

    def __init__(
        self,
        name: str,
        shape: str,
        dimensions: np.array,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        if type(dimensions) in [int, float, np.float64]:
            self.dimensions = np.array([deepcopy(dimensions)] * 3)
        else:
            assert len(dimensions) == 3, "Position is invalid"
            self.dimensions = deepcopy(dimensions)

        self.shape = deepcopy(shape)
        self.name = deepcopy(name)
        self.position = deepcopy(position)
        self.rotation = deepcopy(rotation)
        self.strand = strand
        self.chain = chain
        self.index = index

    def translate(self, translation: np.array) -> None:
        """
        Molecule.translate(translation)

        :param translation: Translate the molecule by (x, y, z)
        """
        self.position = self.position + translation
        return None

    def rotate(self, rotation: np.array) -> None:
        """
        Molecule.rotate(rotation)

        Rotate molecule by [X_angle, Y_angle, Z_angle]

        :param rotation: Euler angles for rotation
        """
        if rotation.size == 3:
            rmatrix = rot.eulerMatrix(rotation[0], rotation[1], rotation[2])
        elif rotation.shape == (3, 3):
            rmatrix = rotation
        else:
            return NotImplementedError("The rotation was invalid")
        # assert rotation.size == 3, "Rotation array must be length-3"
        oldrotation = rot.eulerMatrix(*self.rotation)
        newrotation = rmatrix
        self.rotation = rot.getEulerAngles(np.dot(newrotation, oldrotation))
        return None

    def to_text(self, seperator: str = " ") -> str:
        """Return a text description of the molecule
        Molecule.toText(seperator=" ")

        :param seperator: seperation character
        """
        return (
            seperator.join(
                [
                    self.name,
                    self.shape,
                    str(self.chain),
                    str(self.strand),
                    str(self.index),
                    " ".join(map(str, self.dimensions)),
                    " ".join(map(str, self.position)),
                    " ".join(map(str, self.rotation)),
                ]
            )
            + "\n"
        )

    def to_series(self) -> pd.Series:
        """Convert molecule to a pandas series

        :return: Series representation of the molecule
        """
        return pd.Series(
            {
                "name": self.name,
                "shape": self.shape,
                "chain_idx": self.chain,
                "strand_idx": self.strand,
                "bp_idx": self.index,
                "size_x": self.dimensions[0],
                "size_y": self.dimensions[1],
                "size_z": self.dimensions[2],
                "pos_x": self.position[0],
                "pos_y": self.position[1],
                "pos_z": self.position[2],
                "rot_x": self.rotation[0],
                "rot_y": self.rotation[1],
                "rot_z": self.rotation[2],
            }
        )


# Define some standard molecules
class Guanine(Molecule):
    """
    Guanine molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        super().__init__(
            "Guanine",
            "ellipse",
            GUANINE_SIZE,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )


class Adenine(Molecule):
    """Adenine molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """Constructor"""
        super().__init__(
            "Adenine",
            "ellipse",
            ADENINE_SIZE,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )


class Thymine(Molecule):
    """Thymine molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        super().__init__(
            "Thymine",
            "ellipse",
            THYMINE_SIZE,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )


class Cytosine(Molecule):
    """
    Cytosine molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        super().__init__(
            "Cytosine",
            "ellipse",
            CYTOSINE_SIZE,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )


class DNASugar(Molecule):
    """
    DNASugar molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        super().__init__(
            "Sugar",
            "sphere",
            SUGAR_RADIUS,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )


class Triphosphate(Molecule):
    """
    Triphosphate molecule

    :param strand: strand ID
    :param chain: Chain ID
    :param position: position array (3-vector)
    :param rotation: rotation array (euler angles)
    :param index: base pait index
    """

    def __init__(
        self,
        strand: int = -1,
        chain: int = -1,
        position: np.array = np.zeros(3),
        rotation: np.array = np.zeros(3),
        index: int = 0,
    ):
        """constructor"""
        super().__init__(
            "Phosphate",
            "sphere",
            PHOSPHATE_RADIUS,
            strand=strand,
            chain=chain,
            position=position,
            rotation=rotation,
            index=index,
        )
