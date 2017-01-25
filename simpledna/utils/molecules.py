"""
molecules.py

Define the molecules that make up a simple DNA chain
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
from copy import deepcopy
from . import dnapositions as dpos
from . import rotations as rot

# Physical parameters, all in Angstrom
GUANINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(dpos.GUANINE)\
    .find_equivalent_half_lengths()
ADENINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(dpos.ADENINE)\
    .find_equivalent_half_lengths()
THYMINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(dpos.THYMINE)\
    .find_equivalent_half_lengths()
CYTOSINE_SIZE = dpos.MoleculeFromAtoms.from_cylindrical(dpos.CYTOSINE)\
    .find_equivalent_half_lengths()

# GUANINE_SIZE[2] = 0.1*GUANINE_SIZE[2]
# ADENINE_SIZE[2] = 0.1*ADENINE_SIZE[2]
# THYMINE_SIZE[2] = 0.1*THYMINE_SIZE[2]
# CYTOSINE_SIZE[2] = 0.1*CYTOSINE_SIZE[2]

SUGAR_RADIUS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.DEOXYRIBOSE)\
    .find_equivalent_radius()
PHOSPHATE_RADIUS = dpos.MoleculeFromAtoms.from_cylindrical(dpos.PHOSPHATE)\
    .find_equivalent_radius()


class Molecule(object):
    def __init__(self, name, shape, dimensions, strand=-1, chain=-1,
                 position=np.zeros(3), rotation=np.zeros(3), index=0):
        """
        Molecule(name, shape, dimensions, strand=-1, chain=-1,
                 position=np.zeros(3), rotation=np.zeros(3)):

        Create a molecule
        args:
            name:       molecule name
            shape:      shape of molecule
            dimensions: dimensions of molecule shape
                        (3-vector for xyz-dims, int/float for radius)

        kwargs:
            strand:     int to hold an ID related to a localised strand
            chain:      int to hold an ID related to a macromolecule/protein or
                        DNA chain
            position:   3-vector for molecule position relative to global axis
            rotation:   3-vector of euler angles (radians) for molecule
                        rotation relative to the global xyz axis
            index:      index of base pair along chain
        """
        if type(dimensions) in [int, float, np.float64]:
            self.dimensions = np.array([deepcopy(dimensions)] * 3)
        else:
            assert len(dimensions) == 3, 'Position is invalid'
            self.dimensions = deepcopy(dimensions)

        self.shape = deepcopy(shape)
        self.name = deepcopy(name)
        self.position = deepcopy(position)
        self.rotation = deepcopy(rotation)
        self.strand = strand
        self.chain = chain
        self.index = index

    def translate(self, translation):
        """
        Molecule.translate(translation)

        Translate the molecule by (x, y, z)
        """
        self.position = self.position + translation
        return None

    def rotate(self, rotation):
        """
        Molecule.rotate(rotation)

        Rotate molecule by [X_angle, Y_angle, Z_angle]
        """
        assert len(rotation) == 3, "Rotation array must be length-3"
        oldrotation = rot.eulerMatrix(*self.rotation)
        newrotation = rot.eulerMatrix(*rotation)
        self.rotation = rot.getEulerAngles(np.dot(newrotation, oldrotation))
        return None

    def to_text(self, seperator=" "):
        """
        Molecule.toText(seperator=" ")
        Return a text description of the molecule
        """
        return seperator.join([self.name, self.shape, str(self.chain),
                               str(self.strand), str(self.index),
                               " ".join(map(str, self.dimensions)),
                               " ".join(map(str, self.position)),
                               " ".join(map(str, self.rotation))]) + "\n"


# Define some standard molecules
class Guanine(Molecule):
    """
    Guanine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(Guanine, self).__init__("Guanine", "ellipse", GUANINE_SIZE,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation,
                                      index=index)


class Adenine(Molecule):
    """
    Adenine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(Adenine, self).__init__("Adenine", "ellipse", ADENINE_SIZE,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation,
                                      index=index)


class Thymine(Molecule):
    """
    Thymine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(Thymine, self).__init__("Thymine", "ellipse", THYMINE_SIZE,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation,
                                      index=index)


class Cytosine(Molecule):
    """
    Cytosine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(Cytosine, self).__init__("Cytosine", "ellipse", CYTOSINE_SIZE,
                                       strand=strand, chain=chain,
                                       position=position, rotation=rotation,
                                       index=index)


class DNASugar(Molecule):
    """
    DNASugar molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(DNASugar, self).__init__("Sugar", "sphere", SUGAR_RADIUS,
                                       strand=strand, chain=chain,
                                       position=position, rotation=rotation,
                                       index=index)


class Triphosphate(Molecule):
    """
    Triphosphate molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3), index=0):
        super(Triphosphate, self).__init__("Phosphate", "sphere",
                                           PHOSPHATE_RADIUS,
                                           strand=strand, chain=chain,
                                           position=position,
                                           rotation=rotation, index=index)
