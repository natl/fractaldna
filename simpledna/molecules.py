"""
molecules.py

Define the molecules that make up a simple DNA chain
"""
from __future__ import division, unicode_literals, print_function

import numpy as np
from copy import deepcopy

# Physical parameters
GUANINE_RADIUS = 1  # Angstrom
ADENINE_RADIUS = 1  # Angstrom
THYMINE_RADIUS = 1  # Angstrom
CYTOSINE_RADIUS = 1  # Angstrom

SUGAR_RADIUS = 1  # Angstrom
PHOSPHATE_RADIUS = 1  # Angstrom


class Molecule(object):
    def __init__(self, name, shape, dimensions, strand=-1, chain=-1,
                 position=np.zeros(3), rotation=np.zeros(3)):
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
        """
        if type(dimensions) in [int, float]:
            self.dimensions = np.array([deepcopy(dimensions)]*3)
        else:
            assert len(dimensions) == 3, 'Position is invalid'
            self.dimensions = deepcopy(dimensions)

        self.shape = deepcopy(shape)
        self.name = deepcopy(name)
        self.position = deepcopy(position)
        self.rotation = deepcopy(rotation)
        self.strand = -1
        self.chain = -1

    def translate(self, translation):
        """
        Molecule.translate(translation)

        Translate the molecule by (x, y, z)
        """
        self.position = self.position + translation
        return None

    def toText(self, seperator=" "):
        """
        Molecule.toText(seperator=" ")
        Return a text description of the molecule
        """
        return seperator.join([self.name, self.shape, int(self.chain),
                               int(self.strand),
                               " ".join(map(str, self.dimensions)),
                               " ".join(map(str, self.position)),
                               " ".join(map(str, self.rotation))])


# Define some standard molecules
class Guanine(Molecule):
    """
    Guanine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(Guanine, self).__init__("Guanine", "sphere", GUANINE_RADIUS,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation)


# Define some standard molecules
class Adenine(Molecule):
    """
    Adenine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(Adenine, self).__init__("Adenine", "sphere", ADENINE_RADIUS,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation)


# Define some standard molecules
class Thymine(Molecule):
    """
    Thymine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(Thymine, self).__init__("Thymine", "sphere", THYMINE_RADIUS,
                                      strand=strand, chain=chain,
                                      position=position, rotation=rotation)


# Define some standard molecules
class Cytosine(Molecule):
    """
    Cytosine molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(Cytosine, self).__init__("Cytosine", "sphere", CYTOSINE_RADIUS,
                                       strand=strand, chain=chain,
                                       position=position, rotation=rotation)


# Define some standard molecules
class DNASugar(Molecule):
    """
    DNASugar molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(DNASugar, self).__init__("DNASugar", "sphere", SUGAR_RADIUS,
                                       strand=strand, chain=chain,
                                       position=position, rotation=rotation)


# Define some standard molecules
class Triphosphate(Molecule):
    """
    Triphosphate molecule
    """

    def __init__(self, strand=-1, chain=-1, position=np.zeros(3),
                 rotation=np.zeros(3)):
        super(Triphosphate, self).__init__("Triphosphate", "sphere",
                                           PHOSPHATE_RADIUS,
                                           strand=strand, chain=chain,
                                           position=position,
                                           rotation=rotation)
