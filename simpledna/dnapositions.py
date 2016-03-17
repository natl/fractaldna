"""
dnapositions.py

Specification of DNA molecules in a variety of co-ordinate systems
"""
from __future__ import division, unicode_literals, print_function
from copy import deepcopy
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Start by specifying the molecules in cylindrical co-ords
# From Arnott and Hukins, 1972, Biochem and Biophys Res Comms, 47, 6, p1504.

# B-DNA geometry
# Each array contains r (angstroms), theta (deg), z (angstroms) position
PHOSPHATE = {"O1": np.array([ 8.75,  97.4,  3.63]),
             "O2": np.array([10.20,  91.1,  1.86]),
             "O3": np.array([ 8.82, 103.3,  1.29]),
             "P1": np.array([ 8.91,  95.2,  2.08]),
             "O4": np.array([ 7.73,  88.0,  1.83])}

DEOXYRIBOSE = {"C5": np.array([7.70, 79.8,  2.77]),
               "O5": np.array([6.22, 66.0,  1.83]),
               "C4": np.array([7.59, 69.9,  2.04]),
               "C3": np.array([8.20, 69.9,  0.64]),
               "C2": np.array([7.04, 73.2, -0.24]),
               "C1": np.array([5.86, 67.4,  0.47])}

ADENINE = {"N9": np.array([4.63,  76.6, 0.42]),
           "C8": np.array([4.84,  93.0, 0.50]),
           "N7": np.array([3.95, 105.4, 0.43]),
           "C5": np.array([2.74,  94.0, 0.28]),
           "N6": np.array([1.83, 154.0, 0.14]),
           "C6": np.array([1.41, 107.2, 0.15]),
           "N1": np.array([0.86,  40.1, 0.03]),
           "C2": np.array([2.17,  30.6, 0.04]),
           "N3": np.array([3.24,  47.0, 0.16]),
           "C4": np.array([3.33,  70.5, 0.28])}

GUANINE = {"N9": np.array([4.63,  76.6,  0.42]),
           "C8": np.array([4.82,  93.2,  0.50]),
           "N7": np.array([3.92, 105.7,  0.42]),
           "C5": np.array([2.70,  94.0,  0.28]),
           "O6": np.array([1.71, 154.6,  0.13]),
           "C6": np.array([1.39, 109.3,  0.15]),
           "N1": np.array([0.92,  37.9,  0.03]),
           "N2": np.array([3.01,   4.2, -0.10]),
           "C2": np.array([2.28,  28.7,  0.03]),
           "N3": np.array([3.29,  46.7,  0.16]),
           "C4": np.array([3.33,  70.3,  0.28])}

CYTOSINE = {"N1": np.array([4.63,  76.6,  0.42]),
            "C6": np.array([4.99,  92.2,  0.52]),
            "C5": np.array([4.35, 107.0,  0.47]),
            "N4": np.array([2.76, 136.6,  0.27]),
            "C4": np.array([2.94, 110.0,  0.32]),
            "N3": np.array([2.31,  83.9,  0.22]),
            "O2": np.array([3.69,  47.9,  0.18]),
            "C2": np.array([3.40,  67.4,  0.27])}

THYMINE = {"N1": np.array([4.63,  76.6, 0.42]),
           "C6": np.array([5.01,  92.3, 0.52]),
           "Me": np.array([5.40, 119.8, 0.58]),
           "C5": np.array([4.38, 106.9, 0.47]),
           "O4": np.array([2.82, 136.3, 0.27]),
           "C4": np.array([2.98, 111.9, 0.32]),
           "N3": np.array([2.36,  85.2, 0.23]),
           "O2": np.array([3.64,  47.8, 0.18]),
           "C2": np.array([3.42,  67.3, 0.27])}

# Entry of these numbers double-checked 16-MAR-2016 by NJL

# Van der Waals radius of varius elements in Angstrom.
# From Bondi (1964), J. Phys. Chem.; and
# Kammeyer & Whitman (1972), J. Chem. Phys.
RADIUS = {"H": 1.2,
          "C": 1.7,
          "O": 1.4,
          "N": 1.5,
          "P": 1.9,
          "Me": 2.1}

COLORS = {"H": "white",
          "C": "grey",
          "O": "red",
          "N": "skyblue",
          "P": "goldenrod",
          "Me": "grey",
          "PHOSPHATE": "yellow",
          "DEOXYRIBOSE": "black",
          "ADENINE": "orange",
          "GUANINE": "green",
          "CYTOSINE": "red",
          "THYMINE": "blue"}

LETTERS = re.compile("[A-Za-z]+")


def opposite_pair(base):
    if base == THYMINE:
        return ADENINE
    elif base == ADENINE:
        return THYMINE
    elif base == GUANINE:
        return CYTOSINE
    elif base == CYTOSINE:
        return GUANINE
    else:
        return None


def base_name(base):
        if base == THYMINE:
            return "THYMINE"
        elif base == ADENINE:
            return "ADENINE"
        elif base == GUANINE:
            return "GUANINE"
        elif base == CYTOSINE:
            return "CYTOSINE"
        else:
            return None


class MoleculeFromAtoms(object):
    def __init__(self, atoms):
        """
        MoleculeFromAtoms(atoms)
        A molecule created from a dictionary of cartesian atom positions
        """
        self.atoms = deepcopy(atoms)

    @classmethod
    def from_cylindrical(cls, atoms):
        """
        MoleculeFromAtoms.from_cylindrical(atoms)
        Make a MoleculeFromAtoms instance from a list of atoms in cylindrical
        coords (r, theta phi)

        Note:
        Theta is in degrees
        """
        cylindrical = deepcopy(atoms)
        cartesian = {}
        for (name, pos) in cylindrical.items():
            z = pos[2]
            y = pos[0] * np.sin(np.pi * pos[1] / 180.)
            x = pos[0] * np.cos(np.pi * pos[1] / 180.)
            cartesian[name] = np.array([x, y, z])

        return cls(cartesian)

    def find_center(self):
        """
        c = MoleculeFromAtoms.find_center()

        Find the center of the atoms that constitute this molecule
        """
        c = np.zeros([3])
        for pos in self.atoms.values():
            c += pos
        return c / len(self)

    def find_half_lengths(self):
        """
        half_lengths = MoleculeFromAtoms.find_half_lengths()

        Return an array of the half lengths of a box that encloses the molecule
        in xyz. This is not a minimum bounding volume
        """
        c = self.find_center()
        extents = []
        for (atom, pos) in self.atoms.items():
            pos -= c
            rad = RADIUS[LETTERS.match(atom).group()]
            x = (pos[0] - rad) if pos[0] <= 0 else (pos[0] + rad)
            y = (pos[1] - rad) if pos[1] <= 0 else (pos[1] + rad)
            z = (pos[2] - rad) if pos[2] <= 0 else (pos[2] + rad)
            extents.append([x, y, z])

        extents = np.array(extents)
        max_extents = np.max(extents, axis=0)
        min_extents = np.min(extents, axis=0)
        return 0.5 * (max_extents - min_extents)

    def find_radius(self):
        """
        r = MoleculeFromAtoms.find_radius()

        Return the minimum radius that encloses this molecule
        """
        c = self.find_center()
        radii = []
        for (atom, pos) in self.atoms.items():
            pos -= c
            rad = RADIUS[LETTERS.match(atom).group()]
            rad += np.sqrt(np.sum(pos * pos))
            radii.append(rad)

        return np.max(radii)

    def to_plot(self):
        """
        """
        atomsets = {}
        for (atom, pos) in self.atoms.items():
            a = LETTERS.match(atom).group()
            if a not in atomsets:
                rad = RADIUS[a] if a in RADIUS else 2
                col = COLORS[a] if a in COLORS else "blue"
                atomsets[a] = {"radius": rad, "positions": [],
                               "color": col}
            atomsets[a]["positions"].append(pos)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for atomset in atomsets.values():
            pos = np.array(atomset["positions"])
            size = 3.14 * atomset["radius"] * atomset["radius"]
            color = atomset["color"]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20 * size, c=color)

        return fig

    def __len__(self):
        """
        len(MoleculeFromAtoms)
        returns number of atoms
        """
        return self.atoms.__len__()


class DoubleStrand(object):
    """
    Double strand of DNA (for testing)
    """
    def __init__(self):
        sequence = [THYMINE, GUANINE, ADENINE, CYTOSINE, THYMINE] * 2
        self.atoms = {}
        for ii in range(len(sequence)):
            for (k, v) in sequence[ii].items():
                x = v[0]
                y = v[1] + 36 * ii
                z = v[2] + 3.4 * ii
                self.atoms[k + "_LEFTBP" + str(ii)] = np.array([x, y, z])

            for (k, v) in DEOXYRIBOSE.items():
                x = v[0]
                y = v[1] + 36 * ii
                z = v[2] + 3.4 * ii
                self.atoms[k + "_LEFTDO" + str(ii)] = np.array([x, y, z])

            for (k, v) in PHOSPHATE.items():
                x = v[0]
                y = v[1] + 36 * ii
                z = v[2] + 3.4 * ii
                self.atoms[k + "_LEFTPO" + str(ii)] = np.array([x, y, z])

            for (k, v) in opposite_pair(sequence[ii]).items():
                x = v[0]
                y = -v[1] + 36 * ii
                z = -v[2] + 3.4 * ii
                self.atoms[k + "_RIGHTBP" + str(ii)] = np.array([x, y, z])

            for (k, v) in DEOXYRIBOSE.items():
                x = v[0]
                y = -v[1] + 36 * ii
                z = -v[2] + 3.4 * ii
                self.atoms[k + "_RIGHTDO" + str(ii)] = np.array([x, y, z])

            for (k, v) in PHOSPHATE.items():
                x = v[0]
                y = -v[1] + 36 * ii
                z = -v[2] + 3.4 * ii
                self.atoms[k + "_RIGHTPO" + str(ii)] = np.array([x, y, z])

        return None


class DoubleStrandMolecules(object):
    """
    Double strand of DNA (for testing)
    """
    def __init__(self):
        sequence = [THYMINE, GUANINE, ADENINE, CYTOSINE, THYMINE] * 2
        self.atoms = {}
        for ii in range(len(sequence)):
            # left
            bp = sequence[ii]
            name = base_name(bp)

            c = MoleculeFromAtoms(bp).find_center()
            self.atoms[name + "_LEFT" + str(ii)] = self.shift(c, ii)
            c = MoleculeFromAtoms(DEOXYRIBOSE).find_center()
            self.atoms["DEOXYRIBOSE_LEFT" + str(ii)] = self.shift(c, ii)
            c = MoleculeFromAtoms(PHOSPHATE).find_center()
            self.atoms["PHOSPHATE_LEFT" + str(ii)] = self.shift(c, ii)

            # right
            bp = opposite_pair(bp)
            name = base_name(bp)
            c = MoleculeFromAtoms(bp).find_center()
            self.atoms[name + "_RIGHT" + str(ii)] = self.shift(c, ii, inv=True)
            c = MoleculeFromAtoms(DEOXYRIBOSE).find_center()
            self.atoms["DEOXYRIBOSE_RIGHT" + str(ii)] = self.shift(c, ii, inv=True)
            c = MoleculeFromAtoms(PHOSPHATE).find_center()
            self.atoms["PHOSPHATE_RIGHT" + str(ii)] = self.shift(c, ii, inv=True)

        return None

    @staticmethod
    def shift(v, ii, inv=False):
        sign = -1 if inv is True else 1
        x = v[0]
        y = sign * v[1] + 36 * ii
        z = sign * v[2] + 3.4 * ii
        return np.array([x, y, z])
