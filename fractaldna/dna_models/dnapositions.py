"""
dnapositions.py

Specification of DNA molecules in a variety of co-ordinate systems
"""

from typing import Dict, List

import pdb
import re
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from fractaldna.utils.logging import logger

# Start by specifying the molecules in cylindrical co-ords
# From Arnott and Hukins, 1972, Biochem and Biophys Res Comms, 47, 6, p1504.

# B-DNA geometry
# Each array contains r (angstroms), theta (deg), z (angstroms) position
PHOSPHATE = {
    "O1": np.array([8.75, 97.4, 3.63]),
    "O2": np.array([10.20, 91.1, 1.86]),
    "O3": np.array([8.82, 103.3, 1.29]),
    "P1": np.array([8.91, 95.2, 2.08]),
    "O4": np.array([7.73, 88.0, 1.83]),
}

DEOXYRIBOSE = {
    "C5": np.array([7.70, 79.8, 2.77]),
    "O5": np.array([6.22, 66.0, 1.83]),
    "C4": np.array([7.59, 69.9, 2.04]),
    "C3": np.array([8.20, 69.9, 0.64]),
    "C2": np.array([7.04, 73.2, -0.24]),
    "C1": np.array([5.86, 67.4, 0.47]),
}

ADENINE = {
    "N9": np.array([4.63, 76.6, 0.42]),
    "C8": np.array([4.84, 93.0, 0.50]),
    "N7": np.array([3.95, 105.4, 0.43]),
    "C5": np.array([2.74, 94.0, 0.28]),
    "N6": np.array([1.83, 154.0, 0.14]),
    "C6": np.array([1.41, 107.2, 0.15]),
    "N1": np.array([0.86, 40.1, 0.03]),
    "C2": np.array([2.17, 30.6, 0.04]),
    "N3": np.array([3.24, 47.0, 0.16]),
    "C4": np.array([3.33, 70.5, 0.28]),
}

GUANINE = {
    "N9": np.array([4.63, 76.6, 0.42]),
    "C8": np.array([4.82, 93.2, 0.50]),
    "N7": np.array([3.92, 105.7, 0.42]),
    "C5": np.array([2.70, 94.0, 0.28]),
    "O6": np.array([1.71, 154.6, 0.13]),
    "C6": np.array([1.39, 109.3, 0.15]),
    "N1": np.array([0.92, 37.9, 0.03]),
    "N2": np.array([3.01, 4.2, -0.10]),
    "C2": np.array([2.28, 28.7, 0.03]),
    "N3": np.array([3.29, 46.7, 0.16]),
    "C4": np.array([3.33, 70.3, 0.28]),
}

CYTOSINE = {
    "N1": np.array([4.63, 76.6, 0.42]),
    "C6": np.array([4.99, 92.2, 0.52]),
    "C5": np.array([4.35, 107.0, 0.47]),
    "N4": np.array([2.76, 136.6, 0.27]),
    "C4": np.array([2.94, 110.0, 0.32]),
    "N3": np.array([2.31, 83.9, 0.22]),
    "O2": np.array([3.69, 47.9, 0.18]),
    "C2": np.array([3.40, 67.4, 0.27]),
}

THYMINE = {
    "N1": np.array([4.63, 76.6, 0.42]),
    "C6": np.array([5.01, 92.3, 0.52]),
    "Me": np.array([5.40, 119.8, 0.58]),
    "C5": np.array([4.38, 106.9, 0.47]),
    "O4": np.array([2.82, 136.3, 0.27]),
    "C4": np.array([2.98, 111.9, 0.32]),
    "N3": np.array([2.36, 85.2, 0.23]),
    "O2": np.array([3.64, 47.8, 0.18]),
    "C2": np.array([3.42, 67.3, 0.27]),
}

# Entry of these numbers double-checked 16-MAR-2016 by NJL

# Van der Waals radius of varius elements in Angstrom.
# From Bondi (1964), J. Phys. Chem.; and
# Kammeyer & Whitman (1972), J. Chem. Phys.
RADIUS = {"H": 1.2, "C": 1.7, "O": 1.4, "N": 1.5, "P": 1.9, "Me": 2.1}

COLORS = {
    "H": "white",
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
    "THYMINE": "blue",
}

LETTERS = re.compile("[A-Za-z]+")


def opposite_pair(base: Dict[str, np.array]):
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


def base_name(base: Dict[str, np.array]):
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


def overlap_volume(pos1: List, pos2: List, r1: float, r2: float) -> float:
    """
    Calculate overlapping volume of two spheres

    overlap_volume(pos1, pos2, r1, r2)

    :param pos1: Position of first molecule (x, y, z)
    :param pos2: Position of second molecule (x, y, z)
    :param r1: Radius of first molecule
    :param r2: Radius of second molecule
    :return: Volume of overlap.
    """
    d = sum((pos1 - pos2) ** 2) ** 0.5
    # check they overlap
    if d >= (r1 + r2):
        return 0
    # check if one entirely holds the other
    if r1 > (d + r2):  # 2 is entirely contained in one
        return 4.0 / 3.0 * np.pi * r2 ** 3
    if r2 > (d + r1):  # 1 is entirely contained in one
        return 4.0 / 3.0 * np.pi * r1 ** 3

    vol = (
        np.pi
        * (r1 + r2 - d) ** 2
        * (d ** 2 + (2 * d * r1 - 3 * r1 ** 2 + 2 * d * r2 - 3 * r2 ** 2) + 6 * r1 * r2)
    ) / (12 * d)
    return vol


def get_p_values(a, b, c, alpha, beta, gamma):
    """
    Helper function for triple_overlap_volume
    """
    t2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    tabg2 = (
        (a + beta + gamma)
        * (-a + beta + gamma)
        * (a - beta + gamma)
        * (a + beta - gamma)
    )

    t = t2 ** 0.5
    tabg = tabg2 ** 0.5

    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    alpha2 = alpha ** 2
    beta2 = beta ** 2
    gamma2 = gamma ** 2

    p1 = ((b2 - c2 + beta2 - gamma2) ** 2 + (t - tabg) ** 2) / (4 * a2) - alpha2  # NOQA
    p2 = ((b2 - c2 + beta2 - gamma2) ** 2 + (t + tabg) ** 2) / (4 * a2) - alpha2  # NOQA

    return p1, p2


def atanpi(val):
    theta = np.arctan(val)
    if theta < 0:
        theta += np.pi
    return theta


def triple_overlap_volume(pos1, pos2, pos3, r1, r2, r3):
    """
    triple_overlap_volume(pos1, pos2, pos3, r1, r2, r3)

    Calculate volume overlapped by 3 spheres
    From Gibson and Scheraga (1987)

    Note:
    There are cases where this formula doesn't work properly, not documented
    in the paper by Gibson and Scheraga.This corresponds to the case where
    the center of one circle lies between the line joining the point of
    intersection of the three cricles, and the line between the center of the
    two other circles.

    This geometry rarely arises for chemical species, but if negative volumes
    start appearing, or other quantities that seem unlikely, a Monte Carlo
    integration can be used (provided in this package).
    """
    a = sum((pos3 - pos2) ** 2) ** 0.5
    b = sum((pos3 - pos1) ** 2) ** 0.5
    c = sum((pos2 - pos1) ** 2) ** 0.5
    if not ((a <= (r3 + r2)) and (b <= (r3 + r1)) and (c <= (r2 + r1))):
        return 0

    # Check if one sphere entirely contains another
    if (r1 > (b + r3)) or (r1 > (c + r2)):  # Circle 1 encloses circle 2/3
        vol = overlap_volume(pos2, pos3, r2, r3)
        return vol
    elif (r2 > (a + r3)) or (r2 > (c + r1)):  # Circle 2 encloses circle 1/3
        vol = overlap_volume(pos1, pos3, r1, r3)
        return vol
    elif (r3 > (b + r1)) or (r3 > (a + r2)):  # Circle 3 encloses circle 1/2
        vol = overlap_volume(pos1, pos2, r1, r2)
        return vol

    if (r1 > b) and (r2 > a):  # Circle C is enclosed by both others
        logger.debug("Warning:: Circle C's center is interior to A and B")
    if (r1 > c) and (r3 > a):  # Circle B is enclosed by both others
        logger.debug("Warning:: Circle B's center is interior to A and C")
    if (r2 > c) and (r3 > b):  # Circle A is enclosed by both others
        logger.debug("Warning:: Circle A's center is interior to B and C")

    alpha = r1
    beta = r2
    gamma = r3

    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    alpha2 = alpha ** 2
    beta2 = beta ** 2
    gamma2 = gamma ** 2

    eps1 = (beta2 - gamma2) / a2
    eps2 = (gamma2 - alpha2) / b2
    eps3 = (alpha2 - beta2) / c2

    w2 = (
        (alpha2 * a2 + beta2 * b2 + gamma2 * c2) * (a2 + b2 + c2)
        - 2 * (alpha2 * a2 ** 2 + beta2 * b2 ** 2 + gamma2 * c2 ** 2)
        + (a2 * b2 * c2) * (eps1 * eps2 + eps2 * eps3 + eps3 * eps1 - 1)
    )

    if w2 > 0:
        w = w2 ** 0.5
        q1 = a * (b2 + c2 - a2 + beta2 + gamma2 - 2.0 * alpha2 + eps1 * (b2 - c2))
        q2 = b * (c2 + a2 - b2 + gamma2 + alpha2 - 2.0 * beta2 + eps2 * (c2 - a2))
        q3 = c * (a2 + b2 - c2 + alpha2 + beta2 - 2.0 * gamma2 + eps3 * (a2 - b2))

        alpha3 = alpha ** 3.0
        beta3 = beta ** 3.0
        gamma3 = gamma ** 3.0
        aw = a * w
        bw = b * w
        cw = c * w

        vol = (
            w / 6.0
            - a
            / 2.0
            * (beta2 + gamma2 - a2 * (1.0 / 6.0 - eps1 ** 2 / 2.0))
            * atanpi(2 * w / q1)  # NOQA
            - b
            / 2.0
            * (gamma2 + alpha2 - b2 * (1.0 / 6.0 - eps2 ** 2 / 2.0))
            * atanpi(2 * w / q2)  # NOQA
            - c
            / 2.0
            * (alpha2 + beta2 - c2 * (1.0 / 6.0 - eps3 ** 2 / 2.0))
            * atanpi(2 * w / q3)  # NOQA
            + (2.0 / 3.0)
            * alpha3
            * (
                atanpi(bw / (alpha * q2) * (1 - eps2))
                + atanpi(cw / (alpha * q3) * (1 + eps3))
            )  # NOQA
            + (2.0 / 3.0)
            * beta3
            * (
                atanpi(cw / (beta * q3) * (1 - eps3))
                + atanpi(aw / (beta * q1) * (1 + eps1))
            )  # NOQA
            + (2.0 / 3.0)
            * gamma3
            * (
                atanpi(aw / (gamma * q1) * (1 - eps1))
                + atanpi(bw / (gamma * q2) * (1 + eps2))
            )
        )  # NOQA

    elif w2 < 0:
        p1, p2 = get_p_values(a, b, c, alpha, beta, gamma)
        p3, p4 = get_p_values(b, c, a, beta, gamma, alpha)
        p5, p6 = get_p_values(c, a, b, gamma, alpha, beta)

        if (p3 > 0) and (p5 > 0):
            if p1 > 0:
                vol = 0
            if p1 < 0:
                vol = overlap_volume(pos2, pos3, r2, r3)
        elif (p1 > 0) and (p5 > 0):  # fill out...
            if p3 > 0:
                vol = 0
            if p3 < 0:
                vol = overlap_volume(pos1, pos3, r1, r3)
        elif (p1 > 0) and (p3 > 0):
            if p5 > 0:
                vol = 0
            if p5 < 0:
                vol = overlap_volume(pos1, pos2, r1, r2)
        elif (p1 > 0) and (p3 < 0) and (p5 < 0):  # NOQA
            vol = (
                overlap_volume(pos1, pos2, r1, r2)
                + overlap_volume(pos1, pos3, r1, r3)
                - 4.0 / 3.0 * np.pi * r1 ** 3.0
            )
        elif (p1 < 0) and (p3 > 0) and (p5 < 0):  # NOQA
            vol = (
                overlap_volume(pos1, pos2, r1, r2)
                + overlap_volume(pos2, pos3, r2, r3)
                - 4.0 / 3.0 * np.pi * r2 ** 3.0
            )
        elif (p1 < 0) and (p3 < 0) and (p5 > 0):  # NOQA
            vol = (
                overlap_volume(pos1, pos3, r1, r3)
                + overlap_volume(pos2, pos3, r2, r3)
                - 4.0 / 3.0 * np.pi * r3 ** 3.0
            )
        else:
            # Fall back to MCMC calculation
            vol = mc_triple_volume(pos1, pos2, pos3, r1, r2, r3)
    else:
        vol = 0

    return vol


def mc_triple_volume(p1, p2, p3, r1, r2, r3, n=1e5):
    # Generate points inside the box containing smallest circle
    # as this is a constraint
    rs = [r1, r2, r3]
    if r1 == min(rs):
        centres = p1
        ranges = 2 * np.ones([3]) * r1
    elif r2 == min(rs):
        centres = p2
        ranges = 2 * np.ones([3]) * r2
    elif r3 == min(rs):
        centres = p3
        ranges = 2 * np.ones([3]) * r3

    in_overlap = 0.0
    in_circle = lambda p, c, r: sum((p - c) ** 2.0) ** 0.5 < r
    for ii in range(int(n)):
        position = (np.random.random(3) - 0.5) * ranges + centres
        in1 = in_circle(position, p1, r1)
        in2 = in_circle(position, p2, r2)
        in3 = in_circle(position, p3, r3)
        if in1 and in2 and in3:
            in_overlap += 1
    vol_total = np.product(ranges)
    return vol_total * in_overlap / n


class MoleculeFromAtoms:
    def __init__(self, atoms: Dict[str, np.array]):
        """
        MoleculeFromAtoms(atoms)
        A molecule created from a dictionary of cartesian atom positions
        """
        self.atoms = deepcopy(atoms)

    @classmethod
    def from_cylindrical(cls, atoms: Dict[str, np.array], inverse: bool = False):
        """Make a MoleculeFromAtoms instance from a list of atoms in cylindrical
        coords (r, theta, phi)

        MoleculeFromAtoms.from_cylindrical(atoms, inverse=False)

        Note:
        Theta is in degrees

        :param inverse: (default False)Set to True to generate a dyadically related
            base pair (negates theta and z)
        """
        cylindrical = deepcopy(atoms)
        cartesian = {}
        sgn = 1 if inverse is False else -1
        for (name, pos) in cylindrical.items():
            z = sgn * pos[2]  # NOQA
            y = pos[0] * np.sin(sgn * np.pi * pos[1] / 180.0)
            x = pos[0] * np.cos(sgn * np.pi * pos[1] / 180.0)
            cartesian[name] = np.array([x, y, z])

        return cls(cartesian)

    def find_center(self) -> float:
        """
        c = MoleculeFromAtoms.find_center()

        Find the barycenter of the atoms that constitute this molecule
        """
        c = np.zeros([3])
        denom = 0
        for (atom, pos) in self.atoms.items():
            r = RADIUS[LETTERS.match(atom).group()]
            c += pos * r
            denom += r
        return c / denom

    def find_half_lengths(self) -> np.array:
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

    def find_equivalent_half_lengths(self) -> np.array:
        """
        l = MoleculeFromAtoms.find_equivalent_half_lengths()

        Find the half lengths scaled to give a volume equal to what the
        constituent molecules occupy
        """
        half_lengths = self.find_half_lengths()
        max_volume = 4.0 / 3.0 * np.pi * np.product(half_lengths)
        equiv_volume = 4.0 / 3.0 * np.pi * self.find_equivalent_radius() ** 3.0
        return half_lengths * (equiv_volume / max_volume) ** (1.0 / 3.0)

    def find_radius(self) -> float:
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

    def find_equivalent_radius(self) -> float:
        """
        r = MoleculeFromAtoms.find_equivalent_radius()

        Return the radius that yields the same volume occupied by all atoms
        """
        vol = 0
        for (atom, pos) in self.atoms.items():
            rad = RADIUS[LETTERS.match(atom).group()]
            vol += 4.0 / 3.0 * np.pi * rad ** 3
        # subtract double overlaps
        for ((a1, p1), (a2, p2)) in combinations(self.atoms.items(), 2):
            r1 = RADIUS[LETTERS.match(a1).group()]
            r2 = RADIUS[LETTERS.match(a2).group()]
            vol -= overlap_volume(p1, p2, r1, r2)
        for ((a1, p1), (a2, p2), (a3, p3)) in combinations(self.atoms.items(), 3):
            # pdb.set_trace()
            r1 = RADIUS[LETTERS.match(a1).group()]
            r2 = RADIUS[LETTERS.match(a2).group()]
            r3 = RADIUS[LETTERS.match(a3).group()]
            vol += triple_overlap_volume(p1, p2, p3, r1, r2, r3)
        return (vol * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)

    def to_plot(self) -> plt.Figure:
        """
        fig = MoleculeFromAtoms.to_plot()

        Returns a matplotlib figure instance of the molecule.
        """
        atomsets = {}
        for (atom, pos) in self.atoms.items():
            a = LETTERS.match(atom).group()
            if a not in atomsets:
                rad = RADIUS[a] if a in RADIUS else 2
                col = COLORS[a] if a in COLORS else "blue"
                atomsets[a] = {"radius": rad, "positions": [], "color": col}
            atomsets[a]["positions"].append(pos)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for atomset in atomsets.values():
            pos = np.array(atomset["positions"])
            size = 3.14 * atomset["radius"] * atomset["radius"]
            color = atomset["color"]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20 * size, c=color)

        return fig

    def __len__(self) -> int:
        """
        len(MoleculeFromAtoms)
        returns number of atoms
        """
        return self.atoms.__len__()


class MoleculeDictionary:
    """
    Base class for test molecules
    """

    def __init__(self):
        self.atoms = {}

    def items(self):
        return self.atoms.items()


class DoubleStrand(MoleculeDictionary):
    """
    Double strand of DNA (for testing)
    """

    def __init__(self):
        super().__init__()
        sequence = [THYMINE, GUANINE, ADENINE, CYTOSINE, THYMINE] * 2
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


class DoubleStrandMolecules(MoleculeDictionary):
    """
    Double strand of DNA (for testing)
    """

    def __init__(self):
        super().__init__()
        sequence = [THYMINE, GUANINE, ADENINE, CYTOSINE, THYMINE] * 2
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
