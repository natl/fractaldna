import random
import unittest
from itertools import permutations

import numpy as np

from fractaldna.structure_models import hilbert as h
from fractaldna.structure_models import voxelisation as vxl


def rotz(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def roty(angle):
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rotx(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def eulerMatrix(angx, angy, angz):
    return np.dot(rotz(angz), np.dot(roty(angy), rotx(angx)))


class TestFractalCreation(unittest.TestCase):
    """ """

    def check_from_seed(self, seed, iterations):
        """
        Test if the fractal is correctly generated by measuring whether
        points are uniformly distributed within subvolumes.
        """
        for nreps in range(1, iterations):
            s = seed
            for ii in range(nreps):
                s = h.iterate_lstring(s)
            arrpath = np.array(h.generate_path(s, distance=1, n=1))
            mins = np.zeros(3)
            maxs = np.zeros(3)
            for ii in range(3):
                mins[ii] = min(arrpath[:, ii])
                maxs[ii] = max(arrpath[:, ii])
            counts = []
            boxsize = (maxs - mins) / nreps
            # Bottom Corner Range
            botCorRange = maxs - mins - boxsize
            for ii in range(10):
                botCor = np.array(
                    [
                        np.random.rand() * botCorRange[0] + mins[0],
                        np.random.rand() * botCorRange[1] + mins[1],
                        np.random.rand() * botCorRange[2] + mins[2],
                    ]
                )
                topCor = botCor + boxsize
                count = 0
                for point in arrpath:
                    if (point >= botCor).all() and (point <= topCor).all():
                        count = count + 1
                counts.append(count)

            totalVolume = (
                (maxs[0] - mins[0]) * (maxs[1] - mins[1]) * (maxs[2] - mins[2])
            )
            boxVolume = boxsize[0] * boxsize[1] * boxsize[2]
            countsExpected = boxVolume / totalVolume * len(arrpath)
            for count in counts:
                self.assertGreater(
                    count,
                    (1 - nreps) / nreps * countsExpected,
                    f"Counts were low for seed {seed}\n"
                    f"N_iterations: {nreps}\n"
                    f"Counts/Expected: {count}/{countsExpected}",
                )
                self.assertLess(
                    count,
                    (1 + nreps) / nreps * countsExpected,
                    f"Counts were high for seed {seed}\n"
                    f"N_iterations: {nreps}\n"
                    f"Counts/Expected: {count}/{countsExpected}",
                )

        return None

    def test_hilbert(self):
        """
        Test if the fractal is correctly generated by measuring whether
        points are uniformly distributed within subvolumes.
        """
        seeds = [h.X, "XFX", h.A, h.B, h.C, h.D]
        for seed in seeds:
            self.check_from_seed(seed, 5)
        return None

    def test_peano(self):
        """
        Test if the fractal is correctly generated by measuring whether
        points are uniformly distributed within subvolumes.
        """
        seeds = [h.P]  # , h.B, h.C, h.D, h.X]
        for seed in seeds:
            self.check_from_seed(seed, 4)
        return None


class TestVoxelCreation(unittest.TestCase):
    """
    Test the hilbert.Voxel class
    """

    xdir = np.array([1, 0, 0])
    ydir = np.array([0, 1, 0])
    zdir = np.array([0, 0, 1])
    dirs = {"x": xdir, "y": ydir, "z": zdir}
    types = vxl.Voxel.types

    def test_makestraight(self):
        pos = np.array(
            [
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
            ]
        )
        for (heading, principal) in permutations(self.dirs, 2):
            if heading != principal:
                vox = vxl.Voxel(
                    pos,
                    self.dirs[heading],
                    self.dirs[principal],
                    self.dirs[heading],
                    self.dirs[principal],
                )
                self.assertEqual(vox.type, self.types["straight"], "straight")

    def test_makestraighttwist(self):
        pos = np.array(
            [
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
            ]
        )
        for (heading, principal) in permutations(self.dirs, 2):
            if heading != principal:
                d = "xyz".replace(heading, "").replace(principal, "")
                outPrincipal = self.dirs[d]
                vox = vxl.Voxel(
                    pos,
                    self.dirs[heading],
                    self.dirs[principal],
                    self.dirs[heading],
                    outPrincipal,
                )
                self.assertEqual(vox.type, self.types["straighttwist"], "straighttwist")

    def test_turns(self):
        pos = np.array(
            [
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
                10 * (random.random() - 0.5),
            ]
        )
        for (inh, inp, outh, outp) in permutations(self.dirs, 4):
            valid = (inh != outh) and (inh != inp) and (outh != outp)
            if valid:
                vox = vxl.Voxel(
                    pos,
                    self.dirs[inh],
                    self.dirs[inp],
                    self.dirs[outh],
                    self.dirs[outp],
                )
                if inp == outp:
                    self.assertEqual(vox.type, self.types["turn"], "turn")
                else:
                    self.assertEqual(vox.type, self.types["turntwist"], "turntwist")

    def test_angles_turn1(self):
        newVoxel = vxl.Voxel(
            np.array([0, 0, 0]),
            self.dirs["y"],
            self.dirs["x"],
            -self.dirs["z"],
            self.dirs["x"],
        )
        angles = np.around(np.array([-np.pi / 2.0, np.pi / 2.0, 0.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        return None

    def test_angles_turn2(self):
        newVoxel = vxl.Voxel(
            np.array([0, 0, 0]),
            self.dirs["z"],
            self.dirs["x"],
            self.dirs["y"],
            self.dirs["x"],
        )
        angles = np.around(np.array([0.0, 0.0, np.pi / 2.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        return None

    def test_angles_straights(self):
        origin = np.array([0, 0, 0])
        # Positive Z
        newVoxel = vxl.Voxel(
            origin, self.dirs["z"], self.dirs["x"], self.dirs["z"], self.dirs["x"]
        )
        angles = np.around(np.array([0.0, 0.0, 0.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        # Negative Z
        newVoxel = vxl.Voxel(
            origin, -self.dirs["z"], self.dirs["x"], -self.dirs["z"], self.dirs["x"]
        )
        angles = np.around(np.array([np.pi, 0.0, 0.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        # Positive X
        newVoxel = vxl.Voxel(
            origin, self.dirs["x"], self.dirs["y"], self.dirs["x"], self.dirs["y"]
        )
        angles = np.around(np.array([np.pi / 2.0, 0, np.pi / 2.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            ((angles == voxelAngles).all() or (angles == -voxelAngles).all()),
            "voxel has incorrect "
            + "euler angles,"
            + "\nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        # Negative X
        newVoxel = vxl.Voxel(
            origin, -self.dirs["x"], self.dirs["y"], -self.dirs["x"], self.dirs["y"]
        )
        angles = np.around(np.array([-np.pi / 2.0, 0, np.pi / 2.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all() or (angles == -voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles,"
            + "\nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        # Positive Y
        newVoxel = vxl.Voxel(
            origin, self.dirs["y"], self.dirs["x"], self.dirs["y"], self.dirs["x"]
        )
        angles = np.around(np.array([-np.pi / 2.0, 0.0, 0.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )

        # Negative Y
        newVoxel = vxl.Voxel(
            origin, -self.dirs["y"], self.dirs["x"], -self.dirs["y"], self.dirs["x"]
        )
        angles = np.around(np.array([+np.pi / 2.0, 0.0, 0.0]), 8)
        voxelAngles = np.around(
            np.array([newVoxel.psi, newVoxel.theta, newVoxel.phi]), 8
        )
        self.assertTrue(
            (angles == voxelAngles).all(),
            "voxel has incorrect "
            + "euler angles, \nExpected: "
            + str(angles)
            + "\nFound:   "
            + str(voxelAngles),
        )


class TestVoxFractalMakeVoxelMethod(unittest.TestCase):
    """
    Test when voxels are made within the VoxelisedFractal class as part of the
    makeVoxel static method

    No rotation matrix tests are implemented yet, but these are tested
    under the VoxelisedFractal tests
    """

    xdir = np.array([1, 0, 0])
    ydir = np.array([0, 1, 0])
    zdir = np.array([0, 0, 1])
    dirs = {"x": xdir, "y": ydir, "z": zdir}
    types = vxl.Voxel.types
    reverseTypes = {val: key for (key, val) in types.items()}
    pos1 = np.array([0, 0, 0])
    vox1 = vxl.Voxel(pos1, dirs["x"], dirs["y"], dirs["x"], dirs["y"])

    def test_straight(self):
        currpos = self.pos1 + self.dirs["x"]
        nextpos = currpos + self.dirs["x"]
        newVoxel = vxl.VoxelisedFractal._makeVoxel(self.vox1, currpos, nextpos)
        self.assertTrue((newVoxel.pos == currpos).all(), "newVoxel position mismatch")
        self.assertTrue(
            (newVoxel.inHeading == self.dirs["x"]).all(), "newVoxel inHeading mismatch"
        )
        self.assertTrue(
            (newVoxel.outHeading == self.dirs["x"]).all(),
            "newVoxel outHeading mismatch",
        )
        self.assertTrue(
            (newVoxel.inPrincipal == self.dirs["y"]).all(),
            "newVoxel inPrincipal mismatch",
        )
        self.assertTrue(
            (newVoxel.outPrincipal == self.dirs["y"]).all(),
            "newVoxel outPrincipal mismatch",
        )
        self.assertEqual(
            newVoxel.type,
            self.types["straight"],
            "expected " + "straight, got " + self.reverseTypes[newVoxel.type],
        )

        return None

    def test_turn(self):
        currpos = self.pos1 + self.dirs["x"]
        nextpos = currpos + self.dirs["z"]
        newVoxel = vxl.VoxelisedFractal._makeVoxel(self.vox1, currpos, nextpos)
        self.assertTrue((newVoxel.pos == currpos).all(), "newVoxel position mismatch")
        self.assertTrue(
            (newVoxel.inHeading == self.dirs["x"]).all(), "newVoxel inHeading mismatch"
        )
        self.assertTrue(
            (newVoxel.outHeading == self.dirs["z"]).all(),
            "newVoxel outHeading mismatch",
        )
        self.assertTrue(
            (newVoxel.inPrincipal == self.dirs["y"]).all(),
            "newVoxel inPrincipal mismatch: " + str(newVoxel.inPrincipal),
        )
        self.assertTrue(
            (newVoxel.outPrincipal == -self.dirs["y"]).all(),
            "newVoxel outPrincipal mismatch: " + str(newVoxel.outPrincipal),
        )
        self.assertEqual(
            newVoxel.type,
            self.types["turn"],
            "expected turn, " "got " + self.reverseTypes[newVoxel.type],
        )

        return None

    def test_turntwist(self):
        currpos = self.pos1 + self.dirs["x"]
        nextpos = currpos + self.dirs["y"]
        newVoxel = vxl.VoxelisedFractal._makeVoxel(self.vox1, currpos, nextpos)
        self.assertTrue((newVoxel.pos == currpos).all(), "newVoxel position mismatch")
        self.assertTrue(
            (newVoxel.inHeading == self.dirs["x"]).all(), "newVoxel inHeading mismatch"
        )
        self.assertTrue(
            (newVoxel.outHeading == self.dirs["y"]).all(),
            "newVoxel outHeading mismatch",
        )
        self.assertTrue(
            (newVoxel.inPrincipal == self.dirs["y"]).all(),
            "newVoxel inPrincipal mismatch: " + str(newVoxel.inPrincipal),
        )
        self.assertTrue(
            (newVoxel.outPrincipal == self.dirs["z"]).all(),
            "newVoxel outPrincipal mismatch: " + str(newVoxel.outPrincipal),
        )
        self.assertEqual(
            newVoxel.type,
            self.types["turntwist"],
            "expected " + "turntwist, got " + self.reverseTypes[newVoxel.type],
        )

        return None


class TestVoxelisedFractalCreation(unittest.TestCase):
    """
    Test creation of the voxelised fractal, with simple test cases
    """

    types = vxl.Voxel.types
    reverseTypes = {val: key for (key, val) in types.items()}

    def test_empty(self):
        frac = vxl.VoxelisedFractal()
        self.assertEqual(frac.fractal, [])

        return None

    def test_straightLString(self):
        s = "F" * 10
        frac = vxl.VoxelisedFractal.fromLString(s)
        for vox in frac.fractal:
            self.assertEqual(vox.type, self.types["straight"])

        return None

    def test_arbitraryFractalLength(self):
        s = r"n<XFn<XFX-Fn>>XFX&F+>>XFX-F>X->"
        for ii in range(2):
            s = h.iterate_lstring(s)
        frac = vxl.VoxelisedFractal.fromLString(s)
        nvoxels = 1
        for char in s:
            if char == "F":
                nvoxels += 1
        self.assertEqual(len(frac.fractal), nvoxels, "checking voxel length")

        return None

    def test_simpleTurnLString(self):
        s = r"FFFnFFFnFFF"
        frac = vxl.VoxelisedFractal.fromLString(s)
        nvoxels = 1
        for char in s:
            if char == "F":
                nvoxels += 1
        self.assertEqual(len(frac.fractal), nvoxels, "checking voxel length")
        exp = (
            ["straight"] * 3 + ["turn"] + ["straight"] * 2 + ["turn"] + ["straight"] * 3
        )

        f = frac.fractal
        for ii in range(len(f)):
            self.assertEqual(
                f[ii].type,
                self.types[exp[ii]],
                "expected " + exp[ii] + ", got " + self.reverseTypes[f[ii].type],
            )

        return None

    def test_simpleHilbert(self):
        s = r"n<XFn<XFX-Fn>>XFX&F+>>XFX-F>X->"
        frac = vxl.VoxelisedFractal.fromLString(s)
        nvoxels = 1
        for char in s:
            if char == "F":
                nvoxels += 1
        self.assertEqual(len(frac.fractal), nvoxels, "checking voxel length")
        exp = [
            "straight",
            "turn",
            "turn",
            "turntwist",
            "turn",
            "turntwist",
            "turn",
            "straight",
        ]

        f = frac.fractal
        for ii in range(len(f)):
            self.assertEqual(
                f[ii].type,
                self.types[exp[ii]],
                "expected "
                + exp[ii]
                + " for voxel "
                + str(ii)
                + ", got "
                + self.reverseTypes[f[ii].type],
            )

        return None

    def test_simpleHilbertRotations(self):
        s = r"n<XFn<XFX-Fn>>XFX&F+>>XFX-F>X->"
        frac = vxl.VoxelisedFractal.fromLString(s)
        nvoxels = 1
        for char in s:
            if char == "F":
                nvoxels += 1
        self.assertEqual(len(frac.fractal), nvoxels, "checking voxel length")
        exp = [
            eulerMatrix(0, 0, 0),
            eulerMatrix(0, 0, np.pi / 2),
            eulerMatrix(-np.pi / 2.0, np.pi / 2.0, 0.0),
            eulerMatrix(np.pi, 0, 0),
            eulerMatrix(np.pi, -np.pi / 2, 0),
            eulerMatrix(0, 0, -np.pi / 2),
            eulerMatrix(0, np.pi / 2, -np.pi / 2),
            eulerMatrix(np.pi, 0, 0),
        ]

        f = frac.fractal
        for ii in range(len(f)):
            tested = np.around(f[ii].rotation, 8)
            expect = np.around(exp[ii], 8)
            self.assertTrue(
                (tested == expect).all(),
                "Unequal rotations in voxel "
                + str(ii)
                + "\nvoxel:\n"
                + str(tested)
                + "\nexpected:\n"
                + str(expect),
            )

        return None

    def test_simpleHilbertEulers(self):
        s = r"n<XFn<XFX-Fn>>XFX&F+>>XFX-F>X->"
        frac = vxl.VoxelisedFractal.fromLString(s)
        nvoxels = 1
        for char in s:
            if char == "F":
                nvoxels += 1
        self.assertEqual(len(frac.fractal), nvoxels, "checking voxel length")
        exp = [
            (0, 0, 0),
            (0, 0, np.pi / 2),
            (-np.pi / 2.0, np.pi / 2.0, 0.0),
            (np.pi, 0, 0),
            (np.pi, -np.pi / 2, 0),
            (0, 0, -np.pi / 2),
            (0, np.pi / 2, -np.pi / 2),
            (np.pi, 0, 0),
        ]

        f = frac.fractal
        for ii in range(len(f)):
            v = f[ii]
            tested = np.around(np.array([v.psi, v.theta, v.phi]), 8)
            # Make all -pi into +pi
            for jj in range(len(tested)):
                if tested[jj] == -round(np.pi, 8):
                    tested[jj] = round(np.pi, 8)

            expect = np.around(np.array(exp[ii]), 8)
            if abs(tested[1]) == round(np.pi / 2.0, 8):
                self.assertTrue(
                    tested[1] == expect[1],
                    "Unequal rotations in voxel "
                    + str(ii)
                    + "\nvoxel:\n"
                    + str(tested)
                    + "\nexpected:\n"
                    + str(expect),
                )
                self.assertTrue(
                    tested[2] - tested[0] == expect[2] - expect[0],
                    "Unequal rotations in voxel "
                    + str(ii)
                    + "\nvoxel:\n"
                    + str(tested)
                    + "\nexpected:\n"
                    + str(expect),
                )
            else:
                self.assertTrue(
                    (tested == expect).all(),
                    "Unequal rotations in voxel "
                    + str(ii)
                    + "\nvoxel:\n"
                    + str(tested)
                    + "\nexpected:\n"
                    + str(expect),
                )

        return None


class TestEulerAngleGeneration(unittest.TestCase):
    def test_recoverEulerAngles(self):
        for ii in range(100):
            angles = np.pi * (np.random.random(3) - 0.5)
            oldpsi = angles[0]
            oldtheta = angles[1]
            oldphi = angles[2]

            newpsi, newtheta, newphi = vxl.getEulerAngles(
                eulerMatrix(oldpsi, oldtheta, oldphi)
            )
            newangles = np.around(np.array([newpsi, newtheta, newphi]), 8)
            oldangles = np.around(np.array([oldpsi, oldtheta, oldphi]), 8)

            self.assertTrue(
                (newangles == oldangles).all(),
                "Euler angles not recovered for: "
                + str(oldangles)
                + "Instead, I got: "
                + str(newangles),
            )


class TestTextOutput(unittest.TestCase):
    types_inverse = {v: k for (k, v) in vxl.Voxel.types.items()}

    def test_straightPath(self):
        s = "F" * 10
        frac = vxl.VoxelisedFractal.fromLString(s)
        text = frac.to_text()
        text = text.split("\n")[1:]
        for (vtext, voxel) in zip(text, frac.fractal):
            vtext = " ".join(vtext.split(" ")[1:])  # remove index
            self.assertEqual(vtext, voxel.to_text(), "voxel text output incorrect")
            self.assertEqual(
                vtext,
                " ".join(
                    [self.types_inverse[voxel.type]]
                    + list(map(str, list(voxel.pos)))
                    + list(map(str, [voxel.psi, voxel.theta, voxel.phi]))
                ),
                "cannot reconstruct voxel text",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
