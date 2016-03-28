from __future__ import division, print_function, unicode_literals

import unittest
import dnachain
import basepair
import dnapositions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA
import pdb


def triple_sphere_plot(p1, p2, p3, r1, r2, r3):
    """
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (r, p, c) in zip([r1, r2, r3], [p1, p2, p3], ['r', 'g', 'b']):
        x, y, z = ellipse_xyz(p, np.ones([3]) * r)
        ax.plot_wireframe(x, y, z, color=c)
    fig.show()

    return fig


def ellipse_xyz(center, extent):
    [a, b, c] = extent
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = a * np.cos(u) * np.sin(v) + center[0]
    y = b * np.sin(u) * np.sin(v) + center[1]
    z = c * np.cos(v) + center[2]

    return x, y, z


class TestDNACreation(unittest.TestCase):
    """
    """
    def test_dnaLength(self):
        """
        """
        genome = "GCGCGCGCTATATATATCTCTCACACA"
        x = dnachain.DNAChain(genome)
        self.assertEqual(len(x.basepairs), len(genome),
                         "Genome is incorrect length")
        nMolecules = sum([len(bp.molecules) for bp in x.basepairs])
        self.assertEqual(nMolecules, len(genome)*6,
                         "Incorrect number of molecules")


class TestBasePairs(unittest.TestCase):
    """
    """
    def test_basepair_creation_rotation_translation(self):
        translation = np.array([1, 2, 3])
        rotation = np.array([0.5, 0.25, 1])*np.pi
        bp1 = basepair.BasePair("G")
        bp2 = basepair.BasePair("G", position=translation,
                                rotation=rotation)
        bp1.rotate(rotation)
        bp1.translate(translation)
        for name in basepair.BasePair.moleculeNames:
            self.assertAlmostEqual(
                np.sum(bp1[name].position - bp2[name].position), 0)
            self.assertAlmostEqual(
                np.sum(bp1[name].rotation - bp2[name].rotation), 0)


class TestPositions(unittest.TestCase):
    """
    """
    def test_overlaps_non_overlapping(self):
        pos1 = np.array([0, 0, 0])
        pos2 = np.array([2, 0, 0])
        pos3 = np.array([0, 0, 2])
        r1 = 1
        r2 = 1
        r3 = 1
        vol = dnapositions.triple_overlap_volume(pos1, pos2, pos3, r1, r2, r3)

        self.assertEqual(vol, 0, "Overlap found when not possible")
        return None

    def test_overlaps(self):
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 0, 0])
        p3 = np.array([0, 1, 0])

        for ii in range(100):
            r1 = 1 * np.random.random()
            r2 = 1 * np.random.random()
            r3 = 1 * np.random.random()
            vol = dnapositions.triple_overlap_volume(p1, p2, p3, r1, r2, r3)

        self.assertGreaterEqual(vol, 0, "Volume is zero")
        return None

    def test_research_paper_value(self):
        p1 = np.array([2. * np.cos(np.arccos(11. / 16.)),
                       2. * np.sin(np.arccos(11. / 16.)), 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([4, 0, 0])
        d12 = np.sum((p1 - p2) ** 2) ** 0.5
        d23 = np.sum((p2 - p3) ** 2) ** 0.5
        d13 = np.sum((p1 - p3) ** 2) ** 0.5
        r1 = 1
        r2 = 2
        r3 = 3
        vol = dnapositions.triple_overlap_volume(p1, p2, p3, r1, r2, r3)
        v_mc = dnapositions.mc_triple_volume(p1, p2, p3, r1, r2, r3)
        v = np.round(vol, 4)
        print("d12: {0} r1: {1}".format(d12, r1))
        print("d23: {0} r2: {1}".format(d23, r2))
        print("d13: {0} r3: {1}".format(d13, r3))
        print("Volume: {0}".format(v))
        print("MC Volume: {0}".format(v_mc))
        self.assertEqual(v, 0.5737, "Volume differs to " +
                         "paper, got {}".format(v))
        return None


if __name__ == '__main__':
    # suite = unittest.TestSuite(None)
    # suite.addTest(TestPositions('test_overlaps_non_overlapping'))
    # suite.run()
    unittest.main(verbosity=2)
